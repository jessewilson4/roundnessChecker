"""
# === UAIPCS START ===
file: utils/database.py
purpose: SQLite database for caching search results, managing history, and tracking keyword approval status
deps: [@sqlite3:library, @json:library, @datetime:library]
funcs:
  - save_search(search_term:str, results:list, stats:dict, batch_id:int) -> int  # side_effect: writes to database
  - get_search(search_id:int) -> dict  # no_side_effect
  - get_search_history(limit:int) -> list  # no_side_effect
  - update_keyword_status(keyword:str, status:str, images_valid:int, images_rejected:int, pagination_offset:int) -> None  # side_effect: updates status
  - get_keywords_needing_images(batch_id:int) -> list  # no_side_effect
  - approve_keyword(keyword:str) -> None  # side_effect: sets approved status
  - reject_image(search_id:int, image_id:str, reason:str) -> None  # side_effect: marks image rejected
classes:
  - Database  # manages SQLite connection and operations
refs:
notes: schema=searches+batches+keyword_status+images, persist=durable, thread_safety=connection_per_operation
# === UAIPCS END ===
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import os


class Database:
    """
    Manages SQLite database for search caching, history, and word-level approval tracking.
    
    Phase 2 additions:
    - keyword_status table for tracking word approval state
    - Image-level rejection tracking
    - Methods for incomplete keyword detection
    """
    
    def __init__(self, db_path: str = './cache/searches.db'):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    
    def _init_database(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Searches table with new multi-metric schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_term TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                num_images INTEGER,
                avg_circularity REAL,
                avg_aspect_ratio REAL,
                avg_eccentricity REAL,
                avg_solidity REAL,
                avg_convexity REAL,
                avg_composite REAL,
                std_composite REAL,
                min_composite REAL,
                max_composite REAL,
                median_composite REAL,
                outliers_removed INTEGER,
                results_json TEXT,
                batch_id INTEGER,
                FOREIGN KEY (batch_id) REFERENCES batches(id)
            )
        ''')
        
        # Batches table for bulk processing
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_keywords INTEGER,
                completed_keywords INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                current_keyword_index INTEGER DEFAULT 0,
                images_per_keyword INTEGER DEFAULT 30,
                keywords_json TEXT
            )
        ''')
        
        # PHASE 2.1: Keyword status tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS keyword_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT UNIQUE NOT NULL,
                batch_id INTEGER,
                status TEXT DEFAULT 'pending_review',
                images_required INTEGER DEFAULT 20,
                images_valid INTEGER DEFAULT 0,
                images_rejected INTEGER DEFAULT 0,
                pagination_offset INTEGER DEFAULT 0,
                last_processed DATETIME,
                approved_date DATETIME,
                FOREIGN KEY (batch_id) REFERENCES batches(id)
            )
        ''')
        
        # Images table with rejection tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_id INTEGER,
                image_id TEXT,
                url TEXT,
                title TEXT,
                thumbnail_path TEXT,
                viz_paths_json TEXT,
                circularity REAL,
                aspect_ratio REAL,
                eccentricity REAL,
                solidity REAL,
                convexity REAL,
                composite REAL,
                area REAL,
                perimeter REAL,
                detection_confidence REAL,
                rank INTEGER,
                status TEXT DEFAULT 'valid',
                rejection_reason TEXT,
                rejection_date DATETIME,
                FOREIGN KEY (search_id) REFERENCES searches(id)
            )
        ''')
        
        # Indexes for fast lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_search_term 
            ON searches(search_term)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON searches(timestamp DESC)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_keyword_status 
            ON keyword_status(keyword, status)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_image_status 
            ON images(search_id, status)
        ''')
        
        conn.commit()
        conn.close()
    
    
    def save_search(
        self, 
        search_term: str, 
        results: List[Dict],
        filtered_results: List[Dict],
        outliers: List[Dict],
        stats: Dict,
        batch_id: Optional[int] = None,
        detection_keyword: Optional[str] = None,
        search_phrase: Optional[str] = None,
        category: Optional[str] = None
    ) -> int:
        """
        Save complete search results to database.
        
        Args:
            search_term: Search query
            results: All results with images
            filtered_results: Results after outlier removal (Phase 1.4: same as results now)
            outliers: Removed outliers (Phase 1.4: empty list)
            stats: Statistical summary
            batch_id: Optional batch ID if part of batch processing
            
        Returns:
            Search ID
        """
        # Default to search_term if new fields not provided (backward compatibility)
        if detection_keyword is None:
            detection_keyword = search_term
        if search_phrase is None:
            search_phrase = search_term
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert search record with new dual-keyword fields
        cursor.execute('''
            INSERT INTO searches (
                search_term, detection_keyword, search_phrase, category,
                num_images, avg_circularity, avg_aspect_ratio,
                avg_eccentricity, avg_solidity, avg_convexity,
                avg_composite, std_composite, min_composite, max_composite,
                median_composite, outliers_removed, results_json, batch_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            search_term,
            detection_keyword,
            search_phrase,
            category,
            len(filtered_results),
            stats['circularity']['mean'],
            stats['aspect_ratio']['mean'],
            stats['eccentricity']['mean'],
            stats['solidity']['mean'],
            stats['convexity']['mean'],
            stats['composite']['mean'],
            stats['composite']['std'],
            stats['composite']['min'],
            stats['composite']['max'],
            stats['composite']['median'],
            len(outliers),
            json.dumps({
                'filtered': filtered_results,
                'outliers': outliers
            }),
            batch_id
        ))
        
        search_id = cursor.lastrowid
        
        # Insert individual image records
        for result in filtered_results:
            cursor.execute('''
                INSERT INTO images (
                    search_id, image_id, url, title, thumbnail_path, viz_paths_json,
                    circularity, aspect_ratio, eccentricity, solidity, convexity,
                    composite, area, perimeter, detection_confidence, rank, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                search_id,
                result.get('image_id'),
                result.get('url'),
                result.get('title'),
                result.get('thumbnail_path'),
                json.dumps(result.get('viz_paths', {})),
                result['circularity'],
                result['aspect_ratio'],
                result['eccentricity'],
                result['solidity'],
                result['convexity'],
                result['composite'],
                result['area'],
                result.get('perimeter', 0),
                result.get('detection_confidence', 0),
                result.get('rank', 0),
                'valid'  # All images start as valid
            ))
        
        conn.commit()
        conn.close()
        
        # Update or create keyword status
        self.update_keyword_status(
            keyword=search_term,
            status='pending_review',
            images_valid=len(filtered_results),
            images_rejected=0,
            pagination_offset=0,
            batch_id=batch_id
        )
        
        return search_id
    
    
    def get_search_history(self, limit: int = 50) -> List[Dict]:
        """
        Get recent search history with summary stats.
        
        Args:
            limit: Maximum number of searches to return
            
        Returns:
            List of search summary dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                s.id, s.search_term, s.timestamp, s.num_images,
                s.avg_composite, s.std_composite, s.median_composite,
                s.avg_circularity, s.avg_aspect_ratio, s.avg_eccentricity,
                s.avg_solidity, s.avg_convexity, s.outliers_removed,
                s.batch_id, b.name as batch_name
            FROM searches s
            LEFT JOIN batches b ON s.batch_id = b.id
            ORDER BY s.timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                'id': row[0],
                'search_term': row[1],
                'timestamp': row[2],
                'num_images': row[3],
                'avg_composite': row[4],
                'std_composite': row[5],
                'median_composite': row[6],
                'avg_circularity': row[7],
                'avg_aspect_ratio': row[8],
                'avg_eccentricity': row[9],
                'avg_solidity': row[10],
                'avg_convexity': row[11],
                'outliers_removed': row[12],
                'batch_id': row[13],
                'batch_name': row[14]
            })
        
        return history
    
    
    def get_search(self, search_id: int) -> Optional[Dict]:
        """
        Get complete search data by ID.
        
        Args:
            search_id: Database search ID
            
        Returns:
            Complete search data with all results
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get search metadata
        cursor.execute('''
            SELECT 
                search_term, timestamp, num_images,
                avg_circularity, avg_aspect_ratio, avg_eccentricity,
                avg_solidity, avg_convexity,
                avg_composite, std_composite,
                min_composite, max_composite, median_composite,
                outliers_removed, results_json
            FROM searches
            WHERE id = ?
        ''', (search_id,))
        
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        # Parse results JSON
        results_data = json.loads(row[14])
        
        search_data = {
            'search_term': row[0],
            'timestamp': row[1],
            'num_images': row[2],
            'stats': {
                'circularity': {'mean': row[3]},
                'aspect_ratio': {'mean': row[4]},
                'eccentricity': {'mean': row[5]},
                'solidity': {'mean': row[6]},
                'convexity': {'mean': row[7]},
                'composite': {
                    'mean': row[8],
                    'std': row[9],
                    'min': row[10],
                    'max': row[11],
                    'median': row[12]
                },
                'total_analyzed': row[2] + row[13],
                'total_valid': row[2],
                'outliers_removed': row[13]
            },
            'filtered_results': results_data['filtered'],
            'outliers': results_data['outliers']
        }
        
        conn.close()
        return search_data
    
    
    # ============= PHASE 2.1: KEYWORD STATUS METHODS =============
    
    def update_keyword_status(
        self,
        keyword: str,
        status: str = None,
        images_valid: int = None,
        images_rejected: int = None,
        images_required: int = None,
        pagination_offset: int = None,
        batch_id: int = None
    ) -> None:
        """
        Update or create keyword status record.
        
        Args:
            keyword: The search term/keyword
            status: Status ('pending_review', 'approved', 'needs_more_images')
            images_valid: Count of valid images
            images_rejected: Count of rejected images
            images_required: Target number of images
            pagination_offset: Resume point for next fetch
            batch_id: Associated batch ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if record exists
        cursor.execute('SELECT id FROM keyword_status WHERE keyword = ?', (keyword,))
        exists = cursor.fetchone()
        
        if exists:
            # Build UPDATE query dynamically
            updates = []
            params = []
            
            if status is not None:
                updates.append('status = ?')
                params.append(status)
            if images_valid is not None:
                updates.append('images_valid = ?')
                params.append(images_valid)
            if images_rejected is not None:
                updates.append('images_rejected = ?')
                params.append(images_rejected)
            if images_required is not None:
                updates.append('images_required = ?')
                params.append(images_required)
            if pagination_offset is not None:
                updates.append('pagination_offset = ?')
                params.append(pagination_offset)
            
            updates.append('last_processed = ?')
            params.append(datetime.now().isoformat())
            
            if updates:
                params.append(keyword)
                cursor.execute(f'''
                    UPDATE keyword_status 
                    SET {', '.join(updates)}
                    WHERE keyword = ?
                ''', params)
        else:
            # Insert new record
            cursor.execute('''
                INSERT INTO keyword_status (
                    keyword, batch_id, status, images_valid, images_rejected, 
                    images_required, pagination_offset, last_processed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                keyword,
                batch_id,
                status or 'pending_review',
                images_valid or 0,
                images_rejected or 0,
                images_required or 20,  # Default to 20 if not specified
                pagination_offset or 0,
                datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
    
    
    def get_keyword_status(self, keyword: str) -> Optional[Dict]:
        """Get status for a specific keyword"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, keyword, batch_id, status, images_required, images_valid,
                   images_rejected, pagination_offset, last_processed, approved_date
            FROM keyword_status
            WHERE keyword = ?
        ''', (keyword,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            'id': row[0],
            'keyword': row[1],
            'batch_id': row[2],
            'status': row[3],
            'images_required': row[4],
            'images_valid': row[5],
            'images_rejected': row[6],
            'pagination_offset': row[7],
            'last_processed': row[8],
            'approved_date': row[9]
        }
    
    
    def get_keywords_needing_images(self, batch_id: int = None) -> List[Dict]:
        """
        Get keywords that need more images (incomplete).
        
        Args:
            batch_id: Optional filter by batch
            
        Returns:
            List of keyword status records needing images
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if batch_id:
            cursor.execute('''
                SELECT id, keyword, batch_id, status, images_required, images_valid,
                       images_rejected, pagination_offset
                FROM keyword_status
                WHERE batch_id = ? 
                  AND (status = 'needs_more_images' 
                       OR (status = 'pending_review' AND images_valid < images_required))
            ''', (batch_id,))
        else:
            cursor.execute('''
                SELECT id, keyword, batch_id, status, images_required, images_valid,
                       images_rejected, pagination_offset
                FROM keyword_status
                WHERE status = 'needs_more_images' 
                   OR (status = 'pending_review' AND images_valid < images_required)
            ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        keywords = []
        for row in rows:
            keywords.append({
                'id': row[0],
                'keyword': row[1],
                'batch_id': row[2],
                'status': row[3],
                'images_required': row[4],
                'images_valid': row[5],
                'images_rejected': row[6],
                'pagination_offset': row[7],
                'deficit': row[4] - row[5]  # How many more needed
            })
        
        return keywords
    
    
    def approve_keyword(self, keyword: str) -> None:
        """Mark keyword as approved"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE keyword_status 
            SET status = 'approved', approved_date = ?
            WHERE keyword = ?
        ''', (datetime.now().isoformat(), keyword))
        
        conn.commit()
        conn.close()
    
    
    # ============= PHASE 2.2: IMAGE REJECTION METHODS =============
    
    def reject_image(self, search_id: int, image_id: str, reason: str = 'user_rejected') -> None:
        """
        Mark an image as rejected.
        
        Args:
            search_id: Search ID
            image_id: Image identifier
            reason: Rejection reason
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Mark image as rejected
        cursor.execute('''
            UPDATE images 
            SET status = 'rejected', 
                rejection_reason = ?,
                rejection_date = ?
            WHERE search_id = ? AND image_id = ?
        ''', (reason, datetime.now().isoformat(), search_id, image_id))
        
        # Get keyword and update counts
        cursor.execute('SELECT search_term FROM searches WHERE id = ?', (search_id,))
        row = cursor.fetchone()
        
        if row:
            keyword = row[0]
            
            # Count valid images
            cursor.execute('''
                SELECT COUNT(*) FROM images 
                WHERE search_id IN (SELECT id FROM searches WHERE search_term = ?)
                  AND status = 'valid'
            ''', (keyword,))
            
            valid_count = cursor.fetchone()[0]
            
            # Count rejected images
            cursor.execute('''
                SELECT COUNT(*) FROM images 
                WHERE search_id IN (SELECT id FROM searches WHERE search_term = ?)
                  AND status = 'rejected'
            ''', (keyword,))
            
            rejected_count = cursor.fetchone()[0]
            
            # Get required count
            cursor.execute('''
                SELECT images_required FROM keyword_status WHERE keyword = ?
            ''', (keyword,))
            
            required_row = cursor.fetchone()
            required_count = required_row[0] if required_row else 20
            
            # Update keyword status
            new_status = 'needs_more_images' if valid_count < required_count else 'pending_review'
            
            cursor.execute('''
                UPDATE keyword_status 
                SET images_valid = ?, images_rejected = ?, status = ?
                WHERE keyword = ?
            ''', (valid_count, rejected_count, new_status, keyword))
        
        conn.commit()
        conn.close()
    
    
    def get_valid_image_count(self, search_id: int) -> int:
        """Count valid (non-rejected) images for a search"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM images 
            WHERE search_id = ? AND status = 'valid'
        ''', (search_id,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    
    def get_valid_image_count_for_keyword(self, keyword: str) -> int:
        """Count valid images across all searches for a keyword"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM images 
            WHERE search_id IN (SELECT id FROM searches WHERE search_term = ?)
              AND status = 'valid'
        ''', (keyword,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
    
    
    # ============= EXISTING BATCH METHODS (unchanged) =============
    
    def search_exists(self, search_term: str) -> bool:
        """Check if a search term has been searched before"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM searches
            WHERE LOWER(search_term) = LOWER(?)
        ''', (search_term,))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    
    def delete_searches(self, search_ids: List[int]) -> int:
        """Delete multiple searches by their IDs"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        placeholders = ','.join('?' * len(search_ids))
        cursor.execute(f'''
            DELETE FROM searches
            WHERE id IN ({placeholders})
        ''', search_ids)
        
        # Also delete associated images
        cursor.execute(f'''
            DELETE FROM images
            WHERE search_id IN ({placeholders})
        ''', search_ids)
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count
    
    
    def create_batch(self, name: str, keywords: List, images_per_keyword: int, category: str = None, format_version: str = 'v2') -> int:
        """
        Create a new batch for bulk processing
        
        Args:
            name: Batch name
            keywords: List of keywords (v1) or list of dicts with keyword data (v2)
            images_per_keyword: Number of images to fetch per keyword
            category: Optional category for grouping
            format_version: 'v1' (simple keywords) or 'v2' (dual-keyword with metadata)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # For v2, extract just the keywords for total count
        if format_version == 'v2' and isinstance(keywords, list) and len(keywords) > 0 and isinstance(keywords[0], dict):
            keyword_list = [kw.get('keyword', kw) for kw in keywords]
        else:
            keyword_list = keywords
        
        cursor.execute('''
            INSERT INTO batches (name, total_keywords, images_per_keyword, keywords_json, category, format_version)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, len(keyword_list), images_per_keyword, json.dumps(keywords), category, format_version))
        
        batch_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Initialize keyword status for each keyword in batch
        for kw in keyword_list:
            keyword_str = kw.get('keyword', kw) if isinstance(kw, dict) else kw
            self.update_keyword_status(
                keyword=keyword_str,
                status='pending_review',
                images_valid=0,
                images_rejected=0,
                pagination_offset=0,
                batch_id=batch_id
            )
        
        return batch_id
    
    
    def get_batch(self, batch_id: int) -> Optional[Dict]:
        """Get batch details"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, created_at, total_keywords, completed_keywords, 
                   status, current_keyword_index, images_per_keyword, keywords_json
            FROM batches
            WHERE id = ?
        ''', (batch_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            'id': row[0],
            'name': row[1],
            'created_at': row[2],
            'total_keywords': row[3],
            'completed_keywords': row[4],
            'status': row[5],
            'current_keyword_index': row[6],
            'images_per_keyword': row[7],
            'keywords': json.loads(row[8])
        }
    
    
    def get_incomplete_batches(self) -> List[Dict]:
        """Get all batches that are not complete"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, created_at, total_keywords, completed_keywords, 
                   status, current_keyword_index, images_per_keyword
            FROM batches
            WHERE status != 'complete'
            ORDER BY created_at DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        batches = []
        for row in rows:
            batches.append({
                'id': row[0],
                'name': row[1],
                'created_at': row[2],
                'total_keywords': row[3],
                'completed_keywords': row[4],
                'status': row[5],
                'current_keyword_index': row[6],
                'images_per_keyword': row[7]
            })
        
        return batches
    
    
    def update_batch_progress(self, batch_id: int, current_index: int, completed: int, status: str = None):
        """Update batch progress"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if status is None:
            cursor.execute('''
                UPDATE batches 
                SET current_keyword_index = ?, completed_keywords = ?
                WHERE id = ?
            ''', (current_index, completed, batch_id))
        else:
            cursor.execute('''
                UPDATE batches 
                SET current_keyword_index = ?, completed_keywords = ?, status = ?
                WHERE id = ?
            ''', (current_index, completed, status, batch_id))
        
        conn.commit()
        conn.close()
    
    
    def update_batch_status(self, batch_id: int, status: str, current_index: int = None, completed: int = None):
        """Update batch status (alias for update_batch_progress with different parameter order)"""
        if current_index is None or completed is None:
            # Status-only update
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('UPDATE batches SET status = ? WHERE id = ?', (status, batch_id))
            conn.commit()
            conn.close()
        else:
            self.update_batch_progress(batch_id, current_index, completed, status)
    
    
    def get_batch_searches(self, batch_id: int) -> List[Dict]:
        """Get all searches for a batch"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                id, search_term, timestamp, num_images,
                avg_composite, std_composite, median_composite,
                avg_circularity, avg_aspect_ratio, avg_eccentricity,
                avg_solidity, avg_convexity, outliers_removed
            FROM searches
            WHERE batch_id = ?
            ORDER BY timestamp ASC
        ''', (batch_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        searches = []
        for row in rows:
            searches.append({
                'id': row[0],
                'search_term': row[1],
                'timestamp': row[2],
                'num_images': row[3],
                'avg_composite': row[4],
                'std_composite': row[5],
                'median_composite': row[6],
                'avg_circularity': row[7],
                'avg_aspect_ratio': row[8],
                'avg_eccentricity': row[9],
                'avg_solidity': row[10],
                'avg_convexity': row[11],
                'outliers_removed': row[12]
            })
        
        return searches
    
    
    def get_all_batches(self) -> List[Dict]:
        """Get all batches"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, created_at, total_keywords, completed_keywords, status
            FROM batches
            ORDER BY created_at DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        batches = []
        for row in rows:
            batches.append({
                'id': row[0],
                'name': row[1],
                'created_at': row[2],
                'total_keywords': row[3],
                'completed_keywords': row[4],
                'status': row[5]
            })
        
        return batches
    
    
    def delete_batch(self, batch_id: int) -> bool:
        """Delete a batch record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM batches WHERE id = ?', (batch_id,))
        cursor.execute('DELETE FROM keyword_status WHERE batch_id = ?', (batch_id,))
        
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return deleted