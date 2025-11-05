"""
SQLite database for caching search results and managing history.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
import os


class Database:
    """
    Manages SQLite database for search caching and history.
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
        
        # Images table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_id INTEGER,
                pexels_id INTEGER,
                url TEXT,
                photographer TEXT,
                thumbnail_path TEXT,
                viz_path TEXT,
                circularity REAL,
                aspect_ratio REAL,
                eccentricity REAL,
                solidity REAL,
                convexity REAL,
                composite REAL,
                area REAL,
                perimeter REAL,
                rank INTEGER,
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
        
        conn.commit()
        conn.close()
    
    
    def save_search(
        self, 
        search_term: str, 
        results: List[Dict],
        filtered_results: List[Dict],
        outliers: List[Dict],
        stats: Dict,
        batch_id: Optional[int] = None
    ) -> int:
        """
        Save complete search results to database.
        
        Args:
            search_term: Search query
            results: All results with images
            filtered_results: Results after outlier removal
            outliers: Removed outliers
            stats: Statistical summary
            batch_id: Optional batch ID if part of batch processing
            
        Returns:
            Search ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert search record
        cursor.execute('''
            INSERT INTO searches (
                search_term, num_images, avg_circularity, avg_aspect_ratio,
                avg_eccentricity, avg_solidity, avg_convexity,
                avg_composite, std_composite, min_composite, max_composite,
                median_composite, outliers_removed, results_json, batch_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            search_term,
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
        
        # Insert image records
        for result in filtered_results:
            cursor.execute('''
                INSERT INTO images (
                    search_id, pexels_id, url, photographer,
                    thumbnail_path, viz_path, circularity, aspect_ratio,
                    eccentricity, solidity, convexity, composite, area, perimeter, rank
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                search_id,
                result.get('pexels_id'),
                result.get('url'),
                result.get('photographer'),
                result.get('thumbnail_path'),
                result.get('viz_path'),
                result['circularity'],
                result['aspect_ratio'],
                result['eccentricity'],
                result['solidity'],
                result['convexity'],
                result['composite'],
                result['area'],
                result.get('perimeter'),
                result['rank']
            ))
        
        conn.commit()
        conn.close()
        
        return search_id
    
    
    def get_search_history(self, limit: int = 50) -> List[Dict]:
        """
        Get recent search history.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of search summaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                id, search_term, timestamp, num_images,
                avg_circularity, avg_aspect_ratio, avg_eccentricity,
                avg_solidity, avg_convexity, avg_composite,
                std_composite, min_composite, max_composite, median_composite,
                outliers_removed, batch_id
            FROM searches
            ORDER BY timestamp DESC
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
                'avg_circularity': row[4],
                'avg_aspect_ratio': row[5],
                'avg_eccentricity': row[6],
                'avg_solidity': row[7],
                'avg_convexity': row[8],
                'avg_composite': row[9],
                'std_composite': row[10],
                'min_composite': row[11],
                'max_composite': row[12],
                'median_composite': row[13],
                'outliers_removed': row[14],
                'batch_id': row[15]
            })
        
        return history
    
    
    def get_all_searches(self, limit: int = 50) -> List[Dict]:
        """
        Alias for get_search_history() for backwards compatibility.
        
        Args:
            limit: Maximum number of results
        """
        return self.get_search_history(limit)
    
    
    def get_previous_searches(self, search_term: str, limit: int = 5) -> List[Dict]:
        """
        Get previous searches for a specific term (for autocomplete).
        
        Args:
            search_term: Search term (case-insensitive partial match)
            limit: Maximum results
            
        Returns:
            List of previous search summaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                id, search_term, timestamp, num_images,
                avg_circularity, avg_composite
            FROM searches
            WHERE LOWER(search_term) LIKE LOWER(?)
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (f'%{search_term}%', limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            results.append({
                'id': row[0],
                'search_term': row[1],
                'timestamp': row[2],
                'num_images': row[3],
                'avg_circularity': row[4] * 100,
                'avg_composite': row[5] * 100
            })
        
        return results
    
    
    def load_search_by_id(self, search_id: int) -> Optional[Dict]:
        """
        Load complete search results by ID.
        
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
    
    
    def search_exists(self, search_term: str) -> bool:
        """
        Check if a search term has been searched before.
        
        Args:
            search_term: Search term
            
        Returns:
            True if exists
        """
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
        """
        Delete multiple searches by their IDs.
        
        Args:
            search_ids: List of search IDs to delete
            
        Returns:
            Number of searches deleted
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete searches
        placeholders = ','.join('?' * len(search_ids))
        cursor.execute(f'''
            DELETE FROM searches
            WHERE id IN ({placeholders})
        ''', search_ids)
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count
    
    
    def update_search(self, search_id: int, filtered_results: List[Dict], outliers: List[Dict], stats: Dict) -> None:
        """
        Update an existing search with new results and stats after deletion.
        
        Args:
            search_id: Search ID to update
            filtered_results: Updated filtered results
            outliers: Updated outliers
            stats: Updated statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update the search
        cursor.execute('''
            UPDATE searches 
            SET 
                num_images = ?,
                avg_circularity = ?,
                avg_aspect_ratio = ?,
                avg_eccentricity = ?,
                avg_solidity = ?,
                avg_convexity = ?,
                avg_composite = ?,
                std_composite = ?,
                min_composite = ?,
                max_composite = ?,
                median_composite = ?,
                outliers_removed = ?,
                results_json = ?
            WHERE id = ?
        ''', (
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
            search_id
        ))
        
        conn.commit()
        conn.close()
    
    
    # ============= BATCH PROCESSING METHODS =============
    
    def create_batch(self, name: str, keywords: List[str], images_per_keyword: int) -> int:
        """
        Create a new batch for bulk processing.
        
        Args:
            name: Batch name
            keywords: List of keywords to process
            images_per_keyword: Number of images per keyword
            
        Returns:
            batch_id
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Remove duplicates while preserving order
        unique_keywords = list(dict.fromkeys(keywords))
        
        cursor.execute('''
            INSERT INTO batches (name, total_keywords, images_per_keyword, keywords_json)
            VALUES (?, ?, ?, ?)
        ''', (
            name,
            len(unique_keywords),
            images_per_keyword,
            json.dumps(unique_keywords)
        ))
        
        batch_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return batch_id
    
    
    def get_batch(self, batch_id: int) -> Optional[Dict]:
        """Get batch information"""
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
    
    
    def update_batch_status(self, batch_id: int, status: str, current_index: int = None, completed: int = None):
        """Update batch processing status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if current_index is not None and completed is not None:
            cursor.execute('''
                UPDATE batches 
                SET status = ?, current_keyword_index = ?, completed_keywords = ?
                WHERE id = ?
            ''', (status, current_index, completed, batch_id))
        else:
            cursor.execute('''
                UPDATE batches 
                SET status = ?
                WHERE id = ?
            ''', (status, batch_id))
        
        conn.commit()
        conn.close()
    
    
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
        
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return deleted

    def get_incomplete_batches(self) -> List[Dict]:
        """Get batches that are incomplete (not 'complete' status)"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, created_at, total_keywords, completed_keywords, 
                   status, current_keyword_index, keywords_json
            FROM batches
            WHERE status != 'complete'
            ORDER BY created_at DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        batches = []
        for row in rows:
            keywords = json.loads(row[7])
            current_keyword = keywords[row[6]] if row[6] < len(keywords) else 'N/A'
            
            batches.append({
                'id': row[0],
                'name': row[1],
                'created_at': row[2],
                'total_keywords': row[3],
                'completed_keywords': row[4],
                'status': row[5],
                'current_keyword_index': row[6],
                'current_keyword': current_keyword,
                'progress_percent': (row[4] / row[3] * 100) if row[3] > 0 else 0
            })
        
        return batches