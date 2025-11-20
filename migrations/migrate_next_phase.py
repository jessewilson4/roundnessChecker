"""
Database Migration Script for NEXT PHASE: Dual-Keyword System
Adds support for separate detection keywords and search phrases
"""

import sqlite3
import os

def migrate_next_phase():
    """Add columns for dual-keyword system"""
    db_path = './cache/searches.db'
    
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Starting NEXT PHASE database migration...")
    
    # Check and add columns to searches table
    cursor.execute("PRAGMA table_info(searches)")
    existing_columns = {row[1] for row in cursor.fetchall()}
    
    if 'detection_keyword' not in existing_columns:
        print("Adding detection_keyword column to searches table...")
        cursor.execute('ALTER TABLE searches ADD COLUMN detection_keyword TEXT')
        # Set default to search_term for existing records
        cursor.execute('UPDATE searches SET detection_keyword = search_term WHERE detection_keyword IS NULL')
        print("✓ Added detection_keyword")
    
    if 'search_phrase' not in existing_columns:
        print("Adding search_phrase column to searches table...")
        cursor.execute('ALTER TABLE searches ADD COLUMN search_phrase TEXT')
        # Set default to search_term for existing records
        cursor.execute('UPDATE searches SET search_phrase = search_term WHERE search_phrase IS NULL')
        print("✓ Added search_phrase")
    
    if 'category' not in existing_columns:
        print("Adding category column to searches table...")
        cursor.execute('ALTER TABLE searches ADD COLUMN category TEXT')
        print("✓ Added category")
    
    # Check and add columns to batches table
    cursor.execute("PRAGMA table_info(batches)")
    existing_batch_columns = {row[1] for row in cursor.fetchall()}
    
    if 'category' not in existing_batch_columns:
        print("Adding category column to batches table...")
        cursor.execute('ALTER TABLE batches ADD COLUMN category TEXT')
        print("✓ Added category to batches")
    
    if 'format_version' not in existing_batch_columns:
        print("Adding format_version column to batches table...")
        cursor.execute('ALTER TABLE batches ADD COLUMN format_version TEXT DEFAULT "v1"')
        # Mark all existing batches as v1
        cursor.execute('UPDATE batches SET format_version = "v1" WHERE format_version IS NULL')
        print("✓ Added format_version to batches")
    
    conn.commit()
    conn.close()
    
    print("\n✅ Migration complete!")
    print("   - searches table: detection_keyword, search_phrase, category")
    print("   - batches table: category, format_version")

if __name__ == '__main__':
    migrate_next_phase()
