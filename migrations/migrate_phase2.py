import sqlite3
import os

DB_PATH = './cache/searches.db'

def migrate():
    if not os.path.exists(DB_PATH):
        print("Database not found, skipping migration (will be created by app)")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("Checking for keyword_status table...")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='keyword_status'")
    if not cursor.fetchone():
        print("Creating keyword_status table...")
        cursor.execute('''
            CREATE TABLE keyword_status (
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
        cursor.execute('CREATE INDEX idx_keyword_status ON keyword_status(keyword, status)')
    else:
        print("keyword_status table already exists.")

    print("Checking images table for new columns...")
    cursor.execute("PRAGMA table_info(images)")
    columns = [info[1] for info in cursor.fetchall()]
    
    if 'status' not in columns:
        print("Adding status column to images...")
        cursor.execute("ALTER TABLE images ADD COLUMN status TEXT DEFAULT 'valid'")
        cursor.execute("CREATE INDEX idx_image_status ON images(search_id, status)")
    
    if 'rejection_reason' not in columns:
        print("Adding rejection_reason column to images...")
        cursor.execute("ALTER TABLE images ADD COLUMN rejection_reason TEXT")
        
    if 'rejection_date' not in columns:
        print("Adding rejection_date column to images...")
        cursor.execute("ALTER TABLE images ADD COLUMN rejection_date DATETIME")

    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
