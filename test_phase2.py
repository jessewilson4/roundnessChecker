import unittest
import os
import shutil
import json
import sys
from unittest.mock import MagicMock

# Mock modules that might be missing or heavy
sys.modules['playwright'] = MagicMock()
sys.modules['playwright.sync_api'] = MagicMock()
sys.modules['segment_anything'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['cv2'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Now import Database
# We need to bypass utils.__init__ if possible, or let the mocks handle the imports
# Since utils.__init__ imports from .google_search which imports playwright, our mock should work.
from utils.database import Database

class TestPhase2(unittest.TestCase):
    def setUp(self):
        self.test_db_path = './test_cache/test.db'
        if os.path.exists('./test_cache'):
            shutil.rmtree('./test_cache')
        os.makedirs('./test_cache')
        self.db = Database(self.test_db_path)
        
    def tearDown(self):
        if os.path.exists('./test_cache'):
            shutil.rmtree('./test_cache')

    def test_keyword_status_lifecycle(self):
        # Create batch
        batch_id = self.db.create_batch("Test Batch", ["apple", "banana"], 10)
        
        # Check initial status
        status = self.db.get_keyword_status("apple")
        self.assertEqual(status['status'], 'pending_review')
        self.assertEqual(status['batch_id'], batch_id)
        
        # Simulate saving search results
        self.db.save_search(
            search_term="apple",
            results=[],
            filtered_results=[{'image_id': 'img1', 'circularity': 0.9, 'aspect_ratio': 0.9, 'eccentricity': 0.1, 'solidity': 0.9, 'convexity': 0.9, 'composite': 0.9, 'area': 100}],
            outliers=[],
            stats={'circularity': {'mean': 0.9}, 'aspect_ratio': {'mean': 0.9}, 'eccentricity': {'mean': 0.1}, 'solidity': {'mean': 0.9}, 'convexity': {'mean': 0.9}, 'composite': {'mean': 0.9, 'std': 0, 'min': 0.9, 'max': 0.9, 'median': 0.9}},
            batch_id=batch_id
        )
        
        # Check status updated
        status = self.db.get_keyword_status("apple")
        self.assertEqual(status['images_valid'], 1)
        
        # Approve keyword
        self.db.approve_keyword("apple")
        status = self.db.get_keyword_status("apple")
        self.assertEqual(status['status'], 'approved')
        self.assertIsNotNone(status['approved_date'])

    def test_image_rejection(self):
        # Create batch
        batch_id = self.db.create_batch("Test Batch", ["car"], 2)
        
        # Save search with 2 images
        search_id = self.db.save_search(
            search_term="car",
            results=[],
            filtered_results=[
                {'image_id': 'img1', 'circularity': 0.9, 'aspect_ratio': 0.9, 'eccentricity': 0.1, 'solidity': 0.9, 'convexity': 0.9, 'composite': 0.9, 'area': 100},
                {'image_id': 'img2', 'circularity': 0.9, 'aspect_ratio': 0.9, 'eccentricity': 0.1, 'solidity': 0.9, 'convexity': 0.9, 'composite': 0.9, 'area': 100}
            ],
            outliers=[],
            stats={'circularity': {'mean': 0.9}, 'aspect_ratio': {'mean': 0.9}, 'eccentricity': {'mean': 0.1}, 'solidity': {'mean': 0.9}, 'convexity': {'mean': 0.9}, 'composite': {'mean': 0.9, 'std': 0, 'min': 0.9, 'max': 0.9, 'median': 0.9}},
            batch_id=batch_id
        )
        
        # Set requirement to 2 images
        self.db.update_keyword_status("car", images_required=2)
        
        # Reject one image
        self.db.reject_image(search_id, 'img1', 'bad quality')
        
        # Check image status
        conn = sqlite3.connect(self.test_db_path)
        c = conn.cursor()
        c.execute("SELECT status, rejection_reason FROM images WHERE image_id = 'img1'")
        row = c.fetchone()
        conn.close()
        self.assertEqual(row[0], 'rejected')
        self.assertEqual(row[1], 'bad quality')
        
        # Check keyword status (should be needs_more_images because 1 valid < 2 required)
        status = self.db.get_keyword_status("car")
        self.assertEqual(status['images_valid'], 1)
        self.assertEqual(status['images_rejected'], 1)
        self.assertEqual(status['status'], 'needs_more_images')

import sqlite3
if __name__ == '__main__':
    unittest.main()
