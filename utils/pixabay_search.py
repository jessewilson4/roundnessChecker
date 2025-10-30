"""
Pixabay API integration for diverse image search
"""

import requests
from typing import List, Dict, Optional


class PixabaySearcher:
    """Search and download images from Pixabay"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://pixabay.com/api/"
    
    def search_images(self, query: str, num_images: int = 30) -> List[Dict]:
        """
        Search Pixabay for images.
        
        Args:
            query: Search term
            num_images: Number of images to retrieve (max 200 per request)
            
        Returns:
            List of image data dictionaries
        """
        results = []
        per_page = min(num_images, 200)  # Pixabay max 200 per page
        
        try:
            params = {
                "key": self.api_key,
                "q": query,
                "per_page": per_page,
                "image_type": "photo",
                "safesearch": "true"
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            hits = data.get('hits', [])
            
            for hit in hits:
                results.append({
                    'id': hit['id'],
                    'url': hit['webformatURL'],  # Web format (~640px) for faster processing
                    'photographer': hit['user'],
                    'photographer_url': f"https://pixabay.com/users/{hit['user']}-{hit['user_id']}/",
                    'source': 'pixabay',
                    'width': hit['webformatWidth'],
                    'height': hit['webformatHeight']
                })
            
            print(f"  âœ“ Found {len(results)} Pixabay images")
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"  âœ— Pixabay search failed: {e}")
            return []
    
    def download_image(self, url: str) -> Optional[bytes]:
        """Download image from URL"""
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"  âœ— Download failed: {e}")
            return None