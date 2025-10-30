"""
Unsplash API integration for high-quality image search
"""

import requests
from typing import List, Dict, Optional


class UnsplashSearcher:
    """Search and download images from Unsplash"""
    
    def __init__(self, access_key: str):
        self.access_key = access_key
        self.base_url = "https://api.unsplash.com"
        self.headers = {
            "Authorization": f"Client-ID {access_key}"
        }
    
    def search_images(self, query: str, num_images: int = 30) -> List[Dict]:
        """
        Search Unsplash for images.
        
        Args:
            query: Search term
            num_images: Number of images to retrieve (max 30 per page)
            
        Returns:
            List of image data dictionaries
        """
        results = []
        per_page = min(num_images, 30)  # Unsplash max 30 per page
        
        try:
            url = f"{self.base_url}/search/photos"
            params = {
                "query": query,
                "per_page": per_page,
                "orientation": "squarish"  # Better for isolated objects
            }
            
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            photos = data.get('results', [])
            
            for photo in photos:
                results.append({
                    'id': photo['id'],
                    'url': photo['urls']['regular'],  # Regular size (~1080px) for faster processing
                    'photographer': photo['user']['name'],
                    'photographer_url': photo['user']['links']['html'],
                    'source': 'unsplash',
                    'width': photo['width'],
                    'height': photo['height']
                })
            
            print(f"  âœ“ Found {len(results)} Unsplash images")
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"  âœ— Unsplash search failed: {e}")
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