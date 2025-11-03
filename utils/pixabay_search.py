"""
# AI_PSEUDOCODE:
# @file: utils/pixabay_search.py
# @purpose: @pixabay_api integration FOR @img search
# @priority_weight: 35% (unlimited requests, good variety)
# @dependencies: [requests]
# 
# ALGO: search_images
#   IN: @term:str, @num:int
#   OUT: [@img_metadata:arr]
#   STEPS:
#     1. query = @term + " object white background"
#     2. @pixabay_api.GET /?key={key}&q={query}
#     3. FILTER: image_type=photo
#     4. RETURN [@img_urls + metadata]
# 
# ALGO: download_image
#   IN: @url:str
#   OUT: @img_bytes:bytes
"""

import requests
import time
from typing import List, Dict, Optional


class PixabaySearcher:
    """
    Interface for searching images using Pixabay API.
    Pixabay provides unlimited free searches with good variety.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Pixabay searcher.
        
        Args:
            api_key: Pixabay API key (get free at pixabay.com/api/docs/)
        """
        self.api_key = api_key
        self.base_url = "https://pixabay.com/api/"
        self.rate_limit_delay = 0.5  # Unlimited, but be respectful
        
    
    def search_images(self, search_term: str, num_images: int = 30) -> List[Dict]:
        """
        Search for images with quality modifiers.
        
        Args:
            search_term: Object to search for (e.g., "ball", "cat")
            num_images: Number of images to retrieve
            
        Returns:
            List of image dictionaries with url, id, photographer info
        """
        print(f"  🔍 Searching Pixabay for: '{search_term}'")
        
        # Build query
        query = f"{search_term} object"
        
        try:
            params = {
                'key': self.api_key,
                'q': query,
                'per_page': min(num_images, 200),  # Pixabay max is 200
                'image_type': 'photo',
                'orientation': 'horizontal',  # Prefer landscapes
                'safesearch': 'true'
            }
            
            response = requests.get(
                self.base_url,
                params=params,
                timeout=10
            )
            
            response.raise_for_status()
            data = response.json()
            
            hits = data.get('hits', [])
            
            # Extract relevant information
            images = []
            for photo in hits:
                images.append({
                    'id': f"pixabay_{photo['id']}",
                    'url': photo['webformatURL'],  # Good quality, reasonable size
                    'original_url': photo['largeImageURL'],
                    'photographer': photo['user'],
                    'photographer_url': f"https://pixabay.com/users/{photo['user']}-{photo['user_id']}/",
                    'width': photo['imageWidth'],
                    'height': photo['imageHeight'],
                    'source': 'pixabay'
                })
            
            print(f"  ✓ Pixabay: {len(images)} images")
            return images
            
        except Exception as e:
            print(f"  ✗ Pixabay search failed: {e}")
            return []
    
    
    def download_image(self, url: str) -> Optional[bytes]:
        """
        Download image from URL.
        
        Args:
            url: Image URL
            
        Returns:
            Image data as bytes, or None if failed
        """
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"  ✗ Failed to download from Pixabay: {e}")
            return None