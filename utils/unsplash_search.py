"""
# AI_PSEUDOCODE:
# @file: utils/unsplash_search.py
# @purpose: @unsplash_api integration FOR @img search
# @priority_weight: 40% (highest quality isolated objects)
# @dependencies: [requests]
# 
# ALGO: search_images
#   IN: @term:str, @num:int
#   OUT: [@img_metadata:arr]
#   STEPS:
#     1. query = @term + " full object isolated white background"
#     2. @unsplash_api.GET /search/photos
#     3. FILTER: orientation=landscape (prefer centered)
#     4. RETURN [@img_urls + metadata]
# 
# ALGO: download_image
#   IN: @url:str
#   OUT: @img_bytes:bytes
#   STEPS:
#     1. GET @url → @response
#     2. RETURN @response.content
"""

import requests
import time
from typing import List, Dict, Optional


class UnsplashSearcher:
    """
    Interface for searching images using Unsplash API.
    Unsplash provides excellent isolated object photography.
    """
    
    def __init__(self, access_key: str):
        """
        Initialize Unsplash searcher.
        
        Args:
            access_key: Unsplash Access Key (get free at unsplash.com/developers)
        """
        self.access_key = access_key
        self.base_url = "https://api.unsplash.com"
        self.headers = {
            'Authorization': f'Client-ID {access_key}'
        }
        self.rate_limit_delay = 1.2  # Seconds between requests (50 req/hour = 72s/req, buffer)
        
    
    def search_images(self, search_term: str, num_images: int = 30) -> List[Dict]:
        """
        Search for images with quality modifiers.
        
        Args:
            search_term: Object to search for (e.g., "ball", "cat")
            num_images: Number of images to retrieve
            
        Returns:
            List of image dictionaries with url, id, photographer info
        """
        print(f"  🔍 Searching Unsplash for: '{search_term}'")
        
        # Build query with quality modifiers
        query = f"{search_term} full object isolated"
        
        try:
            params = {
                'query': query,
                'per_page': min(num_images, 30),  # Unsplash max is 30
                'orientation': 'landscape',  # Prefer centered compositions
                'content_filter': 'high'  # High quality only
            }
            
            response = requests.get(
                f"{self.base_url}/search/photos",
                headers=self.headers,
                params=params,
                timeout=10
            )
            
            response.raise_for_status()
            data = response.json()
            
            results = data.get('results', [])
            
            # Extract relevant information
            images = []
            for photo in results:
                images.append({
                    'id': f"unsplash_{photo['id']}",
                    'url': photo['urls']['regular'],  # High quality, not too large
                    'original_url': photo['urls']['full'],
                    'photographer': photo['user']['name'],
                    'photographer_url': photo['user']['links']['html'],
                    'width': photo['width'],
                    'height': photo['height'],
                    'source': 'unsplash'
                })
            
            print(f"  ✓ Unsplash: {len(images)} images")
            return images
            
        except Exception as e:
            print(f"  ✗ Unsplash search failed: {e}")
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
            print(f"  ✗ Failed to download from Unsplash: {e}")
            return None