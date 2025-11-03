"""
# AI_PSEUDOCODE:
# @file: utils/pexels_search.py
# @purpose: @pexels_api integration FOR @img search
# @priority_weight: 25% (backup source, rate limited)
# @dependencies: [requests]
# 
# MODIFIED: @date 2025-11-02
# CHANGES: add search modifiers FOR better isolated objects
#
# ALGO: search_images
#   IN: @term:str, @num:int
#   OUT: [@img_metadata:arr]
#   STEPS:
#     1. query = @term + " full object white background"  # MODIFIED: add modifiers
#     2. @pexels_api.GET /search
#     3. FILTER: orientation=square (centered objects)
#     4. RETURN [@img_urls + metadata]
"""

import requests
import time
from typing import List, Dict, Optional


class PexelsSearcher:
    """
    Interface for searching images using Pexels API.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Pexels searcher.
        
        Args:
            api_key: Pexels API key (get free at pexels.com/api)
        """
        self.api_key = api_key
        self.base_url = "https://api.pexels.com/v1"
        self.headers = {
            'Authorization': api_key
        }
        self.rate_limit_delay = 1.0  # Seconds between requests
        
    
    def search_images(self, search_term: str, num_images: int = 30) -> List[Dict]:
        """
        Search for images using intelligent query strategies.
        
        Args:
            search_term: Object to search for (e.g., "ball", "cat")
            num_images: Number of images to retrieve
            
        Returns:
            List of image dictionaries with url, id, photographer info
        """
        print(f"  🔍 Searching Pexels for: '{search_term}'")
        
        # MODIFIED: Enhanced query with modifiers for better object isolation
        query = f"{search_term} full object white background"
        
        try:
            results = self._fetch_query(query, num_images)
            print(f"  ✓ Pexels: {len(results)} images")
            return results
                
        except Exception as e:
            print(f"  ✗ Pexels search failed: {e}")
            return []
    
    
    def _fetch_query(self, query: str, per_page: int) -> List[Dict]:
        """
        Fetch images for a specific query.
        
        Args:
            query: Search query string
            per_page: Number of results to fetch
            
        Returns:
            List of image result dictionaries
        """
        params = {
            'query': query,
            'per_page': min(per_page, 80),  # Pexels max is 80
            'orientation': 'square',  # MODIFIED: Prefer centered objects
            'size': 'large'  # Get high resolution for better edge detection
        }
        
        response = requests.get(
            f"{self.base_url}/search",
            headers=self.headers,
            params=params,
            timeout=10
        )
        
        response.raise_for_status()
        data = response.json()
        
        photos = data.get('photos', [])
        
        # Extract relevant information
        results = []
        for photo in photos:
            results.append({
                'id': photo['id'],
                'url': photo['src']['medium'],  # Use medium size for faster processing
                'original_url': photo['src']['original'],
                'photographer': photo['photographer'],
                'photographer_url': photo['photographer_url'],
                'width': photo['width'],
                'height': photo['height'],
                'source': 'pexels'
            })
        
        return results
    
    
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
            print(f"  ✗ Failed to download image: {e}")
            return None