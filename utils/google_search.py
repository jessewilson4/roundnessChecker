"""
# AI_PSEUDOCODE:
# @file: utils/google_search.py
# @purpose: Google Custom Search API image searcher w/ negative keywords
# 
# FLOW: search(@term, @count) → GET w/ excludeTerms → parse JSON → RETURN [@img_urls]
# 
# FEATURES:
#   - negative keywords (excludeTerms)
#   - imgSize filter (medium+)
#   - 10 results per call
#   - aspect ratio filter
# 
# API: https://www.googleapis.com/customsearch/v1
# PARAMS:
#   key → API_KEY
#   cx → SEARCH_ENGINE_ID
#   q → search_term
#   searchType=image
#   imgSize=medium|large|xlarge
#   excludeTerms="hand hands person people"
#   num=10
"""

import requests
from typing import List, Dict, Optional
import time


class GoogleSearcher:
    """Google Custom Search API image searcher"""
    
    def __init__(self, api_key: str, search_engine_id: str):
        """
        Initialize Google Custom Search API client.
        
        Args:
            api_key: Google API key
            search_engine_id: Custom Search Engine ID (cx)
        """
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
        # Negative keywords to filter out
        self.exclude_terms = "hand hands person people face closeup close-up texture pattern background"
        
    def search_images(self, search_term: str, num_images: int = 10) -> List[Dict]:
        """
        Search for images using Google Custom Search API.
        
        Args:
            search_term: Object to search for
            num_images: Number of images (must be multiple of 10: 10, 20, 30)
            
        Returns:
            List of image dicts with url, width, height, thumbnail
        """
        if num_images % 10 != 0:
            raise ValueError("num_images must be multiple of 10 (10, 20, 30)")
        
        all_results = []
        num_calls = num_images // 10
        
        for i in range(num_calls):
            start_index = (i * 10) + 1
            results = self._single_search(search_term, start_index)
            all_results.extend(results)
            
            # Rate limiting
            if i < num_calls - 1:
                time.sleep(0.5)
        
        return all_results[:num_images]
    
    def _single_search(self, search_term: str, start_index: int = 1) -> List[Dict]:
        """
        Perform single API call (returns 10 results).
        
        Args:
            search_term: Search query
            start_index: Starting result index (1-based)
            
        Returns:
            List of image result dicts
        """
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': search_term,
            'searchType': 'image',
            'imgSize': 'medium',  # medium, large, xlarge
            'num': 10,
            'start': start_index,
            'excludeTerms': self.exclude_terms,
            'safe': 'active',
            'fileType': 'jpg,png'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get('items', []):
                # Extract image info
                image_data = item.get('image', {})
                
                result = {
                    'url': item.get('link'),
                    'thumbnail_url': image_data.get('thumbnailLink'),
                    'width': image_data.get('width', 0),
                    'height': image_data.get('height', 0),
                    'thumbnail_width': image_data.get('thumbnailWidth', 0),
                    'thumbnail_height': image_data.get('thumbnailHeight', 0),
                    'title': item.get('title', ''),
                    'source': 'google',
                    'context_url': image_data.get('contextLink', ''),
                    'mime': item.get('mime', '')
                }
                
                # Filter by aspect ratio (avoid extreme panoramas)
                if result['width'] > 0 and result['height'] > 0:
                    aspect_ratio = result['width'] / result['height']
                    if 0.5 <= aspect_ratio <= 2.0:
                        results.append(result)
            
            return results
            
        except requests.exceptions.RequestException as e:
            print(f"Google API error: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []