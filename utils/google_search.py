'''
# === UAIPCS START ===
file: utils/google_search.py
purpose: Google Custom Search API client for image search with negative keyword filtering to find isolated, whole objects (~1024px), with detailed logging for troubleshooting
deps: [@requests:library, @typing:library, @time:library]
funcs:
  - search_images(search_term:str, num_images:int=10) -> list  # side_effect: API calls, rate limiting, single-page troubleshooting
  - _single_search(search_term:str, start_index:int=1) -> list  # side_effect: API call, logs API results and filtered results
  - download_image(url:str, fallback_url:str=None) -> bytes  # side_effect: network I/O
classes:
  - GoogleSearcher  # manages API client with negative keywords filtering and size/aspect filtering
refs:
notes: api=https://www.googleapis.com/customsearch/v1, ratelimit=500ms_between_calls, excludes=hand|hands|person|people|face|closeup|texture|pattern|background, imgsize=large, min_width=1024, maxresults=10_per_call, single_page_logging=True
# === UAIPCS END ===
'''

import requests
from typing import List, Dict, Optional
import time


class GoogleSearcher:
    """Google Custom Search API image searcher for ~1024px images with negative keyword filtering and detailed logging"""
    
    def __init__(self, api_key: str, search_engine_id: str):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.exclude_terms = "hand hands person people face closeup close-up texture pattern background"
    
    def search_images(self, search_term: str, num_images: int = 10) -> List[Dict]:
        """Search for images and return filtered list (~1024px, correct aspect ratio) with logging (single page)"""
        if num_images != 10:
            print("⚠️ For troubleshooting, only 10 items are fetched per call.")
            num_images = 10
        
        print(f"📥 Fetching {num_images} images for search term: '{search_term}'")
        
        api_results, filtered_results = self._single_search(search_term)
        
        print(f"✅ Total images returned after filtering: {len(filtered_results)}\n")
        return filtered_results
    
    def _single_search(self, search_term: str, start_index: int = 1) -> (List[Dict], List[Dict]):
        """Perform a single API call with detailed logging for troubleshooting purposes"""
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': search_term,
            'searchType': 'image',
            'imgSize': 'large',
            'num': 10,
            'start': start_index,
            'excludeTerms': self.exclude_terms,
            'safe': 'active'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            api_results = data.get('items', [])
            filtered_results = []
            
            for item in api_results:
                image_data = item.get('image', {})
                width = image_data.get('width', 0)
                height = image_data.get('height', 0)
                
                print(f"  🔹 Image: {item.get('link')}")
                print(f"    width={width}, height={height}")
                print(f"    thumbnail={image_data.get('thumbnailLink')}")
                print(f"    context={image_data.get('contextLink')}")
                
                if width >= 1024:
                    aspect_ratio = width / height if height > 0 else 0
                    if 0.5 <= aspect_ratio <= 2.0:
                        filtered_results.append({
                            'url': item.get('link'),
                            'thumbnail_url': image_data.get('thumbnailLink'),
                            'width': width,
                            'height': height,
                            'thumbnail_width': image_data.get('thumbnailWidth', 0),
                            'thumbnail_height': image_data.get('thumbnailHeight', 0),
                            'title': item.get('title', ''),
                            'source': 'google',
                            'context_url': image_data.get('contextLink', ''),
                            'mime': item.get('mime', ''),
                            'image_id': f"google_{item.get('link', '')[-20:]}"
                        })
                else:
                    print("    ❌ Skipped (width < 1024)")
            
            print(f"  🔹 {len(filtered_results)} items passed filtering\n")
            return api_results, filtered_results
        
        except requests.exceptions.RequestException as e:
            print(f"Google API error: {e}")
            return [], []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return [], []
    
    def download_image(self, url: str, fallback_url: str = None) -> Optional[bytes]:
        """Download image from URL with browser-like headers to reduce 403 errors"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',
            'Referer': 'https://www.google.com/'
        }
        urls_to_try = [url]
        if fallback_url:
            urls_to_try.append(fallback_url)
        
        for u in urls_to_try:
            try:
                resp = requests.get(u, headers=headers, timeout=15)
                resp.raise_for_status()
                return resp.content
            except Exception as e:
                print(f"  ✗ Failed to download image from {u}: {e}")
        
        return None
