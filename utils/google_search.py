"""
# === UAIPCS START ===
file: utils/google_search.py
purpose: Google Custom Search API client for image search using xxlarge size parameter, downloads full-size images directly from link field using fresh Playwright browser instance per batch
deps: [@requests:library, @typing:library, @time:library, @playwright:library]
funcs:
  - search_images(search_term:str, num_images:int=10) -> list  # side_effect: API calls, rate limiting
  - _single_search(search_term:str, start_index:int=1) -> tuple  # side_effect: API call, logs results
  - download_images_batch(image_results:list) -> list  # side_effect: network I/O, creates fresh browser per call
classes:
  - GoogleSearcher  # manages API client and Playwright batch downloads with fresh browser per batch
refs:
notes: api=https://www.googleapis.com/customsearch/v1, ratelimit=500ms_between_calls, imgsize=xxlarge, maxresults=10_per_call, browser=fresh_instance_per_batch, downloads=direct_from_link_field_via_page_goto
# === UAIPCS END ===
"""

import requests
from typing import List, Dict, Tuple
import time
from playwright.sync_api import Browser, sync_playwright


class GoogleSearcher:
    """Google Custom Search API image searcher - downloads full-size images directly using single browser instance"""
    
    def __init__(self, api_key: str, search_engine_id: str):
        self.api_key = api_key
        self.search_engine_id = search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between API calls
    
    def search_images(self, search_term: str, num_images: int = 10, start_offset: int = 0, use_config_modifiers: bool = True) -> List[Dict]:
        """
        Search for images using multiple API calls if needed (Google returns max 10 per call)
        
        Args:
            search_term: Search query
            num_images: Total number requested (will make multiple API calls if > 10)
            start_offset: Starting offset for pagination
            use_config_modifiers: If True, apply positive/negative keywords from config.json
            
        Returns:
            List of image metadata dicts with 'url', 'title', 'width', 'height', etc.
        """
        all_results = []
        calls_needed = (num_images + 9) // 10  # Round up: 30 images = 3 calls
        
        print(f"📥 Fetching up to {num_images} images for: '{search_term}' ({calls_needed} API calls)")
        
        seen_urls = set()
        
        for call_num in range(calls_needed):
            start_index = start_offset + (call_num * 10) + 1
            api_results, filtered_results = self._single_search(search_term, start_index, use_config_modifiers)
            
            # Deduplicate by URL
            for result in filtered_results:
                if result['url'] not in seen_urls:
                    seen_urls.add(result['url'])
                    all_results.append(result)
            
            # Stop if we got fewer than 10 (no more results available)
            if len(api_results) < 10:
                break
            
            # Stop if we have enough unique results
            if len(all_results) >= num_images:
                break
        
        print(f"   ✓ Deduplicated: {len(all_results)} unique images from {len(seen_urls)} total")
        
        # Return only requested amount
        return all_results[:num_images]
    
    def _single_search(self, search_term: str, start_index: int = 1, use_config_modifiers: bool = True) -> Tuple[List[Dict], List[Dict]]:
        """
        Perform single API call - returns all results (no filtering)
        
        Args:
            search_term: Search query
            start_index: Starting index for pagination (1-based)
            use_config_modifiers: If True, apply positive/negative keywords from config
            
        Returns:
            Tuple of (raw_api_results, formatted_results)
        """
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        # Conditionally load config and build enhanced query
        if use_config_modifiers:
            from utils.config import get_setting
            positive_keywords = get_setting('search.positive_keywords', [])
            negative_keywords = get_setting('search.negative_keywords', [])
            
            # CRITICAL FIX: Quote multi-word search terms to keep them together
            # "Golf R" should search for the exact phrase, not "Golf" OR "R"
            if ' ' in search_term:
                enhanced_query = f'"{search_term}"'
            else:
                enhanced_query = search_term
            
            # Add positive keywords
            if positive_keywords:
                enhanced_query += ' ' + ' '.join(positive_keywords)
            
            # Add negative keywords
            if negative_keywords:
                enhanced_query += ' ' + ' '.join(f'-{kw}' for kw in negative_keywords)
            
            print(f"🔍 Enhanced query: '{enhanced_query}'")
        else:
            # Use search_term as-is without config modifiers
            enhanced_query = search_term
            print(f"🔍 Query (no modifiers): '{enhanced_query}'")
        
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': enhanced_query,
            'searchType': 'image',
            'imgSize': 'xxlarge',  # Largest available
            'num': 10,
            'start': start_index,
            'safe': 'active'
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            self.last_request_time = time.time()
            response.raise_for_status()
            data = response.json()
            
            api_results = data.get('items', [])
            filtered_results = []
            
            print(f"\n📊 Google API returned {len(api_results)} results")
            print("="*80)
            
            for idx, item in enumerate(api_results, 1):
                image_data = item.get('image', {})
                width = image_data.get('width', 0)
                height = image_data.get('height', 0)
                byte_size = image_data.get('byteSize', 0)
                
                print(f"\n  [{idx}] {item.get('title', 'Untitled')[:50]}")
                print(f"      URL: {item.get('link')[:70]}...")
                print(f"      Size: {width}x{height} ({byte_size} bytes)")
                print(f"      Context: {image_data.get('contextLink', '')[:60]}...")
                
                filtered_results.append({
                    'url': item.get('link'),  # Direct link to full-size image
                    'thumbnail_url': image_data.get('thumbnailLink'),
                    'context_url': image_data.get('contextLink', ''),
                    'width': width,
                    'height': height,
                    'byte_size': byte_size,
                    'title': item.get('title', ''),
                    'source': 'google',
                    'mime': item.get('mime', ''),
                    'image_id': f"google_{hash(item.get('link', ''))}"
                })
                print(f"      ✅ Added to download queue")
            
            print("="*80)
            print(f"\n✅ {len(filtered_results)} images queued\n")
            return api_results, filtered_results
        
        except requests.exceptions.RequestException as e:
            print(f"❌ Google API error: {e}")
            return [], []
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return [], []
    
    def download_images_batch(self, image_results: List[Dict]) -> List[Tuple[Dict, bytes]]:
        """
        Download multiple images using a fresh browser instance for this batch.
        Based on test_google_download.py working implementation.
        
        Args:
            image_results: List of image metadata dicts with 'url' field
            
        Returns:
            List of tuples (img_data, image_bytes)
        """
        if not image_results:
            return []
        
        downloads = []
        
        print(f"⬇️  Downloading {len(image_results)} images...")
        print("="*80)
        
        # Create fresh browser for this batch
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0',
                viewport={'width': 1920, 'height': 1080},
                ignore_https_errors=True
            )
            page = context.new_page()
            
            for img_data in image_results:
                url = img_data['url']
                print(f"\n  🌐 {url[:70]}...")
                
                try:
                    # Direct download via page.goto() - same as test_google_download.py
                    response = page.goto(url, wait_until='load', timeout=15000)
                    
                    if response and response.status == 200:
                        content_type = response.headers.get('content-type', '').lower()
                        
                        if 'image' in content_type:
                            image_bytes = response.body()
                            size_kb = len(image_bytes) // 1024
                            print(f"      ✅ Downloaded {size_kb}KB")
                            downloads.append((img_data, image_bytes))
                        else:
                            print(f"      ❌ Not an image: {content_type}")
                    else:
                        status = response.status if response else "No response"
                        print(f"      ❌ Status {status}")
                        
                except Exception as e:
                    print(f"      ❌ Error: {str(e)[:60]}")
                
                time.sleep(0.3)  # Rate limiting between downloads
            
            # Close browser and context
            context.close()
            browser.close()
        
        print("="*80)
        print(f"\n✅ Downloaded {len(downloads)}/{len(image_results)} images\n")
        
        return downloads