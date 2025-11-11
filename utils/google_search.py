'''
# === UAIPCS START ===
file: utils/google_search.py
purpose: Google Custom Search API client for image search using xxlarge size parameter, downloads full-size images directly from link field using single reusable Playwright browser instance
deps: [@requests:library, @typing:library, @time:library, @playwright:library]
funcs:
  - search_images(search_term:str, num_images:int=10) -> list  # side_effect: API calls, rate limiting
  - _single_search(search_term:str, start_index:int=1) -> tuple  # side_effect: API call, logs results
  - download_images_batch(image_results:list, browser:Browser) -> list  # side_effect: network I/O, reuses browser
classes:
  - GoogleSearcher  # manages API client and Playwright batch downloads with single browser instance
refs:
notes: api=https://www.googleapis.com/customsearch/v1, ratelimit=500ms_between_calls, imgsize=xxlarge, maxresults=10_per_call, browser=single_reusable_instance, downloads=direct_from_link_field
# === UAIPCS END ===
'''

import requests
from typing import List, Dict, Optional, Tuple
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
        self._browser = None
        self._playwright = None
    
    def _ensure_browser(self):
        """Lazy initialization of browser - creates on first use"""
        if self._browser is None:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(headless=True)
        return self._browser
    
    def close_browser(self):
        """Close browser instance - call when done with all downloads"""
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None
    
    def search_images(self, search_term: str, num_images: int = 10) -> List[Dict]:
        """Search for images - hard limited to 10 per API call"""
        num_images = 10  # Google API limit per request
        
        print(f"📥 Fetching up to {num_images} images for: '{search_term}'")
        
        api_results, filtered_results = self._single_search(search_term)
        
        print(f"✅ {len(filtered_results)} images ready for download\n")
        return filtered_results
    
    def _single_search(self, search_term: str, start_index: int = 1) -> Tuple[List[Dict], List[Dict]]:
        """Perform single API call - returns all results (no filtering)"""
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        
        params = {
            'key': self.api_key,
            'cx': self.search_engine_id,
            'q': search_term,
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
        Download multiple images using single browser instance
        
        Args:
            image_results: List of image metadata dicts with 'url' field
            
        Returns:
            List of tuples (img_data, image_bytes)
        """
        if not image_results:
            return []
        
        browser = self._ensure_browser()
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0',
            viewport={'width': 1920, 'height': 1080},
            ignore_https_errors=True
        )
        page = context.new_page()
        
        downloads = []
        
        print(f"⬇️  Downloading {len(image_results)} images...")
        print("="*80)
        
        for img_data in image_results:
            url = img_data['url']
            print(f"\n  🌐 {url[:70]}...")
            
            try:
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
        
        context.close()
        print("="*80)
        print(f"\n✅ Downloaded {len(downloads)}/{len(image_results)} images\n")
        
        return downloads