"""
Test script for Google Custom Search + Playwright image downloading
Downloads 10 images for 'dog' to cache/google_test/ for manual inspection
"""

import requests
import os
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import time

# Configuration
GOOGLE_API_KEY = "AIzaSyCaJNp5o4gS4V8TmKrAyc0YZkcaUSg3w8w"
GOOGLE_CSE_ID = "018409522456625749151:7mwyuw1w4bq"
OUTPUT_DIR = "./cache/google_test"
SEARCH_TERM = "isolated dog"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)


def search_google_images(search_term, num_results=10):
    """Search Google Custom Search API"""
    print(f"\n{'='*80}")
    print(f"SEARCHING GOOGLE FOR: '{search_term}'")
    print(f"{'='*80}\n")
    
    params = {
        'key': GOOGLE_API_KEY,
        'cx': GOOGLE_CSE_ID,
        'q': search_term,
        'searchType': 'image',
        'imgSize': 'xxlarge',  # Use xxlarge instead of large
        'num': 10,
        'safe': 'active'
    }
    
    response = requests.get("https://www.googleapis.com/customsearch/v1", params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    # Dump full JSON for inspection
    import json
    json_output = os.path.join(OUTPUT_DIR, 'api_response.json')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(json_output, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"📄 Full API response saved to: {json_output}\n")
    
    items = data.get('items', [])
    print(f"✅ API returned {len(items)} results\n")
    
    results = []
    for idx, item in enumerate(items, 1):
        image_data = item.get('image', {})
        result = {
            'index': idx,
            'url': item.get('link'),
            'context_url': image_data.get('contextLink', ''),
            'title': item.get('title', 'Untitled'),
            'metadata_width': image_data.get('width', 0),
            'metadata_height': image_data.get('height', 0),
            'byte_size': image_data.get('byteSize', 0),
            'thumbnail_url': image_data.get('thumbnailLink', ''),
            'mime': item.get('mime', '')
        }
        results.append(result)
        
        print(f"[{idx}] {result['title'][:60]}")
        print(f"    Direct URL: {result['url'][:70]}...")
        print(f"    Context: {result['context_url'][:70]}...")
        print(f"    Metadata: {result['metadata_width']}x{result['metadata_height']} ({result['byte_size']} bytes)")
        print(f"    MIME: {result['mime']}")
        print()
    
    return results


def download_images_with_playwright(results, browser):
    """Download all images using a single browser instance"""
    context = browser.new_context(
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0',
        viewport={'width': 1920, 'height': 1080},
        ignore_https_errors=True
    )
    page = context.new_page()
    
    downloaded = []
    
    for result in results:
        idx = result['index']
        url = result['url']
        
        print(f"\n{'─'*80}")
        print(f"[{idx}] {result['title'][:60]}")
        print(f"    URL: {url[:70]}...")
        print(f"    Expected: {result['metadata_width']}x{result['metadata_height']} ({result['byte_size']} bytes)")
        
        extension = url.split('.')[-1].split('?')[0][:4]
        output_path = os.path.join(OUTPUT_DIR, f"image_{idx:02d}.{extension}")
        
        try:
            print(f"    📥 Downloading...")
            response = page.goto(url, wait_until='load', timeout=15000)
            
            if response and response.status == 200:
                content_type = response.headers.get('content-type', '').lower()
                
                if 'image' in content_type:
                    image_bytes = response.body()
                    size_kb = len(image_bytes) // 1024
                    
                    print(f"    ✅ Downloaded {size_kb}KB")
                    print(f"    Content-Type: {content_type}")
                    
                    with open(output_path, 'wb') as f:
                        f.write(image_bytes)
                    
                    print(f"    💾 Saved: {output_path}")
                    downloaded.append((result, size_kb))
                else:
                    print(f"    ❌ Not an image: {content_type}")
            else:
                status = response.status if response else "No response"
                print(f"    ❌ Failed: Status {status}")
                
        except Exception as e:
            print(f"    ❌ Error: {str(e)[:60]}")
        
        time.sleep(0.3)  # Rate limiting
    
    context.close()
    return downloaded


def main():
    print(f"\nGoogle Image Download Test")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Search term: {SEARCH_TERM}")
    
    # Search
    results = search_google_images(SEARCH_TERM)
    
    if not results:
        print("\n❌ No results from Google API")
        return
    
    # Download with single browser instance
    print(f"\n{'='*80}")
    print(f"DOWNLOADING {len(results)} IMAGES")
    print(f"{'='*80}")
    
    with sync_playwright() as p:
        print("\n🎭 Launching Playwright browser (single instance)...")
        browser = p.chromium.launch(headless=True)
        
        downloaded = download_images_with_playwright(results, browser)
        
        browser.close()
        print("\n🎭 Browser closed")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total images: {len(results)}")
    print(f"✅ Downloaded: {len(downloaded)} ({len(downloaded)/len(results)*100:.0f}%)")
    print(f"❌ Failed: {len(results) - len(downloaded)}")
    
    if downloaded:
        total_kb = sum(kb for _, kb in downloaded)
        print(f"\nTotal size: {total_kb}KB ({total_kb/1024:.1f}MB)")
        print(f"Average size: {total_kb/len(downloaded):.0f}KB per image")
    
    print(f"\n📁 Images saved to: {OUTPUT_DIR}")
    print(f"   Run: ls -lh {OUTPUT_DIR}")
    print(f"\n📄 API response: {OUTPUT_DIR}/api_response.json")


if __name__ == '__main__':
    main()