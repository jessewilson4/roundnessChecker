"""
Pexels API integration for image search.
Searches for high-quality isolated object images.
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
        print(f"\nðŸ” Searching Pexels for: '{search_term}'")
        
        # Build multiple query variations for best results
        queries = self._build_queries(search_term)
        
        all_results = []
        seen_ids = set()
        
        for query in queries:
            if len(all_results) >= num_images:
                break
                
            print(f"  Trying query: '{query}'")
            
            try:
                results = self._fetch_query(query, num_images - len(all_results))
                
                # Deduplicate by photo ID
                for result in results:
                    if result['id'] not in seen_ids:
                        all_results.append(result)
                        seen_ids.add(result['id'])
                
                print(f"  âœ“ Found {len(results)} images (total: {len(all_results)})")
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                print(f"  âœ— Query failed: {e}")
                continue
        
        print(f"âœ“ Retrieved {len(all_results)} total images")
        return all_results[:num_images]
    
    
    def _build_queries(self, search_term: str) -> List[str]:
        """
        Build query variations for objects.
        
        Args:
            search_term: Base search term
            
        Returns:
            List of query strings to try
        """
        # Simplified: just use the plain search term
        return [
            f"{search_term}",
        ]
    
    
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
            'orientation': 'square',  # Prefer centered objects
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
                'height': photo['height']
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
            print(f"  âœ— Failed to download image: {e}")
            return None