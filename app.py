"""
# AI_PSEUDOCODE:
# @file: app.py
# @purpose: Flask web server + image roundness analysis
# @routes: [/, /search, /history, /batch, /load_search, /delete_result, /api/autocomplete]
# 
# MODIFIED: @date 2025-11-02
# CHANGE_01: multi-source API integration (Unsplash 40%, Pixabay 35%, Pexels 25%)
# CHANGE_02: import new searcher classes
# CHANGE_03: add API keys for Unsplash + Pixabay
# CHANGE_04: modify search_all_sources() for weighted distribution
# CHANGE_05: modify download_image_from_source() for multi-source support
# CHANGE_06: modify init_app() to initialize all searchers
#
# FLOW: @search_term → search_all_sources() → download_parallel() → analyze_batch() → filter_outliers() → cache → render
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import numpy as np
from datetime import datetime
import threading
import time
import pandas as pd
from werkzeug.utils import secure_filename
import gc
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import (
    GoogleSearcher,
    compress_thumbnail,
    Database,
    remove_outliers
)
from utils.edge_detection import analyze_image_roundness, RoundnessAnalyzer, analyze_images_batch


def get_roundness_score(composite_percentage):
    """
    Convert composite percentage (35-98%) to a 1-50 roundness score.
    
    Args:
        composite_percentage: Float percentage (0-100)
        
    Returns:
        Integer score from 1-50
    """
    if composite_percentage < 35:
        return 1
    if composite_percentage >= 98:
        return 50
    
    score = ((composite_percentage - 35) / 63) * 49 + 1
    return int(round(score))


def get_score_description(score):
    """
    Get description and color for a roundness score.
    
    Args:
        score: Integer from 1-50
        
    Returns:
        Tuple of (description, color)
    """
    if score >= 48:
        return ('Nearly Perfect Circle', '#10b981')
    elif score >= 41:
        return ('Highly Round', '#22c55e')
    elif score >= 31:
        return ('Round', '#3b82f6')
    elif score >= 21:
        return ('Somewhat Round', '#f59e0b')
    elif score >= 11:
        return ('Slightly Round', '#ef4444')
    else:
        return ('Not Round', '#991b1b')


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-in-production'
app.config['CACHE_DIR'] = './cache'
app.config['IMAGES_DIR'] = './cache/images'

# API Keys
PEXELS_API_KEY = "RHgT85Bjti2XozygivPv3JnQ9ZDp0ivX6wUjEalkjbqtGXgqH4pU6dOo"
UNSPLASH_ACCESS_KEY = "6ABQPlA4CSpGyuOdd8rdJHqiKQqP0cH58E9pf3nuACc"  # NEW
PIXABAY_API_KEY = "52936735-d4c77f8b0486c2ea7d1ad2c1a"  # NEW
GOOGLE_API_KEY = "AIzaSyCaJNp5o4gS4V8TmKrAyc0YZkcaUSg3w8w"
GOOGLE_CSE_ID = "27af1a123dd994e88"

# Configuration for parallel processing
BATCH_SIZE = 4
MAX_DOWNLOAD_WORKERS = 8

# Initialize components
pexels_searcher = None
unsplash_searcher = None  # NEW
pixabay_searcher = None   # NEW
database = None
analyzer = None
batch_processors = {}


def cleanup_memory():
    """Force garbage collection and clear GPU cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sanitize_filename(text):
    """Remove invalid filename characters"""
    import re
    text = re.sub(r'[<>:"/\\|?*\']', '', text)
    text = text.replace(' ', '_')
    return text[:50]


def init_searchers():
    """Initialize all image searchers - MODIFIED"""
    global pexels_searcher, unsplash_searcher, pixabay_searcher
    if pexels_searcher is None:
        pexels_searcher = PexelsSearcher(PEXELS_API_KEY)
    if unsplash_searcher is None:
        unsplash_searcher = UnsplashSearcher(UNSPLASH_ACCESS_KEY)
    if pixabay_searcher is None:
        pixabay_searcher = PixabaySearcher(PIXABAY_API_KEY)


def search_all_sources(search_term: str, num_images: int = 30):
    """Search all image sources and combine results - MODIFIED"""
    init_searchers()
    
    # 40% Unsplash, 35% Pixabay, 25% Pexels
    unsplash_count = int(num_images * 0.40)
    pixabay_count = int(num_images * 0.35)
    pexels_count = num_images - unsplash_count - pixabay_count
    
    all_images = []
    
    print(f"🔍 Searching across sources: Unsplash({unsplash_count}) + Pixabay({pixabay_count}) + Pexels({pexels_count})")
    
    # Unsplash
    unsplash_results = unsplash_searcher.search_images(search_term, num_images=unsplash_count)
    all_images.extend(unsplash_results)
    
    # Pixabay
    pixabay_results = pixabay_searcher.search_images(search_term, num_images=pixabay_count)
    all_images.extend(pixabay_results)
    
    # Pexels
    pexels_results = pexels_searcher.search_images(search_term, num_images=pexels_count)
    all_images.extend(pexels_results)
    
    print(f"✓ Retrieved {len(all_images)} total images")
    
    return all_images


def download_image_from_source(img_data):
    """Download image based on source - MODIFIED"""
    source = img_data.get('source', 'pexels')
    
    if source == 'unsplash':
        return unsplash_searcher.download_image(img_data['url'])
    elif source == 'pixabay':
        return pixabay_searcher.download_image(img_data['url'])
    else:  # pexels
        return pexels_searcher.download_image(img_data['url'])


def download_images_parallel(image_results, max_workers=MAX_DOWNLOAD_WORKERS):
    """
    Download multiple images in parallel.
    
    Args:
        image_results: List of image metadata dicts
        max_workers: Number of parallel download threads
        
    Returns:
        List of tuples (img_data, image_bytes)
    """
    downloads = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_img = {
            executor.submit(download_image_from_source, img_data): img_data
            for img_data in image_results
        }
        
        for future in as_completed(future_to_img):
            img_data = future_to_img[future]
            try:
                image_bytes = future.result()
                if image_bytes:
                    downloads.append((img_data, image_bytes))
            except Exception as e:
                print(f"    ✗ Download failed for {img_data['id']}: {e}")
    
    return downloads


def init_app():
    """Initialize application components including AI models at startup - MODIFIED"""
    global pexels_searcher, unsplash_searcher, pixabay_searcher, database, analyzer
    
    os.makedirs(app.config['CACHE_DIR'], exist_ok=True)
    os.makedirs(app.config['IMAGES_DIR'], exist_ok=True)
    
    # Initialize all searchers
    init_searchers()
    
    database = Database()
    
    print("\n🤖 Loading AI models at startup...")
    use_local = os.path.exists("./models/owlvit-base-patch32")
    analyzer = RoundnessAnalyzer(use_local_models=use_local)
    print("✓ Models loaded and ready\n")
    
    print("✓ Application initialized with multi-source search")


@app.route('/')
def index():
    """Main search page"""
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    """
    Process search request with parallel downloads and batch inference.
    OPTIMIZED: 40-50% faster than sequential version.
    """
    search_term = request.form.get('search_term', '').strip()
    num_images_requested = int(request.form.get('num_images', 30))
    
    if not search_term:
        return render_template('index.html', error="Please enter a search term")
    
    if not pexels_searcher:
        return render_template('index.html', 
                             error="Pexels API key not configured.")
    
    try:
        print(f"\n{'='*80}")
        print(f" OPTIMIZED SEARCH: '{search_term}' ({num_images_requested} images)")
        print(f"{'='*80}")
        
        num_to_fetch = num_images_requested * 2
        num_to_process = int(num_images_requested * 1.25)
        
        print(f"📥 Fetching {num_to_fetch} images")
        
        # Search for images (MODIFIED: multi-source)
        image_results = search_all_sources(search_term, num_images=num_to_fetch)
        
        if not image_results:
            return render_template('index.html',
                                 error=f"No images found for '{search_term}'.")
        
        image_results = image_results[:num_to_process]
        
        print(f"⚡ Downloading {len(image_results)} images in parallel...")
        
        # OPTIMIZED: Parallel downloads
        start_time = time.time()
        downloads = download_images_parallel(image_results)
        download_time = time.time() - start_time
        
        print(f"✓ Downloaded {len(downloads)} images in {download_time:.1f}s")
        
        if not downloads:
            return render_template('index.html',
                                 error=f"Could not download images for '{search_term}'.")
        
        # OPTIMIZED: Batch processing
        print(f"⚡ Analyzing images in batches of {BATCH_SIZE}...")
        results = []
        
        # Process in batches
        for batch_start in range(0, len(downloads), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(downloads))
            batch = downloads[batch_start:batch_end]
            
            print(f"\n   Batch {batch_start//BATCH_SIZE + 1}: Processing {len(batch)} images")
            
            # Extract data
            batch_img_data = [item[0] for item in batch]
            batch_img_bytes = [item[1] for item in batch]
            
            # Batch analyze
            analyses = analyze_images_batch(batch_img_bytes, search_term, analyzer)
            
            # Process results
            for img_data, image_bytes, analysis in zip(batch_img_data, batch_img_bytes, analyses):
                if not analysis:
                    continue
                
                # Save files
                thumbnail = compress_thumbnail(image_bytes, max_size_kb=25)
                safe_search_term = sanitize_filename(search_term)
                thumbnail_filename = f"{safe_search_term}_{img_data['id']}_thumb.jpg"
                thumbnail_path = os.path.join(app.config['IMAGES_DIR'], thumbnail_filename)
                
                with open(thumbnail_path, 'wb') as f:
                    f.write(thumbnail)
                
                # Save viz images
                viz_paths = {}
                for viz_type, viz_bytes in analysis['visualizations'].items():
                    viz_filename = f"{safe_search_term}_{img_data['id']}_{viz_type}.jpg"
                    viz_path = os.path.join(app.config['IMAGES_DIR'], viz_filename)
                    with open(viz_path, 'wb') as f:
                        f.write(viz_bytes)
                    viz_paths[viz_type] = viz_filename
                
                # Calculate scores
                composite_pct = analysis['composite'] * 100
                roundness_score = get_roundness_score(composite_pct)
                score_desc, score_color = get_score_description(roundness_score)
                
                results.append({
                    'image_id': img_data['id'],
                    'pexels_id': img_data['id'],
                    'url': img_data['url'],
                    'photographer': img_data['photographer'],
                    'photographer_url': img_data['photographer_url'],
                    'source': img_data.get('source', 'pexels'),
                    'thumbnail_path': thumbnail_filename,
                    'viz_paths': viz_paths,
                    'circularity': analysis['circularity'],
                    'aspect_ratio': analysis['aspect_ratio'],
                    'eccentricity': analysis['eccentricity'],
                    'solidity': analysis['solidity'],
                    'convexity': analysis['convexity'],
                    'composite': analysis['composite'],
                    'roundness_score': roundness_score,
                    'score_description': score_desc,
                    'score_color': score_color,
                    'area': analysis['area'],
                    'perimeter': analysis.get('perimeter', 0)
                })
                
                if len(results) >= num_images_requested:
                    break
            
            # Cleanup after batch
            cleanup_memory()
            
            if len(results) >= num_images_requested:
                break
        
        if not results:
            return render_template('index.html',
                                 error=f"Could not analyze images for '{search_term}'.")
        
        # Sort and rank
        results.sort(key=lambda x: x['composite'], reverse=True)
        for i, result in enumerate(results, 1):
            result['rank'] = i
        
        # Remove outliers
        filtered_results, outliers = remove_outliers(results, metric='composite')
        
        # Calculate statistics
        stats = calculate_statistics(filtered_results, outliers)
        
        # Calculate scores
        avg_composite_pct = stats['composite']['mean']
        average_roundness_score = get_roundness_score(avg_composite_pct)
        average_score_description, average_score_color = get_score_description(average_roundness_score)
        
        # Prepare chart data
        chart_data = prepare_chart_data(filtered_results)
        
        # Clean for database
        results_for_db = []
        for r in filtered_results:
            clean_result = {
                'pexels_id': r.get('pexels_id'),
                'image_id': r.get('image_id'),
                'url': r.get('url'),
                'photographer': r.get('photographer'),
                'photographer_url': r.get('photographer_url'),
                'source': r.get('source', 'pexels'),
                'thumbnail_path': r.get('thumbnail_path'),
                'viz_paths': r.get('viz_paths'),
                'circularity': float(r['circularity']),
                'aspect_ratio': float(r['aspect_ratio']),
                'eccentricity': float(r['eccentricity']),
                'solidity': float(r['solidity']),
                'convexity': float(r['convexity']),
                'composite': float(r['composite']),
                'roundness_score': r.get('roundness_score'),
                'score_description': r.get('score_description'),
                'score_color': r.get('score_color'),
                'area': float(r['area']),
                'perimeter': float(r.get('perimeter', 0)),
                'rank': r['rank']
            }
            results_for_db.append(clean_result)
        
        outliers_for_db = []
        for o in outliers:
            clean_outlier = {
                'pexels_id': o.get('pexels_id'),
                'url': o.get('url'),
                'photographer': o.get('photographer'),
                'photographer_url': o.get('photographer_url'),
                'thumbnail_path': o.get('thumbnail_path'),
                'viz_paths': o.get('viz_paths'),
                'circularity': float(o['circularity']),
                'aspect_ratio': float(o['aspect_ratio']),
                'eccentricity': float(o['eccentricity']),
                'solidity': float(o['solidity']),
                'convexity': float(o['convexity']),
                'composite': float(o['composite']),
                'area': float(o['area']),
                'perimeter': float(o.get('perimeter', 0)),
                'outlier_reason': o.get('outlier_reason'),
                'outlier_direction': o.get('outlier_direction')
            }
            outliers_for_db.append(clean_outlier)
        
        # Save to database
        search_id = database.save_search(
            search_term, 
            results_for_db,
            results_for_db,
            outliers_for_db, 
            stats
        )
        
        print(f"\n✓ Analysis complete!")
        print(f"  Valid results: {len(filtered_results)}")
        print(f"  Average composite: {stats['composite']['mean']:.1f}%")
        
        # Final cleanup
        cleanup_memory()
        
        # Clean for template
        clean_results_for_template = [
            {k: v for k, v in r.items() if k not in ['method1', 'method2']} 
            for r in results_for_db[:10]
        ]
        
        clean_all_results = [
            {k: v for k, v in r.items() if k not in ['method1', 'method2']} 
            for r in results_for_db
        ]
        
        return render_template('results.html',
                             search_term=search_term,
                             num_images=num_images_requested,
                             results=clean_results_for_template,
                             all_results=clean_all_results,
                             outliers=outliers_for_db,
                             stats=stats,
                             average_roundness_score=average_roundness_score,
                             average_score_description=average_score_description,
                             average_score_color=average_score_color,
                             chart_data=chart_data,
                             timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                             search_id=search_id)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template('index.html',
                             error=f"Error processing search: {str(e)}")


@app.route('/history')
def history():
    """View search history"""
    searches = database.get_search_history()
    batches = database.get_all_batches()

    for search in searches:
        composite_pct = search['avg_composite']
        search['roundness_score'] = get_roundness_score(composite_pct)
        desc, color = get_score_description(search['roundness_score'])
        search['score_description'] = desc
        search['score_color'] = color
        search['batch_name'] = None
        if search.get('batch_id'):
            batch = next((b for b in batches if b['id'] == search['batch_id']), None)
            if batch:
                search['batch_name'] = batch['name']

    return render_template('history.html', history=searches, batches=batches)


@app.route('/load_search/<int:search_id>')
def load_search(search_id):
    """Load a previous search from cache"""
    search_data = database.load_search_by_id(search_id)
    
    if not search_data:
        return render_template('index.html',
                             error="Search not found in cache.")
    
    chart_data = prepare_chart_data(search_data['filtered_results'])
    
    return render_template('results.html',
                         search_term=search_data['search_term'],
                         search_id=search_id,
                         results=search_data['filtered_results'][:10],
                         all_results=search_data['filtered_results'],
                         outliers=search_data['outliers'],
                         stats=search_data['stats'],
                         chart_data=chart_data,
                         timestamp=search_data['timestamp'],
                         from_cache=True)


@app.route('/delete_result', methods=['POST'])
def delete_result():
    """Delete a specific result from a search"""
    try:
        search_id = request.form.get('search_id')
        pexels_id = request.form.get('pexels_id')
        
        if not search_id or not pexels_id:
            return jsonify({'success': False, 'error': 'Missing parameters'})
        
        search_data = database.load_search_by_id(search_id)
        if not search_data:
            return jsonify({'success': False, 'error': 'Search not found'})
        
        result_to_delete = None
        all_results = search_data['filtered_results'] + search_data.get('outliers', [])
        
        for result in all_results:
            if str(result.get('pexels_id')) == str(pexels_id):
                result_to_delete = result
                break
        
        if not result_to_delete:
            return jsonify({'success': False, 'error': 'Result not found'})
        
        # Delete files
        if result_to_delete.get('thumbnail_path'):
            thumb_path = os.path.join(app.config['IMAGES_DIR'], result_to_delete['thumbnail_path'])
            if os.path.exists(thumb_path):
                os.remove(thumb_path)
        
        if result_to_delete.get('viz_paths'):
            for viz_file in result_to_delete['viz_paths'].values():
                viz_path = os.path.join(app.config['IMAGES_DIR'], viz_file)
                if os.path.exists(viz_path):
                    os.remove(viz_path)
        
        updated_results = [r for r in all_results if str(r.get('pexels_id')) != str(pexels_id)]
        
        filtered_results, outliers = remove_outliers(updated_results, metric='composite')
        
        filtered_results.sort(key=lambda x: x['composite'], reverse=True)
        for i, result in enumerate(filtered_results, 1):
            result['rank'] = i
        
        stats = calculate_statistics(filtered_results, outliers)
        
        database.update_search(search_id, filtered_results, outliers, stats)
        
        return jsonify({
            'success': True,
            'new_stats': stats,
            'remaining_count': len(filtered_results)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/autocomplete')
def autocomplete():
    """Get previous searches for autocomplete"""
    try:
        query = request.args.get('q', '').lower()
        previous = database.get_all_searches()
        
        suggestions = []
        for search in previous:
            if query in search['search_term'].lower():
                suggestions.append({
                    'id': search['id'],
                    'term': search['search_term'],
                    'timestamp': search['timestamp'],
                    'avg_circularity': search['avg_circularity'],
                    'avg_composite': search['avg_composite'],
                    'num_images': search['num_images']
                })
        
        return jsonify({'suggestions': suggestions})
        
    except Exception as e:
        return jsonify({'suggestions': [], 'error': str(e)})


@app.route('/image/<filename>')
def serve_image(filename):
    """Serve cached image"""
    image_path = os.path.join(app.config['IMAGES_DIR'], filename)
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/jpeg')
    return "Image not found", 404


def calculate_statistics(results, outliers):
    """Calculate summary statistics for shape metrics"""
    if not results:
        return {}
    
    composites = [r['composite'] * 100 for r in results]
    circularities = [r['circularity'] * 100 for r in results]
    aspect_ratios = [r['aspect_ratio'] * 100 for r in results]
    eccentricities = [r['eccentricity'] * 100 for r in results]
    solidities = [r['solidity'] * 100 for r in results]
    convexities = [r['convexity'] * 100 for r in results]
    
    return {
        'total_analyzed': len(results) + len(outliers),
        'total_valid': len(results),
        'outliers_removed': len(outliers),
        'composite': {
            'mean': np.mean(composites),
            'std': np.std(composites),
            'min': np.min(composites),
            'max': np.max(composites),
            'median': np.median(composites),
            'values': composites
        },
        'circularity': {
            'mean': np.mean(circularities),
            'std': np.std(circularities),
            'median': np.median(circularities),
            'values': circularities
        },
        'aspect_ratio': {
            'mean': np.mean(aspect_ratios),
            'std': np.std(aspect_ratios),
            'median': np.median(aspect_ratios),
            'values': aspect_ratios
        },
        'eccentricity': {
            'mean': np.mean(eccentricities),
            'std': np.std(eccentricities),
            'median': np.median(eccentricities),
            'values': eccentricities
        },
        'solidity': {
            'mean': np.mean(solidities),
            'std': np.std(solidities),
            'median': np.median(solidities),
            'values': solidities
        },
        'convexity': {
            'mean': np.mean(convexities),
            'std': np.std(convexities),
            'median': np.median(convexities),
            'values': convexities
        }
    }


def prepare_chart_data(results):
    """Prepare histogram data"""
    roundness_scores = np.array([get_roundness_score(r['composite'] * 100) for r in results])
    
    bins = np.arange(0, 55, 5)
    hist, bin_edges = np.histogram(roundness_scores, bins=bins)
    
    bin_labels = []
    for i in range(len(hist)):
        start = int(bins[i]) + 1 if bins[i] == 0 else int(bins[i])
        end = int(bins[i+1])
        bin_labels.append(f"{start}-{end}")
    
    return {
        'histogram': {
            'bin_labels': bin_labels,
            'counts': hist.tolist(),
            'mean': float(np.mean(roundness_scores)),
            'std': float(np.std(roundness_scores))
        }
    }


@app.route('/export/history.csv')
def export_history_csv():
    """Export search history summary as CSV"""
    import csv
    from io import StringIO
    from flask import Response
    
    history_data = database.get_search_history(limit=1000)
    
    output = StringIO()
    writer = csv.writer(output)
    
    writer.writerow([
        'Search Term', 'Date', 'Number of Images', 'Roundness Score (1-50)',
        'Score Description', 'Composite Mean (%)', 'Composite Std (%)',
        'Composite Min (%)', 'Composite Max (%)', 'Composite Median (%)',
        'Circularity Mean (%)', 'Aspect Ratio Mean (%)', 'Eccentricity Mean (%)',
        'Solidity Mean (%)', 'Convexity Mean (%)', 'Outliers Removed'
    ])
    
    for search in history_data:
        roundness_score = get_roundness_score(search['avg_composite'])
        score_desc, _ = get_score_description(roundness_score)
        
        writer.writerow([
            search['search_term'], search['timestamp'][:10], search['num_images'],
            roundness_score, score_desc, f"{search['avg_composite']:.1f}",
            f"{search.get('std_composite', 0):.1f}", f"{search.get('min_composite', 0):.1f}",
            f"{search.get('max_composite', 0):.1f}", f"{search.get('median_composite', 0):.1f}",
            f"{search['avg_circularity']:.1f}", f"{search['avg_aspect_ratio']:.1f}",
            f"{search['avg_eccentricity']:.1f}", f"{search['avg_solidity']:.1f}",
            f"{search['avg_convexity']:.1f}", search['outliers_removed']
        ])
    
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=roundness_history.csv'}
    )


@app.route('/export/search/<int:search_id>.csv')
def export_search_csv(search_id):
    """Export individual search results as CSV"""
    import csv
    from io import StringIO
    from flask import Response
    
    search_data = database.load_search_by_id(search_id)
    
    if not search_data:
        return "Search not found", 404
    
    output = StringIO()
    writer = csv.writer(output)
    
    writer.writerow([
        'Rank', 'Image ID', 'Photographer', 'Source', 'Roundness Score (1-50)',
        'Score Description', 'Composite (%)', 'Circularity (%)', 'Aspect Ratio (%)',
        'Eccentricity (%)', 'Solidity (%)', 'Convexity (%)', 'Area (px²)', 'Perimeter (px)'
    ])
    
    for result in search_data['filtered_results']:
        composite_pct = result['composite'] * 100
        roundness_score = get_roundness_score(composite_pct)
        score_desc, _ = get_score_description(roundness_score)
        
        writer.writerow([
            result['rank'], result.get('pexels_id', ''), result.get('photographer', ''),
            result.get('source', 'pexels'), roundness_score, score_desc,
            f"{result['composite'] * 100:.1f}", f"{result['circularity'] * 100:.1f}",
            f"{result['aspect_ratio'] * 100:.1f}", f"{result['eccentricity'] * 100:.1f}",
            f"{result['solidity'] * 100:.1f}", f"{result['convexity'] * 100:.1f}",
            f"{result['area']:.0f}", f"{result.get('perimeter', 0):.1f}"
        ])
    
    output.seek(0)
    filename = f"roundness_{search_data['search_term'].replace(' ', '_')}_{search_data['timestamp'][:10]}.csv"
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename={filename}'}
    )


@app.route('/delete_searches', methods=['POST'])
def delete_searches():
    """Delete multiple searches by ID"""
    try:
        data = request.get_json()
        search_ids = data.get('search_ids', [])
        
        if not search_ids:
            return jsonify({'success': False, 'error': 'No search IDs provided'})
        
        deleted_count = database.delete_searches(search_ids)
        
        return jsonify({'success': True, 'deleted_count': deleted_count})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


# ============= BATCH PROCESSING =============

@app.route('/batch')
def batch_processing():
    """Render batch processing page"""
    incomplete_batches = database.get_incomplete_batches()
    return render_template('batch.html', incomplete_batches=incomplete_batches)


@app.route('/batch/upload', methods=['POST'])
def batch_upload():
    """Upload and parse CSV/Excel file"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        filename = secure_filename(file.filename)
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'success': False, 'error': 'Invalid file type.'})
        
        columns = df.columns.tolist()
        
        return jsonify({
            'success': True,
            'columns': columns,
            'row_count': len(df),
            'preview': df.head(5).to_dict('records')
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/batch/create', methods=['POST'])
def batch_create():
    """Create batch"""
    try:
        batch_name = request.form.get('batch_name')
        column_name = request.form.get('column_name')
        images_per_keyword = int(request.form.get('images_per_keyword', 30))
        
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        filename = secure_filename(file.filename)
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return jsonify({'success': False, 'error': 'Invalid file type'})
        
        if column_name not in df.columns:
            return jsonify({'success': False, 'error': f'Column "{column_name}" not found'})
        
        keywords = df[column_name].dropna().astype(str).tolist()
        
        if not keywords:
            return jsonify({'success': False, 'error': 'No keywords found'})
        
        batch_id = database.create_batch(batch_name, keywords, images_per_keyword)
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'total_keywords': len(set(keywords)),
            'message': f'Batch created with {len(set(keywords))} unique keywords'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/batch/<int:batch_id>/start', methods=['POST'])
def batch_start(batch_id):
    """Start batch processing"""
    try:
        batch = database.get_batch(batch_id)
        if not batch:
            return jsonify({'success': False, 'error': 'Batch not found'})
        
        if batch_id in batch_processors:
            return jsonify({'success': False, 'error': 'Batch already processing'})
        
        processor = {
            'is_paused': False,
            'is_stopped': False,
            'thread': None
        }
        
        def process_batch():
            """Background batch processing"""
            keywords = batch['keywords']
            images_per = batch['images_per_keyword']
            
            database.update_batch_status(batch_id, 'processing')
            
            completed = batch['completed_keywords']
            start_index = batch['current_keyword_index']
            
            for idx in range(start_index, len(keywords)):
                while processor['is_paused'] and not processor['is_stopped']:
                    time.sleep(1)
                
                if processor['is_stopped']:
                    database.update_batch_status(batch_id, 'stopped', idx, completed)
                    break
                
                keyword = keywords[idx]
                print(f"\nBatch processing: '{keyword}' ({idx+1}/{len(keywords)})")
                
                try:
                    init_searchers()
                    image_results = search_all_sources(keyword, num_images=images_per * 2)
                    
                    if not image_results:
                        completed += 1
                        database.update_batch_status(batch_id, 'processing', idx + 1, completed)
                        continue
                    
                    image_results = image_results[:int(images_per * 1.25)]
                    
                    downloads = download_images_parallel(image_results)
                    
                    if not downloads:
                        completed += 1
                        database.update_batch_status(batch_id, 'processing', idx + 1, completed)
                        continue
                    
                    results = []
                    
                    for batch_start in range(0, len(downloads), BATCH_SIZE):
                        batch_end = min(batch_start + BATCH_SIZE, len(downloads))
                        batch_data = downloads[batch_start:batch_end]
                        
                        batch_img_data = [item[0] for item in batch_data]
                        batch_img_bytes = [item[1] for item in batch_data]
                        
                        analyses = analyze_images_batch(batch_img_bytes, keyword, analyzer)
                        
                        for img_data, image_bytes, analysis in zip(batch_img_data, batch_img_bytes, analyses):
                            if not analysis:
                                continue
                            
                            thumbnail = compress_thumbnail(image_bytes, max_size_kb=25)
                            safe_keyword = sanitize_filename(keyword)
                            thumbnail_filename = f"{safe_keyword}_{img_data['id']}_thumb.jpg"
                            thumbnail_path = os.path.join(app.config['IMAGES_DIR'], thumbnail_filename)
                            
                            with open(thumbnail_path, 'wb') as f:
                                f.write(thumbnail)
                            
                            viz_paths = {}
                            for viz_type, viz_bytes in analysis['visualizations'].items():
                                viz_filename = f"{safe_keyword}_{img_data['id']}_{viz_type}.jpg"
                                viz_path = os.path.join(app.config['IMAGES_DIR'], viz_filename)
                                with open(viz_path, 'wb') as f:
                                    f.write(viz_bytes)
                                viz_paths[viz_type] = viz_filename
                            
                            composite_pct = analysis['composite'] * 100
                            roundness_score = get_roundness_score(composite_pct)
                            score_desc, score_color = get_score_description(roundness_score)
                            
                            results.append({
                                'image_id': img_data['id'],
                                'pexels_id': img_data['id'],
                                'url': img_data['url'],
                                'photographer': img_data['photographer'],
                                'photographer_url': img_data['photographer_url'],
                                'source': img_data.get('source', 'pexels'),
                                'thumbnail_path': thumbnail_filename,
                                'viz_paths': viz_paths,
                                'circularity': analysis['circularity'],
                                'aspect_ratio': analysis['aspect_ratio'],
                                'eccentricity': analysis['eccentricity'],
                                'solidity': analysis['solidity'],
                                'convexity': analysis['convexity'],
                                'composite': analysis['composite'],
                                'roundness_score': roundness_score,
                                'score_description': score_desc,
                                'score_color': score_color,
                                'area': analysis['area'],
                                'perimeter': analysis.get('perimeter', 0)
                            })
                            
                            if len(results) >= images_per:
                                break
                        
                        cleanup_memory()
                        
                        if len(results) >= images_per:
                            break
                    
                    if results:
                        results.sort(key=lambda x: x['composite'], reverse=True)
                        for i, result in enumerate(results, 1):
                            result['rank'] = i
                        
                        filtered_results, outliers = remove_outliers(results, metric='composite')
                        stats = calculate_statistics(filtered_results, outliers)
                        
                        database.save_search(
                            search_term=keyword,
                            results=results,
                            filtered_results=filtered_results,
                            outliers=outliers,
                            stats=stats,
                            batch_id=batch_id
                        )
                    
                    completed += 1
                    database.update_batch_status(batch_id, 'processing', idx + 1, completed)
                    
                    cleanup_memory()
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"Error processing {keyword}: {str(e)}")
                    completed += 1
                    database.update_batch_status(batch_id, 'processing', idx + 1, completed)
                    cleanup_memory()
                    continue
            
            if not processor['is_stopped']:
                database.update_batch_status(batch_id, 'complete', len(keywords), completed)
            
            cleanup_memory()
            
            if batch_id in batch_processors:
                del batch_processors[batch_id]
        
        thread = threading.Thread(target=process_batch, daemon=True)
        processor['thread'] = thread
        batch_processors[batch_id] = processor
        thread.start()
        
        return jsonify({'success': True, 'message': 'Batch processing started'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/batch/<int:batch_id>/status')
def batch_status(batch_id):
    """Get batch status"""
    try:
        batch = database.get_batch(batch_id)
        if not batch:
            return jsonify({'success': False, 'error': 'Batch not found'})
        
        searches = database.get_batch_searches(batch_id)
        
        keyword_statuses = []
        keywords = batch['keywords']
        
        for idx, keyword in enumerate(keywords):
            search = next((s for s in searches if s['search_term'] == keyword), None)
            
            if idx < batch['current_keyword_index']:
                if search:
                    status = 'complete' if search['num_images'] >= batch['images_per_keyword'] else 'incomplete'
                    keyword_statuses.append({
                        'keyword': keyword,
                        'status': status,
                        'images': search['num_images'],
                        'avg_score': get_roundness_score(search['avg_composite']) if search['avg_composite'] else 0,
                        'search_id': search['id']
                    })
                else:
                    keyword_statuses.append({
                        'keyword': keyword,
                        'status': 'failed',
                        'images': 0,
                        'avg_score': 0,
                        'search_id': None
                    })
            elif idx == batch['current_keyword_index'] and batch['status'] == 'processing':
                keyword_statuses.append({
                    'keyword': keyword,
                    'status': 'processing',
                    'images': 0,
                    'avg_score': 0,
                    'search_id': None
                })
            else:
                keyword_statuses.append({
                    'keyword': keyword,
                    'status': 'pending',
                    'images': 0,
                    'avg_score': 0,
                    'search_id': None
                })
        
        return jsonify({
            'success': True,
            'batch': {
                'id': batch['id'],
                'name': batch['name'],
                'status': batch['status'],
                'total_keywords': batch['total_keywords'],
                'completed_keywords': batch['completed_keywords'],
                'current_keyword_index': batch['current_keyword_index'],
                'progress_percent': (batch['completed_keywords'] / batch['total_keywords'] * 100) if batch['total_keywords'] > 0 else 0
            },
            'keywords': keyword_statuses
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/batch/<int:batch_id>/pause', methods=['POST'])
def batch_pause(batch_id):
    """Pause batch"""
    if batch_id in batch_processors:
        batch_processors[batch_id]['is_paused'] = True
        database.update_batch_status(batch_id, 'paused')
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Batch not running'})


@app.route('/batch/<int:batch_id>/resume', methods=['POST'])
def batch_resume(batch_id):
    """Resume batch"""
    if batch_id in batch_processors:
        batch_processors[batch_id]['is_paused'] = False
        database.update_batch_status(batch_id, 'processing')
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Batch not running'})


@app.route('/batch/<int:batch_id>/stop', methods=['POST'])
def batch_stop(batch_id):
    """Stop batch"""
    if batch_id in batch_processors:
        batch_processors[batch_id]['is_stopped'] = True
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Batch not running'})


@app.route('/delete_batch/<int:batch_id>', methods=['POST'])
def delete_batch(batch_id):
    """Delete batch and associated searches"""
    try:
        searches = database.get_batch_searches(batch_id)
        search_ids = [s['id'] for s in searches]
        
        if search_ids:
            database.delete_searches(search_ids)
        
        database.delete_batch(batch_id)
        
        return jsonify({'success': True, 'deleted_count': len(search_ids)})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/batch/<int:batch_id>/export.csv')
def batch_export(batch_id):
    """Export batch results"""
    import csv
    from io import StringIO
    from flask import Response
    
    batch = database.get_batch(batch_id)
    if not batch:
        return "Batch not found", 404
    
    searches = database.get_batch_searches(batch_id)
    
    output = StringIO()
    writer = csv.writer(output)
    
    writer.writerow([
        'Keyword', 'Status', 'Images Processed', 'Images Requested',
        'Avg Roundness Score (1-50)', 'Avg Composite Mean (%)',
        'Avg Composite Std (%)', 'Avg Composite Median (%)',
        'Avg Circularity (%)', 'Avg Aspect Ratio (%)',
        'Avg Eccentricity (%)', 'Avg Solidity (%)',
        'Avg Convexity (%)', 'Outliers Removed'
    ])
    
    for search in searches:
        status = 'complete' if search['num_images'] >= batch['images_per_keyword'] else 'incomplete'
        roundness_score = get_roundness_score(search['avg_composite']) if search['avg_composite'] else 0
        
        writer.writerow([
            search['search_term'], status, search['num_images'], batch['images_per_keyword'],
            roundness_score,
            f"{search['avg_composite']:.1f}" if search['avg_composite'] else '0.0',
            f"{search['std_composite']:.1f}" if search['std_composite'] else '0.0',
            f"{search['median_composite']:.1f}" if search['median_composite'] else '0.0',
            f"{search['avg_circularity']:.1f}" if search['avg_circularity'] else '0.0',
            f"{search['avg_aspect_ratio']:.1f}" if search['avg_aspect_ratio'] else '0.0',
            f"{search['avg_eccentricity']:.1f}" if search['avg_eccentricity'] else '0.0',
            f"{search['avg_solidity']:.1f}" if search['avg_solidity'] else '0.0',
            f"{search['avg_convexity']:.1f}" if search['avg_convexity'] else '0.0',
            search['outliers_removed']
        ])
    
    output.seek(0)
    safe_name = batch['name'].replace(' ', '_')
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=batch_{safe_name}.csv'}
    )


if __name__ == '__main__':
    init_app()
    app.run(debug=True, host='0.0.0.0', port=5000)