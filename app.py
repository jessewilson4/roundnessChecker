"""
# === UAIPCS START ===
file: app.py
purpose: Flask web application for image roundness analysis using Google Custom Search API with OWL-ViT object detection and SAM segmentation, includes batch processing, comprehensive export functionality, and admin configuration interface
deps: [@flask:framework, @google_custom_search_api:service/external, @sqlite:storage, @transformers:library, @segment_anything:library, @torch:library, @opencv:library, @pandas:library, @numpy:library, @config:module/local]
funcs:
  - get_roundness_score(composite_percentage:float) -> int  # no_side_effect
  - get_score_description(score:int) -> tuple  # no_side_effect
  - sanitize_filename(text:str) -> str  # no_side_effect
  - cleanup_memory() -> None  # side_effect: garbage collection and GPU cache clear
  - init_searchers() -> None  # side_effect: initializes global google_searcher
  - search_all_sources(search_term:str, num_images:int) -> list  # side_effect: API calls
  - download_images_parallel(image_results:list) -> list  # side_effect: network I/O via GoogleSearcher
  - init_app() -> None  # side_effect: initializes globals, loads AI models, creates directories
  - calculate_statistics(results:list, outliers:list) -> dict  # no_side_effect
  - prepare_chart_data(results:list) -> dict  # no_side_effect
  - admin_page() -> str  # no_side_effect: renders admin template
  - admin_save() -> dict  # side_effect: saves config to disk
  - admin_reset() -> dict  # side_effect: resets config to defaults
refs:
  - config.py::load_config
  - config.py::save_config
  - config.py::get_setting
  - utils/google_search.py::GoogleSearcher
  - utils/database.py::Database
  - utils/edge_detection.py::RoundnessAnalyzer
  - utils/edge_detection.py::analyze_images_batch
  - templates/index.html
  - templates/results.html
  - templates/history.html
  - templates/batch.html
  - templates/admin.html
notes: perf=cold=3-5s|hot=<1s, persist=durable, concur=not_thread_safe, volatiledeps=[@google_api], memory=high_during_batch, config=json_file_based
# === UAIPCS END ===
"""

from flask import Flask, render_template, request, jsonify, send_file, Response
import os
import re
import numpy as np
from datetime import datetime
import threading
import time
import pandas as pd
from werkzeug.utils import secure_filename
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from io import StringIO
from utils.config import load_config, save_config, get_setting, reload_config, DEFAULT_CONFIG
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
        Tuple of (description, color_class)
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

# Google API Keys
GOOGLE_API_KEY = "AIzaSyCaJNp5o4gS4V8TmKrAyc0YZkcaUSg3w8w"
GOOGLE_CSE_ID = "018409522456625749151:7mwyuw1w4bq"

# Load configuration
BATCH_SIZE = get_setting('image_processing.batch_size', 4)

# Initialize components
google_searcher = None
database = None
analyzer = None
batch_processors = {}


def cleanup_memory():
    """Force garbage collection and clear CUDA cache"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass  # No torch available


def sanitize_filename(text):
    """Remove invalid filename characters"""
    import re
    text = re.sub(r'[<>:"/\\|?*\']', '', text)
    text = text.replace(' ', '_')
    return text[:50]


def init_searchers():
    """Initialize Google searcher"""
    global google_searcher
    if google_searcher is None:
        google_searcher = GoogleSearcher(GOOGLE_API_KEY, GOOGLE_CSE_ID)


def search_all_sources(search_term: str, num_images: int = 30, start_offset: int = 0, use_config_modifiers: bool = True):
    """Search Google for images"""
    init_searchers()
    return google_searcher.search_images(search_term, num_images, start_offset, use_config_modifiers)


def download_images_parallel(image_results):
    """
    Download multiple images using GoogleSearcher's batch method with internal browser management.
    
    Args:
        image_results: List of image metadata dicts
        
    Returns:
        List of tuples (img_data, image_bytes)
    """
    return google_searcher.download_images_batch(image_results)


def calculate_statistics(results, outliers):
    """Calculate comprehensive statistics"""
    if not results:
        return {}
    
    metrics = ['composite', 'circularity', 'aspect_ratio', 'eccentricity', 'solidity', 'convexity']
    stats = {}
    
    for metric in metrics:
        values = [r[metric] * 100 for r in results]
        stats[metric] = {
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    # Add totals for template
    stats['total_analyzed'] = len(results) + len(outliers)
    stats['total_valid'] = len(results)
    stats['outliers_removed'] = len(outliers)
    
    return stats


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


def init_app():
    """Initialize application components including AI models at startup"""
    global google_searcher, database, analyzer
    
    os.makedirs(app.config['CACHE_DIR'], exist_ok=True)
    os.makedirs(app.config['IMAGES_DIR'], exist_ok=True)
    
    # Initialize Google searcher
    init_searchers()
    
    database = Database()
    
    print("\n🤖 Loading AI models at startup...")
    use_local = os.path.exists("./models/owlvit-base-patch32")
    analyzer = RoundnessAnalyzer(use_local_models=use_local)
    print("✓ Models loaded and ready\n")
    
    print("✓ Application initialized with Google Custom Search")


@app.route('/')
def index():
    """Home page"""
    config = load_config()
    return render_template('index.html', config=config)


@app.route('/search', methods=['POST'])
def search():
    """
    Process search request with batch downloads and inference.
    Uses retry loop with pagination to fetch enough valid results.
    """
    search_term = request.form.get('search_term', '').strip()
    num_images_requested = int(request.form.get('num_images', 30))
    
    if not search_term:
        return render_template('index.html', error="Please enter a search term")
    
    if not google_searcher:
        return render_template('index.html', 
                             error="Google API not configured.")
    
    try:
        print(f"\n{'='*80}")
        print(f" SEARCH: '{search_term}' ({num_images_requested} images)")
        print(f"{'='*80}")
        
        # Phase 1.3: Add download limit to prevent runaway loops
        max_images_to_download = num_images_requested * 3  # Cap at 3x requested
        
        # Retry loop with pagination
        seen_urls = set()
        results = []
        pagination_offset = 0
        max_fetch_attempts = 5
        total_downloaded = 0
        
        print(f"⚙️  Settings: max_attempts={max_images_to_download}, batch_size=10\n")
        
        while len(results) < num_images_requested and total_downloaded < max_images_to_download and pagination_offset < max_fetch_attempts * 10:
            still_needed = num_images_requested - len(results)
            
            print(f"\n📥 Need {still_needed} more valid results. Fetching batch (offset={pagination_offset})...")
            
            # Fetch with pagination offset
            image_results = search_all_sources(search_term, num_images=10, start_offset=pagination_offset)
            
            if not image_results:
                print("⚠️  No more results from API")
                break
            
            # Filter already-tried URLs
            new_results = [r for r in image_results if r['url'] not in seen_urls]
            seen_urls.update(r['url'] for r in image_results)
            
            if not new_results:
                print("⚠️  All results already tried")
                pagination_offset += 10
                continue
            
            print(f"✓ Got {len(new_results)} new unique images")
            
            # Download
            print(f"⬇️  Downloading {len(new_results)} images...")
            print("="*80)
            downloads = download_images_parallel(new_results)
            print("="*80)
            
            if not downloads:
                print("⚠️  No successful downloads")
                pagination_offset += 10
                continue
            
            print(f"✅ Downloaded {len(downloads)} images")
            total_downloaded += len(downloads)
            
            # Phase 1.2: SEQUENTIAL PROCESSING (not batches)
            # Process one image at a time with immediate feedback
            print(f"\n🔬 Analyzing {len(downloads)} images sequentially...")
            
            for idx, (img_data, image_bytes) in enumerate(downloads, 1):
                # Check if we already have enough before processing
                if len(results) >= num_images_requested:
                    print(f"\n🎯 Target reached: {len(results)}/{num_images_requested} valid images")
                    break
                
                # Analyze single image with detailed logging
                image_id = f"{img_data.get('title', 'unknown')[:30]}_{img_data['image_id']}"
                analysis = analyze_image_roundness(image_bytes, search_term, analyzer, image_id)
                
                if not analysis:
                    # Logging already done in analyze_image_roundness
                    continue
                
                # Save thumbnail
                thumbnail = compress_thumbnail(image_bytes, max_size_kb=25)
                safe_search_term = sanitize_filename(search_term)
                thumbnail_filename = f"{safe_search_term}_google_{img_data['image_id']}_thumb.jpg"
                thumbnail_path = os.path.join(app.config['IMAGES_DIR'], thumbnail_filename)
                
                with open(thumbnail_path, 'wb') as f:
                    f.write(thumbnail)
                
                # Save visualization images
                viz_paths = {}
                for viz_type, viz_bytes in analysis['visualizations'].items():
                    viz_filename = f"{safe_search_term}_google_{img_data['image_id']}_{viz_type}.jpg"
                    viz_path = os.path.join(app.config['IMAGES_DIR'], viz_filename)
                    with open(viz_path, 'wb') as f:
                        f.write(viz_bytes)
                    viz_paths[viz_type] = viz_filename
                
                # Calculate scores
                composite_pct = analysis['composite'] * 100
                roundness_score = get_roundness_score(composite_pct)
                score_desc, score_color = get_score_description(roundness_score)
                
                # Build complete result object
                results.append({
                    'image_id': img_data['image_id'],
                    'url': img_data['url'],
                    'title': img_data.get('title', ''),
                    'source': 'google',
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
                    'perimeter': analysis.get('perimeter', 0),
                    'detection_confidence': analysis.get('detection_confidence', 0)
                })
                
                print(f"     📊 Progress: {len(results)}/{num_images_requested} valid images collected")
                
                # Memory cleanup after each image
                cleanup_memory()
            
            # Check download limit
            if total_downloaded >= max_images_to_download:
                print(f"\n⚠️  Hit download limit: {total_downloaded}/{max_images_to_download} images tried")
                print(f"   Collected {len(results)} valid images from {total_downloaded} attempts")
                print(f"   Detection rate: {len(results)/total_downloaded*100:.1f}%")
                if len(results) < num_images_requested:
                    print(f"   💡 Tip: Try adjusting confidence threshold or search terms")
                break
            
            # Exit main loop if we have enough results
            if len(results) >= num_images_requested:
                break
            
            # Move to next page
            pagination_offset += 10
        
        if not results:
            cleanup_memory()
            return render_template('index.html',
                                 error=f"No objects detected in images for '{search_term}'.")
        
        print(f"\n{'='*80}")
        print(f"✅ COMPLETED: {len(results)} valid images from {total_downloaded} downloads")
        print(f"   Detection success rate: {len(results)/total_downloaded*100:.1f}%")
        print(f"{'='*80}\n")
        
        # Phase 1.4: NO automatic outlier filtering - user will review
        print(f"📊 Keeping all {len(results)} detected images (no auto-filtering)")
        filtered_results = results
        outliers = []  # No automatic outlier removal
        
        # Sort and rank
        filtered_results.sort(key=lambda x: x['composite'], reverse=True)
        for i, result in enumerate(filtered_results, 1):
            result['rank'] = i
        
        # Limit final results
        final_results = filtered_results[:num_images_requested]
        
        # Calculate statistics
        stats = calculate_statistics(filtered_results, outliers)
        chart_data = prepare_chart_data(filtered_results)
        
        # Calculate average score
        if final_results:
            avg_composite = stats['composite']['mean']
            avg_score = get_roundness_score(avg_composite)
            avg_description, avg_color = get_score_description(avg_score)
        else:
            avg_score = 0
            avg_description = "N/A"
            avg_color = "#000000"
        
        # Clean results for JSON serialization (convert numpy types to Python types)
        clean_results = []
        for r in filtered_results:
            clean_result = {
                'image_id': r.get('image_id'),
                'pexels_id': r.get('image_id'),  # Use image_id as pexels_id for JS compatibility
                'url': r.get('url'),
                'title': r.get('title'),
                'photographer': r.get('title', 'Google'),  # Use title as photographer fallback
                'photographer_url': r.get('url'),  # Link to source
                'source': r.get('source', 'google'),
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
            clean_results.append(clean_result)
        
        clean_outliers = []
        for o in outliers:
            clean_outlier = {
                'image_id': o.get('image_id'),
                'pexels_id': o.get('image_id'),  # Use image_id as pexels_id for JS compatibility
                'url': o.get('url'),
                'title': o.get('title'),
                'photographer': o.get('title', 'Google'),
                'photographer_url': o.get('url'),
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
            clean_outliers.append(clean_outlier)
        
        # Cache results
        search_id = database.save_search(
            search_term=search_term,
            results=results,
            filtered_results=clean_results,
            outliers=clean_outliers,
            stats=stats
        )
        
        cleanup_memory()
        
        print(f"✓ Search complete!\n")
        
        return render_template('results.html',
                             search_term=search_term,
                             config=load_config(),
                             results=clean_results[:10],
                             all_results=clean_results,
                             outliers=clean_outliers,
                             stats=stats,
                             chart_data=chart_data,
                             average_roundness_score=avg_score,
                             average_score_description=avg_description,
                             average_score_color=avg_color,
                             search_id=search_id,
                             timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                             num_images=num_images_requested)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        cleanup_memory()
        return render_template('index.html',
                             error=f"An error occurred: {str(e)}")


@app.route('/history')
def history():
    """Search history page"""
    history_data = database.get_search_history(limit=50)
    all_batches = database.get_all_batches()
    # Filter out empty batches
    batches = [b for b in all_batches if b['total_keywords'] > 0]
    
    # Add roundness scores
    for search in history_data:
        search['roundness_score'] = get_roundness_score(search['avg_composite'])
        search['score_description'], search['score_color'] = get_score_description(search['roundness_score'])
    
    return render_template('history.html', history=history_data, batches=batches)


@app.route('/load/<int:search_id>')
def load_search(search_id):
    """Load previously cached search"""
    search_data = database.get_search(search_id)
    
    if not search_data:
        return render_template('index.html', error="Search not found")
    
    # Database returns 'filtered_results' not 'results'
    results = search_data['filtered_results']
    outliers = search_data['outliers']
    
    # Add scores if not already present
    for result in results:
        if 'roundness_score' not in result:
            composite_pct = result['composite'] * 100
            result['roundness_score'] = get_roundness_score(composite_pct)
            desc, color = get_score_description(result['roundness_score'])
            result['score_description'] = desc
            result['score_color'] = color
    
    # Recalculate stats
    stats = calculate_statistics(results, outliers)
    chart_data = prepare_chart_data(results)
    
    if results:
        avg_composite = stats['composite']['mean']
        avg_score = get_roundness_score(avg_composite)
        avg_description, avg_color = get_score_description(avg_score)
    else:
        avg_score = 0
        avg_description = "N/A"
        avg_color = "#000000"
    
    return render_template('results.html',
                         search_term=search_data['search_term'],
                         results=results[:10],
                         all_results=results,
                         outliers=outliers,
                         stats=stats,
                         chart_data=chart_data,
                         average_roundness_score=avg_score,
                         average_score_description=avg_description,
                         average_score_color=avg_color,
                         search_id=search_id,
                         timestamp=search_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                         num_images=len(results))
    
    # Check if this search is part of a batch and add navigation context
    batch_context = None
    conn = sqlite3.connect(database.db_path)
    cursor = conn.cursor()
    
    # Get batch_id from the search
    cursor.execute('SELECT batch_id FROM searches WHERE id = ?', (search_id,))
    row = cursor.fetchone()
    
    if row and row[0]:
        batch_id = row[0]
        batch = database.get_batch(batch_id)
        
        if batch:
            keywords = batch['keywords']
            search_term = search_data['search_term']
            
            try:
                current_index = keywords.index(search_term)
                
                # Get prev and next search IDs
                prev_search_id = None
                next_search_id = None
                
                if current_index > 0:
                    prev_keyword = keywords[current_index - 1]
                    cursor.execute('SELECT id FROM searches WHERE search_term = ? AND batch_id = ? ORDER BY id DESC LIMIT 1', 
                                 (prev_keyword, batch_id))
                    prev_row = cursor.fetchone()
                    if prev_row:
                        prev_search_id = prev_row[0]
                
                if current_index < len(keywords) - 1:
                    next_keyword = keywords[current_index + 1]
                    cursor.execute('SELECT id FROM searches WHERE search_term = ? AND batch_id = ? ORDER BY id DESC LIMIT 1', 
                                 (next_keyword, batch_id))
                    next_row = cursor.fetchone()
                    if next_row:
                        next_search_id = next_row[0]
                
                batch_context = {
                    'batch_id': batch_id,
                    'batch_name': batch['name'],
                    'current_position': current_index + 1,
                    'total_keywords': len(keywords),
                    'prev_search_id': prev_search_id,
                    'next_search_id': next_search_id
                }
            except ValueError:
                # Keyword not found in batch (shouldn't happen)
                pass
    
    conn.close()
    
    return render_template('results.html',
                         search_term=search_data['search_term'],
                         results=results[:10],
                         all_results=results,
                         outliers=outliers,
                         stats=stats,
                         chart_data=chart_data,
                         average_roundness_score=avg_score,
                         average_score_description=avg_description,
                         average_score_color=avg_color,
                         search_id=search_id,
                         timestamp=search_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                         num_images=len(results),
                         batch_context=batch_context)


@app.route('/autocomplete')
def autocomplete():
    """
    Autocomplete endpoint for search suggestions.
    Uses cached keywords from database.
    """
    query = request.args.get('q', '').lower().strip()
    
    if not query or len(query) < 2:
        return jsonify([])
    
    # Get previous searches from database
    searches = database.get_search_history(limit=100)
    keywords = [s['search_term'] for s in searches]
    
    # Filter matches
    matches = [k for k in keywords if query in k.lower()]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_matches = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            unique_matches.append(m)
    
    return jsonify(unique_matches[:10])


@app.route('/image/<filename>')
def serve_image(filename):
    """Serve cached image"""
    image_path = os.path.join(app.config['IMAGES_DIR'], filename)
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/jpeg')
    return "Image not found", 404


@app.route('/export/history.csv')
def export_history_csv():
    """Export search history summary as CSV"""
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
            search['search_term'],
            search['timestamp'],
            search['num_images'],
            roundness_score,
            score_desc,
            f"{search['avg_composite']:.1f}",
            f"{search['std_composite']:.1f}",
            f"{search['min_composite']:.1f}",
            f"{search['max_composite']:.1f}",
            f"{search['median_composite']:.1f}",
            f"{search['avg_circularity']:.1f}",
            f"{search['avg_aspect_ratio']:.1f}",
            f"{search['avg_eccentricity']:.1f}",
            f"{search['avg_solidity']:.1f}",
            f"{search['avg_convexity']:.1f}",
            search['outliers_removed']
        ])
    
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=search_history.csv'}
    )


@app.route('/export/<int:search_id>/csv')
def export_search_csv(search_id):
    """Export detailed results for a specific search"""
    search_data = database.get_search(search_id)
    
    if not search_data:
        return "Search not found", 404
    
    output = StringIO()
    writer = csv.writer(output)
    
    writer.writerow([
        'Rank', 'Image ID', 'Source', 'Roundness Score (1-50)', 'Score Description',
        'Composite (%)', 'Circularity (%)', 'Aspect Ratio (%)', 'Eccentricity (%)',
        'Solidity (%)', 'Convexity (%)', 'Area (px²)', 'Perimeter (px)'
    ])
    
    for result in search_data['results']:
        composite_pct = result['composite'] * 100
        roundness_score = get_roundness_score(composite_pct)
        score_desc, _ = get_score_description(roundness_score)
        
        writer.writerow([
            result.get('rank', ''), result.get('image_id', ''), result.get('source', 'google'),
            roundness_score, score_desc,
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


@app.route('/delete', methods=['POST'])
def delete_searches():
    """Delete multiple searches"""
    try:
        data = request.get_json()
        search_ids = data.get('search_ids', [])
        
        if not search_ids:
            return jsonify({'success': False, 'error': 'No searches selected'})
        
        database.delete_searches(search_ids)
        
        return jsonify({'success': True, 'deleted_count': len(search_ids)})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


# ============= BATCH PROCESSING =============

@app.route('/batch')
def batch_processing():
    """Render batch processing page"""
    all_incomplete = database.get_incomplete_batches()
    # Filter out empty batches
    incomplete_batches = [b for b in all_incomplete if b['total_keywords'] > 0]
    return render_template('batch.html', incomplete_batches=incomplete_batches)


@app.route('/batch/upload', methods=['POST'])
def batch_upload():
    """Upload and parse CSV/Excel file with fixed format validation"""
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
            return jsonify({'success': False, 'error': 'Invalid file type. Use CSV or Excel.'})
        
        # Validate required headers for v2 format
        required_headers = {'Keyword', 'Phrase', 'Category', 'Search Value'}
        actual_headers = set(df.columns)
        
        if not required_headers.issubset(actual_headers):
            missing = required_headers - actual_headers
            return jsonify({
                'success': False,
                'error': f'Missing required columns: {", ".join(missing)}. Expected: Keyword, Phrase, Category, Search Value'
            })
        
        # Stop at first empty row (where Keyword is empty/NaN)
        if df['Keyword'].isna().any():
            first_empty_idx = df[df['Keyword'].isna()].index[0]
            df = df.iloc[:first_empty_idx]
            print(f"Stopped at first empty row (index {first_empty_idx})")
        
        # Extract unique categories for display
        categories = df['Category'].dropna().unique().tolist()
        
        return jsonify({
            'success': True,
            'format': 'v2',
            'row_count': len(df),
            'categories': categories,
            'preview': df.head(5).to_dict('records')
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/batch/create', methods=['POST'])
def batch_create():
    """Create batch from v2 format CSV"""
    try:
        batch_name = request.form.get('batch_name', '').strip()
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
        
        # Validate required headers
        required_headers = {'Keyword', 'Phrase', 'Category', 'Search Value'}
        if not required_headers.issubset(set(df.columns)):
            return jsonify({'success': False, 'error': 'Missing required columns'})
        
        # Stop at first empty row
        if df['Keyword'].isna().any():
            first_empty_idx = df[df['Keyword'].isna()].index[0]
            df = df.iloc[:first_empty_idx]
        
        # Build keywords data structure (v2 format)
        keywords_data = []
        for _, row in df.iterrows():
            keyword = str(row['Keyword']).strip()
            phrase = str(row['Phrase']).strip()
            category = str(row['Category']).strip()
            search_value = str(row['Search Value']).strip() if pd.notna(row['Search Value']) else ''
            
            # Clean up search_value: collapse multiple quotes to single quotes
            # Handles """word""" -> "word", ""word"" -> "word", etc.
            # This ensures literal search phrases are correctly formatted for Google
            if search_value:
                search_value = re.sub(r'"{2,}', '"', search_value)
            
            # Fallback to keyword if search_value is empty
            if not search_value:
                search_value = keyword
            
            keywords_data.append({
                'keyword': keyword,
                'phrase': phrase,
                'category': category,
                'search_value': search_value
            })
        
        if not keywords_data:
            return jsonify({'success': False, 'error': 'No valid keywords found'})
        
        # Use category as batch name if not provided
        category = keywords_data[0]['category']
        if not batch_name:
            batch_name = category
        
        # Create batch with v2 format
        batch_id = database.create_batch(
            name=batch_name,
            keywords=keywords_data,
            images_per_keyword=images_per_keyword,
            category=category,
            format_version='v2'
        )
        
        return jsonify({
            'success': True,
            'batch_id': batch_id,
            'total_keywords': len(keywords_data),
            'category': category,
            'message': f'Batch created with {len(keywords_data)} keywords'
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
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
            'is_stopped': False,
            'is_paused': False,
            'thread': None
        }
        
        def process_batch():
            """Background batch processing"""
            # Re-fetch batch data inside thread to avoid closure issues
            batch = database.get_batch(batch_id)
            if not batch:
                print(f"❌ Batch {batch_id} not found")
                return
            
            keywords = batch['keywords']
            images_per = batch['images_per_keyword']
            completed = 0
            
            database.update_batch_status(batch_id, 'processing', 0, 0)
            
            for idx, keyword_data in enumerate(keywords):
                if processor['is_stopped']:
                    database.update_batch_status(batch_id, 'stopped', idx, completed)
                    break
                
                while processor['is_paused']:
                    time.sleep(1)
                    if processor['is_stopped']:
                        database.update_batch_status(batch_id, 'stopped', idx, completed)
                        break
                
                if processor['is_stopped']:
                    break
                
                # Handle both v1 (string) and v2 (dict) formats
                if isinstance(keyword_data, dict):
                    # V2 format with metadata
                    keyword = keyword_data['keyword']
                    search_value = keyword_data['search_value']
                    category = keyword_data.get('category', '')
                    use_config_modifiers = False  # V2 batches don't use config modifiers
                else:
                    # V1 format (simple string)
                    keyword = keyword_data
                    search_value = keyword
                    category = None
                    use_config_modifiers = True  # V1 batches use config modifiers
                
                print(f"\n{'='*80}")
                print(f"BATCH [{idx+1}/{len(keywords)}]: {keyword}")
                if search_value != keyword:
                    print(f"  Search Query: {search_value[:100]}..." if len(search_value) > 100 else f"  Search Query: {search_value}")
                print(f"{'='*80}")
                
                try:
                    # Retry loop with pagination (similar to standard search)
                    results = []
                    seen_urls = set()
                    pagination_offset = 0
                    max_fetch_attempts = 5
                    total_downloaded = 0
                    max_images_to_download = images_per * 3  # Cap at 3x requested
                    
                    while len(results) < images_per and total_downloaded < max_images_to_download and pagination_offset < max_fetch_attempts * 10:
                        if processor['is_stopped']:
                            break
                        
                        still_needed = images_per - len(results)
                        print(f"  📥 Need {still_needed} more valid results. Fetching batch (offset={pagination_offset})...")
                        
                        # Search using search_value with appropriate modifier setting
                        image_results = search_all_sources(search_value, num_images=10, start_offset=pagination_offset, use_config_modifiers=use_config_modifiers)
                        
                        if not image_results:
                            print(f"  ⚠️  No more results from API for '{keyword}'")
                            break
                        
                        # Filter already-tried URLs
                        new_results = [r for r in image_results if r['url'] not in seen_urls]
                        seen_urls.update(r['url'] for r in image_results)
                        
                        if not new_results:
                            print("  ⚠️  All results already tried")
                            pagination_offset += 10
                            continue
                        
                        # Download
                        downloads = download_images_parallel(new_results)
                        total_downloaded += len(downloads)
                        
                        if not downloads:
                            print(f"  ⚠️  No successful downloads for '{keyword}' (offset {pagination_offset})")
                            pagination_offset += 10
                            continue
                        
                        # Process in batches with full result construction
                        print(f"  🔬 Analyzing {len(downloads)} images...")
                        
                        for batch_start in range(0, len(downloads), BATCH_SIZE):
                            batch_end = min(batch_start + BATCH_SIZE, len(downloads))
                            batch = downloads[batch_start:batch_end]
                            
                            # Extract data
                            batch_img_data = [item[0] for item in batch]
                            batch_img_bytes = [item[1] for item in batch]
                            
                            # Batch analyze
                            batch_ids = [item['image_id'] for item in batch_img_data]
                            analyses = analyze_images_batch(batch_img_bytes, keyword, analyzer, image_ids=batch_ids)
                            
                            # Process results
                            for img_data, image_bytes, analysis in zip(batch_img_data, batch_img_bytes, analyses):
                                if not analysis:
                                    continue
                                
                                # Save thumbnail
                                thumbnail = compress_thumbnail(image_bytes, max_size_kb=25)
                                safe_keyword = sanitize_filename(keyword)
                                thumbnail_filename = f"{safe_keyword}_{img_data['image_id']}_thumb.jpg"
                                thumbnail_path = os.path.join(app.config['IMAGES_DIR'], thumbnail_filename)
                                
                                with open(thumbnail_path, 'wb') as f:
                                    f.write(thumbnail)
                                
                                # Save visualization images
                                viz_paths = {}
                                for viz_type, viz_bytes in analysis['visualizations'].items():
                                    viz_filename = f"{safe_keyword}_{img_data['image_id']}_{viz_type}.jpg"
                                    viz_path = os.path.join(app.config['IMAGES_DIR'], viz_filename)
                                    with open(viz_path, 'wb') as f:
                                        f.write(viz_bytes)
                                    viz_paths[viz_type] = viz_filename
                                
                                # Calculate scores
                                composite_pct = analysis['composite'] * 100
                                roundness_score = get_roundness_score(composite_pct)
                                score_desc, score_color = get_score_description(roundness_score)
                                
                                # Build complete result object
                                results.append({
                                    'image_id': img_data['image_id'],
                                    'pexels_id': img_data['image_id'],
                                    'url': img_data['url'],
                                    'title': img_data.get('title', ''),
                                    'photographer': img_data.get('title', 'Google'),
                                    'photographer_url': img_data.get('url'),
                                    'source': 'google',
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
                            
                            # Cleanup after batch
                            cleanup_memory()
                            
                            if len(results) >= images_per:
                                break
                        
                        # Move to next page if we still need images
                        if len(results) < images_per:
                            pagination_offset += 10
                    
                    if not results:
                        print(f"  ✗ No valid detections for '{keyword}' after trying {total_downloaded} images")
                        database.update_batch_status(batch_id, 'processing', idx + 1, completed)
                        continue
                    
                    # Sort and rank
                    results.sort(key=lambda x: x['composite'], reverse=True)
                    for i, result in enumerate(results, 1):
                        result['rank'] = i
                    
                    # Filter outliers and clean for database
                    # PHASE 1.3: NO automatic outlier filtering - keep everything
                    # filtered, outliers_list = remove_outliers(results, metric='composite')
                    filtered = results
                    outliers_list = []
                    
                    # Clean results for database (convert numpy types)
                    clean_filtered = []
                    for r in filtered[:images_per]:
                        clean_result = {
                            'image_id': r['image_id'],
                            'pexels_id': r['pexels_id'],
                            'url': r['url'],
                            'title': r['title'],
                            'photographer': r['photographer'],
                            'photographer_url': r['photographer_url'],
                            'source': r['source'],
                            'thumbnail_path': r['thumbnail_path'],
                            'viz_paths': r['viz_paths'],
                            'circularity': float(r['circularity']),
                            'aspect_ratio': float(r['aspect_ratio']),
                            'eccentricity': float(r['eccentricity']),
                            'solidity': float(r['solidity']),
                            'convexity': float(r['convexity']),
                            'composite': float(r['composite']),
                            'roundness_score': r['roundness_score'],
                            'score_description': r['score_description'],
                            'score_color': r['score_color'],
                            'area': float(r['area']),
                            'perimeter': float(r['perimeter']),
                            'rank': r['rank']
                        }
                        clean_filtered.append(clean_result)
                    
                    clean_outliers = []
                    for o in outliers_list:
                        clean_outlier = {
                            'image_id': o['image_id'],
                            'pexels_id': o['pexels_id'],
                            'url': o['url'],
                            'title': o['title'],
                            'photographer': o['photographer'],
                            'photographer_url': o['photographer_url'],
                            'thumbnail_path': o['thumbnail_path'],
                            'viz_paths': o['viz_paths'],
                            'circularity': float(o['circularity']),
                            'aspect_ratio': float(o['aspect_ratio']),
                            'eccentricity': float(o['eccentricity']),
                            'solidity': float(o['solidity']),
                            'convexity': float(o['convexity']),
                            'composite': float(o['composite']),
                            'area': float(o['area']),
                            'perimeter': float(o['perimeter']),
                            'outlier_reason': o.get('outlier_reason'),
                            'outlier_direction': o.get('outlier_direction')
                        }
                        clean_outliers.append(clean_outlier)
                    
                    if clean_filtered:
                        stats = calculate_statistics(clean_filtered, clean_outliers)
                        
                        # Save with dual-keyword fields for v2 format
                        database.save_search(
                            search_term=keyword,
                            results=results,
                            filtered_results=clean_filtered,
                            outliers=clean_outliers,
                            stats=stats,
                            batch_id=batch_id,
                            detection_keyword=keyword,  # Simple keyword for OWL-ViT
                            search_phrase=search_value,  # Complex query for Google
                            category=category if isinstance(keyword_data, dict) else None
                        )
                        
                        print(f"  ✓ Saved {len(clean_filtered)} results")
                    
                    completed += 1
                    database.update_batch_status(batch_id, 'processing', idx + 1, completed)
                    
                    cleanup_memory()
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"  ✗ Error processing '{keyword}': {e}")
                    import traceback
                    traceback.print_exc()
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
        
        for idx, keyword_data in enumerate(keywords):
            # Handle both v1 (string) and v2 (dict) formats
            if isinstance(keyword_data, dict):
                keyword = keyword_data['keyword']
            else:
                keyword = keyword_data
            
            search = next((s for s in searches if s['search_term'] == keyword), None)
            
            if idx < batch['current_keyword_index']:
                if search:
                    # Check actual status from DB
                    kw_status = database.get_keyword_status(keyword)
                    status = kw_status['status'] if kw_status else ('complete' if search['num_images'] >= batch['images_per_keyword'] else 'incomplete')
                    
                    keyword_statuses.append({
                        'keyword': keyword,
                        'status': status,
                        'images': search['num_images'],
                        'avg_score': get_roundness_score(search['avg_composite']) if search['avg_composite'] else 0,
                        'search_id': search['id']
                    })
                else:
                    # No search found - failed to process
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



@app.route('/api/keyword/<keyword>/approve', methods=['POST'])
def api_approve_keyword(keyword):
    """Approve a keyword"""
    try:
        database.approve_keyword(keyword)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/image/<image_id>/reject', methods=['POST'])
def api_reject_image(image_id):
    """Reject an image"""
    try:
        data = request.get_json() or {}
        reason = data.get('reason', 'user_rejected')
        search_id = data.get('search_id')
        
        if not search_id:
            # Try to find search_id from image_id if not provided
            # This requires a DB lookup not currently in Database class, 
            # so we rely on frontend passing search_id for now
            return jsonify({'success': False, 'error': 'search_id required'})
            
        database.reject_image(int(search_id), image_id, reason)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})




@app.route('/api/keywords/bulk-approve', methods=['POST'])
def api_bulk_approve_keywords():
    """Approve multiple keywords at once"""
    try:
        data = request.get_json() or {}
        keywords = data.get('keywords', [])
        
        if not keywords:
            return jsonify({'success': False, 'error': 'No keywords provided'})
        
        approved_count = 0
        for keyword in keywords:
            try:
                database.approve_keyword(keyword)
                approved_count += 1
            except Exception as e:
                print(f"Error approving keyword '{keyword}': {e}")
        
        return jsonify({
            'success': True,
            'approved_count': approved_count,
            'total': len(keywords)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ============= ADMIN SETTINGS =============

@app.route('/admin')
def admin_page():
    """Admin settings page"""
    config = load_config()
    return render_template('admin.html', config=config)


@app.route('/admin/save', methods=['POST'])
def admin_save():
    """Save admin settings"""
    try:
        settings = request.get_json()
        save_config(settings)
        reload_config()  # Reload config cache
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/admin/reset', methods=['POST'])
def admin_reset():
    """Reset settings to defaults"""
    try:
        save_config(DEFAULT_CONFIG)
        reload_config()  # Reload config cache
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/admin/delete_all', methods=['POST'])
def admin_delete_all():
    """Delete all searches and batches"""
    try:
        # Get all searches
        all_searches = database.get_search_history(limit=10000)
        search_ids = [s['id'] for s in all_searches]
        
        # Get all batches
        all_batches = database.get_all_batches()
        batch_ids = [b['id'] for b in all_batches]
        
        # Delete searches
        if search_ids:
            database.delete_searches(search_ids)
        
        # Delete batches
        for batch_id in batch_ids:
            database.delete_batch(batch_id)
        
        return jsonify({'success': True, 'deleted_searches': len(search_ids), 'deleted_batches': len(batch_ids)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    init_app()
    app.run(debug=True, host='0.0.0.0', port=5000)