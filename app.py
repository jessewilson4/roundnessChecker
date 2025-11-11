'''
# === UAIPCS START ===
file: app.py
purpose: Flask web application for image roundness analysis using Google Custom Search API with OWL-ViT object detection and SAM segmentation, includes batch processing and comprehensive export functionality
deps: [@flask:framework, @google_custom_search_api:service/external, @sqlite:storage, @transformers:library, @segment_anything:library, @torch:library, @opencv:library, @pandas:library, @numpy:library]
funcs:
  - get_roundness_score(composite_percentage:float) -> int  # no_side_effect
  - get_score_description(score:int) -> tuple  # no_side_effect
  - sanitize_filename(text:str) -> str  # no_side_effect
  - cleanup_memory() -> None  # side_effect: garbage collection and GPU cache clear
  - init_searchers() -> None  # side_effect: initializes global google_searcher
  - search_all_sources(search_term:str, num_images:int) -> list  # side_effect: API calls
  - download_images_parallel(image_results:list, max_workers:int) -> list  # side_effect: network I/O
  - init_app() -> None  # side_effect: initializes globals, loads AI models, creates directories
  - calculate_statistics(results:list, outliers:list) -> dict  # no_side_effect
  - prepare_chart_data(results:list) -> dict  # no_side_effect
refs:
  - utils/google_search.py::GoogleSearcher
  - utils/database.py::Database
  - utils/edge_detection.py::RoundnessAnalyzer
  - utils/edge_detection.py::analyze_images_batch
  - templates/index.html
  - templates/results.html
  - templates/history.html
  - templates/batch.html
notes: perf=cold=3-5s|hot=<1s, persist=durable, concur=not_thread_safe, volatiledeps=[@google_api], memory=high_during_batch
# === UAIPCS END ===
'''

from flask import Flask, render_template, request, jsonify, send_file, Response
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
import csv
from io import StringIO
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

BATCH_SIZE = 4
MAX_DOWNLOAD_WORKERS = 8

# Initialize components
google_searcher = None
database = None
analyzer = None
batch_processors = {}


def cleanup_memory():
    """Force garbage collection and clear CUDA cache"""
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
    """Initialize Google searcher"""
    global google_searcher
    if google_searcher is None:
        google_searcher = GoogleSearcher(GOOGLE_API_KEY, GOOGLE_CSE_ID)


def search_all_sources(search_term: str, num_images: int = 30):
    """Search Google for images with negative keywords for isolated objects"""
    init_searchers()
    return google_searcher.search_images(search_term, num_images)


def download_images_parallel(image_results, max_workers=MAX_DOWNLOAD_WORKERS):
    """
    Download multiple images using GoogleSearcher's batch method with single browser instance
    
    Args:
        image_results: List of image metadata dicts
        max_workers: Unused (kept for compatibility)
        
    Returns:
        List of tuples (img_data, image_bytes)
    """
    return google_searcher.download_images_batch(image_results)


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
    """Main search page"""
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    """
    Process search request with parallel downloads and batch inference.
    TROUBLESHOOTING MODE: Hard limit 10 images to conserve API quota.
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
        print(f" TROUBLESHOOTING SEARCH: '{search_term}'")
        print(f"{'='*80}")
        
        # TROUBLESHOOTING: Hard limit to 10 to conserve API quota
        num_to_fetch = 10
        num_to_process = 10
        
        print(f"📥 Fetching {num_to_fetch} images from Google (troubleshooting mode)")
        
        # Search for images
        image_results = search_all_sources(search_term, num_images=num_to_fetch)
        
        if not image_results:
            return render_template('index.html',
                                 error=f"No images found for '{search_term}'.")
        
        print(f"\n✓ API returned {len(image_results)} images that passed filtering")
        
        # Download in parallel
        print(f"\n⬇️  Downloading {len(image_results)} images with Playwright...")
        print("="*80)
        downloads = download_images_parallel(image_results)
        print("="*80)
        
        if not downloads:
            return render_template('index.html',
                                 error=f"Failed to download any images for '{search_term}'.")
        
        print(f"\n✅ Successfully downloaded {len(downloads)}/{len(image_results)} images")
        
        # Process in batches
        print(f"\n🔬 Analyzing {len(downloads)} images in batches of {BATCH_SIZE}...")
        
        results = analyze_images_batch(
            downloads,
            search_term,
            analyzer
        )
        
        if not results:
            cleanup_memory()
            return render_template('index.html',
                                 error=f"No objects detected in images for '{search_term}'.")
        
        print(f"✓ Got {len(results)} successful detections")
        
        # Add scores and rankings
        for result in results:
            composite_pct = result['composite'] * 100
            result['roundness_score'] = get_roundness_score(composite_pct)
            desc, color = get_score_description(result['roundness_score'])
            result['score_description'] = desc
            result['score_color'] = color
        
        # Filter outliers
        print("\n📈 Filtering statistical outliers...")
        filtered_results, outliers = remove_outliers(results, metric='composite')
        print(f"✓ Kept {len(filtered_results)} results, removed {len(outliers)} outliers")
        
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
        
        # Cache results
        search_id = database.save_search(
            search_term=search_term,
            results=results,
            filtered_results=filtered_results,
            outliers=outliers,
            stats=stats
        )
        
        cleanup_memory()
        
        print(f"✓ Search complete! Showing {len(final_results)} results\n")
        
        return render_template('results.html',
                             search_term=search_term,
                             num_images=num_images_requested,
                             results=final_results[:10],
                             all_results=filtered_results,
                             outliers=outliers,
                             stats=stats,
                             average_roundness_score=avg_score,
                             average_score_description=avg_description,
                             average_score_color=avg_color,
                             chart_data=chart_data,
                             timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                             search_id=search_id)
    
    except Exception as e:
        print(f"\n✗ Search failed: {e}")
        import traceback
        traceback.print_exc()
        cleanup_memory()
        return render_template('index.html',
                             error=f"Search failed: {str(e)}")


@app.route('/history')
def history():
    """Display search history"""
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
    """Load cached search results"""
    search_data = database.load_search_by_id(search_id)
    
    if not search_data:
        return render_template('index.html', error="Search not found")
    
    # Calculate stats
    filtered_results = search_data['filtered_results']
    outliers = search_data['outliers']
    
    stats = calculate_statistics(filtered_results, outliers)
    chart_data = prepare_chart_data(filtered_results)
    
    # Calculate average score
    if filtered_results:
        avg_composite = stats['composite']['mean']
        avg_score = get_roundness_score(avg_composite)
        avg_description, avg_color = get_score_description(avg_score)
    else:
        avg_score = 0
        avg_description = "N/A"
        avg_color = "#000000"
    
    return render_template('results.html',
                         search_term=search_data['search_term'],
                         num_images=search_data['num_images'],
                         results=filtered_results[:10],
                         all_results=filtered_results,
                         outliers=outliers,
                         stats=stats,
                         average_roundness_score=avg_score,
                         average_score_description=avg_description,
                         average_score_color=avg_color,
                         chart_data=chart_data,
                         timestamp=search_data['timestamp'],
                         search_id=search_id)


@app.route('/delete_result/<int:search_id>', methods=['POST'])
def delete_result(search_id):
    """Delete a search result"""
    success = database.delete_search(search_id)
    return jsonify({'success': success})


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


@app.route('/api/autocomplete')
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
    search_data = database.load_search_by_id(search_id)
    
    if not search_data:
        return "Search not found", 404
    
    output = StringIO()
    writer = csv.writer(output)
    
    writer.writerow([
        'Rank', 'Image ID', 'Source', 'Roundness Score (1-50)',
        'Score Description', 'Composite (%)', 'Circularity (%)', 'Aspect Ratio (%)',
        'Eccentricity (%)', 'Solidity (%)', 'Convexity (%)', 'Area (px²)', 'Perimeter (px)'
    ])
    
    for result in search_data['filtered_results']:
        composite_pct = result['composite'] * 100
        roundness_score = get_roundness_score(composite_pct)
        score_desc, _ = get_score_description(roundness_score)
        
        writer.writerow([
            result['rank'], result.get('image_id', ''), result.get('source', 'google'),
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
            'batch_id': batch_id,
            'is_running': True,
            'is_paused': False,
            'is_stopped': False,
            'thread': None
        }
        
        def process_batch():
            global analyzer, database
            
            keywords = batch['keywords']
            images_per = batch['images_per_keyword']
            completed = batch['completed_keywords']
            start_idx = batch['current_keyword_index']
            
            for idx in range(start_idx, len(keywords)):
                if processor['is_stopped']:
                    break
                
                while processor['is_paused'] and not processor['is_stopped']:
                    time.sleep(1)
                
                if processor['is_stopped']:
                    break
                
                keyword = keywords[idx]
                
                try:
                    print(f"\n[Batch {batch_id}] Processing {idx+1}/{len(keywords)}: {keyword}")
                    
                    # Search
                    image_results = search_all_sources(keyword, num_images=images_per * 2)
                    
                    if not image_results:
                        print(f"  ✗ No images found for '{keyword}'")
                        completed += 1
                        database.update_batch_status(batch_id, 'processing', idx + 1, completed)
                        continue
                    
                    # Download
                    downloads = download_images_parallel(image_results)
                    
                    if not downloads:
                        print(f"  ✗ Failed to download images for '{keyword}'")
                        completed += 1
                        database.update_batch_status(batch_id, 'processing', idx + 1, completed)
                        continue
                    
                    # Analyze
                    results = analyze_images_batch(
                        downloads[:int(images_per * 1.25)],
                        keyword,
                        analyzer,
                        batch_size=BATCH_SIZE
                    )
                    
                    if not results:
                        print(f"  ✗ No objects detected for '{keyword}'")
                        completed += 1
                        database.update_batch_status(batch_id, 'processing', idx + 1, completed)
                        continue
                    
                    # Add scores
                    for result in results:
                        composite_pct = result['composite'] * 100
                        result['roundness_score'] = get_roundness_score(composite_pct)
                        desc, color = get_score_description(result['roundness_score'])
                        result['score_description'] = desc
                        result['score_color'] = color
                    
                    # Filter and save
                    filtered, outliers_list = remove_outliers(results, metric='composite')
                    
                    if filtered:
                        filtered.sort(key=lambda x: x['composite'], reverse=True)
                        for i, result in enumerate(filtered, 1):
                            result['rank'] = i
                        
                        stats = calculate_statistics(filtered, outliers_list)
                        
                        database.save_search(
                            search_term=keyword,
                            results=results,
                            filtered_results=filtered[:images_per],
                            outliers=outliers_list,
                            stats=stats,
                            batch_id=batch_id
                        )
                        
                        print(f"  ✓ Saved {len(filtered[:images_per])} results")
                    
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