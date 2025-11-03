#!/usr/bin/env python3
"""
Standalone HTML Report Exporter
Exports roundness analysis data to a self-contained single-page HTML application
"""

import sqlite3
import json
import os
import shutil
from datetime import datetime
from typing import List, Dict, Optional


class HTMLExporter:
    def __init__(self, db_path: str = './cache/searches.db', images_dir: str = './cache/images'):
        self.db_path = db_path
        self.images_dir = images_dir
        
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at {db_path}")
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"Images directory not found at {images_dir}")
    
    def get_all_individual_searches(self) -> List[Dict]:
        """Get all searches that are NOT part of a batch"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, search_term, timestamp, num_images,
                   avg_composite, std_composite, median_composite,
                   avg_circularity, avg_aspect_ratio, avg_eccentricity,
                   avg_solidity, avg_convexity, outliers_removed, results_json
            FROM searches
            WHERE batch_id IS NULL
            ORDER BY timestamp DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        searches = []
        for row in rows:
            results_data = json.loads(row[13]) if row[13] else {'filtered': [], 'outliers': []}
            searches.append({
                'id': row[0],
                'search_term': row[1],
                'timestamp': row[2],
                'num_images': row[3],
                'avg_composite': row[4],
                'std_composite': row[5],
                'median_composite': row[6],
                'avg_circularity': row[7],
                'avg_aspect_ratio': row[8],
                'avg_eccentricity': row[9],
                'avg_solidity': row[10],
                'avg_convexity': row[11],
                'outliers_removed': row[12],
                'results': results_data['filtered']
            })
        
        return searches
    
    def get_all_batches(self) -> List[Dict]:
        """Get all batches"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, created_at, total_keywords, completed_keywords, status
            FROM batches
            ORDER BY created_at DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{
            'id': row[0],
            'name': row[1],
            'created_at': row[2],
            'total_keywords': row[3],
            'completed_keywords': row[4],
            'status': row[5]
        } for row in rows]
    
    def get_batch_searches(self, batch_id: int) -> List[Dict]:
        """Get all searches for a specific batch"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, search_term, timestamp, num_images,
                   avg_composite, std_composite, median_composite,
                   avg_circularity, avg_aspect_ratio, avg_eccentricity,
                   avg_solidity, avg_convexity, outliers_removed, results_json
            FROM searches
            WHERE batch_id = ?
            ORDER BY timestamp ASC
        ''', (batch_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        searches = []
        for row in rows:
            results_data = json.loads(row[13]) if row[13] else {'filtered': [], 'outliers': []}
            searches.append({
                'id': row[0],
                'search_term': row[1],
                'timestamp': row[2],
                'num_images': row[3],
                'avg_composite': row[4],
                'std_composite': row[5],
                'median_composite': row[6],
                'avg_circularity': row[7],
                'avg_aspect_ratio': row[8],
                'avg_eccentricity': row[9],
                'avg_solidity': row[10],
                'avg_convexity': row[11],
                'outliers_removed': row[12],
                'results': results_data['filtered']
            })
        
        return searches
    
    def get_roundness_score(self, composite_percentage: float) -> int:
        """Convert composite percentage to 1-50 roundness score"""
        if composite_percentage < 35:
            return 1
        if composite_percentage >= 98:
            return 50
        score = ((composite_percentage - 35) / 63) * 49 + 1
        return int(round(score))
    
    def copy_images(self, searches: List[Dict], export_images_dir: str) -> None:
        """Copy contour images for all searches"""
        print("Copying images...")
        copied_count = 0
        
        for search in searches:
            for result in search.get('results', []):
                viz_paths = result.get('viz_paths', {})
                contour_filename = viz_paths.get('contour')
                
                if contour_filename:
                    src = os.path.join(self.images_dir, contour_filename)
                    dst = os.path.join(export_images_dir, contour_filename)
                    
                    if os.path.exists(src):
                        shutil.copy2(src, dst)
                        copied_count += 1
        
        print(f"  Copied {copied_count} images")
    
    def generate_data_json(self, searches: List[Dict]) -> str:
        """Generate data.json content"""
        export_data = {
            'export_date': datetime.now().isoformat(),
            'total_searches': len(searches),
            'searches': []
        }
        
        for search in searches:
            search_data = {
                'search_term': search['search_term'],
                'timestamp': search['timestamp'],
                'num_images': search['num_images'],
                'avg_composite': search['avg_composite'],
                'std_composite': search['std_composite'],
                'median_composite': search['median_composite'],
                'avg_circularity': search['avg_circularity'],
                'avg_aspect_ratio': search['avg_aspect_ratio'],
                'avg_eccentricity': search['avg_eccentricity'],
                'avg_solidity': search['avg_solidity'],
                'avg_convexity': search['avg_convexity'],
                'outliers_removed': search['outliers_removed'],
                'roundness_score': self.get_roundness_score(search['avg_composite']),
                'images': []
            }
            
            for result in search.get('results', []):
                viz_paths = result.get('viz_paths', {})
                contour_filename = viz_paths.get('contour')
                
                if contour_filename:
                    composite_pct = result['composite'] * 100
                    search_data['images'].append({
                        'filename': contour_filename,
                        'roundness_score': self.get_roundness_score(composite_pct),
                        'composite': result['composite'],
                        'rank': result.get('rank', 0),
                        'photographer': result.get('photographer', ''),
                        'circularity': result.get('circularity', 0),
                        'aspect_ratio': result.get('aspect_ratio', 0),
                        'eccentricity': result.get('eccentricity', 0),
                        'solidity': result.get('solidity', 0),
                        'convexity': result.get('convexity', 0)
                    })
            
            export_data['searches'].append(search_data)
        
        return json.dumps(export_data, indent=2)
    
    def generate_html(self, data_json: str) -> str:
        """Generate the single-page HTML application with embedded data"""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Roundness Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}
        
        header p {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        /* Index Page Styles */
        .search-bar {{
            margin-bottom: 30px;
        }}
        
        .search-bar input {{
            width: 100%;
            padding: 15px;
            font-size: 1.1rem;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            transition: border-color 0.3s;
        }}
        
        .search-bar input:focus {{
            outline: none;
            border-color: #667eea;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        th {{
            background: #f8fafc;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            color: #475569;
            border-bottom: 2px solid #e2e8f0;
            cursor: pointer;
            user-select: none;
        }}
        
        th:hover {{
            background: #f1f5f9;
        }}
        
        th.sortable::after {{
            content: " ⇅";
            opacity: 0.3;
        }}
        
        th.sort-asc::after {{
            content: " ↑";
            opacity: 1;
        }}
        
        th.sort-desc::after {{
            content: " ↓";
            opacity: 1;
        }}
        
        td {{
            padding: 15px;
            border-bottom: 1px solid #e2e8f0;
        }}
        
        tr:hover {{
            background: #f8fafc;
            cursor: pointer;
        }}
        
        .score-badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.9rem;
        }}
        
        .score-high {{
            background: #d1fae5;
            color: #065f46;
        }}
        
        .score-medium {{
            background: #fef3c7;
            color: #92400e;
        }}
        
        .score-low {{
            background: #fee2e2;
            color: #991b1b;
        }}
        
        /* Detail Page Styles */
        .detail-header {{
            margin-bottom: 30px;
        }}
        
        .back-btn {{
            display: inline-block;
            padding: 10px 20px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            margin-bottom: 20px;
            transition: background 0.3s;
        }}
        
        .back-btn:hover {{
            background: #5568d3;
        }}
        
        .nav-buttons {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 20px 0;
            gap: 20px;
        }}
        
        .nav-btn {{
            display: inline-block;
            padding: 10px 20px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            transition: background 0.3s;
            font-weight: 500;
        }}
        
        .nav-btn:hover {{
            background: #5568d3;
        }}
        
        .nav-btn.disabled {{
            background: #cbd5e1;
            cursor: not-allowed;
            pointer-events: none;
        }}
        
        .nav-spacer {{
            flex: 1;
        }}
        
        .detail-title {{
            font-size: 2.5rem;
            color: #1e293b;
            margin-bottom: 20px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }}
        
        .stat-card {{
            background: #f8fafc;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        
        .stat-label {{
            font-size: 0.9rem;
            color: #64748b;
            margin-bottom: 8px;
        }}
        
        .stat-value {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #1e293b;
        }}
        
        .images-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-top: 30px;
        }}
        
        .image-card {{
            background: #f8fafc;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        
        .image-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        }}
        
        .image-card img {{
            width: 100%;
            height: 250px;
            object-fit: cover;
            display: block;
        }}
        
        .image-info {{
            padding: 15px;
            text-align: center;
        }}
        
        .image-score {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #667eea;
            margin-bottom: 5px;
        }}
        
        .image-rank {{
            font-size: 0.9rem;
            color: #64748b;
        }}
        
        .loading {{
            text-align: center;
            padding: 60px 20px;
            color: #64748b;
            font-size: 1.2rem;
        }}
        
        .error {{
            text-align: center;
            padding: 60px 20px;
            color: #dc2626;
            font-size: 1.2rem;
        }}
        
        .no-results {{
            text-align: center;
            padding: 60px 20px;
            color: #64748b;
            font-size: 1.2rem;
        }}
        
        @media (max-width: 768px) {{
            .images-grid {{
                grid-template-columns: repeat(2, 1fr);
            }}
            
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
            
            header h1 {{
                font-size: 1.8rem;
            }}
            
            .detail-title {{
                font-size: 1.8rem;
            }}
            
            .nav-buttons {{
                flex-direction: column;
                gap: 10px;
            }}
            
            .nav-btn {{
                width: 100%;
                text-align: center;
            }}
            
            .nav-spacer {{
                display: none;
            }}
        }}
        
        @media (max-width: 480px) {{
            .images-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🔍 Roundness Analysis Report</h1>
            <p id="export-date"></p>
        </header>
        <div class="content" id="app">
            <div class="loading">Loading data...</div>
        </div>
    </div>

    <script>
        // Embedded data (no external fetch needed - works with file:// protocol)
        const appData = {data_json};
        
        // Global state
        let currentSort = {{ column: null, direction: 'asc' }};
        let searchFilter = '';

        // Initialize on page load
        function init() {{
            // Set export date in header
            const exportDate = new Date(appData.export_date);
            document.getElementById('export-date').textContent = 
                `Exported: ${{exportDate.toLocaleString()}}`;
            
            // Initialize router
            window.addEventListener('hashchange', router);
            router();
        }}

        // Router
        function router() {{
            const hash = window.location.hash.slice(1);
            
            if (!hash) {{
                renderIndex();
            }} else {{
                renderDetail(hash);
            }}
        }}

        // Render index page
        function renderIndex() {{
            const searches = getFilteredSearches();
            
            const html = `
                <div class="search-bar">
                    <input 
                        type="text" 
                        id="search-input" 
                        placeholder="Filter searches by name..." 
                        value="${{searchFilter}}"
                    >
                </div>
                
                ${{searches.length === 0 ? 
                    '<div class="no-results">No searches found</div>' :
                    `<table>
                        <thead>
                            <tr>
                                <th class="sortable" data-column="search_term">Search Term</th>
                                <th class="sortable" data-column="timestamp">Date</th>
                                <th class="sortable" data-column="num_images">Images</th>
                                <th class="sortable" data-column="roundness_score">Score (1-50)</th>
                                <th class="sortable" data-column="avg_composite">Mean (%)</th>
                                <th class="sortable" data-column="median_composite">Median (%)</th>
                                <th class="sortable" data-column="std_composite">Std Dev (%)</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${{searches.map(s => `
                                <tr onclick="window.location.hash='${{encodeURIComponent(s.search_term)}}'">
                                    <td><strong>${{escapeHtml(s.search_term)}}</strong></td>
                                    <td>${{new Date(s.timestamp).toLocaleDateString()}}</td>
                                    <td>${{s.num_images}}</td>
                                    <td>
                                        <span class="score-badge ${{getScoreClass(s.roundness_score)}}">
                                            ${{s.roundness_score}}
                                        </span>
                                    </td>
                                    <td>${{s.avg_composite.toFixed(1)}}</td>
                                    <td>${{s.median_composite.toFixed(1)}}</td>
                                    <td>${{s.std_composite.toFixed(1)}}</td>
                                </tr>
                            `).join('')}}
                        </tbody>
                    </table>`
                }}
            `;
            
            document.getElementById('app').innerHTML = html;
            
            // Add event listeners
            document.getElementById('search-input')?.addEventListener('input', (e) => {{
                searchFilter = e.target.value;
                renderIndex();
            }});
            
            document.querySelectorAll('th.sortable').forEach(th => {{
                th.addEventListener('click', () => {{
                    const column = th.dataset.column;
                    if (currentSort.column === column) {{
                        currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
                    }} else {{
                        currentSort.column = column;
                        currentSort.direction = 'asc';
                    }}
                    renderIndex();
                }});
                
                if (th.dataset.column === currentSort.column) {{
                    th.classList.add(currentSort.direction === 'asc' ? 'sort-asc' : 'sort-desc');
                }}
            }});
        }}

        // Render detail page
        function renderDetail(searchTerm) {{
            const decodedTerm = decodeURIComponent(searchTerm);
            const allSearches = getFilteredSearches();
            const currentIndex = allSearches.findIndex(s => s.search_term === decodedTerm);
            const search = allSearches[currentIndex];
            
            if (!search) {{
                document.getElementById('app').innerHTML = 
                    '<div class="error">Search not found</div>';
                return;
            }}
            
            // Determine prev/next
            const hasPrev = currentIndex > 0;
            const hasNext = currentIndex < allSearches.length - 1;
            const prevSearch = hasPrev ? allSearches[currentIndex - 1] : null;
            const nextSearch = hasNext ? allSearches[currentIndex + 1] : null;
            
            const navButtons = `
                <div class="nav-buttons">
                    ${{hasPrev ? 
                        `<a href="#${{encodeURIComponent(prevSearch.search_term)}}" class="nav-btn">← Prev: ${{escapeHtml(prevSearch.search_term)}}</a>` :
                        '<span class="nav-btn disabled">← Prev</span>'
                    }}
                    <div class="nav-spacer"></div>
                    ${{hasNext ? 
                        `<a href="#${{encodeURIComponent(nextSearch.search_term)}}" class="nav-btn">Next: ${{escapeHtml(nextSearch.search_term)}} →</a>` :
                        '<span class="nav-btn disabled">Next →</span>'
                    }}
                </div>
            `;
            
            const html = `
                <div class="detail-header">
                    <a href="#" class="back-btn">← Back to Index</a>
                    <h1 class="detail-title">${{escapeHtml(search.search_term)}}</h1>
                </div>
                
                ${{navButtons}}
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Roundness Score</div>
                        <div class="stat-value">${{search.roundness_score}}/50</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Mean Composite</div>
                        <div class="stat-value">${{search.avg_composite.toFixed(1)}}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Median Composite</div>
                        <div class="stat-value">${{search.median_composite.toFixed(1)}}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Std Deviation</div>
                        <div class="stat-value">${{search.std_composite.toFixed(1)}}%</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Images Analyzed</div>
                        <div class="stat-value">${{search.num_images}}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Outliers Removed</div>
                        <div class="stat-value">${{search.outliers_removed}}</div>
                    </div>
                </div>
                
                <h2 style="margin-bottom: 20px; color: #1e293b;">Image Results (${{search.images.length}})</h2>
                
                <div class="images-grid">
                    ${{search.images.map(img => `
                        <div class="image-card">
                            <img src="images/${{img.filename}}" alt="Rank ${{img.rank}}">
                            <div class="image-info">
                                <div class="image-score">${{img.roundness_score}}/50</div>
                                <div class="image-rank">Rank #${{img.rank}}</div>
                            </div>
                        </div>
                    `).join('')}}
                </div>
                
                ${{navButtons}}
            `;
            
            document.getElementById('app').innerHTML = html;
        }}

        // Helper functions
        function getFilteredSearches() {{
            let searches = [...appData.searches];
            
            // Apply search filter
            if (searchFilter) {{
                const filter = searchFilter.toLowerCase();
                searches = searches.filter(s => 
                    s.search_term.toLowerCase().includes(filter)
                );
            }}
            
            // Apply sorting
            if (currentSort.column) {{
                searches.sort((a, b) => {{
                    let aVal = a[currentSort.column];
                    let bVal = b[currentSort.column];
                    
                    if (currentSort.column === 'timestamp') {{
                        aVal = new Date(aVal);
                        bVal = new Date(bVal);
                    }}
                    
                    if (aVal < bVal) return currentSort.direction === 'asc' ? -1 : 1;
                    if (aVal > bVal) return currentSort.direction === 'asc' ? 1 : -1;
                    return 0;
                }});
            }}
            
            return searches;
        }}

        function getScoreClass(score) {{
            if (score >= 40) return 'score-high';
            if (score >= 25) return 'score-medium';
            return 'score-low';
        }}

        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}

        // Initialize
        init();
    </script>
</body>
</html>'''
    
    def export(self, searches: List[Dict], export_name: str) -> str:
        """
        Export searches to HTML report
        
        Args:
            searches: List of search dictionaries
            export_name: Name for the export (used in folder name)
        
        Returns:
            Path to created export directory
        """
        # Create exports directory
        exports_dir = './exports'
        os.makedirs(exports_dir, exist_ok=True)
        
        # Create timestamped export directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        export_dir = os.path.join(exports_dir, f'{export_name}_{timestamp}')
        os.makedirs(export_dir, exist_ok=True)
        
        # Create images subdirectory
        export_images_dir = os.path.join(export_dir, 'images')
        os.makedirs(export_images_dir, exist_ok=True)
        
        print(f"\nExporting to: {export_dir}")
        
        # Copy images
        self.copy_images(searches, export_images_dir)
        
        # Generate data JSON
        print("Generating embedded data...")
        data_json = self.generate_data_json(searches)
        
        # Generate index.html with embedded data
        print("Generating index.html...")
        html_content = self.generate_html(data_json)
        with open(os.path.join(export_dir, 'index.html'), 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return export_dir


def main():
    print("=" * 60)
    print("HTML REPORT EXPORTER")
    print("=" * 60)
    print()
    
    # Initialize exporter
    try:
        exporter = HTMLExporter()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("\nPlease run this script from the same directory as your Flask app.")
        return
    
    # Get available data
    individual_searches = exporter.get_all_individual_searches()
    batches = exporter.get_all_batches()
    
    print(f"Available data:")
    print(f"  - {len(individual_searches)} individual searches")
    print(f"  - {len(batches)} batches")
    print()
    
    # Prompt user for selection
    print("What would you like to export?")
    print("  1. All searches (individual + all batches)")
    print("  2. Individual searches only")
    print("  3. Specific batch")
    print()
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == '1':
        # Export everything
        print("\nExporting all searches...")
        all_searches = individual_searches.copy()
        
        for batch in batches:
            batch_searches = exporter.get_batch_searches(batch['id'])
            all_searches.extend(batch_searches)
        
        export_dir = exporter.export(all_searches, 'all_searches')
        
    elif choice == '2':
        # Export individual searches only
        if not individual_searches:
            print("\nNo individual searches found.")
            return
        
        print(f"\nExporting {len(individual_searches)} individual searches...")
        export_dir = exporter.export(individual_searches, 'individual_searches')
        
    elif choice == '3':
        # Export specific batch
        if not batches:
            print("\nNo batches found.")
            return
        
        print("\nAvailable batches:")
        for i, batch in enumerate(batches, 1):
            status = batch['status']
            completion = f"{batch['completed_keywords']}/{batch['total_keywords']}"
            print(f"  {i}. {batch['name']} ({completion} - {status})")
        print()
        
        batch_choice = input(f"Enter batch number (1-{len(batches)}): ").strip()
        
        try:
            batch_idx = int(batch_choice) - 1
            if batch_idx < 0 or batch_idx >= len(batches):
                print("Invalid batch number.")
                return
            
            selected_batch = batches[batch_idx]
            batch_searches = exporter.get_batch_searches(selected_batch['id'])
            
            if not batch_searches:
                print(f"\nNo searches found for batch '{selected_batch['name']}'")
                return
            
            print(f"\nExporting batch: {selected_batch['name']}")
            print(f"  {len(batch_searches)} searches")
            
            safe_name = selected_batch['name'].replace(' ', '_').replace('/', '_')
            export_dir = exporter.export(batch_searches, f'batch_{safe_name}')
            
        except ValueError:
            print("Invalid input.")
            return
    
    else:
        print("Invalid choice.")
        return
    
    print()
    print("=" * 60)
    print("✓ EXPORT COMPLETE!")
    print("=" * 60)
    print(f"\nExport location: {os.path.abspath(export_dir)}")
    print("\nTo view the report:")
    print(f"  1. Open the folder: {os.path.abspath(export_dir)}")
    print(f"  2. Double-click 'index.html' to open in your browser")
    print("\nThe report is self-contained and can be shared as a single folder.")
    print()


if __name__ == '__main__':
    main()