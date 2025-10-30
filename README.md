# Image Roundness Analyzer v2.0

A web application that analyzes object roundness using real images from Pexels API and proper edge detection techniques.

## 🎯 What's New in v2.0

### Major Improvements
- ✅ **Real Images**: Uses Pexels API instead of COCO dataset masks
- ✅ **True Edge Detection**: Canny + Sobel on grayscale images (not mask boundaries)
- ✅ **Smart Caching**: SQLite database stores all results for instant re-access
- ✅ **4-Panel Visualization**: See the complete analysis pipeline
- ✅ **Search History**: Track and reload previous searches
- ✅ **Enhanced Autocomplete**: Shows previous searches with scores

### Why v2?
The original COCO approach had rough segmentation masks (rectangles around objects), which didn't provide true edge information. v2 uses actual photographs with proper edge detection, matching the research methodology more closely.

## 📊 Features

- Search for any object using natural language
- Analyzes 30 high-quality images per search
- Auto-adaptive Canny edge detection
- Sobel gradient analysis on grayscale edges
- Statistical outlier removal (IQR method)
- Interactive charts and visualizations
- Complete pipeline view (Original → Grayscale → Edges → Contour)
- Search history with instant loading
- Smart autocomplete with previous results

## 🔬 Methodology

Based on peer-reviewed research:
- **Porto et al. (2025)**: Shape angularity in writing systems
- **Watier (2024)**: Measures of angularity in digital images

### Metrics

**1. First-Order Entropy** (60% weight)
- Measures uniformity of edge orientations using Sobel gradients
- Lower entropy = concentrated edge directions = smooth curves = **ROUND**
- Higher entropy = varied edge directions = angular = **NOT ROUND**
- Correlation: r = -0.84 with human judgment

**2. Circularity** (40% weight)
- Formula: `4π × Area / Perimeter²`
- 1.0 = perfect circle
- Lower values = more irregular shapes

**Composite Score**:
```
Composite = 0.60 × (1 - Entropy) + 0.40 × Circularity
```

Entropy is inverted because lower values indicate roundness.

## 🚀 Setup

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone/download this folder**

2. **Get Pexels API Key** (FREE!)
   - Go to: https://www.pexels.com/api/
   - Sign up (takes 30 seconds)
   - Copy your API key

3. **Add API Key to app.py**
   ```python
   # Open app.py and replace this line:
   PEXELS_API_KEY = "YOUR_PEXELS_API_KEY_HERE"
   # With your actual key:
   PEXELS_API_KEY = "your-actual-key-here"
   ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open browser**
   ```
   http://localhost:5000
   ```

## 📖 Usage

### First Search
1. Enter an object name (e.g., "apple", "ball", "cat")
2. Click Search
3. Wait 20-40 seconds while images are analyzed
4. View ranked results with charts

### Exploring Results
- **Top 10 Table**: See the roundest objects
- **View Pipeline**: Click button to see 4-panel analysis
- **Charts**: Distribution, comparisons, scatter plots
- **All Results**: Expand to see complete rankings
- **Outliers**: View statistically filtered objects

### Using Search History
- Type a previously searched term
- Autocomplete shows past searches with scores
- Click a previous search to load instantly (no re-analysis)
- Or press Enter to run a fresh search with new images

### History Page
- Click "Search History" in header
- View all past searches
- Click any row to reload that search

## 🎨 Understanding the Pipeline

### 4-Panel Visualization
When you click "View Pipeline" on any result:

1. **Original Image** - The photo from Pexels
2. **Grayscale** - Converted for edge detection
3. **Canny Edges** - Auto-adaptive edge detection (white on black)
4. **Contour** - Main object boundary (green overlay)

This shows exactly what the algorithm analyzed!

## 📁 Project Structure

```
roundness_app_v2/
├── app.py                      # Main Flask application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── utils/
│   ├── __init__.py
│   ├── pexels_search.py       # Pexels API integration
│   ├── edge_detection.py       # Canny + Sobel analysis
│   └── database.py             # SQLite caching
│
├── templates/
│   ├── index.html             # Search page
│   ├── results.html           # Results with charts
│   └── history.html           # Search history
│
├── static/
│   ├── css/style.css          # Complete styling
│   └── js/main.js             # Charts & interactions
│
└── cache/                      # Created automatically
    ├── searches.db            # SQLite database
    └── images/                # Cached thumbnails & visualizations
```

## 🔧 Configuration

### Pexels API
- **Free tier**: 200 requests/hour
- **Sufficient for**: Personal use, testing
- **Upgrade if needed**: Contact Pexels for higher limits

### Search Settings
Edit `app.py` to adjust:
- `num_images=30` - Number of images to analyze
- `max_size_kb=25` - Thumbnail compression
- Canny parameters in `edge_detection.py`

## 🎯 Expected Results

### Very Round Objects (80-95%)
- Balls, oranges, coins, plates
- Low entropy (concentrated edge directions)
- High circularity (compact shapes)

### Moderately Round (50-70%)
- Apples, clocks, pizza
- Mixed smooth and angular features

### Angular Objects (20-40%)
- Stop signs, boxes, books
- High entropy (varied edge directions)
- Lower circularity

### Irregular Objects (30-50%)
- Cats, dogs, people
- High variation between instances
- Depends on pose/view

## 🐛 Troubleshooting

### "Pexels API key not configured"
- Edit `app.py`
- Replace `YOUR_PEXELS_API_KEY_HERE` with your actual key
- Restart Flask

### "No images found"
- Try a more common object ("ball" vs "spherical object")
- Check Pexels has photos of that object
- Try different search terms

### Port 5000 in use
```python
# In app.py, change last line:
app.run(debug=True, host='0.0.0.0', port=5001)
```

### Slow analysis
- Normal: 20-40 seconds for 30 images
- Depends on internet speed (downloading images)
- Subsequent searches are instant (cached)

## 📊 Performance

- **First search**: 20-40 seconds (downloading + analysis)
- **Cached search**: <1 second (instant load)
- **Database size**: ~5-10 MB per 100 searches
- **Image cache**: ~25 KB per image (compressed thumbnails)

## 🚧 Future Enhancements

- [ ] Multiple API sources (Unsplash, Google Custom Search)
- [ ] Export results to CSV/PDF
- [ ] Batch comparison mode
- [ ] Custom threshold adjustments
- [ ] Background removal option
- [ ] Mobile app

## 📚 Research References

1. **Porto, A., et al. (2025)**  
   "Glyph norming: Human and computational measurements of shape angularity in writing systems"  
   *Behavior Research Methods*, 57:173

2. **Watier, N. (2024)**  
   "Measures of angularity in digital images"  
   *Behavior Research Methods*, 56(7), 7126-7151

3. **Canny, J. (1986)**  
   "A Computational Approach to Edge Detection"  
   *IEEE Transactions on Pattern Analysis and Machine Intelligence*

## ⚖️ Credits

- **Images**: Pexels.com (free stock photos)
- **Photographers**: Credited on each image
- **Methodology**: Porto et al., Watier et al.
- **Edge Detection**: OpenCV library

## 📝 License

Educational and research purposes. 

**Pexels**: Images licensed under Pexels License (free for personal/commercial use with attribution)

## 🤝 Contributing

This is a personal research tool. Feel free to fork and modify!

---

**Built with**: Python, Flask, OpenCV, Pexels API, SQLite, Chart.js  
**Version**: 2.0  
**Last Updated**: January 2025