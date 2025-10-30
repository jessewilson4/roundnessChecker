#!/bin/bash

# Image Roundness Analyzer v2.0 - Directory Structure Setup
# Creates all directories and empty files for the project

echo "=================================="
echo "Image Roundness Analyzer v2.0"
echo "Building Directory Structure..."
echo "=================================="

# Create main project directory
PROJECT_DIR="roundness_app_v2"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# Create subdirectories
echo "Creating directories..."
mkdir -p utils
mkdir -p templates
mkdir -p static/css
mkdir -p static/js
mkdir -p cache/images

# Create Python files
echo "Creating Python files..."
touch app.py
touch requirements.txt
touch utils/__init__.py
touch utils/pexels_search.py
touch utils/edge_detection.py
touch utils/database.py

# Create HTML templates
echo "Creating HTML templates..."
touch templates/index.html
touch templates/results.html
touch templates/history.html

# Create CSS and JS files
echo "Creating CSS and JS files..."
touch static/css/style.css
touch static/js/main.js

# Create documentation files
echo "Creating documentation..."
touch README.md
touch QUICKSTART.md
touch PROJECT_COMPLETE.md

# Create .gitignore
echo "Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Flask
instance/
.webassets-cache

# Cache directory (but keep the structure)
cache/*.db
cache/images/*.jpg
cache/images/*.jpeg
cache/images/*.png

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Environment
.env
EOF

# Create directory tree visualization
echo ""
echo "=================================="
echo "✓ Directory structure created!"
echo "=================================="
echo ""
echo "Structure:"
tree -L 3 2>/dev/null || find . -type d -o -type f | sed 's|[^/]*/| |g'

echo ""
echo "Next steps:"
echo "1. Copy the provided files into this structure"
echo "2. Get your Pexels API key from https://www.pexels.com/api/"
echo "3. Add API key to app.py"
echo "4. Run: pip install -r requirements.txt"
echo "5. Run: python app.py"
echo ""
echo "Files ready in: $(pwd)"
echo ""