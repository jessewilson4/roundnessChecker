"""
Object Detection and Segmentation using Grounding DINO + SAM
OPTIMIZED: More accurate detection for challenging/obscure objects

This module handles:
1. Open-vocabulary object detection (Grounding DINO) - Better accuracy than OWL-ViT
2. Object segmentation (SAM)
3. Edge detection with contour smoothing
4. Roundness metric calculation
"""

# CRITICAL: Disable CUDA before any imports
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # For Mac M1/M2

import cv2
import numpy as np
from PIL import Image
import io
import torch
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import tempfile

# Force CPU mode for PyTorch BEFORE any other imports
torch.set_default_device('cpu')
if hasattr(torch, 'set_default_dtype'):
    torch.set_default_dtype(torch.float32)

# Smarter CUDA blocking - allows imports but blocks actual usage
_original_cuda = torch.cuda

class FakeCudaDeviceProperties:
    """Fake device properties for PyTorch imports"""
    def __init__(self):
        self.name = "CPU-only mode"
        self.major = 0
        self.minor = 0
        self.total_memory = 0
        self.multi_processor_count = 0

class FakeCuda:
    """Fake CUDA module that allows imports but blocks actual GPU usage"""
    _CudaDeviceProperties = FakeCudaDeviceProperties  # For PyTorch internal imports
    
    @staticmethod
    def is_available():
        return False
    
    @staticmethod
    def device_count():
        return 0
    
    @staticmethod
    def current_device():
        return None
    
    @staticmethod
    def get_device_name(device=0):
        return "CPU-only mode"
    
    @staticmethod
    def get_device_properties(device=0):
        return FakeCudaDeviceProperties()
    
    @staticmethod
    def set_device(device):
        pass  # Silently ignore
    
    def __getattr__(self, name):
        # Return dummy values for properties PyTorch checks during import
        if name in ('_CudaDeviceProperties', 'Event', 'Stream'):
            return type(name, (), {})
        # For everything else, return False/None/0
        return None

# Replace torch.cuda with fake module
torch.cuda = FakeCuda()

# Import Grounding DINO
try:
    from groundingdino.util.inference import load_model, predict
    from groundingdino.util import box_ops
    DINO_AVAILABLE = True
except ImportError:
    print("⚠️  Grounding DINO not installed.")
    print("   Install with: pip install groundingdino-py")
    DINO_AVAILABLE = False

# Import SAM
from segment_anything import sam_model_registry, SamPredictor
from skimage import measure


def compress_thumbnail(image_data: bytes, max_size_kb: int = 25) -> bytes:
    """
    Compress image to target file size, handling RGBA images.
    
    Args:
        image_data: Original image bytes
        max_size_kb: Target size in KB
        
    Returns:
        Compressed JPEG bytes
    """
    img = Image.open(io.BytesIO(image_data))
    
    # Convert RGBA/LA/P to RGB (JPEG doesn't support transparency)
    if img.mode in ('RGBA', 'LA', 'P'):
        background = Image.new('RGB', img.size, (255, 255, 255))
        if img.mode == 'P':
            img = img.convert('RGBA')
        background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to reasonable dimensions
    max_dim = 400
    ratio = min(max_dim / img.width, max_dim / img.height)
    if ratio < 1:
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Compress with quality adjustment
    quality = 85
    while quality > 20:
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, optimize=True)
        size_kb = len(buffer.getvalue()) / 1024
        
        if size_kb <= max_size_kb:
            return buffer.getvalue()
        
        quality -= 10
    
    # Return best effort
    return buffer.getvalue()


def remove_outliers(results: list, metric: str = 'composite') -> Tuple[list, list]:
    """
    Remove statistical outliers using IQR method.
    
    Args:
        results: List of result dictionaries
        metric: Metric to use for outlier detection
        
    Returns:
        Tuple of (filtered_results, outliers)
    """
    if len(results) < 4:
        return results, []
    
    values = [r[metric] for r in results]
    
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    filtered = []
    outliers = []
    
    for r in results:
        if lower_bound <= r[metric] <= upper_bound:
            filtered.append(r)
        else:
            r['outlier_reason'] = 'statistical'
            r['outlier_direction'] = 'low' if r[metric] < lower_bound else 'high'
            outliers.append(r)
    
    return filtered, outliers


def auto_canny(image, sigma=0.33):
    """
    Placeholder for compatibility - not used in Grounding DINO version.
    Kept for backward compatibility with existing imports.
    """
    pass


class RoundnessAnalyzer:
    """
    Analyzer for detecting and measuring object roundness using:
    - Grounding DINO for open-vocabulary object detection (more accurate than OWL-ViT)
    - SAM for precise segmentation
    - Canny edge detection with morphological smoothing
    
    OPTIMIZED: Better accuracy for challenging/obscure objects
    """
    
    def __init__(self, use_local_models=False):
        """
        Initialize the analyzer with models.
        
        Args:
            use_local_models: If True, load from ./models/ directory
        """
        self.dino_model = None
        self.sam_predictor = None
        self.use_local_models = use_local_models
        self._load_models()
    
    def _load_models(self):
        """Load Grounding DINO and SAM models"""
        print("🤖 Loading AI models...")
        
        if not DINO_AVAILABLE:
            raise ImportError("Grounding DINO not available. Install with: pip install groundingdino-py")
        
        # Grounding DINO model
        print("   Loading Grounding DINO (CPU mode)...")
        
        # Download checkpoint if needed
        checkpoint_path = "groundingdino_swint_ogc.pth"
        if not Path(checkpoint_path).exists():
            print(f"   Downloading Grounding DINO checkpoint (~700MB)...")
            import urllib.request
            url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            try:
                urllib.request.urlretrieve(url, checkpoint_path)
            except Exception as e:
                print(f"   ⚠️  Download failed: {e}")
                print(f"   Please manually download from: {url}")
                raise
        
        # Find config file in the installed package
        try:
            import groundingdino
            package_path = Path(groundingdino.__file__).parent
            config_path = package_path / "config" / "GroundingDINO_SwinT_OGC.py"
            
            if not config_path.exists():
                # Try alternative locations
                possible_paths = [
                    package_path / "config" / "GroundingDINO_SwinT_OGC.py",
                    Path("groundingdino/config/GroundingDINO_SwinT_OGC.py"),
                    Path("venv/lib/python3.12/site-packages/groundingdino/config/GroundingDINO_SwinT_OGC.py")
                ]
                
                for path in possible_paths:
                    if path.exists():
                        config_path = path
                        break
                else:
                    raise FileNotFoundError(
                        f"Could not find GroundingDINO config file. "
                        f"Searched in: {package_path / 'config'}"
                    )
            
            print(f"   Using config: {config_path}")
            
            # Load model with CPU - CRITICAL: specify device='cpu'
            self.dino_model = load_model(
                model_config_path=str(config_path),
                model_checkpoint_path=checkpoint_path,
                device='cpu'  # Force CPU
            )
            self.dino_model.to('cpu')
            self.dino_model.eval()
            
            print("   ✓ Grounding DINO loaded (CPU mode)")
            
        except Exception as e:
            print(f"   ✗ Error loading Grounding DINO: {e}")
            print(f"\n   Troubleshooting:")
            print(f"   1. Check if groundingdino is installed: pip show groundingdino-py")
            print(f"   2. Try reinstalling: pip install --force-reinstall groundingdino-py")
            print(f"   3. Or install from source: pip install git+https://github.com/IDEA-Research/GroundingDINO.git")
            raise
        
        # SAM model
        print("   Loading SAM...")
        sam_checkpoint = "sam_vit_b_01ec64.pth"
        
        if not Path(sam_checkpoint).exists():
            print(f"   Downloading SAM checkpoint (375MB)...")
            import urllib.request
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            urllib.request.urlretrieve(url, sam_checkpoint)
        
        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        sam.to('cpu')  # Force CPU mode for SAM
        self.sam_predictor = SamPredictor(sam)
        
        print("   ✓ Models loaded (CPU mode)\n")
    
    def detect_object(self, image_pil: Image.Image, target_text: str, threshold: float = 0.25) -> Optional[Dict]:
        """
        Detect object in image using Grounding DINO.
        
        Args:
            image_pil: PIL Image
            target_text: Object description (e.g., "cat", "orb", "round stone")
            threshold: Confidence threshold (DINO default is 0.25, lower than OWL-ViT)
            
        Returns:
            Detection dict with bbox and confidence, or None if not found
        """
        # Grounding DINO works better with descriptive prompts
        text_queries = [
            target_text,
            f"a {target_text}",
            f"{target_text} object",
        ]
        
        all_detections = []
        
        for text_prompt in text_queries:
            try:
                # CRITICAL FIX: Always pass device='cpu' to predict function
                boxes, logits, phrases = predict(
                    model=self.dino_model,
                    image=image_pil,
                    caption=text_prompt,
                    box_threshold=threshold,
                    text_threshold=0.25,
                    device='cpu'  # CRITICAL: Force CPU device
                )
                
                # Ensure outputs are on CPU
                if hasattr(boxes, 'cpu'):
                    boxes = boxes.cpu()
                if hasattr(logits, 'cpu'):
                    logits = logits.cpu()
                
                # Convert boxes from normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]
                H, W = image_pil.height, image_pil.width
                boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
                
                for box, score in zip(boxes_xyxy, logits):
                    all_detections.append({
                        'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        'confidence': float(score)
                    })
            except Exception as e:
                print(f"    ⚠ Query '{text_prompt}' failed: {e}")
                continue
        
        if not all_detections:
            return None
        
        # Sort by confidence and remove overlapping detections
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Simple NMS
        filtered = []
        for det in all_detections:
            is_dup = False
            for existing in filtered:
                if self._boxes_overlap(det['bbox'], existing['bbox']) > 0.5:
                    is_dup = True
                    break
            if not is_dup:
                filtered.append(det)
        
        return filtered[0] if filtered else None
    
    def detect_objects_batch(self, images_pil: List[Image.Image], target_text: str, threshold: float = 0.25) -> List[Optional[Dict]]:
        """
        Batch detect objects - processes sequentially with Grounding DINO.
        (DINO doesn't have native batch support like OWL-ViT)
        
        Args:
            images_pil: List of PIL Images
            target_text: Object description
            threshold: Confidence threshold
            
        Returns:
            List of detection dicts (or None for each image)
        """
        results = []
        for img in images_pil:
            if img is not None:
                results.append(self.detect_object(img, target_text, threshold))
            else:
                results.append(None)
        return results
    
    def _boxes_overlap(self, box1: list, box2: list) -> float:
        """Calculate IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def segment_object(self, image: np.ndarray, bbox: list) -> np.ndarray:
        """
        Segment object using SAM given a bounding box.
        
        Args:
            image: Image array (H, W, 3)
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Binary mask (H, W) as uint8
        """
        self.sam_predictor.set_image(image)
        
        # Convert bbox to SAM format
        input_box = np.array(bbox)
        
        masks, scores, _ = self.sam_predictor.predict(
            box=input_box,
            multimask_output=True
        )
        
        # Return best mask (highest score)
        best_idx = np.argmax(scores)
        mask = masks[best_idx].astype(np.uint8) * 255
        
        return mask
    
    def analyze_roundness(self, image: np.ndarray, mask: np.ndarray) -> Optional[Dict]:
        """
        Analyze object roundness from segmentation mask.
        
        Args:
            image: Original image (H, W, 3)
            mask: Binary mask (H, W)
            
        Returns:
            Dictionary with roundness metrics and visualizations
        """
        # Smooth mask with morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_smooth = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        main_contour = max(contours, key=cv2.contourArea)
        
        # Get region properties
        labeled = measure.label(mask_smooth)
        props = measure.regionprops(labeled)[0]
        
        # Calculate metrics
        area = props.area
        perimeter = cv2.arcLength(main_contour, True)
        
        if perimeter == 0:
            return None
        
        # 1. CIRCULARITY: 4π * area / perimeter²
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        circularity = min(circularity, 1.0)
        
        # 2. ASPECT RATIO from fitted ellipse
        # 3. ECCENTRICITY from fitted ellipse
        if len(main_contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(main_contour)
                major_axis = max(ellipse[1])
                minor_axis = min(ellipse[1])
                
                if major_axis > 0:
                    aspect_ratio_raw = minor_axis / major_axis
                    aspect_ratio = aspect_ratio_raw
                else:
                    aspect_ratio = 0.0
                    
                # Eccentricity from ellipse
                if major_axis > 0:
                    eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2)
                    eccentricity = 1.0 - eccentricity  # Invert so 1.0 = perfect circle
                else:
                    eccentricity = 0.0
            except:
                # Fallback to regionprops
                if props.minor_axis_length > 0:
                    aspect_ratio = props.minor_axis_length / props.major_axis_length
                else:
                    aspect_ratio = 0.0
                eccentricity = 1.0 - props.eccentricity
                ellipse = None
        else:
            # Fallback
            if props.minor_axis_length > 0:
                aspect_ratio = props.minor_axis_length / props.major_axis_length
            else:
                aspect_ratio = 0.0
            eccentricity = 1.0 - props.eccentricity
            ellipse = None
        
        # 4. SOLIDITY: Area / Convex hull area
        solidity = props.solidity
        
        # 5. CONVEXITY: Convex hull perimeter / actual perimeter
        hull = cv2.convexHull(main_contour)
        hull_perimeter = cv2.arcLength(hull, True)
        if perimeter > 0:
            convexity = hull_perimeter / perimeter
            convexity = min(convexity, 1.0)
        else:
            convexity = 0.0
        
        # COMPOSITE SCORE (weighted combination)
        composite = (
            0.30 * circularity +
            0.25 * aspect_ratio +
            0.20 * eccentricity +
            0.15 * solidity +
            0.10 * convexity
        )
        
        # Create boundary visualization
        boundary_edges = np.zeros_like(mask_smooth)
        cv2.drawContours(boundary_edges, [main_contour], -1, 255, 2)
        
        # Light smoothing for visualization
        epsilon = 0.002 * perimeter
        smooth_contour = cv2.approxPolyDP(main_contour, epsilon, True)
        
        result = {
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'eccentricity': eccentricity,
            'solidity': solidity,
            'convexity': convexity,
            'composite': composite,
            'area': area,
            'perimeter': perimeter,
            'edges_raw': boundary_edges,
            'edges_closed': boundary_edges,
            'contour_raw': main_contour,
            'contour_smooth': smooth_contour,
            'mask': mask
        }
        
        if ellipse is not None:
            result['ellipse'] = ellipse
        
        return result


def analyze_image_roundness(image_data: bytes, search_term: str, analyzer: RoundnessAnalyzer) -> Optional[Dict]:
    """
    Complete pipeline: detect, segment, and analyze roundness.
    
    Args:
        image_data: Raw image bytes
        search_term: Object to detect
        analyzer: RoundnessAnalyzer instance
        
    Returns:
        Dictionary with metrics and visualizations, or None if object not found
    """
    try:
        img_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_array = np.array(img_pil)
        
        detection = analyzer.detect_object(img_pil, search_term)
        
        if detection is None:
            return None
        
        mask = analyzer.segment_object(img_array, detection['bbox'])
        result = analyzer.analyze_roundness(img_array, mask)
        
        if result is None:
            return None
        
        viz = create_visualizations(img_array, detection, mask, result)
        result['visualizations'] = viz
        result['detection_confidence'] = detection['confidence']
        
        return result
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return None


def analyze_images_batch(
    images_data: List[bytes], 
    search_term: str, 
    analyzer: RoundnessAnalyzer
) -> List[Optional[Dict]]:
    """
    Batch process multiple images (sequential with Grounding DINO).
    
    Args:
        images_data: List of raw image bytes
        search_term: Object to detect
        analyzer: RoundnessAnalyzer instance
        
    Returns:
        List of analysis results (or None for failed images)
    """
    try:
        # Load all images
        images_pil = []
        images_array = []
        
        print(f"   Loading {len(images_data)} images...")
        for idx, img_data in enumerate(images_data):
            try:
                img_pil = Image.open(io.BytesIO(img_data)).convert('RGB')
                # Validate image
                if img_pil.width > 0 and img_pil.height > 0:
                    print(f"   Image {idx+1}: {img_pil.width}x{img_pil.height}")
                    images_pil.append(img_pil)
                    images_array.append(np.array(img_pil))
                else:
                    print(f"    ⚠ Invalid image {idx+1}")
                    images_pil.append(None)
                    images_array.append(None)
            except Exception as e:
                print(f"    ⚠ Failed to load image {idx+1}: {e}")
                images_pil.append(None)
                images_array.append(None)
        
        # Process sequentially (DINO doesn't batch well)
        results = []
        for i, (img_pil, img_array) in enumerate(zip(images_pil, images_array)):
            if img_pil is None or img_array is None:
                results.append(None)
                continue
            
            try:
                # Detect
                detection = analyzer.detect_object(img_pil, search_term)
                if detection is None:
                    results.append(None)
                    continue
                
                # Segment
                mask = analyzer.segment_object(img_array, detection['bbox'])
                
                # Analyze
                analysis = analyzer.analyze_roundness(img_array, mask)
                if analysis is None:
                    results.append(None)
                    continue
                
                # Visualize
                viz = create_visualizations(img_array, detection, mask, analysis)
                analysis['visualizations'] = viz
                analysis['detection_confidence'] = detection['confidence']
                
                results.append(analysis)
                
            except Exception as e:
                print(f"    ✗ Error processing image {i+1}: {e}")
                results.append(None)
        
        return results
        
    except Exception as e:
        print(f"    ✗ Batch error: {e}")
        return [None] * len(images_data)


def create_visualizations(
    original: np.ndarray,
    detection: Dict,
    mask: np.ndarray,
    analysis: Dict
) -> Dict[str, bytes]:
    """Create visualization images"""
    viz = {}
    
    # Original with bbox
    img_bbox = original.copy()
    x1, y1, x2, y2 = detection['bbox']
    cv2.rectangle(img_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
    viz['original'] = array_to_jpeg(img_bbox)
    
    # Grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    viz['grayscale'] = array_to_jpeg(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    
    # SAM mask
    masked_img = original.copy()
    masked_img[mask == 0] = masked_img[mask == 0] * 0.3
    viz['segmentation'] = array_to_jpeg(masked_img)
    
    # Edges
    viz['edges_raw'] = array_to_jpeg(cv2.cvtColor(analysis['edges_raw'], cv2.COLOR_GRAY2RGB))
    viz['edges_closed'] = array_to_jpeg(cv2.cvtColor(analysis['edges_closed'], cv2.COLOR_GRAY2RGB))
    
    # Contour
    contour_img = original.copy()
    cv2.drawContours(contour_img, [analysis['contour_smooth']], -1, (0, 255, 0), 2)
    viz['contour'] = array_to_jpeg(contour_img)
    
    return viz


def array_to_jpeg(array: np.ndarray, quality: int = 85) -> bytes:
    """Convert numpy array to JPEG bytes"""
    img = Image.fromarray(array)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    return buffer.getvalue()