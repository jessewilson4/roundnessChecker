"""
# === UAIPCS START ===
# file: utils/edge_detection.py
# purpose: Object detection + segmentation + roundness analysis with detailed logging
# deps: [@transformers:library, @segment_anything:library, @torch:library, @opencv:library, @numpy:library, @PIL:library, @config:module/local]
# funcs:
#   - analyze_image_roundness(image_data:bytes, search_term:str, analyzer:RoundnessAnalyzer, image_id:str=None) -> dict  # no_side_effect
#   - analyze_images_batch(images_data:list, search_term:str, analyzer:RoundnessAnalyzer, image_ids:list=None) -> list  # no_side_effect
#   - create_visualizations(image:ndarray, detection:dict, mask:ndarray, result:dict) -> dict  # no_side_effect
#   - compress_thumbnail(image_bytes:bytes, max_size_kb:int) -> bytes  # no_side_effect
# classes:
#   - RoundnessAnalyzer  # Main analysis class wrapping OWL-ViT and SAM
# refs:
#   - utils/config.py::get_setting
# notes: resolution=1024px|detection_filters=[confidence,closeup]|logging=per_image_detailed
# === UAIPCS END ===
"""

import cv2
import numpy as np
from PIL import Image
import io
import torch
import gc
from typing import Optional, Dict, Tuple, List
from pathlib import Path

# Import models
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from segment_anything import sam_model_registry, SamPredictor


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
    
    # SAM mask visualization
    masked_img = original.copy()
    masked_img[mask == 0] = masked_img[mask == 0] * 0.3
    viz['segmentation'] = array_to_jpeg(masked_img)
    
    # Raw edges
    viz['edges_raw'] = array_to_jpeg(cv2.cvtColor(analysis['edges_raw'], cv2.COLOR_GRAY2RGB))
    
    # Closed edges
    viz['edges_closed'] = array_to_jpeg(cv2.cvtColor(analysis['edges_closed'], cv2.COLOR_GRAY2RGB))
    
    # Final contour
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


def auto_canny(image, sigma=0.33):
    """Placeholder for compatibility - not used in new system"""
    pass


class RoundnessAnalyzer:
    """
    Analyzer for detecting and measuring object roundness using:
    - OWL-ViT for open-vocabulary object detection
    - SAM for precise segmentation
    - Canny edge detection with morphological smoothing
    
    OPTIMIZED: Now supports batch inference for parallel processing
    """
    
    def __init__(self, use_local_models=False):
        """
        Initialize the analyzer with models.
        
        Args:
            use_local_models: If True, load from ./models/, else from HuggingFace
        """
        print("🤖 Loading AI models...")
        
        if use_local_models:
            owl_path = "./models/owlvit-base-patch32"
            sam_path = "./models/sam_vit_h_4b8939.pth"
            print(f"   Loading OWL-ViT from: {owl_path}")
        else:
            owl_path = "google/owlvit-base-patch32"
            sam_path = None
            print(f"   Loading OWL-ViT from: {owl_path}")
        
        self.processor = OwlViTProcessor.from_pretrained(owl_path)
        self.model = OwlViTForObjectDetection.from_pretrained(owl_path)
        
        # SAM model
        print("   Loading SAM...")
        sam_checkpoint = "models/sam_vit_b_01ec64.pth" if use_local_models else "sam_vit_b_01ec64.pth"
        
        if not Path(sam_checkpoint).exists():
            print(f"   Downloading SAM checkpoint (375MB)...")
            import urllib.request
            Path(sam_checkpoint).parent.mkdir(exist_ok=True)
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            urllib.request.urlretrieve(url, sam_checkpoint)
        
        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
            
        if torch.cuda.is_available():
            sam = sam.cuda()
            
        self.predictor = SamPredictor(sam)
        
        print("   ✓ Models loaded\n")
    
    def detect_object(self, image: Image.Image, text_query: str, threshold: float = 0.15) -> Optional[Dict]:
        """
        Detect object in image using OWL-ViT.
        Returns None if no detection found.
        """
        inputs = self.processor(text=[[text_query]], images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=threshold,
            target_sizes=target_sizes
        )[0]
        
        if len(results['boxes']) == 0:
            return None
        
        # Get highest confidence detection
        best_idx = torch.argmax(results['scores']).item()
        box = results['boxes'][best_idx].int().tolist()
        confidence = results['scores'][best_idx].item()
        
        return {
            'bbox': box,
            'confidence': confidence,
            'label': text_query
        }
    
    def detect_objects_batch(self, images: List[Image.Image], text_query: str, threshold: float = 0.15) -> List[Optional[Dict]]:
        """
        Batch detect objects across multiple images.
        """
        if not images:
            return []
        
        inputs = self.processor(
            text=[[text_query]] * len(images),
            images=images,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        target_sizes = torch.tensor([img.size[::-1] for img in images])
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=threshold,
            target_sizes=target_sizes
        )
        
        detections = []
        for result in results:
            if len(result['boxes']) == 0:
                detections.append(None)
            else:
                best_idx = torch.argmax(result['scores']).item()
                box = result['boxes'][best_idx].int().tolist()
                confidence = result['scores'][best_idx].item()
                
                detections.append({
                    'bbox': box,
                    'confidence': confidence,
                    'label': text_query
                })
        
        return detections
    
    def segment_object(self, image: np.ndarray, bbox: list) -> np.ndarray:
        """
        Segment object using SAM given a bounding box.
        Uses bbox as prompt for more accurate segmentation.
        """
        self.predictor.set_image(image)
        
        # Use the BBOX as the prompt (more reliable than center point)
        x1, y1, x2, y2 = bbox
        input_box = np.array([x1, y1, x2, y2])
        
        masks, scores, _ = self.predictor.predict(
            box=input_box,
            multimask_output=True
        )
        
        # Get best mask (highest score)
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        
        return mask.astype(np.uint8) * 255
    
    def analyze_roundness(self, image: np.ndarray, mask: np.ndarray) -> Optional[Dict]:
        """
        Analyze roundness metrics from segmentation mask using Canny edge detection.
        """
        # Smooth mask with larger elliptical kernel to remove watermarks/noise
        # Increased from 3x3 to 7x7 to better filter out attached text/logos
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_smooth = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_smooth)
        mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_CLOSE, kernel_smooth)
        
        # Get edges directly from the mask (silhouette) to avoid internal lines
        # We use Canny on the binary mask to get a clean boundary line for visualization
        edges_raw = cv2.Canny(mask_smooth, 100, 200)
        edges_closed = edges_raw.copy() # No need to close if we use the mask
        
        # Find contours from the MASK, not the edges of the texture
        # RETR_EXTERNAL ensures we only get the outer boundary
        contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Largest contour
        main_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(main_contour)
        
        if area < 100:
            return None
            
        # Create a clean mask with ONLY the main contour
        # This prevents multiple objects from being grouped together or analyzed
        clean_mask = np.zeros_like(mask)
        cv2.drawContours(clean_mask, [main_contour], -1, (255), thickness=cv2.FILLED)
        
        # Update the mask to be the clean one
        mask = clean_mask
        
        # Smooth contour
        perimeter_raw = cv2.arcLength(main_contour, True)
        # Reduced epsilon to 0.0005 (0.05%) to keep curves smoother and less faceted
        epsilon = 0.0005 * perimeter_raw
        contour_smooth = cv2.approxPolyDP(main_contour, epsilon, True)
        
        # Calculate metrics
        perimeter = cv2.arcLength(contour_smooth, True)
        
        if perimeter == 0:
            return None
        
        # Circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        circularity = min(circularity, 1.0)
        
        # Aspect ratio from ellipse
        aspect_ratio = 0
        eccentricity = 0
        if len(contour_smooth) >= 5:
            ellipse = cv2.fitEllipse(contour_smooth)
            (cx, cy), (MA, ma), angle = ellipse
            if max(MA, ma) > 0:
                aspect_ratio = min(MA, ma) / max(MA, ma)
                if MA > ma:
                    eccentricity = np.sqrt(1 - (ma/MA)**2) if MA > 0 else 0
                else:
                    eccentricity = np.sqrt(1 - (MA/ma)**2) if ma > 0 else 0
                eccentricity = 1 - eccentricity  # Invert so rounder = higher
        
        # Solidity
        hull = cv2.convexHull(contour_smooth)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Convexity
        hull_perimeter = cv2.arcLength(hull, True)
        convexity = hull_perimeter / perimeter if perimeter > 0 else 0.0
        convexity = 1.0 / convexity if convexity > 0 else 0.0
        convexity = min(convexity, 1.0)
        
        # Composite score
        composite = (
            0.30 * circularity +
            0.25 * aspect_ratio +
            0.20 * eccentricity +
            0.15 * solidity +
            0.10 * convexity
        )
        
        return {
            'circularity': float(circularity),
            'aspect_ratio': float(aspect_ratio),
            'eccentricity': float(eccentricity),
            'solidity': float(solidity),
            'convexity': float(convexity),
            'composite': float(composite),
            'area': float(area),
            'perimeter': float(perimeter),
            'edges_raw': edges_raw,
            'edges_closed': edges_closed,
            'contour_raw': main_contour,
            'contour_smooth': contour_smooth,
            'mask': mask
        }


def analyze_image_roundness(
    image_data: bytes, 
    search_term: str, 
    analyzer: RoundnessAnalyzer,
    image_id: str = "unknown"
) -> Optional[Dict]:
    """
    Complete pipeline: detect, segment, and analyze roundness.
    
    Args:
        image_data: Raw image bytes
        search_term: Object to detect (e.g., "cat", "pitchfork")
        analyzer: RoundnessAnalyzer instance
        image_id: Image identifier for logging
        
    Returns:
        Dictionary with metrics and visualizations, or None if object not found
    """
    try:
        # Load and validate image
        img_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Resize to 1024px max dimension
        if max(img_pil.width, img_pil.height) > 1024:
            ratio = 1024 / max(img_pil.width, img_pil.height)
            new_size = (int(img_pil.width * ratio), int(img_pil.height * ratio))
            img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)
        
        img_array = np.array(img_pil)
        
        # Load config thresholds
        from utils.config import get_setting
        confidence_threshold = get_setting('detection.confidence_threshold', 0.05)
        closeup_threshold = get_setting('detection.closeup_threshold', 0.90)
        
        # Detect object
        if image_id:
            print(f"  🔍 Analyzing image {image_id}...")
            
        detection = analyzer.detect_object(img_pil, search_term, threshold=confidence_threshold)
        
        if detection is None:
            if image_id:
                print(f"  ✗ No object detected in {image_id}")
            return None
        
        # Check confidence
        if detection['confidence'] < confidence_threshold:
            if image_id:
                print(f"  ✗ Low confidence ({detection['confidence']:.2f}) in {image_id}")
            return None
        
        # Check if closeup (bbox covers too much of image)
        bbox = detection['bbox']
        bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        img_area = img_array.shape[0] * img_array.shape[1]
        if bbox_area / img_area > closeup_threshold:
            if image_id:
                print(f"  ✗ Closeup detected (coverage {bbox_area/img_area:.2f}) in {image_id}")
            return None
        
        # Segment object
        mask = analyzer.segment_object(img_array, bbox)
        
        # Analyze roundness
        result = analyzer.analyze_roundness(img_array, mask)
        
        if result is None:
            return None
        
        # Create visualizations
        viz = create_visualizations(img_array, detection, mask, result)
        result['visualizations'] = viz
        result['detection_confidence'] = detection['confidence']
        
        # Memory cleanup
        del mask, result['mask']
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return result
        
    except Exception as e:
        return None


def analyze_images_batch(
    images_data: List[bytes], 
    search_term: str, 
    analyzer: RoundnessAnalyzer,
    image_ids: List[str] = None
) -> List[Optional[Dict]]:
    """
    Process multiple images sequentially (kept for backward compatibility).
    
    Args:
        images_data: List of raw image bytes
        search_term: Object to detect
        analyzer: RoundnessAnalyzer instance
        image_ids: Optional list of image identifiers for logging
        
    Returns:
        List of analysis results (or None for failed images)
    """
    if image_ids is None:
        image_ids = [f"image_{i}" for i in range(len(images_data))]
    
    # Process sequentially
    results = []
    for img_data, img_id in zip(images_data, image_ids):
        result = analyze_image_roundness(img_data, search_term, analyzer, img_id)
        results.append(result)
    
    return results