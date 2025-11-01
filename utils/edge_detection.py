"""
Object Detection and Segmentation using OWL-ViT + SAM
OPTIMIZED: Added batch inference for parallel processing

This module handles:
1. Open-vocabulary object detection (OWL-ViT) - NOW WITH BATCH SUPPORT
2. Object segmentation (SAM)
3. Edge detection with contour smoothing
4. Roundness metric calculation
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
            use_local_models: If True, load from ./models/ directory
        """
        self.processor = None
        self.owl_model = None
        self.sam_predictor = None
        self.use_local_models = use_local_models
        self._load_models()
    
    def _load_models(self):
        """Load OWL-ViT and SAM models (lazy loading)"""
        print("🤖 Loading AI models...")
        
        # OWL-ViT model path
        owl_path = "./models/owlvit-base-patch32" if self.use_local_models else "google/owlvit-base-patch32"
        
        print(f"   Loading OWL-ViT from: {owl_path}")
        self.processor = OwlViTProcessor.from_pretrained(owl_path, token=False, local_files_only=self.use_local_models)
        self.owl_model = OwlViTForObjectDetection.from_pretrained(owl_path, token=False, local_files_only=self.use_local_models)
        
        # SAM model
        print("   Loading SAM...")
        sam_checkpoint = "models/sam_vit_b_01ec64.pth" if self.use_local_models else "sam_vit_b_01ec64.pth"
        
        if not Path(sam_checkpoint).exists():
            print(f"   Downloading SAM checkpoint (375MB)...")
            import urllib.request
            Path(sam_checkpoint).parent.mkdir(exist_ok=True)
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            urllib.request.urlretrieve(url, sam_checkpoint)
        
        sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
        self.sam_predictor = SamPredictor(sam)
        
        print("   ✓ Models loaded\n")
    
    def detect_object(self, image_pil: Image.Image, target_text: str, threshold: float = 0.05) -> Optional[Dict]:
        """
        Detect object in image using OWL-ViT.
        
        Args:
            image_pil: PIL Image
            target_text: Object description (e.g., "cat", "orb", "round stone")
            threshold: Confidence threshold
            
        Returns:
            Detection dict with bbox and confidence, or None if not found
        """
        # Try multiple query variations
        text_queries = [
            [target_text],
            [f"a {target_text}"],
            [f"a photo of a {target_text}"],
        ]
        
        all_detections = []
        
        for queries in text_queries:
            inputs = self.processor(text=queries, images=image_pil, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.owl_model(**inputs)
            
            target_sizes = torch.Tensor([image_pil.size[::-1]])
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold
            )[0]
            
            boxes = results["boxes"].cpu().numpy()
            scores = results["scores"].cpu().numpy()
            
            for box, score in zip(boxes, scores):
                all_detections.append({
                    'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    'confidence': float(score)
                })
        
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
    
    def detect_objects_batch(self, images_pil: List[Image.Image], target_text: str, threshold: float = 0.05) -> List[Optional[Dict]]:
        """
        OPTIMIZED: Batch detect objects in multiple images simultaneously.
        
        Args:
            images_pil: List of PIL Images
            target_text: Object description
            threshold: Confidence threshold
            
        Returns:
            List of detection dicts (or None for each image)
        """
        if not images_pil:
            return []
        
        # Fallback to sequential if batch fails
        try:
            return self._detect_objects_batch_impl(images_pil, target_text, threshold)
        except RuntimeError as e:
            # Known transformers library bug with empty batch results
            if 'shape' in str(e) and 'invalid for input of size' in str(e):
                # Silently fall back - this is expected for some queries
                pass
            else:
                print(f"    ⚠️  Unexpected batch error: {e}")
        except Exception as e:
            print(f"    ⚠️  Batch detection failed: {e}")
        
        # Sequential processing fallback
        results = []
        for img in images_pil:
            try:
                detection = self.detect_object(img, target_text, threshold)
                results.append(detection)
            except:
                results.append(None)
        return results
    
    def _detect_objects_batch_impl(self, images_pil: List[Image.Image], target_text: str, threshold: float = 0.05) -> List[Optional[Dict]]:
        """Internal batch detection implementation"""
        
        # Try multiple query variations
        text_queries = [
            [target_text],
            [f"a {target_text}"],
            [f"a photo of a {target_text}"],
        ]
        
        results_per_image = [[] for _ in images_pil]
        batch_failed = False
        
        for query_idx, queries in enumerate(text_queries):
            try:
                inputs = self.processor(text=queries, images=images_pil, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.owl_model(**inputs)
                
                # If we got here, batch processing worked!
            except RuntimeError as model_error:
                # Batch processing failed - this is the transformers library bug
                if 'shape' in str(model_error) and 'invalid for input of size' in str(model_error):
                    # This is the known batch bug - raise immediately to trigger sequential fallback
                    raise
                else:
                    # Unknown error - try next query
                    continue
            except Exception as model_error:
                # Unknown error - try next query
                continue
            
            # Get target sizes for all images
            target_sizes = torch.Tensor([img.size[::-1] for img in images_pil])
            
            # Post-process for each image
            batch_results = self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold
            )
            
            for img_idx, result in enumerate(batch_results):
                boxes = result["boxes"].cpu().numpy()
                scores = result["scores"].cpu().numpy()
                
                for box, score in zip(boxes, scores):
                    results_per_image[img_idx].append({
                        'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        'confidence': float(score)
                    })
        
        # Process results for each image
        final_results = []
        for detections in results_per_image:
            if not detections:
                final_results.append(None)
                continue
            
            # Sort by confidence and remove overlapping detections
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Simple NMS
            filtered = []
            for det in detections:
                is_dup = False
                for existing in filtered:
                    if self._boxes_overlap(det['bbox'], existing['bbox']) > 0.5:
                        is_dup = True
                        break
                if not is_dup:
                    filtered.append(det)
            
            final_results.append(filtered[0] if filtered else None)
        
        return final_results
    
    def _boxes_overlap(self, box1: List[int], box2: List[int]) -> float:
        """Calculate IoU between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
        
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def segment_object(self, image_array: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Segment object using SAM.
        
        Args:
            image_array: RGB image array
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Binary mask
        """
        self.sam_predictor.set_image(image_array)
        
        input_box = np.array(bbox)
        masks, _, _ = self.sam_predictor.predict(
            box=input_box,
            multimask_output=False
        )
        
        return masks[0]
    
    def analyze_roundness(self, image_array: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Analyze roundness using comprehensive shape descriptors with ellipse fitting.
        Uses fitted ellipse for true geometric properties, ignoring pixel-level noise.
        
        Args:
            image_array: RGB image array
            mask: Binary segmentation mask
            
        Returns:
            Dictionary with metrics and composite score
        """
        from skimage.measure import regionprops
        
        # IMPROVED: More aggressive smoothing to remove noise
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_smooth = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_smooth)
        mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_CLOSE, kernel_smooth)
        
        kernel_round = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_CLOSE, kernel_round)
        
        # Extract ONLY the outer boundary with approximation to smooth
        contours, _ = cv2.findContours(mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour (outer boundary)
        main_contour = max(contours, key=cv2.contourArea)
        
        # Apply additional contour smoothing using approximation
        epsilon = 0.001 * cv2.arcLength(main_contour, True)
        main_contour = cv2.approxPolyDP(main_contour, epsilon, True)
        
        # Basic area and perimeter from smoothed contour
        area = cv2.contourArea(main_contour)
        perimeter = cv2.arcLength(main_contour, True)
        
        if area < 100 or perimeter == 0:
            return None
        
        # Get region properties for additional metrics
        try:
            props = regionprops(mask_smooth)[0]
        except:
            return None
        
        # Calculate CIRCULARITY from actual contour (4πA/P²)
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        circularity = min(circularity, 1.0)
        
        # FIT ELLIPSE for aspect ratio and eccentricity only
        if len(main_contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(main_contour)
                (center_x, center_y), (minor_axis, major_axis), angle = ellipse
                
                # Ensure major >= minor
                if minor_axis > major_axis:
                    major_axis, minor_axis = minor_axis, major_axis
                
                # Use ellipse-based metrics for aspect ratio and eccentricity
                if major_axis > 0:
                    aspect_ratio = minor_axis / major_axis
                    eccentricity_raw = np.sqrt(1 - (minor_axis / major_axis) ** 2)
                    eccentricity = 1.0 - eccentricity_raw
                else:
                    aspect_ratio = 0.0
                    eccentricity = 0.0
            except:
                # Fallback to regionprops if ellipse fitting fails
                if props.minor_axis_length > 0:
                    aspect_ratio_raw = props.major_axis_length / props.minor_axis_length
                    aspect_ratio = 1.0 / aspect_ratio_raw if aspect_ratio_raw > 0 else 0.0
                else:
                    aspect_ratio = 0.0
                eccentricity = 1.0 - props.eccentricity
                ellipse = None
        else:
            # Fallback to regionprops
            if props.minor_axis_length > 0:
                aspect_ratio_raw = props.major_axis_length / props.minor_axis_length
                aspect_ratio = 1.0 / aspect_ratio_raw if aspect_ratio_raw > 0 else 0.0
            else:
                aspect_ratio = 0.0
            eccentricity = 1.0 - props.eccentricity
            ellipse = None
        
        # 4. SOLIDITY: Area/Convex hull area (detects indentations)
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
        
        # Light smoothing for visualization only
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
        
        # Add ellipse if fitted successfully
        if ellipse is not None:
            result['ellipse'] = ellipse
        
        return result


def analyze_image_roundness(image_data: bytes, search_term: str, analyzer: RoundnessAnalyzer) -> Optional[Dict]:
    """
    Complete pipeline: detect, segment, and analyze roundness.
    
    Args:
        image_data: Raw image bytes
        search_term: Object to detect (e.g., "cat", "orb")
        analyzer: RoundnessAnalyzer instance
        
    Returns:
        Dictionary with metrics and visualizations, or None if object not found
    """
    try:
        # Load image
        img_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # OPTIMIZATION: Resize to 512px max dimension if larger
        if max(img_pil.width, img_pil.height) > 512:
            ratio = 512 / max(img_pil.width, img_pil.height)
            new_size = (int(img_pil.width * ratio), int(img_pil.height * ratio))
            img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)
        
        img_array = np.array(img_pil)
        
        # Detect object
        detection = analyzer.detect_object(img_pil, search_term)
        
        if detection is None:
            return None
        
        # Segment object
        mask = analyzer.segment_object(img_array, detection['bbox'])
        
        # Analyze roundness
        result = analyzer.analyze_roundness(img_array, mask)
        
        if result is None:
            return None
        
        # Create visualizations
        viz = create_visualizations(img_array, detection, mask, result)
        result['visualizations'] = viz
        result['detection_confidence'] = detection['confidence']
        
        # MEMORY CLEANUP
        del mask, result['mask']
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
    OPTIMIZED: Batch process multiple images with parallel detection.
    
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
        
        for img_data in images_data:
            try:
                img_pil = Image.open(io.BytesIO(img_data)).convert('RGB')
                
                # OPTIMIZATION: Resize to 512px max dimension if larger
                if max(img_pil.width, img_pil.height) > 512:
                    ratio = 512 / max(img_pil.width, img_pil.height)
                    new_size = (int(img_pil.width * ratio), int(img_pil.height * ratio))
                    img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)
                
                images_pil.append(img_pil)
                images_array.append(np.array(img_pil))
            except:
                images_pil.append(None)
                images_array.append(None)
        
        # Batch detect objects (with fallback to sequential)
        valid_indices = [i for i, img in enumerate(images_pil) if img is not None]
        valid_images = [images_pil[i] for i in valid_indices]
        
        if not valid_images:
            return [None] * len(images_data)
        
        detections_batch = analyzer.detect_objects_batch(valid_images, search_term)
        
        # Map back to original indices
        detections = [None] * len(images_data)
        for i, idx in enumerate(valid_indices):
            detections[idx] = detections_batch[i]
        
        # Process each image (segmentation and analysis must be sequential)
        results = []
        for i, (img_array, detection) in enumerate(zip(images_array, detections)):
            if img_array is None or detection is None:
                results.append(None)
                continue
            
            try:
                # Segment object
                mask = analyzer.segment_object(img_array, detection['bbox'])
                
                # Analyze roundness
                analysis = analyzer.analyze_roundness(img_array, mask)
                
                if analysis is None:
                    results.append(None)
                    continue
                
                # Create visualizations
                viz = create_visualizations(img_array, detection, mask, analysis)
                analysis['visualizations'] = viz
                analysis['detection_confidence'] = detection['confidence']
                
                results.append(analysis)
                
                # MEMORY CLEANUP: Release memory after each image
                del mask, analysis['mask']  # Remove large arrays
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    ✗ Error processing image {i}: {e}")
                results.append(None)
        
        # AGGRESSIVE MEMORY CLEANUP: Force garbage collection after batch completes
        del images_pil, images_array, detections
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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