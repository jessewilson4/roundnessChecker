"""
Object Detection and Segmentation using OWL-ViT + SAM
Open-vocabulary object detection (OWL-ViT)
2. Object segmentation (SAM)
3. Edge detection with contour smoothing
4. Roundness metric calculation
"""

import cv2
import numpy as np
from PIL import Image
import io
import torch
from typing import Optional, Dict, Tuple, List
from pathlib import Path

# Import models
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from segment_anything import sam_model_registry, SamPredictor


class RoundnessAnalyzer:
    """
    Analyzer for detecting and measuring object roundness using:
    - OWL-ViT for open-vocabulary object detection
    - SAM for precise segmentation
    - Canny edge detection with morphological smoothing
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
    
    def detect_object(self, image_pil: Image.Image, target_text: str, threshold: float = 0.1) -> Optional[Dict]:
        """
        Detect object in image using OWL-ViT.
        
        Args:
            image_pil: PIL Image
            target_text: Text description of object
            threshold: Confidence threshold
            
        Returns:
            Detection dict or None
        """
        inputs = self.processor(text=[[target_text]], images=image_pil, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.owl_model(**inputs)
        
        target_sizes = torch.tensor([image_pil.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=threshold
        )
        
        boxes, scores = results[0]["boxes"], results[0]["scores"]
        
        if len(boxes) == 0:
            return None
        
        best_idx = scores.argmax()
        box = boxes[best_idx].numpy()
        score = scores[best_idx].item()
        
        return {
            'bbox': box.astype(int).tolist(),
            'confidence': score,
            'center': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        }
    
    def segment_object(self, image_np: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Segment object using SAM with bounding box prompt.
        
        Args:
            image_np: Image as numpy array (RGB)
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Binary mask
        """
        self.sam_predictor.set_image(image_np)
        
        masks, scores, _ = self.sam_predictor.predict(
            box=np.array(bbox),
            multimask_output=True
        )
        
        best_idx = scores.argmax()
        return masks[best_idx]
    
    def analyze_edges(self, image: np.ndarray, mask: np.ndarray) -> Dict:
        """
        Analyze edges within segmentation mask with smoothing.
        
        Args:
            image: Original image (BGR)
            mask: Binary mask
            
        Returns:
            Dict with edges, contour, and metrics
        """
        # Smooth mask
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_smooth = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_smooth)
        mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_CLOSE, kernel_smooth)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        masked_gray = gray * mask_smooth
        
        # Blur
        blurred = cv2.GaussianBlur(masked_gray, (5, 5), 0)
        
        # Auto Canny
        v = np.median(blurred[blurred > 0]) if np.any(blurred > 0) else 128
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges_raw = cv2.Canny(blurred, lower, upper)
        
        # Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        edges_closed = cv2.morphologyEx(edges_raw, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Largest contour
        main_contour = max(contours, key=cv2.contourArea)
        
        # Smooth contour
        perimeter_raw = cv2.arcLength(main_contour, True)
        epsilon = 0.002 * perimeter_raw
        contour_smooth = cv2.approxPolyDP(main_contour, epsilon, True)
        
        # Calculate metrics
        area = cv2.contourArea(contour_smooth)
        perimeter = cv2.arcLength(contour_smooth, True)
        
        if perimeter == 0:
            return None
        
        # Circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
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
                eccentricity = 1 - eccentricity
        
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
            'contour_smooth': contour_smooth,
        }
    
    def analyze(self, image_pil: Image.Image, target_text: str) -> Optional[Dict]:
        """
        Full pipeline: detect -> segment -> analyze.
        
        Args:
            image_pil: PIL Image
            target_text: Object description
            
        Returns:
            Complete analysis dict or None
        """
        # Detect
        detection = self.detect_object(image_pil, target_text)
        if detection is None:
            return None
        
        # Convert to numpy
        image_np = np.array(image_pil)
        
        # Segment
        mask = self.segment_object(image_np, detection['bbox'])
        
        # Analyze edges
        analysis = self.analyze_edges(image_np, mask)
        if analysis is None:
            return None
        
        return {
            'detection': detection,
            'mask': mask,
            **analysis
        }


def analyze_image_roundness(image_pil: Image.Image, target_text: str, analyzer: RoundnessAnalyzer) -> Optional[Dict]:
    """
    Convenience function for analyzing image roundness.
    
    Args:
        image_pil: PIL Image
        target_text: Object description
        analyzer: RoundnessAnalyzer instance
        
    Returns:
        Analysis dict or None
    """
    return analyzer.analyze(image_pil, target_text)


def generate_visualizations(original: np.ndarray, detection: Dict, mask: np.ndarray, analysis: Dict) -> Dict[str, bytes]:
    """
    Generate visualization images.
    
    Args:
        original: Original image (RGB)
        detection: Detection dict
        mask: Binary mask
        analysis: Analysis dict
        
    Returns:
        Dict of {name: jpeg_bytes}
    """
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