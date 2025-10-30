#!/usr/bin/env python3
"""
POC: Open-Vocabulary Object Detection + Segmentation Test Script

Uses OWL-ViT (open-vocabulary) instead of YOLO to detect ANY object by text description.
Works with any search term, not limited to 80 COCO classes.

Usage:
    python test_owl_vit.py <object_name> <image_path>
    
Example:
    python test_owl_vit.py orb test_orb.jpg
    python test_owl_vit.py "round stone" test_stone.jpg
    python test_owl_vit.py "glass sphere" test_sphere.jpg
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from segment_anything import sam_model_registry, SamPredictor


def load_models(model_path="google/owlvit-base-patch32"):
    """
    Load OWL-ViT and SAM models
    
    Args:
        model_path: Path to OWL-ViT model (default: download from HuggingFace)
                    Can use local path like "./owlvit-base-patch32" if downloaded manually
    """
    print("Loading models...")
    
    # Load OWL-ViT (open-vocabulary detection)
    print(f"  Loading OWL-ViT from: {model_path}")
    if not model_path.startswith("./") and not model_path.startswith("/"):
        print("    (First run will download ~1.8GB model)")
    
    try:
        processor = OwlViTProcessor.from_pretrained(model_path, token=False, local_files_only=False)
        owl_model = OwlViTForObjectDetection.from_pretrained(model_path, token=False, local_files_only=False)
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        print("\nTo use a local model:")
        print("  1. Download manually: python download_owlvit.py")
        print("  2. Run: python test_owl_vit.py orb image.jpg --local")
        raise
    
    # Load SAM
    print("  Loading SAM (ViT-B)...")
    sam_checkpoint = "sam_vit_b_01ec64.pth"
    if not Path(sam_checkpoint).exists():
        print(f"  Downloading SAM checkpoint (375MB)...")
        import urllib.request
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        urllib.request.urlretrieve(url, sam_checkpoint)
    
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam_predictor = SamPredictor(sam)
    
    print("✓ Models loaded\n")
    return processor, owl_model, sam_predictor


def detect_objects_owl(processor, model, image_pil, target_text, threshold=0.1):
    """
    Use OWL-ViT to detect objects matching text description.
    
    Args:
        processor: OWL-ViT processor
        model: OWL-ViT model
        image_pil: PIL Image
        target_text: Text description (e.g., "orb", "round stone", "glass sphere")
        threshold: Confidence threshold (default 0.1 - lower than YOLO since OWL-ViT scores differently)
    
    Returns:
        List of detections with bounding boxes and scores
    """
    print(f"🔍 Running OWL-ViT detection (looking for '{target_text}')...")
    start = time.time()
    
    # Prepare text queries - try multiple variations for better results
    text_queries = [
        [target_text],  # Exact term
        [f"a {target_text}"],  # With article
        [f"a photo of a {target_text}"],  # More descriptive
    ]
    
    all_detections = []
    
    # Try each query variation
    for queries in text_queries:
        inputs = processor(text=queries, images=image_pil, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get predictions
        target_sizes = torch.Tensor([image_pil.size[::-1]])
        results = processor.post_process_object_detection(
            outputs=outputs, 
            target_sizes=target_sizes, 
            threshold=threshold
        )[0]
        
        # Collect detections
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        
        for box, score in zip(boxes, scores):
            all_detections.append({
                'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                'confidence': float(score),
                'query': queries[0]
            })
    
    # Sort by confidence and remove duplicates (boxes that overlap significantly)
    all_detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Simple NMS (non-maximum suppression) to remove duplicates
    filtered_detections = []
    for det in all_detections:
        is_duplicate = False
        for existing in filtered_detections:
            if boxes_overlap(det['bbox'], existing['bbox']) > 0.5:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_detections.append(det)
    
    elapsed = time.time() - start
    print(f"   ⏱️  OWL-ViT: {elapsed:.3f}s")
    print(f"   ✓ Found {len(filtered_detections)} potential match(es)")
    
    if filtered_detections:
        print(f"   ✓ Best match: {filtered_detections[0]['confidence']:.1%} confidence")
    
    return filtered_detections, elapsed


def boxes_overlap(box1, box2):
    """Calculate IoU (Intersection over Union) between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    
    # Calculate union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def segment_with_sam(sam_predictor, image, bbox):
    """
    Use SAM to segment object within bounding box.
    
    Returns binary mask.
    """
    print(f"   Running SAM segmentation...")
    start = time.time()
    
    # Set image for SAM
    sam_predictor.set_image(image)
    
    # Convert bbox to SAM format
    x1, y1, x2, y2 = bbox
    input_box = np.array([x1, y1, x2, y2])
    
    # Get mask
    masks, scores, _ = sam_predictor.predict(
        box=input_box,
        multimask_output=False
    )
    
    mask = masks[0]
    elapsed = time.time() - start
    print(f"   ⏱️  SAM: {elapsed:.3f}s")
    
    return mask, elapsed


def analyze_edge_detection(image, mask):
    """
    Run edge detection on masked region with contour smoothing.
    """
    print(f"   Running edge detection on masked region...")
    start = time.time()
    
    # Step 1: Smooth the mask using morphological operations
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_smooth = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_smooth)
    mask_smooth = cv2.morphologyEx(mask_smooth, cv2.MORPH_CLOSE, kernel_smooth)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply smoothed mask
    masked_gray = gray * mask_smooth
    
    # Blur
    blurred = cv2.GaussianBlur(masked_gray, (5, 5), 0)
    
    # Canny edge detection
    v = np.median(blurred[blurred > 0]) if np.any(blurred > 0) else 128
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(blurred, lower, upper)
    
    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        
        # Step 2: Smooth the contour using Douglas-Peucker approximation
        perimeter_raw = cv2.arcLength(main_contour, True)
        epsilon = 0.002 * perimeter_raw
        smooth_contour = cv2.approxPolyDP(main_contour, epsilon, True)
        
        # Calculate metrics on smoothed contour
        area = cv2.contourArea(smooth_contour)
        perimeter = cv2.arcLength(smooth_contour, True)
        
        # Circularity
        if perimeter > 0:
            circularity = (4 * np.pi * area) / (perimeter ** 2)
        else:
            circularity = 0
        
        contour_info = {
            'raw': main_contour,
            'smooth': smooth_contour
        }
    else:
        contour_info = None
        area = 0
        perimeter = 0
        circularity = 0
    
    elapsed = time.time() - start
    print(f"   ⏱️  Edge Detection: {elapsed:.3f}s")
    print(f"   ✓ Smoothing applied")
    
    return {
        'edges': edges_closed,
        'contour': contour_info,
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity
    }


def create_visualization(image, detection, mask, edge_result):
    """Create side-by-side comparison visualization"""
    h, w = image.shape[:2]
    
    # Create 2x3 grid
    vis = np.zeros((h*2, w*3, 3), dtype=np.uint8)
    
    # 1. Original with OWL-ViT bbox
    img1 = image.copy()
    x1, y1, x2, y2 = detection['bbox']
    cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img1, f"{detection['confidence']:.1%} confident", 
                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    vis[0:h, 0:w] = img1
    
    # 2. SAM mask
    img2 = image.copy()
    img2[mask == 0] = img2[mask == 0] * 0.3
    vis[0:h, w:w*2] = img2
    
    # 3. Edge detection
    edges_rgb = cv2.cvtColor(edge_result['edges'], cv2.COLOR_GRAY2BGR)
    vis[0:h, w*2:w*3] = edges_rgb
    
    # 4. Raw contour
    img4 = image.copy()
    if edge_result['contour'] is not None and edge_result['contour']['raw'] is not None:
        cv2.drawContours(img4, [edge_result['contour']['raw']], -1, (0, 0, 255), 2)
        cv2.putText(img4, "RAW", (10, h-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    vis[h:h*2, 0:w] = img4
    
    # 5. Smoothed contour
    img5 = image.copy()
    if edge_result['contour'] is not None and edge_result['contour']['smooth'] is not None:
        cv2.drawContours(img5, [edge_result['contour']['smooth']], -1, (0, 255, 0), 2)
        cv2.putText(img5, "SMOOTHED", (10, h-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    vis[h:h*2, w:w*2] = img5
    
    # 6. Comparison
    img6 = image.copy()
    if edge_result['contour'] is not None:
        if edge_result['contour']['raw'] is not None:
            cv2.drawContours(img6, [edge_result['contour']['raw']], -1, (0, 0, 255), 1)
        if edge_result['contour']['smooth'] is not None:
            cv2.drawContours(img6, [edge_result['contour']['smooth']], -1, (0, 255, 0), 2)
    cv2.putText(img6, "Red=Raw, Green=Smooth", (10, h-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
    vis[h:h*2, w*2:w*3] = img6
    
    # Add labels
    cv2.putText(vis, "1. OWL-ViT Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, "2. SAM Segmentation", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, "3. Edge Detection", (w*2+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, "4. Raw Contour", (10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, "5. Smoothed Contour", (w+10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, "6. Comparison", (w*2+10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis


def main():
    # Check for --local flag
    use_local = "--local" in sys.argv
    if use_local:
        sys.argv.remove("--local")
    
    if len(sys.argv) != 3:
        print("Usage: python test_owl_vit.py <object_name> <image_path> [--local]")
        print("\nExamples:")
        print("  python test_owl_vit.py orb test_orb.jpg")
        print("  python test_owl_vit.py \"round stone\" test_stone.jpg")
        print("  python test_owl_vit.py sphere test_sphere.jpg --local  # Use local model")
        print("\nNote: Works with ANY object description (not limited to 80 classes)")
        sys.exit(1)
    
    target_text = sys.argv[1]
    image_path = sys.argv[2]
    
    # Validate inputs
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)
    
    print("="*80)
    print(f"POC: Open-Vocabulary Object Detection + Segmentation")
    print(f"Target: '{target_text}'")
    print(f"Image: {image_path}")
    print("="*80 + "\n")
    
    # Load image
    image_cv = cv2.imread(image_path)
    if image_cv is None:
        print(f"❌ Error: Could not load image: {image_path}")
        sys.exit(1)
    
    image_pil = Image.open(image_path).convert("RGB")
    
    print(f"✓ Image loaded: {image_cv.shape[1]}x{image_cv.shape[0]}\n")
    
    # Load models
    model_path = "./owlvit-base-patch32" if use_local else "google/owlvit-base-patch32"
    processor, owl_model, sam_predictor = load_models(model_path)
    
    # Detect objects with OWL-ViT
    detections, owl_time = detect_objects_owl(processor, owl_model, image_pil, target_text)
    
    if not detections:
        print(f"\n❌ No '{target_text}' detected in image")
        print(f"   OWL-ViT couldn't find any instances matching '{target_text}'")
        print(f"   Try:")
        print(f"     - A different image")
        print(f"     - A more specific description (e.g., 'blue orb' instead of 'orb')")
        print(f"     - A more generic description (e.g., 'sphere' instead of 'crystal orb')")
        sys.exit(0)
    
    print(f"\n✓ Processing best detection (confidence: {detections[0]['confidence']:.1%})\n")
    
    # Segment with SAM
    mask, sam_time = segment_with_sam(sam_predictor, image_cv, detections[0]['bbox'])
    
    # Analyze edges
    edge_result = analyze_edge_detection(image_cv, mask)
    
    # Create visualization
    print("\n📊 Creating visualization...")
    vis = create_visualization(image_cv, detections[0], mask, edge_result)
    
    # Save results
    output_path = f"test_result_{target_text.replace(' ', '_')}.jpg"
    cv2.imwrite(output_path, vis)
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"✓ Object detected: '{target_text}' ({detections[0]['confidence']:.1%} confidence)")
    print(f"✓ Segmentation completed")
    print(f"✓ Edge analysis completed")
    print(f"\nMetrics:")
    print(f"  Circularity: {edge_result['circularity']:.3f} ({edge_result['circularity']*100:.1f}%)")
    print(f"  Area: {edge_result['area']:.0f} px²")
    print(f"  Perimeter: {edge_result['perimeter']:.1f} px")
    print(f"\nTiming:")
    print(f"  OWL-ViT detection: {owl_time:.3f}s")
    print(f"  SAM segmentation: {sam_time:.3f}s")
    print(f"  Total: {owl_time + sam_time:.3f}s per image")
    print(f"\n✓ Visualization saved: {output_path}")
    print("="*80)
    
    # Display (if possible)
    try:
        cv2.imshow('Open-Vocabulary Detection Test', vis)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("\n(Could not display image - running headless)")


if __name__ == "__main__":
    main()