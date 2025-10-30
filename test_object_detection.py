#!/usr/bin/env python3
"""
POC: Object Detection + Segmentation Test Script

Tests YOLO object detection and SAM segmentation on a single image
to evaluate performance before integrating into main app.

Usage:
    python test_object_detection.py <object_name> <image_path>
    
Example:
    python test_object_detection.py cat test_cat.jpg
    python test_object_detection.py ball test_ball.jpg
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# COCO class names that YOLO can detect
COCO_CLASSES = {
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
}


def load_models():
    """Load YOLO and SAM models"""
    print("Loading models...")
    
    # Load YOLOv8 (will auto-download on first run)
    print("  Loading YOLOv8n (nano - fastest)...")
    yolo = YOLO('yolov8n.pt')
    
    # Load SAM (will auto-download on first run)
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
    return yolo, sam_predictor


def detect_objects(yolo, image, target_class):
    """
    Use YOLO to detect objects in image.
    
    Returns list of detections for target class with bounding boxes.
    """
    print(f"🔍 Running YOLO object detection (looking for '{target_class}')...")
    start = time.time()
    
    results = yolo(image, verbose=False)
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id]
            confidence = float(box.conf[0])
            
            if class_name == target_class or (target_class == 'ball' and class_name == 'sports ball'):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
    
    elapsed = time.time() - start
    print(f"   ⏱️  YOLO: {elapsed:.3f}s")
    print(f"   ✓ Found {len(detections)} {target_class}(s)")
    
    return detections, elapsed


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
    # This removes pixel noise and creates smoother edges
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
        # Epsilon = 0.002 of perimeter (preserves corners, smooths curves)
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
        
        # Store both contours for visualization
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
    print(f"   ✓ Smoothing applied (removes pixelation)")
    
    return {
        'edges': edges_closed,
        'contour': contour_info,
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity
    }


def create_visualization(image, detection, mask, edge_result):
    """Create side-by-side comparison visualization with before/after smoothing"""
    h, w = image.shape[:2]
    
    # Create 2x3 grid to show smoothing comparison
    vis = np.zeros((h*2, w*3, 3), dtype=np.uint8)
    
    # 1. Original with YOLO bbox
    img1 = image.copy()
    x1, y1, x2, y2 = detection['bbox']
    cv2.rectangle(img1, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img1, f"{detection['class']} {detection['confidence']:.2f}", 
                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    vis[0:h, 0:w] = img1
    
    # 2. SAM mask
    img2 = image.copy()
    img2[mask == 0] = img2[mask == 0] * 0.3  # Darken background
    vis[0:h, w:w*2] = img2
    
    # 3. Edge detection
    edges_rgb = cv2.cvtColor(edge_result['edges'], cv2.COLOR_GRAY2BGR)
    vis[0:h, w*2:w*3] = edges_rgb
    
    # 4. Raw contour (pixelated)
    img4 = image.copy()
    if edge_result['contour'] is not None and edge_result['contour']['raw'] is not None:
        cv2.drawContours(img4, [edge_result['contour']['raw']], -1, (0, 0, 255), 2)
        cv2.putText(img4, "RAW (pixelated)", (10, h-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    vis[h:h*2, 0:w] = img4
    
    # 5. Smoothed contour (cleaned)
    img5 = image.copy()
    if edge_result['contour'] is not None and edge_result['contour']['smooth'] is not None:
        cv2.drawContours(img5, [edge_result['contour']['smooth']], -1, (0, 255, 0), 2)
        cv2.putText(img5, "SMOOTHED (used for calc)", (10, h-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    vis[h:h*2, w:w*2] = img5
    
    # 6. Comparison (both overlaid)
    img6 = image.copy()
    if edge_result['contour'] is not None:
        if edge_result['contour']['raw'] is not None:
            cv2.drawContours(img6, [edge_result['contour']['raw']], -1, (0, 0, 255), 1)
        if edge_result['contour']['smooth'] is not None:
            cv2.drawContours(img6, [edge_result['contour']['smooth']], -1, (0, 255, 0), 2)
    cv2.putText(img6, "Red=Raw, Green=Smooth", (10, h-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    vis[h:h*2, w*2:w*3] = img6
    
    # Add labels
    cv2.putText(vis, "1. YOLO Detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, "2. SAM Segmentation", (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, "3. Edge Detection", (w*2+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, "4. Raw Contour", (10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, "5. Smoothed Contour", (w+10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, "6. Comparison", (w*2+10, h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return vis


def main():
    if len(sys.argv) != 3:
        print("Usage: python test_object_detection.py <object_name> <image_path>")
        print("\nExamples:")
        print("  python test_object_detection.py cat test_cat.jpg")
        print("  python test_object_detection.py ball test_ball.jpg")
        print("  python test_object_detection.py dog mydog.png")
        print(f"\nYOLO can detect these objects: {', '.join(sorted(COCO_CLASSES))}")
        sys.exit(1)
    
    target_class = sys.argv[1].lower()
    image_path = sys.argv[2]
    
    # Validate inputs
    if not Path(image_path).exists():
        print(f"❌ Error: Image not found: {image_path}")
        sys.exit(1)
    
    if target_class not in COCO_CLASSES and target_class != 'ball':
        print(f"⚠️  Warning: '{target_class}' not in COCO classes")
        print(f"   YOLO may not detect it. Available: {', '.join(sorted(COCO_CLASSES))}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    print("="*80)
    print(f"POC: Object Detection + Segmentation Test")
    print(f"Target: {target_class}")
    print(f"Image: {image_path}")
    print("="*80 + "\n")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image: {image_path}")
        sys.exit(1)
    
    print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]}\n")
    
    # Load models
    yolo, sam_predictor = load_models()
    
    # Detect objects
    detections, yolo_time = detect_objects(yolo, image, target_class)
    
    if not detections:
        print(f"\n❌ No '{target_class}' detected in image")
        print(f"   YOLO couldn't find any instances of '{target_class}'")
        print(f"   Try a different image or object class")
        sys.exit(0)
    
    print(f"\n✓ Processing first detection (confidence: {detections[0]['confidence']:.2%})\n")
    
    # Segment with SAM
    mask, sam_time = segment_with_sam(sam_predictor, image, detections[0]['bbox'])
    
    # Analyze edges
    edge_result = analyze_edge_detection(image, mask)
    
    # Create visualization
    print("\n📊 Creating visualization...")
    vis = create_visualization(image, detections[0], mask, edge_result)
    
    # Save results
    output_path = f"test_result_{target_class}.jpg"
    cv2.imwrite(output_path, vis)
    
    # Print summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"✓ Object detected: {detections[0]['class']} ({detections[0]['confidence']:.2%} confidence)")
    print(f"✓ Segmentation completed")
    print(f"✓ Edge analysis completed")
    print(f"\nMetrics:")
    print(f"  Circularity: {edge_result['circularity']:.3f}")
    print(f"  Area: {edge_result['area']:.0f} px²")
    print(f"  Perimeter: {edge_result['perimeter']:.1f} px")
    print(f"\nTiming:")
    print(f"  YOLO detection: {yolo_time:.3f}s")
    print(f"  SAM segmentation: {sam_time:.3f}s")
    print(f"  Total: {yolo_time + sam_time:.3f}s per image")
    print(f"\n✓ Visualization saved: {output_path}")
    print("="*80)
    
    # Display (if possible)
    try:
        cv2.imshow('Object Detection + Segmentation Test', vis)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("\n(Could not display image - running headless)")


if __name__ == "__main__":
    main()