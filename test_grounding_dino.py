"""
Test script for Grounding DINO on CPU
Uses native GroundingDINO library with Swin-T OGC model (matches project implementation)
Run this standalone to verify detection works before integrating
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import requests
import numpy as np
from PIL import Image
from pathlib import Path

# Force CPU mode
torch.set_default_device('cpu')

from groundingdino.util.inference import load_model, predict
from groundingdino.util import box_ops
import groundingdino.datasets.transforms as T

def test_grounding_dino():
    print("🧪 Testing Grounding DINO (Swin-T OGC) on CPU...")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    
    # Load model (matches utils/edge_detection.py implementation)
    print("\n📥 Loading Grounding DINO Swin-T OGC model...")
    
    checkpoint_path = "groundingdino_swint_ogc.pth"
    if not Path(checkpoint_path).exists():
        print(f"   ❌ Checkpoint not found: {checkpoint_path}")
        print(f"   Download from: https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth")
        return False
    
    # Find config file
    try:
        import groundingdino
        package_path = Path(groundingdino.__file__).parent
        config_path = package_path / "config" / "GroundingDINO_SwinT_OGC.py"
        
        if not config_path.exists():
            possible_paths = [
                package_path / "config" / "GroundingDINO_SwinT_OGC.py",
                Path("groundingdino/config/GroundingDINO_SwinT_OGC.py"),
            ]
            
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break
            else:
                print(f"   ❌ Config not found in: {package_path / 'config'}")
                return False
        
        print(f"   Using config: {config_path}")
        
    except ImportError:
        print("   ❌ groundingdino not installed")
        print("   Install with: pip install groundingdino-py")
        return False
    
    # Load model
    model = load_model(
        model_config_path=str(config_path),
        model_checkpoint_path=checkpoint_path,
        device='cpu'
    )
    model.to('cpu')
    model.eval()
    print("   ✓ Model loaded on CPU")
    
    # Load test image
    print("\n🖼️  Loading test image...")
    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image_pil = Image.open(requests.get(image_url, stream=True).raw)
    print(f"   Image size: {image_pil.size}")
    
    # Preprocess image - create transform pipeline
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    image_transformed, _ = transform(image_pil, None)
    
    # Test detection with multiple prompts
    print("\n🔍 Detecting objects...")
    text_queries = ["a cat", "cat", "cat object"]
    
    all_detections = []
    for text_prompt in text_queries:
        try:
            boxes, logits, phrases = predict(
                model=model,
                image=image_transformed,  # Use preprocessed tensor
                caption=text_prompt,
                box_threshold=0.25,
                text_threshold=0.25,
                device='cpu'
            )
            
            # Ensure CPU
            if hasattr(boxes, 'cpu'):
                boxes = boxes.cpu()
            if hasattr(logits, 'cpu'):
                logits = logits.cpu()
            
            # Convert boxes from normalized [cx, cy, w, h] to pixel [x1, y1, x2, y2]
            H, W = image_pil.height, image_pil.width
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
            
            for box, score, phrase in zip(boxes_xyxy, logits, phrases):
                all_detections.append({
                    'bbox': [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                    'confidence': float(score),
                    'phrase': phrase,
                    'query': text_prompt
                })
            
            print(f"   Query '{text_prompt}': {len(boxes)} detections")
            
        except Exception as e:
            print(f"   ⚠ Query '{text_prompt}' failed: {e}")
    
    # Display results
    print(f"\n✅ Total detections: {len(all_detections)}")
    
    if all_detections:
        # Sort by confidence
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        print("\nTop detections:")
        for i, det in enumerate(all_detections[:5]):
            box = det['bbox']
            print(f"   {i+1}. '{det['phrase']}' (query: '{det['query']}')")
            print(f"      Confidence: {det['confidence']:.3f}")
            print(f"      Box: [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
        
        print("\n✨ SUCCESS! Grounding DINO working on CPU")
        return True
    else:
        print("\n❌ No detections found")
        return False

if __name__ == "__main__":
    try:
        success = test_grounding_dino()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n💥 ERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)