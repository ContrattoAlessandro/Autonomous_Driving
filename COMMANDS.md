# Canonical Command Reference (TLR-YOLO-MTL Champion v4)

Execute all commands from the `tl_detection/` directory within the activated virtual environment.

---

## 1. Environment & Setup

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install project dependencies
pip install -r requirements.txt
```

Environment Specifications:
- Python 3.12, PyTorch CUDA 12.x / 13.x, Ultralytics
- Validated on NVIDIA GeForce RTX 5070 (12GB VRAM)

---

## 2. Verification & Smoke Checks

```powershell
# Run the complete unit and integration test suite (154/154 tests)
.\.venv\Scripts\python.exe -m unittest discover -s tests

# Check model architecture graph and P2-P5 pyramid layers
.\.venv\Scripts\python.exe -B -m scripts.check_tlr_yolo_mtl_model

# Check unified multi-task candidate extraction and cross-attention
.\.venv\Scripts\python.exe -B -m scripts.check_tlr_yolo_mtl_unified --device cuda

# Check training loss convergence and backward pass gradients
.\.venv\Scripts\python.exe -B -m scripts.check_tlr_yolo_mtl_training --device cuda --image-size 320
```

---

## 3. Production Model Training (Champion v4)

```powershell
# Launch full 50-epoch joint single-phase training run
.\.venv\Scripts\python.exe -B scripts/train_tlr_yolo_mtl.py `
  --config configs/tlr_yolo11s_champion_v4.yaml `
  --output-dir runs/tlr_yolo11s_champion_v4 `
  --overwrite
```

Training Specifications:
- **Input Size**: `960x1920` (Native 2:1 aspect ratio)
- **Batch Policy**: Physical micro-batch 4, accumulated over 8 steps = Effective batch 32
- **Precision**: Mixed Precision AMP FP16
- **Checkpoints**: Saved to `runs/tlr_yolo11s_champion_v4/weights/` (`best_composite.pt`, `best_tl_detection.pt`, `best_relevance.pt`, `last.pt`)

---

## 4. Standardized Evaluation & Benchmarking

```powershell
# Run official Unified Evaluation Contract on full DTLD validation split
.\.venv\Scripts\python.exe -B scripts/unified_evaluation_contract.py `
  --weights runs/tlr_yolo11s_champion_v4/weights/best_composite.pt `
  --config configs/tlr_yolo11s_champion_v4.yaml
```

---

## 5. Visual Inference & Inspection

```powershell
# Run visual inference on test images with bounding boxes, states, and relevance links
.\.venv\Scripts\python.exe -B scripts/test_model_on_images.py `
  --weights runs/tlr_yolo11s_champion_v4/weights/best_composite.pt `
  --images datasets/tlr_mtl_dtld_paired/images/val `
  --output-dir results/inspect `
  --conf 0.25 `
  --iou 0.45
```
