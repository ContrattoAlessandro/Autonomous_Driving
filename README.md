# TLR-YOLO-MTL: High-Resolution Multi-Task Perception & Geometry-Aware Traffic Light Relevance for Autonomous Driving (Champion v3)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x%20CUDA-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 102/102 Passing](https://img.shields.io/badge/tests-102%2F102%20passing-brightgreen.svg)](tests/)
[![FPS: 36.60 (RTX 5070)](https://img.shields.io/badge/FPS-36.60%20real--time-success.svg)](configs/tlr_yolo11s_champion_v3.yaml)

Official codebase and reference implementation for the Master's Thesis: **"High-Resolution Multi-Task Perception and Geometry-Aware Relational Reasoning for Map-Less Traffic Light Relevance in Autonomous Driving"**.

---

## 1. System Overview

**TLR-YOLO-MTL (Champion v3)** is a unified, camera-only, map-less perception architecture designed to solve the complete traffic signal understanding pipeline for autonomous vehicles:
1. **Distant Sub-8px Object Localization**: Resolves tiny $3\text{--}8\text{ px}$ traffic signals up to $150\text{ meters}$ away via **DySample Dynamic Upsampling ($P3 \to P2$)** and **Size-Adaptive Gaussian NWD Post-Processing** without heavy dense feature pyramids.
2. **Fine-Grained Attribute Classification**: Simultaneously classifies optical signal state (`[Red, Yellow, Green, Off]`), pictogram category (`round` vs `directional`), and multi-label maneuvers (`[Left, Straight, Right]`) via **Task-Gated Fusion** and expanded $5\times5$ ROIAlign.
3. **Contextual Ego-Lane Relevance Reasoning**: Matches traffic lights to the ego-vehicle's active driving corridor by performing **Geometry-Aware Cross-Attention with 14D Relative Spatial Embeddings** on contextual road arrow markings ($M=8$), eliminating cross-lane false alarms without HD maps.
4. **Class-Balanced Optimization**: Eliminates gradient fighting across diverse multi-task heads through static loss weighting and Class-Balanced Focal Softmax ($\beta=0.9999$).
5. **Real-Time Edge Throughput**: Executes the entire multi-task pipeline in **$27.35\text{ ms}$ ($36.60\text{ FPS}$)** on NVIDIA GeForce RTX 5070 GPU (FP16 Tensor Cores).

---

## 2. Champion v3 Architecture

```
                                 INPUT IMAGE I_t (960 x 1920)
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
         Backbone (YOLO11s C2-C5)                               Data Augmentation Pipeline
                    │                                            - E38: Scale-Matched Zoom (Sub-8px)
                    ▼                                            - E38: Paired Copy-Paste (0.30)
         P3 Lateral Path (Stride 8)                              - E39: Photometric Lamp Bloom (|Δh|≤0.004)
                    │                                            - E32: Tri-Tier Hard Negative Mining
                    ▼
         DySample Dynamic Upsampler (E40)
         - 4-Group Content-Aware Sampling
         - Reconnects P3 -> P2
                    │
                    ▼
         P2 High-Resolution Neck (Stride 4)
                    │
         ┌──────────┴─────────────────────────┬─────────────────────────┐
         ▼                                    ▼                         ▼
Joint Detection (P2-P5)              Task-Gated Fusion (E41)   Geometry-Aware Cross-Attention (E42)
  - Traffic Lights (K=32)             - 5x5 ROIAlign (State)     - Query Road Arrows M=8 (E33)
  - Road Arrows (K=32)                - 3x3 ROIAlign (Round/Man) - 14D Relative Spatial Bias
  - Scale-Adaptive NWD TAL (E30)      - Task Gating α_t          - Confidence Gating Anti-Noise
         │                                    │                         │
         ▼                                    ▼                         ▼
Size-Adaptive NWD NMS (E45)          Class-Balanced Focal (E44)Counterfactual Relevance Gate (E43)
  - NWD for boxes < 64 px²            - Smoothing β = 0.9999     - 40/30/15/15 Quota (Cross-Lane)
  - IoU = 0.45 standard               - 4-Class Balanced Pews    - Calibrated Temperature T* = 0.72
```

---

## 3. Master Champions Benchmark Matrix (DTLD Validation Split)

Evaluated under the standardized **Unified Evaluation Contract** on the full DTLD validation set ($5,962$ images, $25,344$ GT traffic lights, $6,108$ GT road arrows):

| Generation | Architecture Description | Composite Score | mAP@50 (Global) | Sub-8px AP (<8px) | Relevance AUPRC | Relevant Red Recall | State Macro-F1 | Latency FP16 (Batch=1) | Edge FPS (RTX 5070) | Status / Verdict |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 **Champion v3** *(Fase 5)* | **DySample + Gated 5x5 ROI + 14D Geom-Attn + CB-Loss + Adapt-NWD** | **`0.8320`** | **`86.80%`** | **`38.60%`** | **`94.80%`** | **`97.50%`** | **`87.20%`** | **`27.35 ms`** | **`36.6 FPS`** | 🏆 **OFFICIAL CHAMPION** |
| 🥈 **Champion v2** *(Fase 5)* | DySample + Task-Gated Fusion + 14D Geometry Attention | `0.8150` | `85.45%` | `34.20%` | `93.40%` | `96.80%` | `85.60%` | `27.10 ms` | `36.9 FPS` | Architectural Baseline |
| 🥉 **Champion v1** *(E36 SFS)* | Resolution 960x1920, Neck P2 high-res, NWD Assigner | `0.7970` | `83.19%` | `29.53%` | `91.11%` | `95.50%` | `84.20%` | `26.81 ms` | `37.3 FPS` | SFS Baseline |
| 4° **Champion v5** *(Fase 8)* | Relay v2, Continuous DFL Refine, Scale Quality, Geom-Attn v2 | `0.7427` | `78.63%` | `21.58%` | `92.34%` | `72.82%` | `66.22%` | `43.38 ms` | `23.1 FPS` | Heavy Refinement Overhead |
| 5° **Champion v4** *(Fase 6)* | C2 $\to$ P2 Feature Relay, Crop Distillation, Sparse Refinement | `0.7382` | `80.53%` | `10.33%` | `90.87%` | `73.22%` | `61.00%` | `23.16 ms` | `43.2 FPS` | Multi-Task Trade-off |
| 6° **Champion v0** *(M2)* | YOLO Baseline, Neck FPN standard, Separate Heads | `0.7012` | `79.20%` | `22.40%` | `86.50%` | `93.20%` | `79.80%` | `25.40 ms` | `39.4 FPS` | Initial Milestone |

---

## 4. Repository Structure

```
tl_detection/
├── configs/                              # Validated configuration files
│   ├── tlr_yolo11s_champion_v3.yaml      # Official Champion v3 production configuration
│   ├── tlr_yolo_mtl_train.yaml           # Joint training baseline configuration
│   ├── tlr_yolo_mtl_data.yaml            # Multi-task dataset configuration
│   └── model/
│       ├── tlr_yolo11s_p2_dysample.yaml  # DySample dynamic upsampler architecture (Champion v3)
│       ├── tlr_yolo11s_p2.yaml           # Standard P2 high-resolution architecture
│       └── tlr_yolo11n_p2.yaml           # Lightweight P2 verification architecture
├── tlr_yolo_mtl/                         # Core Python package
│   ├── model/                            # PyTorch modules (GeometryAttention, DySample, Heads)
│   ├── training/                         # Training engine, Multi-Task losses, Class-Balanced Loss
│   ├── data/                             # Scale-matched zoom, bloom augmentation, hard mining
│   ├── deployment/                       # Size-Adaptive NWD post-processing, ONNX export
│   └── evaluation/                       # Unified evaluation contract, matching, calibration
├── scripts/                              # Canonical training, evaluation, and verification scripts
│   ├── train_tlr_yolo_mtl.py             # Single-phase production training entry point
│   ├── unified_evaluation_contract.py    # Official evaluation benchmark script
│   ├── test_model_on_images.py           # Demo visual inference on driving scenes
│   ├── prepare_dtld_images.py            # DTLD image preparation utility
│   ├── convert_dtld.py                   # DTLD dataset conversion utility
│   └── check_tlr_yolo_mtl_*.py           # Multi-task architecture and candidate flow checks
├── tests/                                # Full unit and integration test suite (102 tests)
├── docs/                                 # Thesis documentation and scientific tickets
│   ├── LLM_PROJECT_CONTEXT.md            # Comprehensive architectural reference
│   └── wayfinder/
│       ├── map.md                        # Research dependency graph & knowledge items
│       └── tickets/                      # Individual ticket reports (E01 to E74)
├── results/                              # Official decreed benchmark telemetry and figures
│   ├── OFFICIAL_CHAMPION_MODEL_DECREE.md # Official Champion v3 proclamation document
│   ├── CHAMPION_MODEL_BENCHMARK_REFERENCE.md
│   ├── audit_e47_champion_v3_lineage.json
│   ├── audit_e47_champion_v3_lineage.png
│   └── champions_benchmark_comparison/   # Comparative charts and master report
├── runs/                                 # Training runs and model checkpoints
├── COMMANDS.md                           # Canonical CLI command reference
└── requirements.txt                      # Environment dependencies
```

---

## 5. Quickstart & Usage

### 5.1 Environment Setup
```powershell
# Create and activate virtual environment (Python 3.12)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Run Verification Tests
```powershell
# Run the complete unit test suite (102 tests)
python -m unittest discover -s tests

# Verify multi-task architecture and candidate flow
python -m scripts.check_tlr_yolo_mtl_unified --device cuda
```

### 5.3 Train Champion v3
```powershell
# Launch full multi-task training on GPU (RTX 5070 / 12GB VRAM)
python scripts/train_tlr_yolo_mtl.py `
  --config configs/tlr_yolo11s_champion_v3.yaml `
  --output-dir runs/tlr_yolo11s_champion_v3 `
  --overwrite
```

### 5.4 Evaluate Validation Benchmark
```powershell
# Run standardized evaluation contract
python scripts/unified_evaluation_contract.py `
  --weights runs/tlr_yolo11s_champion_v3/weights/best_composite.pt `
  --config configs/tlr_yolo11s_champion_v3.yaml
```

### 5.5 Visual Inference on Sample Images
```powershell
# Run visual inference with bounding boxes, states, and relevance links
python scripts/test_model_on_images.py `
  --weights runs/tlr_yolo11s_champion_v3/weights/best_composite.pt `
  --images datasets/tlr_mtl_dtld_paired/images/val `
  --output-dir results/inspect
```

---

## 6. Citation

If you use this repository or its methodology in your research, please cite:
```bibtex
@mastersthesis{contratto2026tlryolo,
  author       = {Alessandro Contratto},
  title        = {High-Resolution Multi-Task Perception and Geometry-Aware Relational Reasoning for Map-Less Traffic Light Relevance in Autonomous Driving},
  school       = {Politecnico di Torino},
  year         = {2026},
  type         = {Master's Thesis}
}
```
