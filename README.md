# TLR-YOLO-MTL: High-Fidelity Multi-Task Perception & Geometry-Aware Traffic Light Relevance for Autonomous Driving (Champion v4)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x%20CUDA-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests: 154/154 Passing](https://img.shields.io/badge/tests-154%2F154%20passing-brightgreen.svg)](tests/)
[![FPS: 36.60 (RTX 5070)](https://img.shields.io/badge/FPS-36.60%20real--time-success.svg)](configs/tlr_yolo11s_champion_v4.yaml)

Official codebase and reference implementation for the Master's Thesis: **"High-Resolution Multi-Task Perception and Geometry-Aware Relational Reasoning for Map-Less Traffic Light Relevance in Autonomous Driving"**.

---

## 1. System Overview

**TLR-YOLO-MTL (Champion v4)** is a unified, camera-only, map-less perception architecture designed to solve the complete traffic signal understanding pipeline for autonomous vehicles:
1. **Distant Sub-8px Object Localization**: Resolves tiny $3\text{--}8\text{ px}$ traffic signals up to $150\text{ meters}$ away without heavy global feature pyramids.
2. **Fine-Grained Attribute Classification**: Simultaneously classifies optical signal state (`[Red, Yellow, Green, Off]`), pictogram category (`round` vs `directional`), and multi-label maneuvers (`[Left, Straight, Right]`).
3. **Contextual Ego-Lane Relevance Reasoning**: Matches traffic lights to the ego-vehicle's active driving corridor by performing Geometry-Aware Cross-Attention with contextual road arrow markings ($M=8$), eliminating cross-lane false alarms without HD maps.
4. **Zero-Overhead Distillation**: Injects high-resolution spatial details and multi-frame temporal stability during training backpropagation while strictly preserving single-frame real-time inference ($36.60\text{ FPS}$ on NVIDIA RTX 5070).

---

## 2. Champion v4 Architecture

```
                                 INPUT IMAGE I_t (960 x 1920)
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
         C2 (Stride 4, Raw Texture)                                 Backbone (YOLO11s C3-C5)
                    │                                                   │
                    │                                                   ▼
                    │                                           P3 Lateral Path (Stride 8)
                    │                                                   │
                    │                                                   ▼
                    │                                        DySample Dynamic Upsampler (E40)
                    │                                                   │
                    └───────────────► Scale-Aware C2 ─► P2 ◄────────────┘
                                     Feature Relay (E51)
                                              │
                                              ▼
                                 P2 High-Resolution (Stride 4)
                                              │
         ┌────────────────────────────────────┼────────────────────────────────────┐
         ▼                                    ▼                                    ▼
Unified Detect (P2-P5)             Task-Gated Fusion (E41)               NWD-Quality Head (E50)
  - Traffic Lights (K=32)            - 5x5 ROIAlign State Head             - s = p^0.7 * q^0.3
  - Road Arrows (K=32)               - Shared Maneuver Head                - Continuous Gaussian NWD
         │                                    │                                    │
         └────────────────────────────────────┴────────────────────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
     Geometry-Aware Cross-Attention (E42)              Sparse Candidate Refinement (E49)
       - 14D Spatial Geometry Bias                       - Top-32 Sub-Grid Regions (<256 px^2)
       - Lane-Level Confidence Gating                    - 7x7 ROIAlign Virtual P1 Stage
       - Ego-Lane Relevance Scoring                      - Sub-Pixel Bounding Box Deltas
                                                         - Residual State Logits
```

---

## 3. Empirical Benchmark Results (DTLD Validation Split)

Evaluated under the standardized **Unified Evaluation Contract** on the full DTLD validation set (5,962 images, 25,344 GT traffic lights, 6,108 GT road arrows):

| Dimension | Metric | Baseline (v0) | Champion v1 (E36) | Champion v3 (E47) | Champion v4 (Final) | Net Lift (v4 vs v0) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Composite Score** | Multi-Task Selection Score | $0.5820$ | $0.6410$ | $0.6720$ | **$0.7018$** | **$+20.6\%$ rel** |
| **Distant Signals (<8px)** | **Sub-8px TL AP@50** | $22.40\%$ | $29.53\%$ | $46.10\%$ | **$55.60\%$** | **$+33.20\%$ ($+148\%$ rel)** |
| | **Sub-4px Recall** | $16.80\%$ | $21.20\%$ | $29.40\%$ | **$41.20\%$** | **$+24.40\%$ ($+145\%$ rel)** |
| | **8–16px TL AP@50** | $58.20\%$ | $65.44\%$ | $78.95\%$ | **$84.30\%$** | **$+26.10\%$** |
| **Global Detection** | **TL AP@50** | $64.10\%$ | $70.31\%$ | $75.48\%$ | **$80.95\%$** | **$+16.85\%$** |
| | **Road Arrow AP@50** | $94.30\%$ | $96.07\%$ | $94.85\%$ | **$94.85\%$** | Stable |
| | **Overall mAP@50** | $79.20\%$ | $83.19\%$ | $85.16\%$ | **$87.90\%$** | **$+8.70\%$** |
| | **Overall mAP@50-95** | $54.10\%$ | $59.12\%$ | $58.82\%$ | **$62.40\%$** | **$+8.30\%$** |
| **Fine-Grained State** | **State Macro-F1** | $79.80\%$ | $84.20\%$ | $91.28\%$ | **$96.10\%$** | **$+16.30\%$** |
| | **Sub-4px State Accuracy** | $48.20\%$ | $62.15\%$ | $72.15\%$ | **$84.80\%$** | **$+36.60\%$** |
| | **Yellow State F1** | $68.40\%$ | $74.80\%$ | $84.79\%$ | **$92.60\%$** | **$+24.20\%$** |
| | **Off State F1** | $63.50\%$ | $70.70\%$ | $86.63\%$ | **$93.90\%$** | **$+30.40\%$** |
| **Ego-Lane Relevance** | **Relevance AUPRC** | $0.8650$ | $0.9111$ | $0.9470$ | **$0.9610$** | **$+0.0960$** |
| | **Relevance Precision** | $78.20\%$ | $83.70\%$ | $91.30\%$ | **$93.80\%$** | **$+15.60\%$** |
| | **Cross-Lane False Positives** | $22.40\%$ | $16.30\%$ | $4.10\%$ | **$2.10\%$** | **$-90.6\%$ rel** |
| **Stability & Safety** | **Sub-Pixel Jitter (RMSE)** | $0.85\text{ px}$ | $0.78\text{ px}$ | $0.76\text{ px}$ | **$0.46\text{ px}$** | **$-45.9\%$ reduction** |
| | **Inter-Frame Flicker Rate** | $21.50\%$ | $18.20\%$ | $14.80\%$ | **$7.90\%$** | **$-63.3\%$ reduction** |
| | **Relevant Red Recall ($\tau_{95}$)**| $78.40\%$ | $95.50\%$ | $96.80\%$ | **$98.80\%$** | **$+20.40\%$ safety floor** |
| **Edge Performance** | **Latency (FP16, RTX 5070)** | **$25.40\text{ ms}$** | **$26.81\text{ ms}$** | **$26.92\text{ ms}$** | **$27.32\text{ ms}$** | $+1.92\text{ ms}$ overhead |
| | **Single-Stream Edge FPS** | **$39.4\text{ FPS}$** | **$37.3\text{ FPS}$** | **$37.15\text{ FPS}$** | **$36.60\text{ FPS}$** | **Automotive Real-Time ($\ge 35\text{ FPS}$)** |

---

## 4. Repository Structure

```
tl_detection/
├── configs/                              # Validated configuration files
│   ├── tlr_yolo11s_champion_v4.yaml      # Official Champion v4 production configuration
│   ├── tlr_yolo11s_champion_final.yaml   # Production alias
│   ├── tlr_yolo11s_champion_v3.yaml      # Phase 5 reference baseline
│   └── model/
│       ├── tlr_yolo11s_p2_relay.yaml     # Champion v4 architecture with C2->P2 Feature Relay
│       ├── tlr_yolo11s_p2_dysample.yaml  # DySample dynamic upsampler architecture
│       └── tlr_yolo11s_p2.yaml           # Standard P2 high-resolution architecture
├── tlr_yolo_mtl/                         # Core Python package
│   ├── model/                            # PyTorch modules (GeometryAttention, DySample, Relay, Heads)
│   ├── training/                         # Training engine, Multi-Task losses, KD modules
│   ├── data/                             # Scale-matched zoom, bloom augmentation, hard mining
│   ├── deployment/                       # Size-Adaptive NWD post-processing, ONNX export
│   └── evaluation/                       # Unified evaluation contract, matching, calibration
├── scripts/                              # Canonical training, evaluation, and audit scripts
│   ├── train_tlr_yolo_mtl.py             # Single-phase production training entry point
│   ├── unified_evaluation_contract.py    # Official evaluation benchmark script
│   ├── test_model_on_images.py           # Demo inference on driving scenes
│   └── audit_e*.py                       # Individual scientific audit scripts (E37 to E52)
├── tests/                                # Full unit and integration test suite (154 tests)
├── docs/                                 # Thesis documentation and scientific tickets
│   └── wayfinder/
│       ├── map.md                        # Research dependency graph & knowledge items
│       └── tickets/                      # Individual ticket reports (E01 to E54)
├── results/                              # Structured benchmark JSON/MD telemetry
│   ├── tlr_yolo11s_champion_v4_summary.json
│   └── tlr_yolo11s_champion_v4_summary.md
├── runs/                                 # Training runs and model checkpoints
│   └── tlr_yolo11s_champion_v4/weights/  # best_composite.pt, best_tl_detection.pt, last.pt
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

# Install PyTorch with CUDA support and dependencies
pip install -r requirements.txt
```

### 5.2 Run Verification Tests
```powershell
# Run the complete unit test suite (154 tests)
python -m unittest discover -s tests

# Verify multi-task architecture and candidate flow
python -m scripts.check_tlr_yolo_mtl_unified
```

### 5.3 Train Champion v4
```powershell
# Launch full multi-task training on GPU (RTX 5070 / 12GB VRAM)
python scripts/train_tlr_yolo_mtl.py `
  --config configs/tlr_yolo11s_champion_v4.yaml `
  --output-dir runs/tlr_yolo11s_champion_v4 `
  --overwrite
```

### 5.4 Evaluate Validation Benchmark
```powershell
# Run standardized evaluation contract
python scripts/unified_evaluation_contract.py `
  --weights runs/tlr_yolo11s_champion_v4/weights/best_composite.pt `
  --config configs/tlr_yolo11s_champion_v4.yaml
```

### 5.5 Visual Inference on Sample Images
```powershell
# Run inference and visualize detection, states, and relevance links
python scripts/test_model_on_images.py `
  --weights runs/tlr_yolo11s_champion_v4/weights/best_composite.pt `
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
