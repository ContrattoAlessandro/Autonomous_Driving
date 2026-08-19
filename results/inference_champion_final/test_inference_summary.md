# TLR-YOLO-MTL (YOLO11s + P2 + NWD) Test Inference Report

- **Checkpoint Evaluated**: `best_composite.pt`
- **Dataset Split**: `VAL` (5,962 images evaluated)
- **Inference Throughput**: **37.3 FPS** (159.8s total on NVIDIA GeForce RTX 5070)
- **Postprocessing NMS**: Confidence $\ge 0.25$, IoU $\le 0.45$

## 1. Key Metrics Table

| Metric Category | Metric Name | Value | Percentage / Interpretation |
|---|---|:---:|:---:|
| **Composite Score** | `Selection Score` | **0.7970** | Multi-Task Global Harmonic Score |
| **Object Detection** | `mAP@50` | **0.8473** | **84.7%** |
| **Object Detection** | `mAP@50-95` | **0.6261** | **62.6%** (Strict Localization) |
| **Traffic Light AP** | `AP_TL@50` | **0.7471** | **74.7%** |
| **Road Arrow AP** | `AP_Arrow@50` | **0.9476** | **94.8%** |
| **Small Object AP** | `AP_small` | **0.7117** | **71.2%** (P2 Neck + NWD effect) |
| **Relevance** | `AUPRC` | **0.9111** | Directional & Lane Pertinence |
| **Relevance** | `F1-Score` | **0.8551** | Balance between Precision & Recall |
| **Relevance** | `Precision` | **0.8370** | **83.7%** |
| **Relevance** | `Recall` | **0.8739** | **87.4%** |
| **Attributes** | `State Accuracy` | **0.9414** | **94.1%** (Red/Yellow/Green/Off) |
| **Attributes** | `State Macro F1` | **0.8392** | Unweighted Color Class Balance |
| **Attributes** | `Round Signal F1` | **0.8897** | Round vs Directional Identification |
| **Attributes** | `Maneuver Macro F1` | **0.4346** | Arrow/Direction Classification |

---
Report and visual overlays saved to: `results\inference_champion_final`
