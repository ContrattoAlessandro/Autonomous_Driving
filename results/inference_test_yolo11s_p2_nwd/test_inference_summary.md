# TLR-YOLO-MTL (YOLO11s + P2 + NWD) Test Inference Report

- **Checkpoint Evaluated**: `best_composite.pt`
- **Dataset Split**: `VAL` (5,962 images evaluated)
- **Inference Throughput**: **54.1 FPS** (110.3s total on NVIDIA GeForce RTX 5070)
- **Postprocessing NMS**: Confidence $\ge 0.25$, IoU $\le 0.45$

## 1. Key Metrics Table

| Metric Category | Metric Name | Value | Percentage / Interpretation |
|---|---|:---:|:---:|
| **Composite Score** | `Selection Score` | **0.8040** | Multi-Task Global Harmonic Score |
| **Object Detection** | `mAP@50` | **0.8447** | **84.5%** |
| **Object Detection** | `mAP@50-95` | **0.5763** | **57.6%** (Strict Localization) |
| **Traffic Light AP** | `AP_TL@50` | **0.7509** | **75.1%** |
| **Road Arrow AP** | `AP_Arrow@50` | **0.9385** | **93.9%** |
| **Small Object AP** | `AP_small` | **0.7322** | **73.2%** (P2 Neck + NWD effect) |
| **Relevance** | `AUPRC` | **0.9139** | Directional & Lane Pertinence |
| **Relevance** | `F1-Score` | **0.8546** | Balance between Precision & Recall |
| **Relevance** | `Precision` | **0.8399** | **84.0%** |
| **Relevance** | `Recall` | **0.8698** | **87.0%** |
| **Attributes** | `State Accuracy` | **0.9505** | **95.1%** (Red/Yellow/Green/Off) |
| **Attributes** | `State Macro F1` | **0.8682** | Unweighted Color Class Balance |
| **Attributes** | `Round Signal F1` | **0.8885** | Round vs Directional Identification |
| **Attributes** | `Maneuver Macro F1` | **0.4374** | Arrow/Direction Classification |

---
Report and visual overlays saved to: `results\inference_test_yolo11s_p2_nwd`
