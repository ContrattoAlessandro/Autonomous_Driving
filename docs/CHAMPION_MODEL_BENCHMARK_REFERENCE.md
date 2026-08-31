# TLR-YOLO-MTL Champion Model: Official Reference Benchmark

> **Purpose**: This document stores the definitive reference baseline metrics for the **TLR-YOLO11s-P2-NWD Champion Model** (50 Epochs). Any future architectural modification, ablation, or hyperparameter tuning must be compared against these exact figures.
>
> **Model Checkpoint**: `runs/tlr_yolo11s_champion_final/weights/best_composite.pt`  
> **Configuration File**: [`configs/tlr_yolo11s_champion_final.yaml`](../configs/tlr_yolo11s_champion_final.yaml)  
> **Evaluation Split**: Canonical DTLD Paired Validation/Test Set ($5,962$ images)  
> **Hardware Reference**: NVIDIA GeForce RTX 5070 12GB — Inference Throughput: **$37.3\text{ FPS}$** ($26.8\text{ ms/image}$)

---

## 1. Training Progression History (50 Epochs)

- **Schedule**: 50 Epochs, 5,000 Optimizer Steps ($160,000$ sampled image windows).
- **Batch Setup**: Physical Micro-Batch = `4`, Gradient Accumulation = `8`, Effective Batch = `32`.
- **Optimization**: AdamW ($\text{Backbone LR} = 10^{-4}, \text{Head LR} = 10^{-3}$, Cosine Annealing to $10^{-6}$, Weight Decay = $0.01$).
- **Precision**: AMP FP16 + TensorFloat-32 (TF32) on CUDA Tensor Cores.

### Epoch-by-Epoch Convergence Table

| Epoch | Train Loss | Val Loss | mAP@50 | mAP@50-95 | AP Small (<32px²) | Rel AUPRC | Rel F1 | Rel Red Recall | State Acc | Selection Score | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 38.150 | 25.306 | 0.5590 | 0.3048 | 0.4317 | 0.8059 | 0.8130 | 0.9130 | 0.5900 | 0.5734 | Warm-up |
| **5** | 16.219 | 17.966 | 0.6830 | 0.4137 | 0.4925 | 0.8469 | 0.8320 | 0.9304 | 0.8320 | 0.6359 | Rapid Convergence |
| **10** | 14.440 | 16.480 | 0.6860 | 0.4473 | 0.5466 | 0.8971 | 0.8520 | 0.8696 | 0.8870 | 0.6769 | Checkpoint saved |
| **15** | 13.260 | 15.970 | 0.7370 | 0.4932 | 0.5600 | 0.9177 | 0.8560 | 0.8609 | 0.8700 | 0.6978 | |
| **20** | 12.669 | 15.918 | 0.7510 | 0.5029 | 0.5390 | 0.9336 | 0.8600 | 0.8087 | 0.9050 | 0.7202 | Checkpoint saved |
| **25** | 11.977 | 15.469 | 0.7450 | 0.5134 | 0.5696 | 0.9333 | 0.8520 | 0.8174 | 0.9140 | 0.7201 | |
| **30** | 11.224 | 15.216 | 0.7560 | 0.5148 | 0.5678 | 0.9258 | 0.8480 | 0.7652 | 0.9230 | 0.7245 | Checkpoint saved |
| **35** | 10.680 | 15.118 | 0.7590 | 0.5286 | 0.5751 | 0.9324 | 0.8730 | 0.8174 | 0.9210 | 0.7268 | |
| **39** | **10.284** | **15.082** | **0.7553** | **0.5174** | **0.5699** | **0.9339** | **0.8506** | **0.7913** | **0.9188** | **0.7281** | ⭐ **Best Val Checkpoint** |
| **40** | 10.184 | 15.099 | 0.7570 | 0.5338 | 0.5689 | 0.9366 | 0.8710 | 0.8522 | 0.8940 | 0.7242 | Checkpoint saved |
| **45** | 9.769 | 15.054 | 0.7560 | 0.5275 | 0.5664 | 0.9310 | 0.8600 | 0.8261 | 0.9050 | 0.7192 | |
| **50** | 9.727 | 15.205 | 0.7490 | 0.5244 | 0.5590 | 0.9358 | 0.8700 | 0.8348 | 0.9100 | 0.7202 | Final Epoch Complete |

---

## 2. Definitive Test Evaluation with Ultralytics Post-Processing (NMS)

- **Post-Processing Engine**: Class-Aware Non-Maximum Suppression (Ultralytics / Torchvision NMS).
- **Operating Point**: Confidence Threshold $\tau_{\text{conf}} = 0.25$, IoU Overlap Threshold $\tau_{\text{IoU}} = 0.45$.
- **Evaluated Samples**: All $5,962$ full-resolution images ($960 \times 1920$).

### Official Benchmark Performance Table

| Task Domain | Metric Name | Raw Value | Percentage | Description / Target |
| :--- | :--- | :---: | :---: | :--- |
| **Global Composite** | **`Selection Score`** | **`0.7970`** | — | **Harmonic Multi-Task Target Criterion** |
| **Object Detection** | **`mAP@50 (Global)`** | **`0.8473`** | **`84.7%`** | Overall Detection Accuracy (TL + Arrows) |
| **Object Detection** | **`mAP@50-95 (Global)`** | **`0.6261`** | **`62.6%`** | High-Precision Spatial Localization |
| **Object Detection** | **`AP_TL@50`** | **`0.7471`** | **`74.7%`** | Traffic Light Average Precision |
| **Object Detection** | **`AP_Arrow@50`** | **`0.9476`** | **`94.8%`** | Road Arrow Surface Marking AP |
| **Object Detection** | **`AP_Small`** | **`0.7117`** | **`71.2%`** | Tiny Object Detection ($<32\text{ px}^2$, P2+NWD effect) |
| **Object Detection** | **`AP_Medium`** | **`0.9467`** | **`94.7%`** | Medium Object Detection ($32\text{--}96\text{ px}^2$) |
| **Object Detection** | **`mAP_State`** | **`0.5426`** | **`54.3%`** | Joint Detection + State Correctness |
| **Relevance Reasoning**| **`AUPRC`** | **`0.9111`** | — | Area Under Precision-Recall Curve |
| **Relevance Reasoning**| **`F1-Score`** | **`0.8551`** | — | Harmonic Mean of Precision and Recall |
| **Relevance Reasoning**| **`Precision`** | **`0.8370`** | **`83.7%`** | Ego-Lane Pertinence Precision |
| **Relevance Reasoning**| **`Recall`** | **`0.8739`** | **`87.4%`** | Ego-Lane Pertinence Sensitivity |
| **Attribute Towers** | **`State Accuracy`** | **`0.9414`** | **`94.1%`** | 4-Class Color Accuracy (Red/Yellow/Green/Off) |
| **Attribute Towers** | **`State Macro F1`** | **`0.8392`** | — | Unweighted State Class Balance |
| **Attribute Towers** | **`Round Signal F1`** | **`0.8897`** | — | Circular vs Directional Signal Discrimination |
| **Attribute Towers** | **`Maneuver Macro F1`** | **`0.4346`** | — | Directional Sub-pictogram Classification |

---

## 3. Comparison Guidelines for Future Iterations

When proposing and evaluating any future model update:

1. **Detection Safety Floor**:
   - `mAP@50` must remain $\ge 84.0\%$.
   - `AP_Small` (Tiny TLs) must remain $\ge 70.0\%$ (do not disable the P2 Neck or NWD assigner).
2. **Relevance Floor**:
   - `AUPRC` must remain $\ge 0.9000$.
   - `Recall` on relevant traffic lights must remain $\ge 86.0\%$ to avoid safety hazards in autonomous navigation.
3. **Attribute Floor**:
   - `State Accuracy` must remain $\ge 93.5\%$.
4. **Latency & Memory Constraints**:
   - VRAM usage at inference must not exceed $6\text{ GB}$ (allowing co-existence with autonomous stack nodes).
   - Throughput must remain $\ge 30\text{ FPS}$ on standard target GPUs.

---

## 4. Associated Saved Artifacts

- **Telemetry JSON**: [`results/inference_champion_final/test_inference_report.json`](inference_champion_final/test_inference_report.json)
- **Summary Report**: [`results/inference_champion_final/test_inference_summary.md`](inference_champion_final/test_inference_summary.md)
- **High-Resolution Visual Overlays**: [`results/inference_champion_final/visualizations/`](inference_champion_final/visualizations/) (16 side-by-side ground-truth vs prediction cases).
