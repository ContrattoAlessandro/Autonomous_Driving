---
title: "W5: Tiny TL Detection Ceiling & P3 Stride-8 Limit Analysis"
type: research
status: closed
blocked_by: ["W1", "W2"]
assignee: "@agent"
---

## Question

Is the perception bottleneck for small traffic lights caused by feature stride 8 (P3) resolution limitations, and what is the granular detection recall/AP profile across fine-grained scale buckets?

## Context & Requirements

1. **Granular Size Breakdown (Beyond Standard COCO Small)**:
   - Calculate Precision, Recall, $AP_{50}$, $AP_{50:95}$, center localization error ($\Delta x, \Delta y$), and bounding box scale error ($\Delta w, \Delta h$) across:
     - Area buckets: $<32, 32\text{--}64, 64\text{--}128, 128\text{--}256, 256\text{--}512, >512\text{ px}^2$.
     - Side buckets: $\min(w,h) < 4, 4\text{--}6, 6\text{--}8, 8\text{--}12, >12\text{ px}$.

2. **Stride 8 (P3) Evaluation**:
   - Compare object bounding box dimensions to stride 8 grid cell coverage ($8 \times 8\text{ px} = 64\text{ px}^2$).
   - Evaluate $Recall_{TL}(size)$ curve:
     - If recall drops sharply for objects $< 64\text{ px}^2$ while remaining high for larger objects, document this as empirical justification for a P2 (stride-4) high-resolution neck ablation.
     - If recall remains consistent across buckets, P3 is confirmed sufficient.

## Empirical Resolution & Diagnostic Summary

- **Evaluated Checkpoint**: Baseline B0 (`runs/tlr_yolo_mtl_single_phase_seed42/weights/best.pt`) on 5,962 validation images (25,344 GT Traffic Lights).
- **Fine-Grained Metric Implementation**:
  - `compute_granular_scale_metrics` added to `tlr_yolo_mtl/evaluation/metrics.py`.
  - Integrated into validation pipeline `tlr_yolo_mtl/evaluation/evaluator.py`.
  - 100% unit test coverage verified in `tests/test_evaluation.py`.

### Key Empirical Findings:

| Scale Metric | $<32\text{ px}^2$ | $32\text{--}64\text{ px}^2$ | $64\text{--}128\text{ px}^2$ | $128\text{--}256\text{ px}^2$ | $256\text{--}512\text{ px}^2$ | $>512\text{ px}^2$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Recall @ 50** | **16.61%** | **45.90%** | **64.31%** | **79.10%** | **88.09%** | **94.35%** |
| **$AP_{50}$** | **18.60%** | **35.05%** | **58.71%** | **78.06%** | **87.95%** | **93.70%** |
| **$AP_{50:95}$** | **5.39%** | **11.84%** | **23.02%** | **37.46%** | **50.14%** | **60.82%** |
| **P3 Cell Ratio ($64\text{ px}^2$)** | **0.26x** | **0.74x** | **1.46x** | **2.88x** | **5.69x** | **21.01x** |

| Min Side Metric | $\min(w,h) < 4\text{ px}$ | $4\text{--}6\text{ px}$ | $6\text{--}8\text{ px}$ | $8\text{--}12\text{ px}$ | $>12\text{ px}$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **Recall @ 50** | **30.53%** | **65.94%** | **77.26%** | **85.26%** | **92.99%** |
| **$AP_{50}$** | **34.85%** | **55.37%** | **76.45%** | **85.07%** | **92.67%** |
| **P3 Stride Ratio ($8\text{ px}$)** | **0.34x** | **0.63x** | **0.87x** | **1.24x** | **2.46x** |

### Architectural Conclusion:
1. **P3 Resolution Limit Confirmed**: Objects smaller than a single P3 grid cell ($<64\text{ px}^2$, 26.8% of dataset) experience a catastrophic recall drop ($\mathbf{16.6\% \text{ to } 45.9\%}$ vs $\mathbf{94.4\%}$ for $>512\text{ px}^2$).
2. **Definitive Justification for P2 Neck**: This empirical evidence formally justifies introducing a high-resolution **P2 (stride-4, $200 \times 400$)** feature neck level to recover sub-grid spatial features for distant traffic signal detection.

### Diagnostic Artifacts Produced:
- **Audit Script**: `scripts/audit_tiny_tl_stride_limit.py`
- **Tabular Report**: `results/audit_tiny_tl_stride_limit.md`
- **JSON Telemetry**: `results/audit_tiny_tl_stride_limit.json`
- **Visualization Plot**: `results/visualizations/w5_tiny_tl_stride_limit.png`

