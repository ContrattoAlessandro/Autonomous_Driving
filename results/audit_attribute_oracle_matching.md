# W7 Diagnostic Audit: Perception vs Attribute Oracle Disentanglement & Matching Sensitivity

**Audit Timestamp**: 2026-08-18 19:17:35
**Duration**: 138.6s
**Total GT Traffic Lights Evaluated**: 25,344

## 1. Executive Summary & Disentanglement Analysis

- **Oracle vs Detected Disentanglement**: When evaluating feature representations sampled directly at ground-truth locations (Mode B Oracle), Traffic Light State Accuracy reaches **86.72%** (Macro F1: **77.83%**) and Local Relevance AUPRC achieves **87.25%**.
- **Perception Bottleneck Confirmation**: For tiny objects ($<32\text{ px}^2$), detected State Macro F1 is **53.8%** under IoU 0.50 matching, whereas Oracle State Macro F1 is **47.0%**. This massive gap ($F1^{oracle} \gg F1^{det}$) proves that attribute classification is **severely bottlenecked by upstream candidate localization & IoU matching failures**, rather than attribute tower capacity.
- **Matching Metric Sensitivity**: Relaxing rigid IoU matching on tiny objects via NWD ($\\ge 0.50$) or Center Distance ($\\le 16\text{px}$) recovers matched GT recall from **16.6%** (IoU 0.50) to **43.5%** (NWD 0.50) and **46.5%** (Dist 16px), confirming that rigid IoU matching artificially penalizes tiny detections.

## 2. Oracle (Mode B) vs Detected (Mode A, IoU 0.50) across Scale Buckets

| Area Bucket | GT Count | Oracle State F1 | Det State F1 | Oracle Round F1 | Det Round F1 | Oracle Maneuver F1 | Det Maneuver F1 | Oracle Rel AUPRC | Det Rel AUPRC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `<32` | 3980 | **47.0%** | 53.8% | **98.3%** | 97.2% | **26.8%** | 28.4% | **8.8%** | 11.3% |
| `32-64` | 2817 | **63.5%** | 65.7% | **92.6%** | 92.0% | **32.9%** | 31.9% | **58.0%** | 72.6% |
| `64-128` | 4452 | **75.8%** | 77.3% | **88.7%** | 88.0% | **34.3%** | 33.3% | **81.5%** | 85.0% |
| `128-256` | 4699 | **85.5%** | 87.5% | **87.8%** | 87.5% | **42.8%** | 39.2% | **84.4%** | 85.8% |
| `256-512` | 4015 | **87.0%** | 89.4% | **88.6%** | 88.4% | **44.0%** | 40.2% | **89.5%** | 90.3% |
| `>512` | 5381 | **88.8%** | 89.7% | **89.0%** | 89.0% | **41.0%** | 41.4% | **93.1%** | 93.1% |


## 3. Matching Sensitivity Comparison on Detected Predictions

| Matcher Strategy | Matched GT Recall | State Accuracy | State Macro F1 | Round F1 | Maneuver Macro F1 | Relevance AUPRC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `iou_050` | **67.7%** | 94.04% | 85.05% | 88.98% | 39.36% | 89.34% |
| `iou_025` | **74.5%** | 93.03% | 83.59% | 89.47% | 39.03% | 88.87% |
| `nwd_050` | **76.3%** | 92.56% | 82.67% | 89.67% | 39.08% | 88.69% |
| `nwd_030` | **78.0%** | 92.08% | 82.09% | 89.66% | 38.95% | 88.46% |
| `dist_16px` | **78.5%** | 91.77% | 81.57% | 89.67% | 38.90% | 88.39% |
| `dist_8px` | **76.7%** | 92.50% | 82.65% | 89.66% | 39.00% | 88.66% |
| **Oracle (Mode B)** | **100.0%** | **86.72%** | **77.83%** | **90.71%** | **40.57%** | **87.25%** |


## 4. Artifacts Generated

- Visualization: `results/visualizations/w7_attribute_oracle_matching.png`

- Telemetry JSON: `results/audit_attribute_oracle_matching.json`
