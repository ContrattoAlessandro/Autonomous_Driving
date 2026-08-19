# W9 Diagnostic Audit: Local Relevance Baseline & Safety-Critical Metrics

**Audit Timestamp**: 2026-08-18 19:30:06
**Duration**: 126.58s
**Total GT Traffic Lights Evaluated**: 25,344 (12,523 Relevant, 3,686 Relevant Red)
**Learned Cross-Attention Alpha Gate**: `-0.031526`

## 1. Executive Summary & Diagnostic Findings

- **Local Relevance Head Ceiling ($\\alpha = 0$)**: The local relevance head alone achieves **89.10% AUPRC** (ROC-AUC: **81.68%**, F1: **84.15%**, ECE: **15.49%**).
- **Contextual Lift vs Local Baseline ($\\Delta AUPRC$)**: With learned cross-attention active, relevance AUPRC is **92.31%**, yielding an overall differential of **+3.22% AUPRC**. Because $\\alpha \\approx 0$ (`-0.0315`), the contextual cross-attention branch remains effectively dormant and local cues dominate relevance decisions.
- **3-Tier Hierarchy Evaluation**: Oracle Relevance (Level 1) reaches **87.25% AUPRC**, Level 2 Detection-Conditioned Relevance achieves **89.10% AUPRC**, and Level 3 End-to-End Relevance + Detection ($s_{det} \\cdot P(rel)$) reaches **23.00% $AP_{50}$** with **94.81%** total relevant GT recall.
- **Safety-Critical Metric (Relevant Red TL Recall)**: Relevant Red TL Recall reaches **75.50%** (Miss Rate: **24.50%**). The waterfall analysis shows that **5.6%** of misses are caused by upstream small-scale detection failures, **3.5%** by state classification errors, and only **15.5%** by relevance misclassification.

## 2. Level 2 Detection-Conditioned Relevance across Granular Slices

| Slice Category | Slice Name | Sample Count | Local AUPRC ($\\alpha=0$) | Ctx AUPRC ($\\alpha_{learned}$) | $\\Delta$ AUPRC | Local ECE | Local Brier |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Overall** | Validation Split | 17,603 | **89.10%** | **92.31%** | **+3.22%** | 15.49% | 0.1826 |
| Arrow Context | Arrows Present | 9,235 | 84.95% | 89.45% | **+4.50%** | 12.17% | 0.1941 |
| Arrow Context | No Arrows | 8,368 | 93.00% | 94.64% | **+1.64%** | 19.45% | 0.1700 |
| Signal Type | Round Signal | 13,423 | 93.42% | 94.73% | **+1.31%** | 21.56% | 0.1685 |
| Signal Type | Directional Arrow Signal | 3,464 | 56.95% | 71.67% | **+14.73%** | 13.08% | 0.2485 |
| Scene Density | Single TL Scene | 375 | 95.95% | 97.23% | **+1.28%** | 24.32% | 0.1384 |
| Scene Density | Multi-TL Scene | 17,228 | 88.80% | 92.12% | **+3.32%** | 15.29% | 0.1836 |
| Area Bucket | `<32` px² | 742 | 12.15% | 19.59% | **+7.44%** | 21.94% | 0.1612 |
| Area Bucket | `32-64` px² | 1,448 | 71.86% | 74.54% | **+2.68%** | 13.16% | 0.2060 |
| Area Bucket | `64-128` px² | 3,003 | 84.43% | 87.91% | **+3.48%** | 14.22% | 0.2043 |
| Area Bucket | `128-256` px² | 3,751 | 85.63% | 89.84% | **+4.21%** | 14.06% | 0.1929 |
| Area Bucket | `256-512` px² | 3,550 | 90.24% | 94.19% | **+3.95%** | 16.69% | 0.1815 |
| Area Bucket | `>512` px² | 5,109 | 93.07% | 95.38% | **+2.31%** | 16.72% | 0.1597 |


## 3. 3-Tier Relevance Evaluation Hierarchy

| Tier Level | Evaluation Description | Primary Metric | Recall on Relevant GT | Optimal Threshold |
|---|---|:---:|:---:|:---:|
| **Level 1 (Oracle)** | Features sampled directly at GT locations (Mode B) | **87.25% AUPRC** | 100.0% (Oracle) | 0.45 |
| **Level 2 (Det-Conditioned Local)** | Local head on IoU $\\ge 0.50$ TP detected boxes | **89.10% AUPRC** | 88.51% (on TPs) | 0.45 |
| **Level 2 (Det-Conditioned Ctx)** | Full model on IoU $\\ge 0.50$ TP detected boxes | **92.31% AUPRC** | 88.00% (on TPs) | 0.45 |
| **Level 3 (End-to-End Local)** | Combined score $s_{det} \\cdot P(rel)_{local}$ on all GTs | **23.00% $AP_{50}$** | **94.81%** (overall) | — |
| **Level 3 (End-to-End Ctx)** | Combined score $s_{det} \\cdot P(rel)_{ctx}$ on all GTs | **23.89% $AP_{50}$** | **94.81%** (overall) | — |


## 4. Safety-Critical Relevant Red Light Waterfall & Attribution

- **Total Relevant Red GT Traffic Lights**: 3,686
- **Relevant Red Recall (@ threshold 0.50)**: **75.50%** (Miss Rate: **24.50%**)
- **Relevant Red Recall (@ threshold 0.30)**: **90.29%**
- **Relevant Red Recall (@ threshold 0.70)**: **1.06%**

### Failure Mode Attribution Waterfall:
| Pipeline Stage | Stage Description | Retained / Lost Count | Retention / Loss % | Cumulative Recall |
|:---:|---|:---:|:---:|:---:|
| **GT Total** | Ground-Truth Relevant Red TLs | 3,686 | 100.0% | 100.0% |
| **Stage 1 (Perception)** | Upstream Detector Miss (IoU < 0.50) | -205 | **-5.56%** | 94.44% |
| **Stage 2 (Candidate)** | Top-32 Candidate Selection Eviction | -0 | **-0.00%** | 94.44% |
| **Stage 3 (State)** | State Head Misclassified ($\\hat{s} \\ne \\text{Red}$) | -127 | **-3.45%** | 90.99% |
| **Stage 4 (Relevance)** | Relevance Head False Negative ($P(rel) < 0.5$) | -571 | **-15.49%** | **75.50%** |
| **Success** | Correctly Detected, Red, and Relevant | **2,783** | **75.50%** | **75.50%** |

### Red Light Relevance Confusion Matrix (@ threshold 0.50):

| Ground Truth \ Prediction | Predicted Relevant | Predicted Irrelevant |
|---|:---:|:---:|
| **Actual Relevant Red** | **TP = 2,783** | **FN = 903** |
| **Actual Irrelevant Red** | **FP = 1,465** | **TN = 3,195** |

## 5. Artifacts Generated

- Visualization: `results/visualizations/w9_local_relevance_safety.png`

- Telemetry JSON: `results/audit_local_relevance_safety.json`
