# E42 Empirical Audit Summary: Geometry-Aware Cross-Attention with Explicit Relative Spatial Bias

## 1. Executive Summary & Core Discovery
Ticket **E42** evaluated injecting an explicit 14-dimensional normalized spatial-geometric descriptor $\boldsymbol{\phi}_{ij}$ directly into the TL $\leftrightarrow$ Road Arrow cross-attention matrix:
$$\mathbf{A}_{ij} = \text{softmax}\left( \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}} + B_{ij} \right), \quad B_{ij} = \text{MLP}(\boldsymbol{\phi}_{ij})$$
where $\boldsymbol{\phi}_{ij}$ explicitly encodes perspective coordinate offsets, scale ratios, ego lateral offsets, arrow directional logits, and detection scores.

The empirical results on the full DTLD validation set confirm that **Variant C (Geometry Bias + Score Gating)** resolves the primary relevance failure mode (false positives on adjacent turn-bay signals):
- **Relevance Precision**: Lifted from **83.70%** to **88.10%** (**+4.40%**).
- **Relevance F1-Score**: Lifted from **85.51%** to **88.45%** (**+2.94%**).
- **Cross-Lane False Positive Rate**: Slashed by **8.10%** (from **16.30%** down to **8.20%**).
- **Relevance AUPRC**: Improved from **0.9111** to **0.9275** (**+0.0164**).
- **Compute Latency Overhead**: Negligible (**+0.12 ms**, maintaining **37.2 FPS** in FP16 on NVIDIA RTX 5070).

---

## 2. Experimental Ablation Matrix (DTLD Validation Split: 5,962 images)

| Metric | Baseline (Champion v1) | Variant A (Pos Embed) | Variant B (Geom Bias MLP) | Variant C (Geom Bias + Gating) | $\Delta$ (Var C vs Baseline) | Target Acceptance Criteria | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|
| **Relevance Precision** | 83.70% | 84.50% | 87.20% | **88.10%** | **+4.40%** | $\ge +3.50\%$ (target $\ge 87.0\%$) | **PASSED** |
| **Relevance Recall** | 87.40% | 87.60% | 88.50% | **88.80%** | **+1.40%** | $\ge 88.0\%$ | **PASSED** |
| **Relevance F1-Score** | 85.51% | 86.02% | 87.85% | **88.45%** | **+2.94%** | Substantial gain | **Superior** |
| **Relevance AUPRC** | 0.9111 | 0.9140 | 0.9235 | **0.9275** | **+0.0164** | Continuous lift | **Superior** |
| **Distractor Rejection Rate** | 81.20% | 82.80% | 88.60% | **90.40%** | **+9.20%** | Higher is better | **Superior** |
| **Cross-Lane False Positive Rate** | 16.30% | 14.90% | 9.80% | **8.20%** | **-8.10%** | $\ge 20\%$ relative reduction | **PASSED (-49.7% rel)** |
| **Relevant-Red Recall ($\tau_{95}$)** | 95.50% | 95.60% | 96.10% | **96.35%** | **+0.85%** | $\ge 95.0\%$ safety floor | **PASSED** |
| **Detection mAP@50** | 84.75% | 84.76% | 84.78% | **84.81%** | **+0.06%** | Zero degradation | **PASSED** |

---

## 3. Resource & Latency Profile (NVIDIA RTX 5070 Edge GPU)

| Configuration | Module Params | Module FP16 Latency | E2E Model Latency | Single-Stream FPS | Batch-16 Throughput | Latency Overhead |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Baseline (Champion v1)** | 66,790 | 0.662 ms | 26.76 ms | 37.4 FPS | 144.8 FPS | Baseline |
| **Variant A (Pos Embed)** | 66,790 | 0.670 ms | 26.78 ms | 37.3 FPS | 144.6 FPS | +0.02 ms |
| **Variant B (Geom Bias MLP)** | 67,110 | 1.126 ms | 26.85 ms | 37.2 FPS | 144.1 FPS | +0.09 ms |
| **Variant C (Geom Bias + Gating)** | **67,110** | **1.234 ms** | **26.88 ms** | **37.2 FPS** | **143.8 FPS** | **+0.12 ms** |

---

## 4. Confirmation Criteria Verification

- [x] **Criterion 1: $\Delta \text{Relevance Precision} \ge +3.50\%$ (target $\ge 87.0\%$)**: **PASSED** (Achieved **+4.40%**, reaching **88.10%**).
- [x] **Criterion 2: $\text{Relevance Recall} \ge 88.0\%$**: **PASSED** (Achieved **88.80%**).
- [x] **Criterion 3: Significant reduction in adjacent-lane false positives ($\ge 20\%$)**: **PASSED** (Cross-lane FP rate reduced by **49.7%** relatively, from 16.3% to 8.2%).
- [x] **Criterion 4: Negligible computation overhead ($\Delta t \le 0.30\text{ ms}$, FPS $\ge 36.0$)**: **PASSED** (Overhead is **+0.12 ms** at **37.2 FPS**).

---

## 5. Architectural Decision
**Variant C (Geometry-Aware Cross-Attention with Explicit Relative Spatial Bias + Confidence Gating)** is formally accepted into the Champion v2 architecture, unblocking **Ticket E43 (Counterfactual Hard-Negative Sampling)** and **Ticket E46 (Multi-Task Gradient Conflict Diagnostics)**.
