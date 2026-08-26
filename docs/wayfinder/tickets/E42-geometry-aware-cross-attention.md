---
title: "E42: Geometry-Aware Cross-Attention with Explicit Relative Spatial Bias"
type: prototype
status: closed
blocked_by:
  - "tickets/E37-evaluation-vs-deployment-operating-points.md"
assignee: "@agent"
---

## Question

Does injecting an explicit relative spatial-geometric inductive bias directly into the TL $\leftrightarrow$ Road Arrow cross-attention matrix:
$$A_{ij} = \frac{\mathbf{q}_i^\top \mathbf{k}_j}{\sqrt{d}} + \text{MLP}(\boldsymbol{\phi}_{ij})$$
where $\boldsymbol{\phi}_{ij} = \left[\frac{\Delta x}{W}, \frac{\Delta y}{H}, \log\left(\frac{w_i}{w_j}\right), \log\left(\frac{h_i}{h_j}\right), y_i, y_j, \frac{x_j - x_{\text{ego}}}{W}, \text{arrow\_dir}, s_i^{\text{TL}}, s_j^{\text{Arr}}\right]$, close the Relevance Precision ($83.7\%$) vs Recall ($87.4\%$) gap by allowing the network to structurally reject visually plausible traffic lights that govern adjacent lanes?

---

## Context & Scientific Motivation

In our multi-task formulation, the vehicle must determine whether a detected traffic light governs its ego-lane by reasoning over road surface arrows and lane geometries. Currently, the cross-attention module relies on learned visual token representations and standard additive positional encodings to implicitly discover geometric relationships.

However, in autonomous driving, road arrows and overhead traffic lights have strong physical and geometric constraints:
- A straight arrow located in lane 1 cannot govern a left-turn traffic light mounted over lane 3.
- Distance, perspective foreshortening, and lateral offsets ($\Delta x, \Delta y$) directly determine physical lane association.

Rooted in the principles of *Relation Networks* (Hu et al., 2018), explicitly injecting geometric priors into the attention weight computation forces the attention mechanism to modulate appearance affinity by physical geometric plausibility. This inductive bias is designed specifically to tackle the primary relevance failure mode: **false positives on adjacent-lane lights**.

---

## Empirical Results: DTLD Validation Split (5,962 images, 25,344 GT TLs)

Evaluated via [scripts/audit_e42_geometry_aware_cross_attention.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/scripts/audit_e42_geometry_aware_cross_attention.py) under the Standardized Evaluation Contract ($\text{conf}_{\text{eval}}=0.001$):

### 1. Multi-Task Relevance & Distractor Discrimination Ablation Matrix

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

### 2. Edge Automotive Latency & Resource Footprint (RTX 5070 Edge GPU)

| Configuration | Module Params | Module FP16 Latency | E2E Model Latency | Single-Stream FPS | Batch-16 Throughput | Latency Overhead | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|
| **Baseline (Champion v1)** | 66,790 | 0.684 ms | 26.76 ms | 37.4 FPS | 144.8 FPS | Baseline | Production standard |
| **Variant A (Pos Embed)** | 66,790 | 0.683 ms | 26.78 ms | 37.3 FPS | 144.6 FPS | +0.02 ms | Marginal lift |
| **Variant B (Geom Bias MLP)** | 67,110 | 1.169 ms | 26.85 ms | 37.2 FPS | 144.1 FPS | +0.09 ms | High spatial selectivity |
| **Variant C (Geom Bias + Gating)** | **67,110** | **1.306 ms** | **26.88 ms** | **37.2 FPS** | **143.8 FPS** | **+0.12 ms** | **ACCEPTED (Champion v2)** |

---

## Acceptance Criteria Verification

- [x] **Criterion 1: $\Delta \text{Relevance Precision} \ge +3.50\%$ (target $\ge 87.0\%$)**: **PASSED** (Achieved **+4.40%**, reaching **88.10%**).
- [x] **Criterion 2: $\text{Relevance Recall} \ge 88.0\%$**: **PASSED** (Achieved **88.80%**).
- [x] **Criterion 3: Significant reduction in adjacent-lane false positives ($\ge 20\%$)**: **PASSED** (Cross-lane FP rate reduced by **49.7%** relatively, from 16.3% down to 8.2%).
- [x] **Criterion 4: Negligible computation overhead ($\Delta t \le 0.30\text{ ms}$, FPS $\ge 36.0$)**: **PASSED** (Overhead is **+0.12 ms** with single-stream **37.2 FPS**).

---

## Architectural Conclusions & Decisions

1. **Resolution of the Relevance Precision/Recall Asymmetry**: Injecting normalized perspective coordinates and scale ratios directly into cross-attention attention weights allows the network to physically rule out traffic signals located across non-contiguous lateral lanes, cutting cross-lane false alarms in half (-49.7%).
2. **Neutral Zero-Initialization Stability**: Zero-initialization of the output linear layer in `GeometryAttentionBiasMLP` ensures that the network starts from an unperturbed neutral attention state, guaranteeing zero destructive interference during transfer and fine-tuning.
3. **Ratification for Champion v2**: Variant C (Geometry-Aware Cross-Attention with Explicit Relative Spatial Bias + Confidence Gating) is formally accepted and ratified into the Champion v2 architecture, unblocking **Ticket E43** and **Ticket E46**.

---

**Status**: Ticket E42 is formally **closed**, unblocking **E43** and **E46**.
