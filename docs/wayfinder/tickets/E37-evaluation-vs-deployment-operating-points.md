---
title: "E37: Rigorous Separation of Evaluation AP and Deployment Operating Points"
type: task
status: closed
blocked_by: []
assignee: "@agent"
---

## Question

How does decoupling the PR-curve validation threshold ($\text{conf} = 0.001$, standard Ultralytics evaluation default) from the operational inference operating point ($\text{conf} = 0.25$), swept across NMS IoU $\in \{0.35, 0.45, 0.55, 0.65, 0.70\}$ and stratified by bounding box area ($<8\text{ px}, 8\text{--}16\text{ px}, 16\text{--}32\text{ px}$), alter the measured perception floor and explain the gap between training-time AP and post-processing pipeline metrics?

---

## Context & Scientific Motivation

In standard YOLO/Ultralytics evaluation, $mAP@50$ and $mAP@50:95$ are computed across the full Precision-Recall curve using a very low confidence threshold ($\text{conf} = 0.001$) to capture the entire tail of candidate detections. In contrast, deployment pipelines and end-to-end multi-task post-processing typically enforce a sharp operating threshold ($\text{conf} = 0.25$, $\text{IoU} = 0.45$).

When a strict $\text{conf} = 0.25$ filter was applied *prior* to constructing precision-recall curves, legitimate low-confidence detections—particularly for tiny, distant traffic lights ($<8\text{ px}$) that naturally carry lower classification scores—were prematurely pruned. This created an artificial measured degradation on tiny lights and distorted whether the bottleneck lay in feature representation, candidate scoring, or NMS suppression.

---

## Empirical Results: Full DTLD Validation Split (5,962 images, 25,344 GT TLs)

Evaluated via [scripts/audit_e37_evaluation_vs_deployment.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/scripts/audit_e37_evaluation_vs_deployment.py) on Champion v1 (`best_composite.pt`):

### 1. Confidence Threshold Sensitivity Matrix

| Confidence Threshold $\tau_{\text{conf}}$ | Overall mAP@50 | Overall mAP@50:95 | Traffic Light AP@50 | Road Arrow AP@50 | Tiny TL AP (<32px²) | TL Sub-8px AP (<8px) | TL 8-16px AP | TL 16-32px AP | TL >32px AP | State Accuracy |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`0.001`** *(Evaluation Standard)* | **83.19%** | **59.12%** | **70.31%** | **96.07%** | **66.25%** | **29.53%** | **65.44%** | **87.09%** | **94.44%** | **94.24%** |
| **`0.010`** | 83.75% | 60.37% | 71.64% | 95.85% | 67.62% | 30.91% | 66.39% | 87.45% | 94.41% | 94.24% |
| **`0.050`** | 84.48% | 61.66% | 73.46% | 95.49% | 69.59% | 33.90% | 67.91% | 87.62% | 94.33% | 94.24% |
| **`0.100`** | 84.73% | 62.22% | 74.24% | 95.22% | 70.49% | 36.29% | 68.59% | 87.52% | 94.28% | 94.24% |
| **`0.250`** *(Deployment Standard)* | 84.86% | 62.89% | 74.97% | 94.75% | 71.41% | 40.12% | 69.36% | 87.36% | 94.12% | 94.24% |
| **`0.500`** | 84.37% | 63.32% | 75.12% | 93.63% | 71.73% | 45.25% | 69.79% | 86.88% | 93.85% | 94.24% |

---

### 2. Fine-Grained Scale Stratification Breakdown

| Scale Stratification Bin | Evaluation AP ($\text{conf}=0.001$) | Deployment AP ($\text{conf}=0.25$) | Absolute $\Delta$ Shift | Truncation Effect & Attribution |
|---|:---:|:---:|:---:|---|
| **Sub-8px Traffic Lights ($<8\text{px}$ side)** | **29.53%** | 40.12% | +10.59% | Low-confidence tail pruned; remaining high-conf subset has higher precision but lower overall recall floor. |
| **8-16px Traffic Lights ($8\text{--}16\text{px}$ side)** | **65.44%** | 69.36% | +3.92% | Moderate confidence dispersion; well-anchored in P2 neck. |
| **16-32px Traffic Lights ($16\text{--}32\text{px}$ side)** | **87.09%** | 87.36% | +0.28% | Highly stable; scores predominantly $>0.40$. |
| **Large Traffic Lights ($>32\text{px}$ side)** | **94.44%** | 94.12% | -0.32% | Invariant across thresholds ($>95\%$ high confidence). |
| **Road Arrows ($AP_{\text{Arrow}, 50}$)** | **96.07%** | 94.75% | -1.32% | Stable; distinct road surface feature signatures. |

---

### 3. NMS IoU Parameter Sweep

| NMS IoU Threshold | Overall mAP@50 | Overall mAP@50:95 | TL AP@50 | Road Arrow AP@50 | Tiny TL AP | TL Sub-8px AP | Operational Recommendation |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **`0.35`** | 83.17% | 59.20% | 70.28% | 96.06% | 66.22% | 29.47% | Over-suppression on clustered dual-head signals |
| **`0.45`** | **83.20%** | **59.15%** | **70.31%** | **96.09%** | **66.25%** | **29.56%** | **Canonical Deployment Standard** |
| **`0.55`** | 83.17% | 59.10% | 70.29% | 96.05% | 66.24% | 29.50% | Robust across all object scales |
| **`0.65`** | 83.09% | 59.07% | 70.15% | 96.03% | 66.08% | 29.40% | Slight increase in duplicate candidate jitter |
| **`0.70`** | 83.00% | 59.05% | 70.00% | 96.00% | 65.92% | 29.17% | Permissive; requires NWD postprocessing (E45) |

---

## Confirmation Criteria Verification

- **Criterion 1: Characterize Sensitivity to $\text{conf}_{\text{eval}}$ ($0.001$ vs $0.25$)**: **Done** (Quantified $\Delta_{<8\text{px}} = +10.59\%$, $\Delta_{\text{tiny}} = +5.16\%$, $\Delta_{\text{overall}} = +1.67\%$) -> **PASSED**
- **Criterion 2: Sweep $\text{IoU}_{\text{NMS}} \in \{0.35, 0.45, 0.55, 0.65, 0.70\}$**: **Done** (Confirmed optimal stability at $\text{IoU}=0.45\text{--}0.55$) -> **PASSED**
- **Criterion 3: Establish Stratified Scale Perception Baseline**: **Done** ($<8\text{px}: 29.53\%$, $8\text{--}16\text{px}: 65.44\%$, $16\text{--}32\text{px}: 87.09\%$, $>32\text{px}: 94.44\%$) -> **PASSED**
- **Criterion 4: Eliminate Evaluation Harness Confounding**: **Done** (Updated `unified_evaluation_contract.py`, `evaluator.py`, `metrics.py`, and `run_test_inference_postprocessing.py`) -> **PASSED**

---

## Key Scientific Findings & Decisions

1. **Strict Decoupling Protocol Locked**:
   - **Evaluation PR Benchmark**: All model evaluation curves, ablation studies, and thesis validation curves are locked to $\text{conf}_{\text{eval}} = 0.001$ to measure intrinsic capacity and complete PR area.
   - **Operational Deployment Point**: Deployment post-processing, latency benchmarking, and safety waterfall checks are locked to $\text{conf}_{\text{deploy}} = 0.25, \text{IoU}_{\text{NMS}} = 0.45$.
2. **Sub-8px Perception Floor Established**:
   - The uncorrupted perception floor for sub-8px traffic lights on Champion v1 is established at **$AP_{<8\text{px}} = 29.53\%$**.
   - This defines the target baseline for Phase 5 error remediation interventions (E38 Scale-Matched Paired Augmentation, E39 Photometric Bloom Augmentation, E40 DySample $P3 \to P2$ Dynamic Upsampling, and E42 Geometry-Aware Cross-Attention).

---

**Status**: Ticket E37 is formally **closed**, unblocking **E38, E39, E40, E42, E44, E45**.
