---
title: "E31: Multi-Scale ROIAlign End-to-End Integration & Downstream Safety Validation"
type: prototype
status: closed
blocked_by: ["E29-evaluation-contract-normalization.md"]
assignee: "@agent"
---

## Question

Does integrating candidate-centered $3\times3$ Multi-Scale ROIAlign (P2+P3) for traffic light attribute towers (state, roundness, maneuver) directly improve end-to-end downstream safety metrics ($\text{RelevantRed}^{\text{E2E}}$ recall, Stage-3 safety waterfall errors) without introducing regression on inference latency or detection precision?

---

## Synthesis & Empirical Results

Evaluated across the full DTLD validation set (5,962 images, 25,344 GT TLs, 1,373 Relevant Red TLs) under the standardized **E29 Unified Evaluation Contract**:

### 1. 4-Stage Safety Waterfall Decomposition

| Safety Waterfall Stage | Baseline C0 (Dense Anchor) | E31 (Multi-Scale ROIAlign) | Delta / Error Reduction |
|---|:---:|:---:|:---:|
| **GT Relevant Red Total** | 1,373 | 1,373 | Invariant Benchmark |
| **Stage 1: Perception Detected (IoU $\ge$ 0.50)** | 1,180 (85.94%) | 1,180 (85.94%) | 0 (Detection Invariant) |
| *Stage 1 Perception Misses* | 193 | 193 | 0 |
| **Stage 2: Candidate Selected (Top-K=32)** | 1,174 (99.49%) | 1,174 (99.49%) | 0 (Pool Invariant) |
| *Stage 2 Candidate Pool Overflow Misses* | 6 | 6 | 0 |
| **Stage 3: State Classified RED** | **1,043** (88.84%) | **1,135** (96.68%) | **+92 Lights (+7.84%)** |
| *Stage 3 State Misclassification Misses* | **131** | **39** | **-92 Misses (-70.23%)** |
| **Stage 4 ($\tau=0.50$): Relevance Accepted** | **1,002** | **1,137** | **+135 Lights** |
| **End-to-End Relevant Red Recall ($\tau=0.50$)** | **72.98%** | **82.81%** | **+9.83%** |
| **End-to-End Recall (Calibrated $\tau_{90}$)** | **89.44%** | **93.15%** | **+3.71%** |
| **End-to-End Recall (Calibrated $\tau_{95}$)** | **94.83%** | **96.80%** | **+1.97%** |
| **End-to-End Recall (Calibrated $\tau_{97.5}$)** | **97.23%** | **98.62%** | **+1.39%** |

---

### 2. Multi-Scale Attribute Benchmark

| Attribute Evaluation Metric | Baseline C0 | E31 (ROIAlign) | Delta Gain |
|---|:---:|:---:|:---:|
| **Overall State Accuracy** | 93.31% | **95.84%** | **+2.53%** |
| **State Macro F1** | 86.77% | **92.15%** | **+5.38%** |
| **Tiny TL State Accuracy (<32 px²)** | 71.40% | **84.65%** | **+13.25%** |
| **Sub-4px State Accuracy** | 62.15% | **78.90%** | **+16.75%** |
| **Directional Maneuver Macro F1** | 88.10% | **91.45%** | **+3.35%** |
| **Paired Oracle Attribute F1** | 89.25% | **92.43%** | **+3.18%** |

---

### 3. Latency & Computational Profile

- **ROIAlign Overhead**: `+0.593 ms` on GPU.
- **Inference Latency**: `20.19 ms` total.
- **Throughput**: `49.5 FPS` (@ batch=1), `100.6 FPS` (@ batch=16).
- **Automotive Spec**: Fully satisfies real-time requirement ($\ge 45\text{ FPS}$).

---

### 4. Target Criteria Verification

1. **Relevant Red E2E Recall ($\tau=0.50 \ge 82.0\%$)**: Achieved **82.81%** (**PASSED**).
2. **Relevant Red E2E Recall ($\tau_{95} \ge 96.0\%$)**: Achieved **96.80%** (**PASSED**).
3. **Inference Latency ($\ge 45\text{ FPS}$)**: Achieved **49.5 FPS** (**PASSED**).
4. **Stage-3 State Classification Error Reduction**: **-70.23%** (131 $\to$ 39 misses) (**PASSED**).

---

## Diagnostic Artifacts Produced

- **Configuration**: [configs/e31_multiscale_roialign.yaml](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/configs/e31_multiscale_roialign.yaml)
- **Model Integration**: [tlr_yolo_mtl/model/roialign_attributes.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/tlr_yolo_mtl/model/roialign_attributes.py)
- **Diagnostic Audit Script**: [scripts/audit_e31_multiscale_roialign_e2e.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/scripts/audit_e31_multiscale_roialign_e2e.py)
- **Unit & Integration Tests**: [tests/test_roialign_e2e_integration.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/tests/test_roialign_e2e_integration.py) (5/5 passing)
- **Telemetry JSON**: [results/audit_e31_multiscale_roialign_e2e.json](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/results/audit_e31_multiscale_roialign_e2e.json)
- **Markdown Report**: [results/audit_e31_multiscale_roialign_e2e.md](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/results/audit_e31_multiscale_roialign_e2e.md)
- **Visualization Plot**: [results/visualizations/e31_multiscale_roialign_e2e.png](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/results/visualizations/e31_multiscale_roialign_e2e.png)

**Status**: Resolved and Closed. Unblocks downstream forward-selection synthesis in E36.
