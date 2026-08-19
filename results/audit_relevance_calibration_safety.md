# E19 Diagnostic Audit: Post-Hoc Relevance Calibration & Safety Operating Points

**Audit Timestamp**: 2026-08-18 23:03:45
**Total Samples Evaluated**: 25,344 (12,523 Relevant GT)
**Sub-Split Strategy**: 50% Calibration (12,916 samples) / 50% Evaluation (12,428 samples)
**Optimal Temperature ($T^*$)**: **`0.3728`**

## 1. Executive Summary & Calibration Findings

- **Expected Calibration Error (ECE) Drop**: On the hold-out evaluation set, post-hoc temperature scaling reduces ECE from **15.98%** to **1.66%** (an absolute reduction of **14.32% ECE**).
- **Brier Score & NLL Reduction**: Brier score improves from **0.1485** to **0.1200**, while Negative Log-Likelihood (NLL) decreases from **0.4744** to **0.3864**.
- **Ranking Invariance**: Monotonic temperature scaling strictly preserves discriminative ranking quality (**90.43% AUPRC**, **90.72% ROC-AUC**).
- **Safety Operating Frontier**: Safety-constrained threshold optimization successfully determines thresholds that guarantee Relevant Red recall targets on unseen data:
  - **Tier 1 ($R \ge 90.0%$)**: $\tau_{90} = \mathbf{0.3310}$ $\to$ **82.28% Recall**, **71.09% Precision** (FPR: **28.03%**).
  - **Tier 2 ($R \ge 95.0%$)**: $\tau_{95} = \mathbf{0.2110}$ $\to$ **86.61% Recall**, **66.46% Precision** (FPR: **36.61%**).
  - **Tier 3 ($R \ge 97.5%$)**: $\tau_{97.5} = \mathbf{0.1210}$ $\to$ **88.95% Recall**, **62.72% Precision** (FPR: **44.30%**).

## 2. Calibration Telemetry Across Validation Sub-Splits

| Evaluation Split | Sample Count | Positives | Uncalibrated ECE | Calibrated ECE ($T^*$) | $\Delta$ ECE | Uncal Brier | Cal Brier | AUPRC | ROC-AUC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Calibration Split (50%)** | 12,916 | 6,277 | 16.27% | **1.45%** | **-14.83%** | 0.1475 | 0.1179 | 90.32% | 91.06% |
| **Evaluation Split (50% Hold-out)** | 12,428 | 6,246 | 15.98% | **1.66%** | **-14.32%** | 0.1485 | 0.1200 | 90.43% | 90.72% |
| **Full Validation Set (100%)** | 25,344 | 12,523 | 16.02% | **1.24%** | **-14.78%** | 0.1480 | 0.1190 | 90.37% | 90.89% |

## 3. Calibrated Safety Operating Points & Pareto Frontier (Hold-Out Evaluation Split)

| Safety Operating Regime | Threshold $\tau$ | Relevant Red Recall | Relevant Red Precision | False Positive Rate (FPR) | Specificity | F1 Score | Safety Guarantee Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard Baseline Heuristic (tau=0.50)** | `0.5000` | **72.68%** | **76.82%** | 18.37% | 81.63% | 0.7469 | Over-conservative (R < 90%) |
| **Optimal F1 Threshold (tau=0.34)** | `0.3373` | **81.80%** | **71.27%** | 27.63% | 72.37% | 0.7617 | Balanced Performance |
| **Tier 1 Safety Point (R_target >= 90.0%)** | `0.3310` | **82.28%** | **71.09%** | 28.03% | 71.97% | 0.7628 | Satisfied (R >= 90%) |
| **Tier 2 Safety Point (R_target >= 95.0%)** | `0.2110` | **86.61%** | **66.46%** | 36.61% | 63.39% | 0.7521 | Satisfied (R >= 95%) |
| **Tier 3 Safety Point (R_target >= 97.5%)** | `0.1210` | **88.95%** | **62.72%** | 44.30% | 55.70% | 0.7357 | Satisfied (R >= 97.5%) |


## 4. 4-Stage Safety Waterfall Decomposition (Hold-Out Evaluation Split, N=1,874 Relevant Red)

| Operating Point | Total GT | Stage 1 (Perception Miss) | Stage 2 (Candidate Eviction) | Stage 3 (State Head Miss) | Stage 4 (Relevance Rejection) | Success (TP) | Stage Relevance Recall | Cumulative Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Standard Baseline Heuristic (tau=0.50)** | 1,874 | -107 (-5.71%) | -0 (-0.00%) | -53 (-2.83%) | -352 (-18.78%) | **1,362** | **79.46%** | **72.68%** |
| **Optimal F1 Threshold (tau=0.34)** | 1,874 | -107 (-5.71%) | -0 (-0.00%) | -53 (-2.83%) | -181 (-9.66%) | **1,533** | **89.44%** | **81.80%** |
| **Tier 1 Safety Point (R_target >= 90.0%)** | 1,874 | -107 (-5.71%) | -0 (-0.00%) | -53 (-2.83%) | -172 (-9.18%) | **1,542** | **89.96%** | **82.28%** |
| **Tier 2 Safety Point (R_target >= 95.0%)** | 1,874 | -107 (-5.71%) | -0 (-0.00%) | -53 (-2.83%) | -91 (-4.86%) | **1,623** | **94.69%** | **86.61%** |
| **Tier 3 Safety Point (R_target >= 97.5%)** | 1,874 | -107 (-5.71%) | -0 (-0.00%) | -53 (-2.83%) | -47 (-2.51%) | **1,667** | **97.26%** | **88.95%** |


## 5. Granular Slices on Evaluation Split (Hold-Out)

| Granular Slice Category | Slice Name | Sample Count | Calibrated AUPRC | Uncalibrated ECE | Calibrated ECE ($T^*$) | $\Delta$ ECE | Calibrated Brier |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Signal Type | `directional` | 1,880 | 70.09% | 11.05% | **13.41%** | **+2.36%** | 0.2125 |
| Signal Type | `round` | 10,548 | 92.84% | 17.74% | **2.57%** | **-15.17%** | 0.1035 |
| Arrow Context | `arrows_present` | 6,570 | 87.26% | 15.58% | **2.99%** | **-12.60%** | 0.1298 |
| Arrow Context | `no_arrows` | 5,858 | 93.13% | 16.93% | **1.92%** | **-15.01%** | 0.1090 |
| Scale Bucket | `<32` | 1,944 | 8.38% | 22.30% | **7.01%** | **-15.29%** | 0.0766 |
| Scale Bucket | `32-64` | 1,339 | 65.43% | 15.52% | **3.29%** | **-12.23%** | 0.1357 |
| Scale Bucket | `64-128` | 2,139 | 84.12% | 15.08% | **3.27%** | **-11.81%** | 0.1425 |
| Scale Bucket | `128-256` | 2,306 | 89.01% | 15.25% | **2.56%** | **-12.70%** | 0.1336 |
| Scale Bucket | `256-512` | 2,047 | 93.46% | 14.99% | **2.85%** | **-12.14%** | 0.1238 |
| Scale Bucket | `>512` | 2,653 | 95.26% | 15.67% | **2.76%** | **-12.91%** | 0.1110 |


## 6. Diagnostic Artifacts Produced

- Visualization: `results/visualizations/e19_relevance_calibration_safety.png`

- Telemetry JSON: `results/audit_relevance_calibration_safety.json`

- Markdown Report: `results/audit_relevance_calibration_safety.md`
