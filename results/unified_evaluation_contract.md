# Unified Evaluation Contract & Cross-Ticket Normalization Report (Ticket E29)

**Generated:** 2026-08-19 13:17:55  
**Canonical Baseline Model ($C_0$):** Run B4 (`YOLO11s + P2 + K_Arrow=32 + NWD-aware TAL`)  
**Primary Benchmark Checkpoint:** `best_composite.pt`  
**Invariant Validation Set:** DTLD full validation split (5,962 images, 25,344 GT TLs)

---

## 1. Locked Baseline $C_0$ Canonical Benchmark Values

These locked values establish the unified reference standard $C_0$ for Phase 4 forward selection:

| Metric Dimension | Canonical $C_0$ Value | Description / Guardrail Standard |
|---|:---:|---|
| **Selection Composite Score** | **0.8039** | Primary multi-task composite metric |
| **mAP@50 (Overall)** | **84.40%** | Joint detection accuracy |
| **mAP@50:95 (Overall)** | **56.60%** | Strict localization quality |
| **AP@50 (Traffic Light)** | **73.73%** | Traffic light detector AP |
| **AP@50 (Road Arrow)** | **95.07%** | Road arrow detector AP ($K_{Arrow}=32$) |
| **Tiny TL Recall ($<32\text{ px}^2$)** | **31.43%** | Perception floor tiny recall |
| **Tiny TL AP@50 ($<32\text{ px}^2$)** | **26.53%** | Perception floor tiny precision |
| **Sub-4px Recall (Side $<4\text{ px}$)** | **44.46%** | Sub-grid anchor allocation recovery |
| **Relevance AUPRC** | **91.61%** | Contextual ranking precision |
| **Relevance F1** | **85.64%** | Standard classification F1 |
| **Relevant Red Recall ($\tau=0.50$)** | **72.98%** | Uncalibrated baseline red recall |
| **State Accuracy** | **94.99%** | Traffic light state classification accuracy |
| **State Macro F1** | **86.77%** | Multi-class state macro F1 |
| **Sub-4px State Accuracy** | **80.46%** | Fine-grained state recognition on $<4\text{ px}$ |
| **Roundness F1** | **88.81%** | Directional vs round distinction |
| **Maneuver Macro F1** | **43.91%** | Multi-label arrow maneuver classification |
| **Batch-1 Latency** | **19.60 ms** | Real-time safety latency (51.0 FPS) |
| **Batch-16 Throughput** | **103.6 FPS** | Batch throughput |

---

## 2. Multi-Checkpoint Diagnostic Matrix (Run B4)

Evaluation across all saved checkpoint types under the exact E29 unified contract:

| Checkpoint | Selection Score | mAP@50 | AP_TL@50 | AP_Arrow@50 | Relevance AUPRC | Rel Red Recall ($\tau=0.50$) | State Acc | State Macro F1 | Tiny Recall | Sub-4px Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `best_composite.pt` | **0.8039** | 84.40% | 73.73% | 95.07% | 91.61% | 72.98% | 94.99% | 86.77% | 31.43% | 44.46% |
| `best_relevance.pt` | **0.8063** | 84.53% | 74.13% | 94.94% | 91.76% | 72.82% | 95.17% | 87.08% | 32.39% | 44.67% |
| `best_tl_detection.pt` | **0.8041** | 84.54% | 73.91% | 95.17% | 91.68% | 73.09% | 95.21% | 87.11% | 32.16% | 44.43% |
| `best_relevant_red_recall.pt` | **0.7201** | 81.26% | 69.60% | 92.93% | 86.27% | 87.44% | 88.42% | 55.81% | 19.87% | 33.08% |
| `last.pt` | **0.8015** | 83.49% | 72.90% | 94.07% | 91.36% | 71.27% | 95.38% | 87.69% | 29.40% | 40.67% |

---

## 3. 50/50 Holdout Temperature Calibration & Safety Operating Points

- **Optimal Fitted Temperature ($T^*$):** `0.7241`
- **Holdout Negative Log-Likelihood (NLL):** `0.5079` $\to$ **`0.4963`**
- **Holdout Expected Calibration Error (ECE):** `12.99%` $\to$ **`8.64%`**
- **Holdout Brier Score:** `0.1498` $\to$ **`0.1387`**

### Calibrated Safety Operating Points Table:

| Operating Point | Target Red Recall | Fitted Threshold ($\tau$) | Calibration Recall | Holdout Recall | Holdout Precision | Safety Guarantee Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **tau_90** | 90.0% | `tau = 0.3834` | 90.01% | **89.41%** | 67.97% | **PASSED** |
| **tau_95** | 95.0% | `tau = 0.3101` | 95.00% | **94.85%** | 64.12% | **PASSED** |
| **tau_97.5** | 97.5% | `tau = 0.2255` | 97.50% | **97.25%** | 59.22% | **PASSED** |

---

## 4. 4-Stage Safety Waterfall Failure Decomposition

Analysis of Relevant Red Traffic Light recall drop-off across architectural stages:

1. **Total Ground Truth Relevant Red TLs:** `3686` (100.0%)
2. **Stage 1 (Perception Detection @ IoU=0.50):** `3502` (95.01%) — Missed `184`
3. **Stage 2 (Top-K Candidate Selection):** `3462` (98.86%) — Missed `40`
4. **Stage 3 (State Classification as Red):** `3364` (97.17%) — Misclassified `98`
5. **Stage 4 (Relevance Gate $\tau=0.50$):** `2617` (77.79%) — Rejected `747`
- **End-to-End Recall:** **`71.00%`** (2617 / 3686)

---

## 5. Artifacts Generated

- Visualizations: `results/visualizations/e29_evaluation_contract_benchmark.png`
- JSON Telemetry: `results/unified_evaluation_contract.json`
- Markdown Report: `results/unified_evaluation_contract.md`
