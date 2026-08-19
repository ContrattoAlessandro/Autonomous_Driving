# Unified Evaluation Contract & Cross-Ticket Normalization Report (Ticket E29)

**Generated:** 2026-08-19 12:43:16  
**Canonical Baseline Model ($C_0$):** Run B4 (`YOLO11s + P2 + K_Arrow=32 + NWD-aware TAL`)  
**Primary Benchmark Checkpoint:** `best_composite.pt`  
**Invariant Validation Set:** DTLD full validation split (5,962 images, 25,344 GT TLs)

---

## 1. Locked Baseline $C_0$ Canonical Benchmark Values

These locked values establish the unified reference standard $C_0$ for Phase 4 forward selection:

| Metric Dimension | Canonical $C_0$ Value | Description / Guardrail Standard |
|---|:---:|---|
| **Selection Composite Score** | **0.7110** | Primary multi-task composite metric |
| **mAP@50 (Overall)** | **72.97%** | Joint detection accuracy |
| **mAP@50:95 (Overall)** | **46.25%** | Strict localization quality |
| **AP@50 (Traffic Light)** | **59.85%** | Traffic light detector AP |
| **AP@50 (Road Arrow)** | **86.09%** | Road arrow detector AP ($K_{Arrow}=32$) |
| **Tiny TL Recall ($<32\text{ px}^2$)** | **0.00%** | Perception floor tiny recall |
| **Tiny TL AP@50 ($<32\text{ px}^2$)** | **0.00%** | Perception floor tiny precision |
| **Sub-4px Recall (Side $<4\text{ px}$)** | **0.00%** | Sub-grid anchor allocation recovery |
| **Relevance AUPRC** | **93.27%** | Contextual ranking precision |
| **Relevance F1** | **85.28%** | Standard classification F1 |
| **Relevant Red Recall ($\tau=0.50$)** | **82.35%** | Uncalibrated baseline red recall |
| **State Accuracy** | **94.09%** | Traffic light state classification accuracy |
| **State Macro F1** | **88.69%** | Multi-class state macro F1 |
| **Sub-4px State Accuracy** | **51.47%** | Fine-grained state recognition on $<4\text{ px}$ |
| **Roundness F1** | **97.18%** | Directional vs round distinction |
| **Maneuver Macro F1** | **0.00%** | Multi-label arrow maneuver classification |
| **Batch-1 Latency** | **19.45 ms** | Real-time safety latency (51.4 FPS) |
| **Batch-16 Throughput** | **103.6 FPS** | Batch throughput |

---

## 2. Multi-Checkpoint Diagnostic Matrix (Run B4)

Evaluation across all saved checkpoint types under the exact E29 unified contract:

| Checkpoint | Selection Score | mAP@50 | AP_TL@50 | AP_Arrow@50 | Relevance AUPRC | Rel Red Recall ($\tau=0.50$) | State Acc | State Macro F1 | Tiny Recall | Sub-4px Recall |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `best_composite.pt` | **0.7110** | 72.97% | 59.85% | 86.09% | 93.27% | 82.35% | 94.09% | 88.69% | 0.00% | 0.00% |
| `best_relevance.pt` | **0.7035** | 69.36% | 59.23% | 79.49% | 94.05% | 85.88% | 94.90% | 89.64% | 0.00% | 0.00% |
| `best_tl_detection.pt` | **0.7016** | 68.37% | 59.40% | 77.33% | 93.45% | 87.06% | 95.75% | 92.48% | 0.00% | 0.00% |
| `best_relevant_red_recall.pt` | **0.6506** | 69.50% | 54.56% | 84.44% | 90.51% | 96.47% | 85.38% | 52.09% | 0.00% | 0.00% |
| `last.pt` | **0.6788** | 64.93% | 56.84% | 73.01% | 91.15% | 84.71% | 94.57% | 90.12% | 0.00% | 0.00% |

---

## 3. 50/50 Holdout Temperature Calibration & Safety Operating Points

- **Optimal Fitted Temperature ($T^*$):** `0.2986`
- **Holdout Negative Log-Likelihood (NLL):** `0.4244` $\to$ **`0.4998`**
- **Holdout Expected Calibration Error (ECE):** `13.70%` $\to$ **`8.42%`**
- **Holdout Brier Score:** `0.1252` $\to$ **`0.1147`**

### Calibrated Safety Operating Points Table:

| Operating Point | Target Red Recall | Fitted Threshold ($\tau$) | Calibration Recall | Holdout Recall | Holdout Precision | Safety Guarantee Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **tau_90** | 90.0% | `tau = 0.2012` | 100.00% | **90.24%** | 100.00% | **PASSED** |
| **tau_95** | 95.0% | `tau = 0.2012` | 100.00% | **90.24%** | 100.00% | **MARGINAL** |
| **tau_97.5** | 97.5% | `tau = 0.2012` | 100.00% | **90.24%** | 100.00% | **MARGINAL** |

---

## 4. 4-Stage Safety Waterfall Failure Decomposition

Analysis of Relevant Red Traffic Light recall drop-off across architectural stages:

1. **Total Ground Truth Relevant Red TLs:** `85` (100.0%)
2. **Stage 1 (Perception Detection @ IoU=0.50):** `84` (98.82%) — Missed `1`
3. **Stage 2 (Top-K Candidate Selection):** `83` (98.81%) — Missed `1`
4. **Stage 3 (State Classification as Red):** `83` (100.00%) — Misclassified `0`
5. **Stage 4 (Relevance Gate $\tau=0.50$):** `70` (84.34%) — Rejected `13`
- **End-to-End Recall:** **`82.35%`** (70 / 85)

---

## 5. Artifacts Generated

- Visualizations: `results/visualizations/e29_evaluation_contract_benchmark.png`
- JSON Telemetry: `results/unified_evaluation_contract.json`
- Markdown Report: `results/unified_evaluation_contract.md`
