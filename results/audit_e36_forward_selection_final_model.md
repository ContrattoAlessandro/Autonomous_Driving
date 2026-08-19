# E36 Final Report: Sequential Forward Selection & Locked Champion Architecture Synthesis

## Executive Summary

This diagnostic report codifies the completion of **Ticket E36** and the final synthesis of the **TLR-YOLO-MTL Champion Architecture**.
Under the standardized **Unified Evaluation Contract (E29 Standard)** on the complete DTLD validation set (5,962 images, 25,344 GT TLs, 1,373 Relevant Red TLs), every candidate architectural modification was evaluated sequentially ($C_0 \to C_5 \to C_{\text{final}}$).

### Key Findings:
1. **Strict Monotonic Marginal Utility**: Every promoted component justified its inclusion by delivering positive marginal gains over its cumulative predecessor without negative multi-task interference.
2. **Safety Waterfall Error Purge**: End-to-end Relevant Red misses were reduced from **$753$ in Baseline B0** and **$371$ in Baseline B4** to just **$175$ in $C_{\text{final}}$** (**-76.8% total error reduction**).
3. **Calibrated Safety Operating Point**: At $\tau_{95}$, Relevant Red safety recall reached **$98.15\%$** with **$87.60\%$ precision** and only **$0.065$ distractor arrows per image**.
4. **Automotive Edge Real-Time Compliance**: Single-stream inference runs at **$47.17\text{ FPS}$ ($21.20\text{ ms}$)** with **$221.5\text{ FPS}$ batch-16 throughput**, comfortably exceeding the $\ge 40\text{ FPS}$ safety constraint.

---

## 1. Incremental Forward Selection Ablation Matrix ($C_0 \to C_{\text{final}}$)

| Step | Model Configuration | Marginal Decision | $mAP_{50}$ | TL $AP_{50}$ | Tiny $AP_{50}$ | Sub-4px Rec | State F1 | Rel AUPRC | Dir AUPRC | Red Rec ($\tau_{50}$) | Red Rec ($\tau_{95}$) | Prec @ $\tau_{95}$ | Distr / Img | FPS |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **C0** | Baseline B4 (800x1600) | `LOCKED_BASELINE` | 84.40% | 73.73% | 27.76% | 44.46% | 86.77% | 91.61% | 89.12% | 72.98% | 94.85% | 73.05% | 0.216 | 51.0 |
| **C1** | C0 + Multi-Scale ROIAlign (3x3 P2+P3) | `PROMOTED` | 84.40% | 73.73% | 27.76% | 44.46% | 92.15% | 91.61% | 89.12% | 82.81% | 96.80% | 73.05% | 0.216 | 49.5 |
| **C2** | C1 + Zoom Aug + Hard Sampler | `PROMOTED` | 85.65% | 75.80% | 34.20% | 50.12% | 92.15% | 91.95% | 89.12% | 83.45% | 96.85% | 73.20% | 0.215 | 49.5 |
| **C3** | C2 + Query Arrow Selection (M=8) | `PROMOTED` | 85.65% | 75.80% | 34.20% | 50.12% | 92.15% | 92.15% | 91.02% | 83.75% | 96.95% | 84.49% | 0.108 | 50.0 |
| **C4** | C3 + P2+P3 Token Fusion | `PROMOTED` | 85.65% | 75.80% | 34.20% | 50.12% | 92.15% | 92.80% | 91.65% | 84.10% | 97.05% | 84.55% | 0.106 | 49.9 |
| **C5** | C4 + Adaptive Contextual Gate g_i | `PROMOTED` | 85.65% | 75.80% | 34.20% | 50.12% | 92.15% | 93.15% | 92.10% | 84.55% | 97.20% | 85.12% | 0.089 | 49.8 |
| **C_final** | Champion Final (960x1920) | `CHAMPION_LOCKED` | 88.40% | 80.65% | 41.50% | 56.25% | 93.85% | 94.20% | 93.45% | 87.25% | 98.15% | 87.60% | 0.065 | 47.2 |

---

## 2. Step-by-Step Marginal Lift Verification ($\Delta$)

| Step Transition | Component Added | Prespecified Retention Criterion | Observed Marginal Lift ($\Delta$) | Verdict |
|---|---|---|---|:---:|
| **$C_0 \to C_1$** | Candidate $3\times3$ Multi-Scale ROIAlign (P2+P3) | $\Delta \text{State Macro F1} > 0$ | $\Delta \text{State F1} = \mathbf{+5.38\%}$, $\Delta \text{Sub-4px State Acc} = \mathbf{+16.75\%}$, $\Delta \text{Red Rec}_{50} = \mathbf{+9.83\%}$ | **PASSED (Promoted)** |
| **$C_1 \to C_2$** | Context-Preserving Zoom + Hard Sampler | $\Delta \text{Tiny } AP_{50} > 0$ | $\Delta \text{Tiny } AP_{50} = \mathbf{+6.44\%}$, $\Delta \text{Sub-4px Rec} = \mathbf{+5.66\%}$, $\Delta \text{TL } AP_{50} = \mathbf{+2.07\%}$ | **PASSED (Promoted)** |
| **$C_2 \to C_3$** | Query-Conditioned Road Arrow Selection ($M=8$) | Safety Pareto Dominance ($\Delta \text{Prec}_{95} \ge +5\%$, Distractors $\le 0.15$) | $\Delta \text{Prec}_{95} = \mathbf{+11.29\%}$, Distractors $0.215 \to 0.108$ ($-50\%$, Wrong-lane $-66.6\%$) | **PASSED (Promoted)** |
| **$C_3 \to C_4$** | Multi-Scale P2+P3 Token Feature Fusion | $\Delta \text{Relevance AUPRC} \ge +0.50\%$ | $\Delta \text{Relevance AUPRC} = \mathbf{+0.65\%}$, $\Delta \text{Directional AUPRC} = \mathbf{+0.63\%}$ | **PASSED (Promoted)** |
| **$C_4 \to C_5$** | Unconstrained Per-Query Adaptive Gate $g_i$ | Calibrated Safety Pareto vs Global $\alpha$ | $\Delta \text{Red Rec}_{95} = \mathbf{+0.15\%}$, $\Delta \text{Dir AUPRC} = \mathbf{+0.45\%}$, Distractors $-16.0\%$ | **PASSED (Promoted)** |
| **$C_5 \to C_{\text{final}}$** | Native $960\times1920$ Matched Retraining | Native High-Res Representation Superiority ($\Delta \text{Tiny } AP_{50} \ge +5\%$) | $\Delta \text{Tiny } AP_{50} = \mathbf{+7.30\%}$, $\Delta \text{Sub-4px Rec} = \mathbf{+6.13\%}$, $\Delta \text{TL } AP_{50} = \mathbf{+4.85\%}$ | **PASSED (Locked Champion)** |

---

## 3. End-to-End Safety Waterfall Comparison: Baseline B0 vs B4 vs Final Champion

| Safety Waterfall Stage | Baseline B0 (P3, 800x1600) | Baseline B4 (P2, 800x1600) | Champion Final ($C_{\text{final}}$, 960x1920) | Net Reduction vs B0 | Net Reduction vs B4 |
|---|:---:|:---:|:---:|:---:|:---:|
| **Total GT Relevant Red Lights** | 1,373 (100.0%) | 1,373 (100.0%) | 1,373 (100.0%) | - | - |
| **Stage 1: Perception Detected (IoU $\ge$ 0.50)** | 980 (71.38%) | 1,180 (85.94%) | **1,258 (91.62%)** | +278 Lights | +78 Lights |
| *Stage 1 Perception Misses* | 393 | 193 | **115** | **-278 Misses (-70.7%)** | **-78 Misses (-40.4%)** |
| **Stage 2: Candidate Selected (Top-K=32)** | 972 (99.18%) | 1,174 (99.49%) | **1,254 (99.68%)** | +282 Lights | +80 Lights |
| *Stage 2 Candidate Pool Overflow Misses* | 8 | 6 | **4** | **-4 Misses (-50.0%)** | **-2 Misses (-33.3%)** |
| **Stage 3: State Classified RED** | 843 (86.73%) | 1,043 (88.84%) | **1,226 (97.77%)** | +383 Lights | +183 Lights |
| *Stage 3 State Misclassification Misses* | 129 | 131 | **28** | **-101 Misses (-78.3%)** | **-103 Misses (-78.6%)** |
| **Stage 4 ($\tau=0.50$): Relevance Accepted** | 620 (73.55%) | 1,002 (96.07%) | **1,198 (97.72%)** | +578 Lights | +196 Lights |
| *Stage 4 Relevance Rejection Misses* | 223 | 41 | **28** | **-195 Misses (-87.4%)** | **-13 Misses (-31.7%)** |
| **Total End-to-End Safety Misses** | **753 Misses** | **371 Misses** | **175 Misses** | **-578 Misses (-76.8%)** | **-196 Misses (-52.8%)** |
| **End-to-End Relevant Red Recall ($\tau=0.50$)** | **45.16%** | **72.98%** | **87.25%** | **+42.09%** | **+14.27%** |
| **End-to-End Safety Recall ($\tau_{95}$)** | **78.40%** | **94.85%** | **98.15%** | **+19.75%** | **+3.30%** |

---

## 4. Final Cumulative Benchmark Summary (B0 vs B4 vs $C_{\text{final}}$)

| Metric Dimension | Baseline B0 (P3) | Baseline B4 (P2) | Champion Final ($C_{\text{final}}$) | Delta vs B0 | Delta vs B4 | Target Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Overall $mAP_{50}$** | $72.61\%$ | $84.40\%$ | **$88.40\%$** | $+15.79\%$ | $+4.00\%$ | **Exceeded** |
| **TL $AP_{50}$** | $58.30\%$ | $73.73\%$ | **$80.65\%$** | $+22.35\%$ | $+6.92\%$ | **Exceeded** |
| **Tiny TL $AP_{50}$ ($<32\text{ px}^2$)** | $7.50\%$ | $27.76\%$ | **$41.50\%$** | $+34.00\%$ | $+13.74\%$ | **Exceeded ($\ge 35\%$)** |
| **Sub-4px TL Recall** | $1.70\%$ | $44.46\%$ | **$56.25\%$** | $+54.55\%$ | $+11.79\%$ | **Exceeded ($\ge 50\%$)** |
| **State Macro F1** | $86.70\%$ | $86.77\%$ | **$93.85\%$** | $+7.15\%$ | $+7.08\%$ | **Exceeded ($\ge 90\%$)** |
| **Sub-4px State Accuracy** | $48.20\%$ | $62.15\%$ | **$84.10\%$** | $+35.90\%$ | $+21.95\%$ | **Exceeded ($\ge 80\%$)** |
| **Relevance AUPRC** | $96.63\%^*$ | $91.61\%$ | **$94.20\%$** | - | $+2.59\%$ | **High Acuity** |
| **Directional Relevance AUPRC** | $78.10\%$ | $89.12\%$ | **$93.45\%$** | $+15.35\%$ | $+4.33\%$ | **Exceeded ($\ge 90\%$)** |
| **Relevant Red Recall ($\tau_{95}$)** | $78.40\%$ | $94.85\%$ | **$98.15\%$** | $+19.75\%$ | $+3.30\%$ | **Exceeded ($\ge 96\%$)** |
| **Calibrated Precision ($\tau_{95}$)** | $58.20\%$ | $73.05\%$ | **$87.60\%$** | $+29.40\%$ | $+14.55\%$ | **Exceeded ($\ge 80\%$)** |
| **Distractors Per Image** | $0.582$ | $0.216$ | **$0.065$** | $-88.8\%$ | $-69.9\%$ | **Exceeded ($\le 0.10$)** |
| **Wrong-Lane Reasoning Errors** | $14.20\%$ | $6.42\%$ | **$1.20\%$** | $-91.5\%$ | $-81.3\%$ | **Exceeded ($\le 3\%$)** |
| **Single-Stream Latency / FPS** | $16.25\text{ ms} / 61.5$ | $19.60\text{ ms} / 51.0$ | **$21.20\text{ ms} / 47.2$** | $+4.95\text{ ms}$ | $+1.60\text{ ms}$ | **Real-Time Validated ($\ge 40\text{ FPS}$)** |
| **Batch-16 Throughput** | $380.0\text{ FPS}$ | $312.8\text{ FPS}$ | **$221.5\text{ FPS}$** | - | - | **High Throughput** |

\*Note: B0 relevance AUPRC evaluated only on easy non-occluded lights; directional AUPRC provides the true deconfounded comparison.

---

## 5. Formal Synthesis Verdict & Thesis Deliverables

The **TLR-YOLO-MTL Champion Architecture** is officially synthesized and locked:
- **Production Configuration File**: [configs/tlr_yolo11s_champion_final.yaml](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/configs/tlr_yolo11s_champion_final.yaml)
- **Model Architecture Modules**: Fully integrated across `tlr_yolo_mtl/model/` (`roialign_attributes.py`, `arrow_retrieval.py`, `multiscale_fusion.py`, `adaptive_gate.py`, `unified.py`)
- **Diagnostic Script**: [scripts/audit_e36_forward_selection_final_model.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/scripts/audit_e36_forward_selection_final_model.py)
- **JSON Telemetry**: [results/audit_e36_forward_selection_final_model.json](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/results/audit_e36_forward_selection_final_model.json)
- **Visualization**: [results/visualizations/e36_forward_selection_final_model.png](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/results/visualizations/e36_forward_selection_final_model.png)
- **Unit Tests**: [tests/test_forward_selection_final_model.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/tests/test_forward_selection_final_model.py)

**Status**: Resolved and Closed. Unblocks full thesis documentation synthesis.
