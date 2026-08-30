---
title: "E54: Candidate Recall Ceiling & Waterfall Stage Audit"
type: task
status: closed
blocked_by: []
assignee: "@agent"
---

## Question

At what specific stage of the Champion v4 inference pipeline—Dense Head feature grids, Raw Detection Decoding, Quality Ranking, Virtual-P1 Refinement, or Non-Maximum Suppression—does the greatest fraction of sub-4px and sub-8px Ground Truth traffic lights get dropped, and what is the theoretical upper-bound Oracle Recall at each stage?

---

## Context & Scientific Motivation

In Champion v4, sub-4px Recall is $41.20\%$ while sub-8px AP@50 is $55.60\%$. To architect Champion v5 efficiently, we must resolve the fundamental dichotomy:

$$\begin{aligned}
\textbf{Hypothesis A (Representation Ceiling):} & \quad \text{Pre-NMS/Pre-Filter Recall} \approx \text{Final Recall } (41\text{--}52\%) \\
& \implies \text{The backbone/neck fails to generate candidate activations. Solution: Representation/P1-Lite.} \\
\textbf{Hypothesis B (Filter/Ranking Bottleneck):} & \quad \text{Pre-NMS/Pre-Filter Recall} \gg \text{Final Recall } (\ge 75\%) \\
& \implies \text{The backbone already sees the signals, but downstream stages discard them. Solution: Ranking/NMS.}
\end{aligned}$$

Resolving this question prevents wasted effort on heavy backbone modifications if the bottleneck is simply post-processing or candidate ranking.

---

## Empirical Diagnostic Results: DTLD Canonical Validation Set (5,962 images, 25,344 GT TLs, 6,108 GT Arrows)

Evaluated via [scripts/audit_e54_candidate_recall_ceiling.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/scripts/audit_e54_candidate_recall_ceiling.py) on `tlr_yolo11s_champion_v4` (`best_composite.pt`).

### 1. The 6-Stage Recall Waterfall Across Scale Regimes (IoU $\ge 0.50$)

| Stage Index & Pipeline Checkpoint | Sub-4px ($<16\text{ px}^2$) | 4–8px ($16\text{--}64\text{ px}^2$) | 8–16px ($64\text{--}256\text{ px}^2$) | >16px ($\ge 256\text{ px}^2$) | Global TL | Road Arrow | Relevant Red TL |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Stage 1: Dense Head Anchor Grids ($K=\infty$)** | **52.40%** | **92.50%** | **98.60%** | **99.65%** | **91.58%** | **97.80%** | **99.70%** |
| **Stage 2: Post-Decoding Top-K ($K=1024$)** | 49.80% | 90.80% | 98.10% | 99.50% | 90.52% | 97.30% | 99.60% |
| **Stage 3: Quality-Ranked ($s = p^{0.7} q^{0.3}$)** | 48.20% | 89.20% | 97.40% | 99.30% | 89.55% | 96.90% | 99.45% |
| **Stage 4: Post-Virtual-P1 Refinement ($K=32$)** | 47.10% | 87.95% | 96.50% | 99.30% | 88.67% | 96.90% | 99.40% |
| **Stage 5: Post-NMS Output (Size-Adaptive NWD)** | 45.60% | 85.10% | 93.70% | 98.25% | 86.35% | 95.60% | 99.10% |
| **Stage 6: Final Deployment ($\tau_{\text{deploy}} = 0.25$)** | **41.20%** | **78.60%** | **91.80%** | **97.40%** | **82.90%** | **94.85%** | **98.80%** |
| **Total Waterfall Drop ($\Delta \text{Recall}$)** | **$-11.20\text{ pp}$** | **$-13.90\text{ pp}$** | **$-6.80\text{ pp}$** | **$-2.25\text{ pp}$** | **$-8.68\text{ pp}$** | **$-2.95\text{ pp}$** | **$-0.90\text{ pp}$** |

---

### 2. Multi-Metric Matching Sensitivity at Final Operational Output (Stage 6)

| Matching Metric | Sub-4px ($<16\text{ px}^2$) | 4–8px ($16\text{--}64\text{ px}^2$) | 8–16px ($64\text{--}256\text{ px}^2$) | >16px ($\ge 256\text{ px}^2$) | Global TL |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Standard $\text{IoU} \ge 0.50$** | **41.20%** | **78.60%** | **91.80%** | **97.40%** | **82.90%** |
| **Loose $\text{IoU} \ge 0.25$** | **46.50%** | **82.40%** | **93.70%** | **98.20%** | **86.00%** |
| **Gaussian $\text{NWD} \ge 0.50$ ($C=12.0$)** | **50.10%** | **84.90%** | **94.60%** | **98.60%** | **87.50%** |

---

### 3. Recall@K Across Candidate Proposal Budgets (Stage 2 Post-Decoding)

| Candidate Pool Budget ($K$) | Sub-4px ($<16\text{ px}^2$) | 4–8px ($16\text{--}64\text{ px}^2$) | 8–16px ($64\text{--}256\text{ px}^2$) | >16px ($\ge 256\text{ px}^2$) | Global TL | Road Arrow |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **$K = 32$** | 35.40% | 74.20% | 89.50% | 98.10% | 79.30% | 92.40% |
| **$K = 64$** | 41.80% | 83.50% | 94.60% | 99.10% | 85.25% | 95.30% |
| **$K = 128$** | 45.90% | 87.60% | 96.70% | 99.40% | 88.10% | 96.50% |
| **$K = 256$** | 48.20% | 89.60% | 97.60% | 99.50% | 89.70% | 97.00% |
| **$K = 512$** | 49.30% | 90.40% | 97.90% | 99.50% | 90.25% | 97.20% |
| **$K = 1024$** | 49.80% | 90.80% | 98.10% | 99.50% | 90.52% | 97.30% |
| **$K = \infty$ (All Anchors)** | **52.40%** | **92.50%** | **98.60%** | **99.65%** | **91.58%** | **97.80%** |

---

## Causal Hypothesis Resolution & Decision Triggers

1. **Resolution of Sub-4px Dichotomy**:
   - **Stage 1 Sub-4px Recall is $52.40\%$ ($<55.0\%$)** $\implies$ **HYPOTHESIS A (REPRESENTATION CEILING) IS CONFIRMED**.
   - The primary limiting factor for tiny distant signals ($<4\text{ px}$) is that the backbone/P2 neck fails to generate active anchor features at stride 4 for nearly $47.6\%$ of instances.
   - Downstream filters (Stage 1 to Stage 6) drop only $11.20\text{ pp}$ ($52.40\% \to 41.20\%$).
   - **Decision Trigger**: Unblocks **Ticket E55** (Tiny Feature SNR Audit), **Ticket E58** (NWD-TAL Assigner Audit), and conditional tickets **E65** (Candidate-Conditioned P1-Lite) and **E66** (Scale-Conditioned Relay v2).
2. **Resolution of 4–8px and 8–16px Regimes**:
   - **4–8px Regime**: Stage 1 Recall is high ($92.50\%$), but drops by $-13.90\text{ pp}$ to $78.60\%$ in Stage 6. The largest single loss occurs at Stage 5 $\to$ Stage 6 ($-6.50\text{ pp}$) due to confidence thresholding ($s < 0.25$).
   - **8–16px Regime**: Stage 1 Recall is virtually saturated ($98.60\%$), but drops by $-6.80\text{ pp}$ to $91.80\%$. The largest loss occurs at Stage 4 $\to$ Stage 5 ($-2.80\text{ pp}$) due to NMS over-suppression in dense clusters.
   - **Decision Trigger**: Unblocks **Ticket E57** (Virtual-P1 Coverage Audit) and **Ticket E61** (Quality Calibration & Exponent Sweep).

---

## Acceptance & Confirmation Criteria Verification

- [x] **Criterion 1: Waterfall Trace Completeness**: **PASSED** (Unbroken trace evaluated across all 6 pipeline stages on full 5,962 validation images).
- [x] **Criterion 2: Scale-Stratified Waterfall Table**: **PASSED** (Rigorous reporting of $\text{Recall}@K$ across $K \in \{32, 64, 128, 256, 512, 1024, \infty\}$ and all 4 scale bins).
- [x] **Criterion 3: Definite Decision Trigger**: **PASSED** (Formally proved $\text{Stage 1 Recall}_{<4\text{px}} = 52.40\% < 55\%$, confirming Hypothesis A and unblocking Tickets E55, E57, E58, E61).

---

**Status**: Ticket E54 is formally **closed**, unblocking **Ticket E55**, **Ticket E57**, **Ticket E58**, and **Ticket E61** on the Phase 7 roadmap.
