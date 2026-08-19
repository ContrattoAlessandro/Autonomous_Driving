---
title: "W6: TaskAlignedAssigner Positive Allocation & NWD vs CIoU Interaction"
type: research
status: closed
blocked_by: ["W1", "W5"]
assignee: "@agent"
---

## Question

Do tiny ground-truth traffic lights receive sufficient positive anchor assignments during TaskAlignedAssigner matching, and are CIoU and NWD bounding box losses cooperating or conflicting during gradient backpropagation?

## Context & Requirements

1. **Assigner Telemetry per GT Instance**:
   - For every GT object during training, record:
     - Number of assigned positive candidate anchors $N_{pos}$.
     - Distribution of positive assignments across pyramid levels (P3, P4, P5).
     - Maximum alignment score: $t = s^\alpha \cdot \text{IoU}^\beta$.
     - Maximum IoU and maximum Normalized Wasserstein Distance (NWD).
   - Trace conditional probability of starvation: $P(N_{pos} = 0 \mid size)$ and expected candidates $\mathbb{E}[N_{pos} \mid size]$.

2. **CIoU vs NWD Gradient Interaction**:
   - Compute individual regression loss components: $\mathcal{L}_{CIoU}$, $\mathcal{L}_{DFL}$, $\mathcal{L}_{NWD}$.
   - For tiny TL batches, compute gradient cosine similarity on the bounding box regression head:
     $$\cos(g_{CIoU}, g_{NWD}) = \frac{g_{CIoU} \cdot g_{NWD}}{\|g_{CIoU}\| \|g_{NWD}\|}$$
   - Interpretation:
     - $\cos \approx +1$: Synergistic optimization.
     - $\cos \approx 0$: Orthogonal / independent targets.
     - $\cos < 0$: Antagonistic conflict.
   - Use findings to justify whether `nwd_weight` tuning ($0.25, 0.5, 1.0$) or NWD-aware TAL assignment is required.

## Empirical Resolution & Diagnostic Summary

- **Evaluated Split**: DTLD Training set (12,004 GT Traffic Lights evaluated across 500 batches with Baseline B0).
- **Assigner Telemetry & Gradient Cosine Script**: `scripts/audit_assigner_nwd_ciou.py`.

### Key Empirical Findings:

1. **Positive Candidate Allocation per Area Bucket**:

| Area Metric | $<32\text{ px}^2$ | $32\text{--}64\text{ px}^2$ | $64\text{--}128\text{ px}^2$ | $128\text{--}256\text{ px}^2$ | $256\text{--}512\text{ px}^2$ | $>512\text{ px}^2$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Starvation Rate $P(N_{pos}=0)$** | **8.6%** | **1.1%** | **0.2%** | **0.0%** | **0.0%** | **0.0%** |
| **Mean Positive Anchors $\mathbb{E}[N_{pos}]$** | **2.29** | **3.72** | **5.49** | **7.14** | **7.44** | **9.89** |
| **P3 Allocation %** | 78.3% | 77.3% | 76.9% | 77.1% | 77.3% | 67.2% |
| **Mean Max IoU with Anchors** | **0.196** | **0.372** | **0.543** | **0.711** | **0.818** | **0.883** |
| **Mean Max NWD with Anchors** | **0.686** | **0.739** | **0.795** | **0.849** | **0.882** | **0.876** |

2. **CIoU vs NWD Gradient Interaction on Regression Head**:
   - **All Batches Mean Cosine**: $\mathbf{+0.6123 \pm 0.1159}$ (**100.0% positive cosine similarity**).
   - **Tiny-TL Batches ($<64\text{ px}^2$) Mean Cosine**: $\mathbf{+0.6007 \pm 0.1201}$ (**100.0% positive cosine similarity**).
   - **Antagonistic Conflict Rate ($\cos < 0$)**: **0.0%**.

### Architectural Conclusion:
- **Gradient Synergy Verified**: CIoU and NWD losses pull in strongly aligned gradient directions ($\cos \approx +0.61$), confirming that `nwd_weight` ($0.5$) provides beneficial smooth regression gradients without destructive backprop conflicts.
- **Assigner Bottleneck Identified**: The standard TaskAlignedAssigner alignment cost $t = s^\alpha \cdot \text{IoU}^\beta$ suffers from IoU collapse on tiny signals (max IoU drops to 0.196), causing 8.6% starvation and starving tiny signals of anchor supervision. An **NWD-aware alignment metric in TAL** or a **P2 stride-4 neck** is empirically indicated.

### Diagnostic Artifacts Produced:
- **Audit Script**: `scripts/audit_assigner_nwd_ciou.py`
- **Tabular Report**: `results/audit_assigner_nwd_ciou.md`
- **JSON Telemetry**: `results/audit_assigner_nwd_ciou.json`
- **Visualization Plot**: `results/visualizations/w6_assigner_allocation_nwd_ciou.png`
