---
title: "W9: Local Relevance Baseline & Safety-Critical Metrics"
type: research
status: closed
blocked_by: ["W1", "W7"]
assignee: "@agent"
---

## Question

What is the baseline performance of the local relevance head when cross-attention is disabled ($\alpha=0$), and how does the model perform under end-to-end and safety-critical red light evaluation metrics?

## Context & Requirements

1. **Local Baseline ($\alpha = 0$)**:
   - Evaluate checkpoint with gate $\alpha$ clamped to 0.
   - Measure: $AUPRC_{rel}$, ROC-AUC, Precision, Recall, F1, calibration / reliability diagram.
   - Sliced by:
     - Arrows present vs Arrows absent: $AUPRC_{local, \text{arrow present}}$ vs $AUPRC_{local, \text{no arrow}}$.
     - Directional vs Round signals.
     - Single TL scene vs Multi-TL scenes.
     - Object size buckets.

2. **End-to-End System Evaluation (3-Tier Metrics)**:
   - **Level 1 (Oracle Relevance)**: GT TLs $\to$ relevance head.
   - **Level 2 (Detection-Conditioned Relevance)**: True Positive detected TLs $\to$ relevance head (current standard).
   - **Level 3 (End-to-End Detection + Relevance)**: Combined confidence score:
     $$s_{relevantTL} = s_{det} \cdot P(relevant)$$
     Generate PR curve over all relevant ground truth traffic lights directly.

3. **Safety-Critical Metric**:
   - Compute $Recall(\text{Relevant Red TL})$ and associated miss rate.
   - Ensure model selection does not trade away relevant red light recall for marginal composite gains.

## Empirical Resolution & Diagnostic Summary

- **Evaluated Checkpoint**: Baseline B0 on 5,962 validation images (25,344 GT Traffic Lights, 12,523 Relevant TLs, 3,686 Relevant Red TLs).
- **Learned Gate Value**: $\alpha = -0.031526$.
- **Diagnostic Script**: `scripts/audit_local_relevance_safety.py`.

### Key Empirical Findings:

1. **Local Baseline Ceiling & Contextual Gain across Granular Slices**:

| Slice Category | Slice Name | Sample Count | Local AUPRC ($\alpha=0$) | Ctx AUPRC ($\alpha_{learned}$) | $\Delta$ AUPRC | Local ECE | Local Brier |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Overall** | Validation Split | 17,603 | **89.10%** | **92.31%** | **+3.22%** | 15.49% | 0.1826 |
| **Arrow Context** | Arrows Present | 9,235 | **84.95%** | **89.45%** | **+4.50%** | 12.17% | 0.1941 |
| **Arrow Context** | No Arrows | 8,368 | **93.00%** | **94.64%** | **+1.64%** | 19.45% | 0.1700 |
| **Signal Type** | Round Signal | 13,423 | **93.42%** | **94.73%** | **+1.31%** | 21.56% | 0.1685 |
| **Signal Type** | Directional Arrow Signal | 3,464 | **56.95%** | **71.67%** | **+14.73%** | 13.08% | 0.2485 |
| **Scene Density** | Single TL Scene | 375 | **95.95%** | **97.23%** | **+1.28%** | 24.32% | 0.1384 |
| **Scene Density** | Multi-TL Scene | 17,228 | **88.80%** | **92.12%** | **+3.32%** | 15.29% | 0.1836 |
| **Area Bucket** | `<32` px² | 742 | **12.15%** | **19.59%** | **+7.44%** | 21.94% | 0.1612 |
| **Area Bucket** | `32-64` px² | 1,448 | **71.86%** | **74.54%** | **+2.68%** | 13.16% | 0.2060 |
| **Area Bucket** | `64-128` px² | 3,003 | **84.43%** | **87.91%** | **+3.48%** | 14.22% | 0.2043 |
| **Area Bucket** | `128-256` px² | 3,751 | **85.63%** | **89.84%** | **+4.21%** | 14.06% | 0.1929 |
| **Area Bucket** | `256-512` px² | 3,550 | **90.24%** | **94.19%** | **+3.95%** | 16.69% | 0.1815 |
| **Area Bucket** | `>512` px² | 5,109 | **93.07%** | **95.38%** | **+2.31%** | 16.72% | 0.1597 |

2. **3-Tier Relevance Evaluation Hierarchy**:

| Tier Level | Evaluation Description | Primary Metric | Recall on Relevant GT | Optimal Threshold |
|---|---|:---:|:---:|:---:|
| **Level 1 (Oracle)** | Features sampled directly at GT locations (Mode B) | **87.25% AUPRC** | 100.0% (Oracle) | 0.45 |
| **Level 2 (Det-Conditioned Local)** | Local head on IoU $\ge 0.50$ TP detected boxes | **89.10% AUPRC** | 88.51% (on TPs) | 0.45 |
| **Level 2 (Det-Conditioned Ctx)** | Full model on IoU $\ge 0.50$ TP detected boxes | **92.31% AUPRC** | 88.00% (on TPs) | 0.45 |
| **Level 3 (End-to-End Local)** | Combined score $s_{det} \cdot P(rel)_{local}$ on all GTs | **23.00% $AP_{50}$** | **94.81%** (overall) | — |
| **Level 3 (End-to-End Ctx)** | Combined score $s_{det} \cdot P(rel)_{ctx}$ on all GTs | **23.89% $AP_{50}$** | **94.81%** (overall) | — |

3. **Safety-Critical Relevant Red Light Waterfall & Attribution**:
- **Total Relevant Red GT Traffic Lights**: 3,686
- **Relevant Red Recall (@ threshold 0.50)**: **75.50%** (Miss Rate: **24.50%**, Success Count: 2,783)
- **Relevant Red Recall (@ threshold 0.30)**: **90.29%** (recovering 14.8% safety recall via threshold tuning)
- **Waterfall Attribution of 903 Missed Relevant Red Lights**:
  - **Stage 1 (Perception / Detection Miss)**: 205 missed (**5.56%** of total GT) due to $<64\text{ px}^2$ stride-8 localization drop.
  - **Stage 2 (Candidate Selection)**: 0 missed (**0.00%**), Top-32 candidate budget imposes zero candidate starvation on detected instances.
  - **Stage 3 (State Head Classification)**: 127 missed (**3.45%** of total GT) misclassified as non-red.
  - **Stage 4 (Relevance Head False Negatives)**: 571 missed (**15.49%** of total GT) predicted $P(rel) < 0.50$.

### Diagnostic Conclusions:
1. **Strong Local Baseline with Contextual Gain on Ambiguity**: The local relevance head is a robust baseline ($89.10\%$ AUPRC), while cross-attention provides a targeted **+14.73% AUPRC lift** on difficult directional signals where arrow context is informative.
2. **Safety-Critical Red Light Operating Point**: Standard 0.50 relevance threshold yields $75.50\%$ Relevant Red Recall; adjusting decision operating point to $0.30$ boosts safety recall to $\mathbf{90.29\%}$.

### Diagnostic Artifacts Produced:
- **Audit Script**: `scripts/audit_local_relevance_safety.py`
- **Tabular Report**: `results/audit_local_relevance_safety.md`
- **JSON Telemetry**: `results/audit_local_relevance_safety.json`
- **Visualization Plot**: `results/visualizations/w9_local_relevance_safety.png`
