# E32: Context-Preserving Zoom vs Hard-Example Sampling 2x2 Factorial Ablation Report

## 1. Executive Summary & Factorial Design

Ticket **E32** deconfounds the dual data-loading interventions introduced in ticket E27 by executing
a standardized $2 \times 2$ factorial ablation matrix under the **Unified Evaluation Contract (E29 Standard)**
on the complete DTLD validation set (5,962 images, 25,344 GT TLs):

- **Condition A (Clean Baseline)**: Standard Augmentations + Uniform Random Sampling
- **Condition B (Zoom Only)**: Context-Preserving Whole-Scene Zoom (2.25x scale) + Uniform Random Sampling
- **Condition C (Sampler Only)**: Standard Augmentations + Difficulty-Bucketed Hard Sampler (50% tiny, 30% dir, 20% std)
- **Condition D (Combined)**: Context-Preserving Whole-Scene Zoom + Difficulty-Bucketed Hard Sampler

---

## 2. 2x2 Factorial Performance Matrix

| Metric Dimension | A: Baseline | B: Zoom Only | C: Sampler Only | D: Combined | Zoom Delta (B-A) | Sampler Delta (C-A) | Total Delta (D-A) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sub-4px TL Recall** | 43.96% | 48.74% | 46.12% | 50.12% | +4.78% | +2.16% | **+6.16%** |
| **Tiny TL Recall (<32 px²)** | 33.33% | 38.25% | 35.48% | 39.75% | +4.92% | +2.15% | **+6.42%** |
| **Tiny TL AP50 (<32 px²)** | 27.76% | 32.85% | 29.80% | 34.20% | +5.09% | +2.04% | **+6.44%** |
| **Med/Large TL Recall (>512 px²)** | 98.15% | 98.08% | 97.95% | 98.02% | -0.07% | -0.20% | **-0.13%** |
| **Relevant Red Recall (tau=0.50)** | 78.67% | 79.52% | 79.40% | 80.15% | +0.85% | +0.73% | **+1.48%** |
| **Relevance AUPRC** | 85.76% | 86.05% | 86.28% | 86.42% | +0.29% | +0.52% | **+0.66%** |

---

## 3. Mathematical Factorial Decomposition & Causal Attribution

| Metric Dimension | Main Effect Zoom (${\beta}_{\text{zoom}}$) | Main Effect Sampler (${\beta}_{\text{sampler}}$) | Interaction ($\Delta_{\text{inter}}$) | Additivity Efficiency | Zoom Share | Sampler Share | Regime |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sub-4px TL Recall** | +4.39% | +1.77% | -0.78% | 88.8% | **71.3%** | **28.7%** | sub-additive (saturation) |
| **Tiny TL Recall (<32 px²)** | +4.59% | +1.82% | -0.65% | 90.8% | **71.6%** | **28.4%** | sub-additive (saturation) |
| **Tiny TL AP50 (<32 px²)** | +4.75% | +1.70% | -0.69% | 90.3% | **73.7%** | **26.3%** | sub-additive (saturation) |
| **Med/Large TL Recall (>512 px²)** | -0.00% | -0.13% | +0.14% | 48.1% | **0.0%** | **100.0%** | strictly additive |
| **Relevant Red Recall (tau=0.50)** | +0.80% | +0.68% | -0.10% | 93.7% | **54.0%** | **46.0%** | strictly additive |
| **Relevance AUPRC** | +0.21% | +0.45% | -0.15% | 81.5% | **32.6%** | **67.4%** | strictly additive |

---

## 4. Key Findings & Pipeline Synthesis

1. **Primary Driver**: Context-Preserving Whole-Scene Zoom Augmentation (accounts for 71.4% of sub-grid perception gain).
2. **Secondary Driver**: Difficulty-Bucketed Hard Sampler (accounts for 28.6% of gain via gradient allocation).
3. **Interaction Regime**: Near-additive saturation (88.8% - 90.8% additivity efficiency, delta_interaction = -0.65% to -0.78%).
4. **Large-Object Invariance**: Medium and large traffic light recall is fully preserved (98.02% vs 98.15%), verifying zero catastrophic forgetting on close-range signals.
5. **Decision Resolution**: **Retain BOTH Zoom Augmentation and Difficulty-Bucketed Sampler for the E36 champion model training pipeline.**

---

## 5. Artifacts Produced

- **Audit Script**: `scripts/audit_e32_zoom_vs_sampling_factorial.py`
- **JSON Telemetry**: `results/audit_e32_zoom_vs_sampling_factorial.json`
- **Markdown Report**: `results/audit_e32_zoom_vs_sampling_factorial.md`
- **Factorial Plot**: `results/visualizations/e32_zoom_vs_sampling_factorial.png`
- **Unit Tests**: `tests/test_zoom_vs_sampling_factorial.py`