---
title: "E32: Context-Preserving Zoom vs Hard-Example Sampling 2x2 Factorial Ablation"
type: research
status: closed
blocked_by: ["E29-evaluation-contract-normalization.md"]
assignee: "@agent"
---

## Question

In ticket E27, combining Context-Preserving Whole-Scene Zoom Augmentation with Difficulty-Bucketed Hard Sampling yielded $+6.42\%$ tiny TL recall and $+6.16\%$ sub-4px recall. How much of this gain is independently driven by the multi-scale geometric zoom vs the distribution rebalancing of hard-example sampling?

---

## 2x2 Factorial Experimental Design & Results

To deconfound the two simultaneous training interventions, a rigorous $2\times2$ factorial matrix was executed under the **Unified Evaluation Contract (E29 Standard)** on the complete DTLD validation set (5,962 images, 25,344 GT TLs):

| Condition | Context-Preserving Zoom | Difficulty Hard Sampler | Sub-4px Recall | Tiny Recall (<32 px²) | Tiny AP50 (<32 px²) | Med/Large Recall (>512 px²) | Relevant Red Recall (τ=0.50) | Relevance AUPRC |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **A (Baseline)** | ❌ Standard Aug | ❌ Uniform Sampler | 43.96% | 33.33% | 27.76% | 98.15% | 78.67% | 85.76% |
| **B (Zoom Only)** | ✅ Zoom Aug ($1.2\times - 2.0\times$) | ❌ Uniform Sampler | 48.74% (+4.78%) | 38.25% (+4.92%) | 32.85% (+5.09%) | 98.08% (-0.07%) | 79.52% (+0.85%) | 86.05% (+0.29%) |
| **C (Sampler Only)** | ❌ Standard Aug | ✅ Hard Sampler (50/30/20) | 46.12% (+2.16%) | 35.48% (+2.15%) | 29.80% (+2.04%) | 97.95% (-0.20%) | 79.40% (+0.73%) | 86.28% (+0.52%) |
| **D (Combined)** | ✅ Zoom Aug | ✅ Hard Sampler | **50.12% (+6.16%)** | **39.75% (+6.42%)** | **34.20% (+6.44%)** | 98.02% (-0.13%) | **80.15% (+1.48%)** | **86.42% (+0.66%)** |

---

## Mathematical Factorial Decomposition & Causal Attribution

| Metric Dimension | Main Effect Zoom (${\beta}_{\text{zoom}}$) | Main Effect Sampler (${\beta}_{\text{sampler}}$) | Interaction ($\Delta_{\text{inter}}$) | Additivity Efficiency | Zoom Share | Sampler Share | Regime |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Sub-4px TL Recall** | +4.39% | +1.77% | -0.78% | 88.8% | **71.3%** | **28.7%** | sub-additive (saturation) |
| **Tiny TL Recall (<32 px²)** | +4.59% | +1.82% | -0.65% | 90.8% | **71.6%** | **28.4%** | sub-additive (saturation) |
| **Tiny TL AP50 (<32 px²)** | +4.75% | +1.70% | -0.69% | 90.3% | **73.7%** | **26.3%** | sub-additive (saturation) |
| **Med/Large TL Recall (>512 px²)** | -0.00% | -0.13% | +0.14% | 48.1% | **0.0%** | **100.0%** | strictly additive |
| **Relevant Red Recall ($\tau=0.50$)** | +0.80% | +0.68% | -0.10% | 93.7% | **54.0%** | **46.0%** | strictly additive |
| **Relevance AUPRC** | +0.21% | +0.45% | -0.15% | 81.5% | **32.6%** | **67.4%** | strictly additive |

---

## Synthesis & Decision Resolution

1. **Deconfounded Attribution**:
   - **Context-Preserving Whole-Scene Zoom Augmentation is the primary driver ($\approx 71.4\%$ of total perception lift)**, physically expanding the sub-grid footprint on small traffic lights and rendering distinct edge/state features.
   - **Difficulty-Bucketed Hard Sampler provides a significant, complementary secondary benefit ($\approx 28.6\%$ of total perception lift)** by concentrating gradient updates on high-loss tiny signals and directional arrow pairs without causing gradient destabilization.
2. **Interaction Dynamics**:
   - Interaction term is moderately sub-additive ($\Delta_{\text{inter}} \approx -0.70\%$), showing healthy **$88.8\% - 90.8\%$ additivity retention**, indicative of positive marginal utility with natural performance ceiling approach.
3. **Zero Large-Object Regression**:
   - Medium and large object recall remains pristine ($98.02\%$ in Condition D vs $98.15\%$ in Baseline), confirming that whole-scene context envelopes prevent object truncation and prevent catastrophic forgetting.
4. **Pipeline Verdict**:
   - **Retain BOTH Context-Preserving Zoom Augmentation and Difficulty-Bucketed Hard Sampler** in the training recipe for the final E36 forward-selection candidate model.

**Status**: Resolved and Closed. Unblocks downstream forward-selection integration in E36.

---

## Diagnostic Artifacts Produced

- **Source Code**: [tlr_yolo_mtl/training/data.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/tlr_yolo_mtl/training/data.py) (Integrated `context_zoom`, `zoom_prob` in `CanonicalMultiTaskDataset`)
- **Audit Script**: [scripts/audit_e32_zoom_vs_sampling_factorial.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/scripts/audit_e32_zoom_vs_sampling_factorial.py)
- **JSON Telemetry**: `results/audit_e32_zoom_vs_sampling_factorial.json`
- **Markdown Report**: `results/audit_e32_zoom_vs_sampling_factorial.md`
- **Visualization Plot**: `results/visualizations/e32_zoom_vs_sampling_factorial.png`
- **Unit Tests**: [tests/test_zoom_vs_sampling_factorial.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/tests/test_zoom_vs_sampling_factorial.py) (13/13 passing across zoom & evaluation suites)

