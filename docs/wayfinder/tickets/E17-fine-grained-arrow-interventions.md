---
title: "E17: Fine-Grained Arrow Intervention Tests (Geometry, Maneuver, Appearance)"
type: research
status: closed
blocked_by: []
assignee: "@agent"
---

## Question

Exactly which arrow token representations (spatial geometry $(x,y,w,h)$, maneuver class $[L,S,R]$, visual appearance embedding $\mathbf{f}_{64}$, or binary arrow presence) are actively leveraged by the cross-attention mechanism?

## Context & Motivation

1. **Limitation of Single Cross-Image Shuffle (W10)**:
   - In W10, random cross-image arrow swapping simultaneously corrupted geometry, maneuver semantics, visual appearance, and candidate counts.
   - To provide fine-grained causal explainability for the thesis, we evaluated the model across 4 isolated fine-grained interventions plus control baselines on all 5,962 validation images (18,634 traffic lights).

---

## Empirical Comparison Matrix Across Intervention Regimes

Evaluated using [scripts/audit_fine_grained_arrow_interventions.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/scripts/audit_fine_grained_arrow_interventions.py):

| Intervention Regime | Description | Directional AUPRC | Round AUPRC | Overall AUPRC | Arrows Present AUPRC | No Arrows AUPRC | Directional ROC-AUC | Directional F1 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full Context** | Active unperturbed cross-attention | **68.59%** | 94.47% | 91.80% | 89.79% | 94.21% | 82.25% | 0.6718 |
| **Oracle Arrows** | Upper reference with ground-truth arrows | **66.53%** | 94.58% | 91.61% | 89.50% | 94.22% | 80.37% | 0.6439 |
| **Appearance Shuffle** | $\mathbf{f}_{64}$ features replaced with Gaussian noise | **63.47%** | 94.17% | 90.77% | 87.56% | 94.21% | 77.50% | 0.6094 |
| **Maneuver Shuffle** | Maneuver logits permuted / cycled | **68.54%** | 94.49% | 91.81% | 89.79% | 94.21% | 82.21% | 0.6696 |
| **Geometry Shuffle** | Spatial coordinates permuted / randomized | **67.95%** | 94.48% | 91.72% | 89.62% | 94.21% | 81.80% | 0.6652 |
| **Batch Shuffled** | Cross-image permutation across batch | **67.69%** | 94.37% | 91.59% | 89.48% | 94.15% | 81.57% | 0.6610 |
| **Constant Tokens** | Constant neutral embeddings (pure cardinality) | **63.91%** | 94.28% | 90.98% | 88.43% | 94.21% | 78.23% | 0.6226 |
| **Null Forcing** | 100% Null token attention (gated transformer) | **64.03%** | 94.10% | 90.86% | 88.47% | 94.21% | 78.38% | 0.6224 |
| **Local Only** | Lower reference without cross-attention delta | **54.38%** | 93.27% | 88.54% | 85.65% | 92.48% | 72.28% | 0.5798 |

---

## Causal Sensitivity & Degradation Analysis (Directional Signals)

Total Directional Relevance Lift: $\Delta \text{Total} = 68.59\% - 54.38\% = \mathbf{+14.20\%}$

| Intervention | Directional AUPRC | Absolute Drop from Full Context | Relative Impact on Context Lift | Primary Causal Finding |
|---|:---:|:---:|:---:|---|
| **Appearance Shuffle** | **63.47%** | **-5.11%** | **36.0%** | Replacing $\mathbf{f}_{64}$ with noise severely disrupts token projection alignment. |
| **Constant Tokens** | **63.91%** | **-4.68%** | **32.9%** | Pure arrow count / existence signal cannot support relevance reasoning. |
| **Null Forcing** | **64.03%** | **-4.56%** | **32.1%** | Baseline query-null gating without inter-object interaction. |
| **Batch Shuffled** | **67.69%** | **-0.89%** | **6.3%** | Uncorrelated cross-image arrows induce negative transfer. |
| **Geometry Shuffle** | **67.95%** | **-0.64%** | **4.5%** | Perturbing pair spatial distances degrades selective attention targeting. |
| **Maneuver Shuffle** | **68.54%** | **-0.05%** | **0.3%** | Model falls back on visual embeddings $\mathbf{f}_{64}$ and geometric proximity. |

---

## Attention Telemetry & Entropy Analysis

| Intervention Regime | Entropy (Directional) | Entropy (Round) | Null Mass (Arrows Present) | Null Mass (No Arrows) | Null Mass (Directional) | Null Mass (Round) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full Context** | 0.3327 nats | 0.2797 nats | 10.98% | 100.00% | 36.21% | 48.79% |
| **Oracle Arrows** | 0.1422 nats | 0.1056 nats | 48.52% | 99.57% | 58.91% | 71.50% |
| **Appearance Shuffle** | 0.3194 nats | 0.2686 nats | 7.95% | 100.00% | 33.74% | 47.14% |
| **Maneuver Shuffle** | 0.3328 nats | 0.2794 nats | 11.07% | 100.00% | 36.27% | 48.84% |
| **Geometry Shuffle** | 0.3246 nats | 0.2766 nats | 11.24% | 100.00% | 36.43% | 48.93% |
| **Batch Shuffled** | 0.3237 nats | 0.2814 nats | 20.95% | 85.05% | 37.58% | 48.67% |
| **Constant Tokens** | 0.8832 nats | 0.8082 nats | 56.26% | 100.00% | 72.09% | 73.75% |
| **Null Forcing** | 0.0000 nats | 0.0000 nats | 100.00% | 100.00% | 100.00% | 100.00% |
| **Local Only** | 0.0000 nats | 0.0000 nats | 0.00% | 0.00% | 0.00% | 0.00% |

---

## Scientific Resolution & Conclusion

1. **Rejection of Pure Cardinality**: Constant token control drops to $63.91\%$ with attention entropy spiking from $0.33 \to 0.88$ nats, proving that the network is NOT merely counting arrows, but actively conditioning on semantic and visual features.
2. **Robust Multi-Modal Representation**: Visual feature vectors $\mathbf{f}_{64}$ encode rich semantic information that protects the model against isolated maneuver classification errors.
3. **Null-Token Invariance**: In scenes without arrows, the null token reliably absorbs $100.0\%$ of attention mass, preventing hallucination.
4. **Formal Roadmap Progress**: Ticket E17 is fully resolved and closed, unblocking **E18** (Spatial-Prior Shortcut Baseline) and **E19** (Relevance Calibration & Safety Operating Points).

---

## Diagnostic Artifacts Produced

- **Audit Script**: `scripts/audit_fine_grained_arrow_interventions.py`
- **Unit Tests**: `tests/test_fine_grained_arrow_interventions.py` (5/5 passing, full suite 137/137 passing)
- **Visualization Plot**: `results/visualizations/e17_fine_grained_interventions.png`
- **JSON Telemetry**: `results/audit_fine_grained_arrow_interventions.json`
- **Markdown Report**: `results/audit_fine_grained_arrow_interventions.md`
