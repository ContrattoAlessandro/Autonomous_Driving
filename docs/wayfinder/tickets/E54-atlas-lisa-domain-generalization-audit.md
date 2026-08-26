---
title: "E54: Cross-Dataset Domain Generalization Audit (ATLAS & LISA)"
type: research
status: open
blocked_by:
  - "tickets/E47-cumulative-champion-v3-integration-lineage-audit.md"
  - "tickets/E48-local-view-tiny-tl-distillation.md"
  - "tickets/E53-scale-conditioned-calibration.md"
assignee: "@agent"
---

## Question

Does the cumulative Champion v3 architecture (and downstream Phase 6 variants with Local-View Distillation and Scale Calibration) transfer zero-shot to external autonomous driving datasets (ATLAS and LISA benchmarks) without catastrophic degradation on tiny instances ($<8\text{ px}$), proving that our tiny-object inductive biases represent general physical principles rather than dataset-specific overfitting to DTLD capture optics?

---

## Context & Scientific Motivation

Throughout Phases 1 through 5, architectural improvements and hyperparameters have been rigorously evaluated on the **DTLD** benchmark (DriveU Traffic Light Dataset, Germany). DTLD provides high-resolution 2-megapixel daylight and dusk urban driving sequences with high-quality traffic light annotations.

However, a critical scientific vulnerability in machine learning thesis research is **dataset-specific over-optimization**:
- If our structural modifications (e.g. DySample $P2$, NWD-TAL, Scale-Matched Sampling, Geometry-Aware Attention) exploit specific lens distortion, camera mounting heights, or German traffic light geometries, they may fail when deployed on US driving data (LISA) or multi-city diverse benchmarks (ATLAS).

To establish scientific validity and publication readiness, Ticket E54 defines a rigorous **Zero-Shot Domain Generalization Matrix**:

### Domain Generalization Evaluation Matrix

| Evaluated Architecture | DTLD (In-Domain) | ATLAS (Cross-Domain City) | LISA (Cross-Domain US) | Generalization Retention |
|:---|:---:|:---:|:---:|:---:|
| **Champion v1 Baseline (E36/E37)** | Ref | Eval | Eval | Baseline Transfer |
| **Champion v3 (E47 Cumulative)** | Target | Eval | Eval | Multi-Component Transfer |
| **Champion v3 + E48 Local-View KD** | Target | Eval | Eval | Distillation Invariance |
| **Champion v3 + E53 Scale-Calibrated** | Target | Eval | Eval | Uncertainty Calibration |

### Evaluated Metrics Across Domains
1. Global Traffic Light $AP@50$ and $mAP@50:95$
2. Scale-Stratified AP ($<8\text{ px}, 8\text{--}16\text{ px}, >16\text{ px}$)
3. Multi-Class State Accuracy / Macro-F1 (Red, Yellow, Green, Off)
4. Domain-Specific Environmental Stratification (Daylight vs Dusk/Night, Urban vs Highway)

---

## Experimental Protocol & Implementation Plan

1. **Dataset Harmonization**:
   - Utilize existing conversion harnesses [scripts/convert_atlas.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/scripts/convert_atlas.py) and [scripts/convert_lisa.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/scripts/convert_lisa.py) with label mapping to the standardized multi-task taxonomy.
2. **Zero-Shot Evaluation Pipeline**:
   - Run inference across ATLAS validation split and LISA daytime/nighttime splits without fine-tuning weights.
3. **Generalization Gap Analysis**:
   - Compute the relative retention ratio: $\mathcal{R}_{\text{domain}} = \frac{AP_{\text{cross}}}{AP_{\text{in-domain}}}$.
   - Identify whether tiny-object gains ($+15\text{--}20\%$ sub-8px AP) transfer proportionally to external datasets.

---

## Acceptance & Confirmation Criteria

- [ ] **Criterion 1: Zero-Shot Generalization Retention**: $\ge 80\%$ relative retention of $AP@50$ and $\ge 75\%$ retention of sub-8px AP gain on ATLAS/LISA compared to DTLD in-domain metrics.
- [ ] **Criterion 2: Superiority Over Champion v1**: Champion v3 must strictly outperform Champion v1 across all three domains (DTLD, ATLAS, LISA) without domain-specific re-tuning.
- [ ] **Criterion 3: State Recognition Robustness**: State Macro-F1 $\ge 85.0\%$ on ATLAS and $\ge 88.0\%$ on LISA under zero-shot transfer.
- [ ] **Criterion 4: Comprehensive Thesis Artifact**: Publication-ready generalization breakdown table and cross-dataset visual comparison galleries.
