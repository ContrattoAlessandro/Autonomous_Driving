---
title: "E53: Scale-Conditioned Confidence Calibration & Safety Waterfall Audit"
type: task
status: open
blocked_by:
  - "tickets/E47-cumulative-champion-v3-integration-lineage-audit.md"
assignee: "@agent"
---

## Question

How does stratifying post-hoc confidence calibration (temperature scaling and vector scaling) across object bounding box scale regimes ($T_{<8\text{px}}, T_{8\text{--}16\text{px}}, T_{>16\text{px}}$) and semantic categories affect Expected Calibration Error (ECE), Brier score, Negative Log-Likelihood (NLL), and the operational safety waterfall for ego-lane relevant-red stop decisions?

---

## Context & Scientific Motivation

In Phase 1/2 (Ticket E19), post-hoc temperature scaling was introduced to calibrate multi-task relevance and state probabilities globally. However, in autonomous driving perception, prediction uncertainty is strongly scale-dependent:
- A $4\times4\text{ px}$ distant traffic light inherently carries higher aleatoric and epistemic uncertainty than a $40\times40\text{ px}$ gantry signal.
- Applying a single global temperature scalar $T_{\text{global}}$ over-calibrates high-resolution confident signals or under-calibrates low-resolution distant signals.

For safety-critical AD stack integration (e.g. planner braking policies), a confidence score of $0.92$ on a $<8\text{ px}$ light must represent the same empirical accuracy as a $0.92$ on a $>32\text{ px}$ light.

### Scale-Conditioned Calibration Formulation

We parameterize post-hoc temperature scaling conditioned on the predicted bounding box area:
$$z_i^{\text{cal}} = \frac{z_i}{T(\text{area}_i)}, \quad T(\text{area}) = \begin{cases} T_{<8\text{px}} & \text{if } \text{area} < 64\text{ px}^2 \\ T_{8\text{-}16\text{px}} & \text{if } 64\le \text{area} < 256\text{ px}^2 \\ T_{>16\text{px}} & \text{if } \text{area} \ge 256\text{ px}^2 \end{cases}$$

Optimization minimizes Negative Log-Likelihood (NLL) on a held-out calibration split:
$$\mathcal{L}_{\text{cal}} = -\sum_{i=1}^{N} \log \sigma\left( \frac{z_i}{T(\text{area}_i)} \right)_{y_i}$$

Evaluation measures:
1. **Expected Calibration Error (ECE)**: $\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$.
2. **Brier Score**: $\text{BS} = \frac{1}{N} \sum_{i=1}^N \|p_i - y_i\|_2^2$.
3. **Safety Waterfall**: Relevant-Red Stop Recall at $\tau_{95} \ge 96.5\%$.

Because calibration applies a simple scalar lookup during post-processing, runtime inference overhead is **$0.00\text{ ms}$**.

---

## Experimental Protocol & Implementation Plan

1. **Calibration Split & Stratification Harness**:
   - Construct scale-stratified calibration loader from validation splits.
2. **Optimization**:
   - Fit scale-conditioned temperature parameters ($T_{<8}, T_{8\text{-}16}, T_{>16}$) and category-specific vector scaling for State and Relevance heads.
3. **Safety Waterfall Verification**:
   - Verify that operating thresholds $\tau_{\text{conf}}=0.25$ and safety threshold $\tau_{95}$ maintain $>96.5\%$ Relevant-Red Recall while eliminating over-confident false alarms.

---

## Acceptance & Confirmation Criteria

- [ ] **Criterion 1: ECE Reduction Across All Scales**:
  - Global ECE reduced by $\ge 40\%$ relative ($\text{ECE} \le 3.5\%$).
  - Sub-8px ECE reduced from $>12.0\%$ to $\le 5.0\%$.
- [ ] **Criterion 2: Negative Log-Likelihood & Brier Score**: Significant reduction in NLL and Brier score on validation set.
- [ ] **Criterion 3: Safety Floor Guarantee**: Relevant-Red Recall @ $\tau_{95} \ge 96.50\%$ strictly preserved.
- [ ] **Criterion 4: Deployment Latency**: Zero runtime overhead ($\Delta t = 0.00\text{ ms}$).
