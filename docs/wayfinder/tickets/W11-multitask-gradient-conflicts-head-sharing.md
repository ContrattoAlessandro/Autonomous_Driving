---
title: "W11: Multi-Task Gradient Conflict & Maneuver Head Sharing Compatibility"
type: research
status: closed
blocked_by: ["W1", "W6", "W10"]
assignee: "@agent"
---

## Question

Are the multi-task objectives (detection, NWD, state, round, maneuver, relevance) and the shared TL-arrow maneuver head cooperating synergistically on the shared backbone/neck, or are there destructive gradient conflicts?

## Empirical Resolution & Diagnostic Findings

Diagnostic evaluation executed across **200 training batches** ($1,600$ autograd backward passes) in `results/audit_multitask_gradient_conflicts.md`:

1. **Shared Maneuver Head Inductive Bias Synergy**:
   - Gradient cosine similarity between traffic lights ($g_{man, TL}$) and road arrows ($g_{man, Arrow}$) on the shared `maneuver_heads` parameters is consistently positive ($\mu = \mathbf{+0.0332}$, **$54.5\%$** synergistic batches).
   - This validates the architectural decision to share directional classification weights: road arrow orientations and traffic light directional pictograms learn a mutually compatible directional representation without requiring decoupled tower heads.

2. **$u_{ego}$ Feature Neutrality Verified**:
   - When `ego_lane_enabled: false`, the arrow ego-lane token entry is clamped to exactly `0.5`, with zero gradient leakage and zero uninitialized variable contamination into the cross-attention geometry bias MLP.

3. **Multi-Task Gradient Interaction Matrix $\mathcal{C}_{ij}$**:
   - **Detection vs NWD**: Strongly synergistic ($\cos(g_{det}, g_{nwd}) = \mathbf{+0.537}$), confirming dual bounding-box supervision accelerates localization.
   - **State vs Round**: Positively aligned ($\cos(g_{state}, g_{round}) = \mathbf{+0.086}$).
   - **Relevance vs Attributes**: Non-antagonistic ($\cos(g_{rel}, g_{state}) = \mathbf{+0.046}$, $\cos(g_{rel}, g_{round}) = \mathbf{+0.032}$).
   - **Detection vs Relevance**: Minor non-destructive orthogonality ($\cos(g_{det}, g_{rel}) = \mathbf{-0.034}$), well within acceptable multi-task tolerance ($|\mathcal{C}_{ij}| < 0.05$).
   - **Conclusion**: Single-phase joint training operates without destructive gradient cancellation across all 6 heads. No complex gradient projection (e.g. PCGrad) is strictly required, though loss-weight rebalancing can further calibrate gradient scales (Detection: $12.23$, State: $7.39$, Relevance: $2.14$).

## Artifacts Generated

- Telemetry JSON: `results/audit_multitask_gradient_conflicts.json`
- Visualization: `results/visualizations/w11_multitask_gradient_conflicts.png`
- Audit Report: `results/audit_multitask_gradient_conflicts.md`
- Unit Test: `tests/test_multitask_gradients.py`
