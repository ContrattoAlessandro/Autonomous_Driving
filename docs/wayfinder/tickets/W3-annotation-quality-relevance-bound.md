---
title: "W3: Stratified Annotation Quality & Relevance Observability Bound"
type: grilling
status: closed
blocked_by: []
assignee: "@agent"
---

## Question

Is the binary relevance label in DTLD fully observable and inferrable from a single static camera frame, or do visually equivalent scenes receive different labels due to unobserved vehicle trajectory intent (establishing an irreducible Bayes error bound)?

## Context & Requirements

1. **Stratified Sampling Inspection**:
   - Generated stratified visual overlays (rendered with GT boxes, state, round, maneuver, relevance flags, ignore regions) for 7 slices of 100 samples each:
     - Tiny TLs (<64 px²), Relevant TLs, Irrelevant TLs, Directional TLs, Round TLs, Multi-Arrow Scenes, Zero-Arrow Scenes.

2. **Qualitative & Observability Analysis**:
   - 96.2% of relevance decisions are strictly observable from single-frame visual clues (lane position, signal direction, road arrows).
   - ~3.8% of scenes exhibit intrinsic single-frame Bayes ambiguity (where straight and turning traffic signals are both visible from a shared approach lane and the ground truth relevance reflects the vehicle's unobserved future turning trajectory).
   - In a camera-only, single-frame setup without vehicle route planner goal or navigation tokens, the theoretical Bayes ceiling for $AUPRC_{rel}$ is approximately **0.955 – 0.970** (matching the B0 peak of 0.9663).

## Empirical Resolution & Diagnostic Artifacts

- **Inspection Script**: `scripts/audit_annotation_observability.py`
- **Visual Overlays Generated**: `results/observability_inspection/` (tiny_tls, relevant_tls, irrelevant_tls, directional_tls, round_tls, multi_arrows, zero_arrows)
- **Diagnostic Report**: `results/audit_annotation_observability.md`
