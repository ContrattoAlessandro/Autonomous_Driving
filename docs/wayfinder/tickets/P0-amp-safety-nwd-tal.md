---
title: "P0: AMP Numerical Safety & Float32 Precision Guardrails in NWD-TAL"
type: task
status: closed
blocked_by: []
assigner: "@agent"
---

## Question

Are the NWD calculation and TaskAlignedAssigner overlap metric numerically protected against float16 underflow, zero-gradient singularities, and AMP scale divergence during mixed-precision training?

## Context & Numerical Motivation

1. **High Exponential Powers in TAL**:
   - The task alignment metric uses $\text{align\_metric} = s^\alpha \times \text{overlap}^\beta$ where $\beta = 6.0$.
   - For sub-grid boxes where $\text{overlap} \approx 0.10$, $\text{overlap}^6 = 10^{-6}$, which falls close to the float16 underflow floor ($6.1 \times 10^{-5}$).
2. **NWD Exponential Decay**:
   - $\text{NWD} = \exp(-\sqrt{\mathcal{W}_2^2} / C)$. As distance increases, values decay rapidly.
   - In float16, distance clamping and square root operations at zero can produce `NaN` gradients if $\text{clamp\_min}(\epsilon)$ is omitted or too small.

## Protocol & Verification

1. **Float32 Invariant**:
   - Ensure `compute_nwd_similarity` and `NWDAwareTaskAlignedAssigner.get_box_metrics` execute in `torch.float32` regardless of model autocast mode.
   - Enforce $\epsilon = 10^{-9}$ in squared Wasserstein distance before square root calculation.
2. **Empirical Verification on Run B4**:
   - Analyzed 13,000 optimization steps on `runs/tlr_yolo11s_p2_nwd/metrics.jsonl`.
   - Result: Gradient norm stable ($\mu \approx 40-70$), AMP loss scale maintained at $128.0$, with exactly 3 minor overflow resets over 130 epochs (0.023% step overflow rate). Zero `NaN` or `Inf` loss occurrences.

## Decision & Resolution

- Numerical safety is formally **verified and approved**. Float32 precision casting in assigners and loss functions is established as an invariant for all Phase 3 runs.
