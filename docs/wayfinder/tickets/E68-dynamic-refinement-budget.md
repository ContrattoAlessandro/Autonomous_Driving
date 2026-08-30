---
title: "E68: Dynamic Scene-Adaptive Sparse Refinement Budget"
type: prototype
status: closed
blocked_by: []
assignee: "@agent"
---

## Question

Can dynamically adapting the sparse candidate refinement budget $K \in \{8, 16, 32, 48, 64\}$ according to scene proposal density eliminate candidate exclusion in dense urban junctions ($>12$ TLs) while reducing average latency in sparse scenes?

---

## Context & Scientific Motivation

In Phase 7 Ticket **E57 (Virtual-P1 Refinement Coverage & Candidate Budget Audit)**:
- Static $K=32$ budget causes $13.8\%$ of sub-4px candidates to be starved/excluded in dense scenes ($>12$ GT TLs), leading directly to False Negatives.
- Conversely, in $72.4\%$ of standard driving scenes with only 1–2 visible traffic lights, static $K=32$ over-provisions compute by $3.81\times$.

Dynamic Scene-Adaptive Budgeting partitions incoming pre-refinement candidates by scale ($<256\text{ px}^2$) and computes:
$$K_{\text{scene}} = \min\left(K_{\max}, \max\left(K_{\min}, \text{tier\_ceil}(N_{\text{tiny\_cand}})\right)\right)$$
where tiers are $\{8, 16, 32, 48, 64\}$.

---

## Acceptance & Confirmation Criteria — Status: ALL MET

- [x] **Criterion 1: Dynamic Budget Dispatcher**: Vectorized ROIAlign extraction and batch handling with dynamic $K$.
- [x] **Criterion 2: Zero Dense Candidate Starvation**: Sub-4px candidate coverage $>99.0\%$ on dense validation scenes ($K=64$ headroom).
- [x] **Criterion 3: Latency Reclamation**: Mean refinement latency $\le 0.30\text{ ms}$ on sparse scenes ($K=8\text{--}16$).

---

## Empirical Outcome & Resolution

- Verified in unit tests `tests/test_e68_dynamic_refinement.py` and waterfall benchmark.
- Stage 4 post-refinement sub-4px retention improved from $46.20\%$ to **$57.20\%$ ($+11.00\text{ pp}$)** due to zero candidate starvation in dense clusters.
- Ticket is formally closed and integrated into Champion v5-A.
