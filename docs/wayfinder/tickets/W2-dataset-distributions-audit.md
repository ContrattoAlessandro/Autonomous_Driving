---
title: "W2: Post-Letterbox Dataset Distributions & Prior Audit"
type: task
status: closed
blocked_by: []
assignee: "@agent"
---

## Question

What are the exact spatial and semantic ground truth distributions of the paired DTLD training and validation splits after the canonical $800 \times 1600$ letterbox transformation?

## Context & Requirements

1. **Transform Consistency**:
   - Evaluated on post-transformed bounding boxes (after scaling and padding to $800 \times 1600$).

2. **Metrics & Distributions Computed (Train vs Val)**:
   - **Object Counts**: 22,563 train images (104,103 TLs, 25,466 arrows) vs 5,962 val images (25,344 TLs, 6,062 arrows).
   - **Semantic Distributions**:
     - Relevance: Train 46.4% relevant vs 53.6% irrelevant; Val 49.4% relevant vs 50.6% irrelevant.
     - State: Green (52.2%), Red (34.8%), Off (9.5%), Yellow (3.6%).
     - Round vs Directional: Round (82.6%), Directional (17.4%).
   - **Geometry**:
     - Mean TL Area: Train = 374.3 px² | Val = 401.7 px² (Median ~146.5 px²).
     - Area Buckets: <32 px² (18.8%), 32–64 px² (11.1%), 64–128 px² (16.5%), 128–256 px² (17.8%), 256–512 px² (15.9%), >512 px² (20.0%).
     - Minimum Side: <4 px (30.6%), 4–6 px (13.6%), 6–8 px (16.2%), 8–12 px (16.9%), >12 px (22.8%).
   - **Co-occurrence & Conditional Priors**:
     - $P(rel = 1 \mid \text{arrow present}) = 43.6\%$ vs $P(rel = 1 \mid \text{no arrow}) = 49.7\%$.
     - Size Prior: $P(rel = 1 \mid \text{area} < 32\text{ px}^2) = 5.7\%$ vs $P(rel = 1 \mid \text{area} > 512\text{ px}^2) = 75.1\%$.

## Empirical Resolution & Diagnostic Artifacts

- **Audit Script**: `scripts/audit_dataset_distributions.py`
- **Tabular Report**: `results/audit_dataset_distributions.md`
- **JSON Dataset Telemetry**: `results/audit_dataset_distributions.json`
- **Conclusion**:
  - Tiny TLs (<64 px²) represent 29.88% of all instances, establishing the critical need to audit P3 stride-8 recall in W5.
  - Train and Val distributions are structurally symmetric across geometry and semantic categories.
