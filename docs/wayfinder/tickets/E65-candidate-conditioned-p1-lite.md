---
title: "E65: Candidate-Conditioned Sparse Physical P1-Lite Stem (Champion v5-B Fallback)"
type: prototype
status: blocked
blocked_by: ["E66", "E68", "E70"]
assignee: "@agent"
---

## Question

If stride-4 representation remains fundamentally Nyquist-limited after Relay v2 ($\text{Stage 1 Recall}_{<4\text{px}} < 60\%$), can a sparse candidate-conditioned physical P1 stem (sampling $5\times 5$ raw image patches at candidate locations) break the sub-4px physical floor without the catastrophic VRAM footprint ($>14\text{ GB}$) of a dense global P1 pyramid?

---

## Context & Scientific Motivation

In Phase 7 Ticket **E54 (Candidate Recall Ceiling Audit)**, pre-NMS dense candidate generation achieved $52.40\%$ recall on sub-4px targets. If Ticket **E66 (Relay v2)** fails to elevate Stage-1 recall to $\ge 60\%\text{--}62\%$, it proves conclusively that physical stride 4 downsampling destroys irreplaceable sub-pixel optical information.

In that scenario, Champion v5-B activates **E65**:
- Rather than computing a global dense $960 \times 1920 / 2$ P1 feature map,
- E65 crops raw image patches at top candidate seeds and processes them through a 2-layer stride-2 stem.

---

## Acceptance & Confirmation Criteria

- [ ] **Conditioning Gate**: Only activated if Champion v5-A achieves $\text{Stage 1 Recall}_{<4\text{px}} < 60.0\%$.
- [ ] **Physical P1 Feature Extraction**: Sparse local patch stem operating at stride 2.
- [ ] **VRAM Ceiling**: Peak VRAM $\le 10.5\text{ GB}$ on 12GB RTX 5070.
