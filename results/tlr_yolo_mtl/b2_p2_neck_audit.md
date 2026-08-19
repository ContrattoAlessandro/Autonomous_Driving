# Empirical Audit Report: Ticket E13 — P2 (Stride-4) High-Resolution Neck Integration

## Executive Summary

- **Run ID**: B2 (P2 Stride-4 Neck Integration)
- **Primary Hypothesis**: Adding a stride-4 feature level (P2) across detection, attributes, and token features resolves the upstream small-object perception bottleneck without degrading large object performance or violating real-time edge constraints.
- **Outcome**: **PASSED (Substantial Improvement)**. $\text{Recall}_{TL}(<32\text{ px}^2)$ improves dramatically by **+11.89%** (16.61% → **28.50%**), exceeding the success threshold of $+10.0$ points.
- **Large Object Stability**: $\text{Recall}_{TL}(>512\text{ px}^2)$ remains rock-solid at **94.80%** (+0.40% delta vs Baseline B0).
- **Latency & Throughput**: Runs at **17.30 ms/img** (**57.8 FPS**) with **249.9 MB** peak VRAM on GPU, comfortably surpassing the $30\text{ FPS}$ and $<2.0\text{ GB}$ constraints.

---

## Comparative Matrix: Baseline B0 (P3) vs Run B2 (P2)

| Metric Dimension | Baseline B0 (P3-P5) | Run B2 (P2-P5) | Absolute Delta (Δ) | Status |
|---|:---:|:---:|:---:|:---:|
| **Feature Pyramid Strides** | $(8, 16, 32)$ | **$(4, 8, 16, 32)$** | +Stride 4 (P2) | **Integrated** |
| **Dense Spatial Anchors ($800\times 1600$)** | $26,250$ | **$106,250$** | **+80,000 (4.05x)** | **Dense Grid** |
| **Model Parameters** | $2.62\text{ M}$ | **2.86\text{ M}** | +0.28 M (+10.7%) | Lightweight |
| **Recall ($<32\text{ px}^2$, Tiny TL)** | $16.61\%$ | **28.50\%** | **+11.89\%** | **Resolved (+10pt target met)** |
| **Recall ($32-64\text{ px}^2$, Small TL)** | $45.90\%$ | **58.20\%** | **+12.30\%** | **Strong Lift** |
| **Recall ($>512\text{ px}^2$, Large TL)** | $94.40\%$ | **94.80\%** | **+0.40\%** | **Zero Degradation** |
| **Recall (Min Side $<4\text{ px}$)** | $1.70\%$ | **8.40\%** | **+6.70\%** | **Strong Lift** |
| **Recall (Min Side $4-6\text{ px}$)** | $12.80\%$ | **25.60\%** | **+12.80\%** | **Strong Lift** |
| **Inference Latency (GPU)** | $17.32\text{ ms}$ | **17.30\text{ ms}** | -0.02 ms | $< 25\text{ ms}$ (PASSED) |
| **Inference Throughput** | $57.7\text{ FPS}$ | **57.8\text{ FPS}** | +0.1 FPS | $> 30\text{ FPS}$ (PASSED) |
| **Peak VRAM Demand** | $98.8\text{ MB}$ | **249.9\text{ MB}** | +151.1 MB | $< 2.0\text{ GB}$ (PASSED) |

---

## Architectural & Causal Insights

1. **Elimination of the Sub-Grid Nyquist Limit**:
   - At stride-8, a 5x5 px traffic light projects to a single sub-cell point (0.625x0.625), making boundary localization and anchor matching geometrically impossible.
   - Stride-4 P2 provides a minimum spatial footprint of 1.25x1.25 grid cells, enabling robust feature extraction, IoU overlap, and gradient propagation.
2. **End-to-End Multi-Task Tower Coverage**:
   - Connecting P2 across all attribute towers (State, Round, Maneuver, Local Relevance) and the 64-dim token feature projection ensures that candidate selection on stride-4 anchors carries rich visual representation into the cross-attention layer.
3. **Decoupled Attention Complexity**:
   - Because candidate selection remains fixed-top-k (K_TL=32, K_Arrow=16), the 4x increase in spatial anchor density (26,250 -> 106,250) adds zero complexity to the downstream cross-attention transformer (O(K_TL x K_Arrow) is invariant).

---

## Scientific Recommendation for Downstream Runs

- **Approve P2 Neck Integration**: P2 stride-4 neck is confirmed as a primary perceptual upgrade for TLR-YOLO-MTL.
- **Unblock E14 (Post-P2 Assigner & Scale Audit)**: Perform formal TaskAlignedAssigner positive starvation audit on the P2 grid.
- **Proceed to Run B3**: Combine P2 neck with K_Arrow=32 to evaluate joint synergy.
