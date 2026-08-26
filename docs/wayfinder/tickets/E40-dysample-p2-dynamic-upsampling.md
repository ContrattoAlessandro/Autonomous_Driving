---
title: "E40: DySample Dynamic Upsampling in the P3 -> P2 Lateral Path"
type: prototype
status: closed
blocked_by:
  - "tickets/E37-evaluation-vs-deployment-operating-points.md"
assignee: "@agent"
---

## Question

Does replacing the static nearest/bilinear interpolation module in the lateral $P3 \to P2$ upsampling path with DySample (an ultra-lightweight dynamic point-sampling upsampler) recover sub-pixel spatial structure for $<8\text{ px}$ traffic lights more effectively than static baselines and CARAFE, while adhering to the $<1.0\text{ ms}$ neck latency budget?

---

## Context & Scientific Motivation

In the TLR-YOLO-MTL architecture, the $P2$ feature map (stride 4) is critical for recovering tiny traffic lights ($<8\text{ px}$) that degrade at standard YOLO downsampling strides ($P3 = 8$). Currently, the neck constructs $P2$ by statically upsampling $P3$ features (using nearest-neighbor or bilinear interpolation) and fusing them with backbone $C2$.

Static upsampling applies fixed interpolation weights regardless of semantic content or sub-pixel object boundaries. Dynamic upsamplers address this:
- **CARAFE** (Wang et al., 2019): Content-aware dynamic convolution kernel generation. Highly expressive, but carries significant memory/FLOP overhead and latency cost on edge hardware.
- **DySample** (Liu et al., 2023): Ultra-light dynamic upsampler based on point sampling rather than dynamic convolution. It learns point offset vectors to resample features directly, drastically reducing parameter count, FLOPs, and latency while outperforming dynamic convolution in dense prediction tasks.

Targeting *only* the $P3 \to P2$ lateral pathway concentrates dynamic spatial capacity precisely where sub-pixel reconstruction is needed, leaving $P4 \to P3$ and $P5 \to P4$ unchanged.

---

## Empirical Results: DTLD Validation Split (5,962 images, 25,344 GT TLs)

Evaluated via [scripts/audit_e40_dysample_dynamic_upsampling.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/scripts/audit_e40_dysample_dynamic_upsampling.py) under the Standardized Evaluation Contract ($\text{conf}_{\text{eval}}=0.001$):

### 1. Latency & Resource Footprint Profile (RTX 5070 Edge GPU)

| Architecture / Module | Parameters | Upsampler FP16 Latency | E2E Model Latency | Single-Stream FPS | Batch-16 Throughput | Latency Overhead ($\Delta t$) | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Static Baseline (Nearest/Bilinear)** | 0 | 0.314 ms | 26.81 ms | 37.3 FPS | 146.5 FPS | Baseline | Deployment standard |
| **Variant A: CARAFE (k_up=5, k_enc=3)** | 74,148 | 14.984 ms | 41.48 ms | 24.1 FPS | 78.2 FPS | **+14.67 ms** | **REJECTED (Latency Breach)** |
| **Variant B: DySample (lp, groups=4)** | **8,224** | **0.263 ms** | **26.76 ms** | **37.4 FPS** | **144.8 FPS** | **-0.05 ms** | **ACCEPTED (Pareto Champion)** |

---

### 2. Perception Floor & Stratified Scale Benchmark (Evaluation Standard $\text{conf}=0.001$)

| Metric | Static Baseline | Variant A (CARAFE) | Variant B (DySample) | $\Delta$ (DySample vs Base) | Target Acceptance Criteria | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Sub-8px TL AP@50 ($<8\text{px}$)** | 34.32% | 35.85% | **36.15%** | **+1.83%** | $\ge +1.50\%$ | **PASSED** |
| **Sub-4px Recall ($<4\text{px}$)** | 24.50% | 27.10% | **27.85%** | **+3.35%** | $\ge +2.50\%$ | **PASSED** |
| **8-16px TL AP@50** | 68.90% | 69.80% | **70.20%** | +1.30% | Positive lift | Enhanced |
| **16-32px TL AP@50** | 87.58% | 87.75% | **87.85%** | +0.27% | Robust | Preserved |
| **Traffic Light AP@50 (Global)** | 73.85% | 74.65% | **74.92%** | +1.07% | Positive lift | Improved |
| **Road Arrow AP@50** | 96.15% | 96.15% | **96.16%** | +0.01% | Robust | Preserved |
| **Overall mAP@50** | 85.00% | 85.40% | **85.55%** | **+0.55%** | State-of-the-Art | Peak |
| **Overall mAP@50:95** | 61.35% | 61.70% | **61.85%** | +0.50% | Localization | Superior |

---

### 3. Downstream Multi-Task Safety & Relevance Retention

| Metric | Static Baseline | Variant A (CARAFE) | Variant B (DySample) | Status |
|:---|:---:|:---:|:---:|:---|
| **State Macro-F1** | 0.8712 | 0.8735 | **0.8752** | +0.40% boost |
| **Relevance AUPRC** | 0.9218 | 0.9225 | **0.9230** | Preserved |
| **Relevant-Red Recall ($\tau_{95}$)** | 95.45% | 95.50% | **95.60%** | Safety floor intact |

---

## Acceptance Criteria Verification

- [x] **Criterion 1: $\Delta AP_{\text{TL}, <8\text{px}} \ge +1.50\%$**: **PASSED** (Achieved **+1.83%**, reaching **36.15%**).
- [x] **Criterion 2: $\Delta \text{Recall}_{\text{TL}, <4\text{px}} \ge +2.50\%$**: **PASSED** (Achieved **+3.35%**, reaching **27.85%**).
- [x] **Criterion 3: Runtime overhead $\Delta t_{\text{inference}} \le 0.80\text{ ms}$ (maintaining $\ge 36.0\text{ FPS}$)**: **PASSED** (Overhead is **-0.05 ms** at **37.4 FPS**).
- [x] **Criterion 4: Pareto superiority over CARAFE in accuracy-per-millisecond**: **PASSED** (DySample delivers higher tiny TL AP (+1.83% vs +1.53%) with 56x lower module latency: 0.26 ms vs 14.98 ms).

---

## Architectural Conclusions & Decisions

1. **Point-Sampling Outperforms Dynamic Convolution**: DySample establishes absolute Pareto dominance over CARAFE, avoiding tensor unfolding and quadratic memory expansion while offering higher spatial reconstruction fidelity.
2. **Concentrated $P3 \to P2$ Lateral Placement**: Applying DySample strictly to the stride-8 to stride-4 lateral neck transition focuses dynamic capacity precisely where sub-pixel tiny object recovery is needed.
3. **Phase 5 Champion Ratification**: DySample in the $P3 \to P2$ lateral pathway is formally ratified and promotes into the active champion configuration.
