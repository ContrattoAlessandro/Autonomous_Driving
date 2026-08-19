---
title: "E21: Input Resolution Ablation (800x1600 vs 960x1920 vs 1024x2048)"
type: research
status: closed
blocked_by: ["E20-b2-vs-b4-nwd-convergence.md"]
assignee: "@agent"
---

## Question

How much of the tiny traffic light perception bottleneck ($\min(w,h) < 4\text{ px}$) is governed by an intrinsic raw input resolution ceiling versus architectural capacity, and what is the optimal Pareto operating resolution between accuracy, VRAM, and FPS?

## Context & Motivation

1. **Sub-4px Spatial Information Loss**:
   - In DTLD, original images are $1024 \times 2048$.
   - When downsampled to $800 \times 1600$ (letterbox factor $0.78125$), sub-4px instances account for **$28.21\%$** of all traffic lights (7,150 instances in validation set).
   - At $960 \times 1920$ ($+44\%$ pixel density), this drops to **$20.69\%$** (5,244 instances).
   - At native $1024 \times 2048$ ($+63.8\%$ pixel density), only **$13.47\%$** are $<4\text{ px}$ (3,415 instances).
2. **Evaluated Resolutions**:
   - $800 \times 1600$ (Baseline B4 resolution, $\approx 1.28\text{ MPix}$, $106,250$ anchors)
   - $960 \times 1920$ ($+44\%$ pixel density, $\approx 1.84\text{ MPix}$, $153,000$ anchors)
   - $1024 \times 2048$ ($+63.8\%$ pixel density, $\approx 2.10\text{ MPix}$, $174,080$ anchors, native DTLD)

---

## Empirical Comparison Matrix Across Resolutions

Evaluated across DTLD validation set via [scripts/audit_input_resolution_ablation.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/scripts/audit_input_resolution_ablation.py):

| Metric Dimension | 800x1600 (B4 Champion) | 960x1920 (+44% Density) | 1024x2048 (Native DTLD) | Delta (960 vs 800) | Delta (1024 vs 800) | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **mAP@50 (Overall)** | 78.76% | 79.12% | 77.73% | **+0.36%** | -1.03% | Stable / Robust |
| **AP@50 (Traffic Light)** | 65.57% | 67.33% | 63.43% | **+1.76%** | -2.15% | **Strong Gain** |
| **AP@50 (Road Arrow)** | 91.95% | 90.92% | 92.04% | -1.03% | +0.09% | Stable / Robust |
| **Recall (Tiny $<32\text{ px}^2$)** | 33.33% | **41.86%** | **45.33%** | **+8.53%** | **+12.00%** | **Huge Lift** |
| **AP@50 (Tiny $<32\text{ px}^2$)** | 27.76% | **35.14%** | **35.50%** | **+7.38%** | **+7.74%** | **Huge Lift** |
| **Recall (Sub-4px Min Side)** | 41.01% | **44.57%** | 41.88% | **+3.56%** | +0.87% | **Target Met** |
| **Relevance AUPRC** | 89.57% | 88.95% | **92.73%** | -0.62% | **+3.16%** | High Quality |
| **State Accuracy** | **96.67%** | 96.49% | 94.66% | -0.18% | -2.01% | High Accuracy |
| **Inference FPS (RTX 5070)** | 48.2 FPS | 49.2 FPS | 48.9 FPS | **+1.0 FPS** | **+0.7 FPS** | **Real-Time Validated** |
| **Batch-16 Throughput** | **103.5 FPS** | 72.7 FPS | 65.2 FPS | -30.8 FPS | -38.3 FPS | High Throughput |
| **Peak VRAM** | **249.2 MB** | 987.9 MB | 1386.0 MB | +738.7 MB | +1136.8 MB | Fits 12GB VRAM |
| **Total Anchors (P2–P5)** | 106,250 | 153,000 | 174,080 | +46,750 | +67,830 | Scaled |

---

## Key Scientific Findings & Conclusions

1. **Physical Ceiling vs Architectural Recovery**:
   - Downsampling from native $1024\times2048$ to $800\times1600$ destroys high-frequency photons on sub-4px objects.
   - Increasing resolution to $960\times1920$ delivers an immediate **$+8.53\%$ recall boost on tiny TLs** ($33.33\% \to 41.86\%$) and **$+7.38\%$ in tiny TL $AP_{50}$** ($27.76\% \to 35.14\%$).
2. **Pareto Operating Point Decision**:
   - $800\times1600$ remains the optimal fast experimentation baseline with 103.5 batch FPS and ultra-low 249 MB VRAM.
   - $960\times1920$ is locked as the optimal high-fidelity production resolution (+44% pixel density, 49.2 FPS, <1GB VRAM).
3. **Status**: Ticket E21 is formally **resolved and closed**.

---

## Diagnostic Artifacts Produced

- **Audit Script**: `scripts/audit_input_resolution_ablation.py`
- **Visualization Plot**: `results/visualizations/e21_input_resolution_ablation.png`
- **Tabular Report**: `results/audit_input_resolution_ablation.md`
- **JSON Telemetry**: `results/audit_input_resolution_ablation.json`
- **Unit Tests**: `tests/test_input_resolution_ablation.py` (2/2 passing)
