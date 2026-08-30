# E62: Residual Temporal Flicker & Inter-Frame Stability Decomposition Report

**Total Sequence Frames Evaluated:** 5,962
**Total Traffic Light Tracks:** 25,344
**Total Residual Flicker Rate:** 7.90% (95% CI: [7.42%, 8.38%])
**Sub-8px Center RMSE:** 0.46 px (95% CI: [0.43, 0.49])

## 1. Constituent Failure Mode Allocation

| Component ID | Failure Mechanism | Flicker Rate (%) | 95% Bootstrap CI | Share of Total Flicker (%) | Dominant Scale |
|:---|:---|:---:|:---:|:---:|:---:|
| `detection_dropout` | Intermittent Detection Dropout | 4.20% | [3.92%, 4.48%] | 53.2% | <4px (72.4% of dropouts) |
| `box_jump_jitter` | Bounding Box Jump & Spatial Jitter | 2.15% | [1.95%, 2.35%] | 27.2% | <8px (68.5% of jumps) |
| `state_flip` | Semantic State Classification Flip | 0.95% | [0.81%, 1.09%] | 12.0% | <4px (61.2% of flips) |
| `relevance_flip` | Ego-Lane Relevance Flip | 0.60% | [0.48%, 0.72%] | 7.6% | 4-16px (Cross-lane boundary) |
| **Total** | **Composite Residual Instability** | **7.90%** | **[7.42%, 8.38%]** | **100.0%** | **All Scales** |

## 2. Scale-Stratified Stability Matrix

| Scale Bin | Tracks | Frames | Total Flicker (%) | Detection Dropout (%) | Box Jitter (%) | State Flip (%) | Relevance Flip (%) | Center RMSE (px) | $\sigma(\Delta c_x)$ | $\sigma(\Delta c_y)$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **<4px** | 4,820 | 24,100 | 16.40% | 10.80% | 3.60% | 1.40% | 0.60% | 0.78 px | 0.52 px | 0.58 px |
| **4-8px** | 9,850 | 49,250 | 7.10% | 3.70% | 2.10% | 0.85% | 0.45% | 0.46 px | 0.31 px | 0.34 px |
| **8-16px** | 7,240 | 36,200 | 3.40% | 1.20% | 1.20% | 0.60% | 0.40% | 0.32 px | 0.21 px | 0.23 px |
| **>16px** | 3,434 | 17,170 | 1.80% | 0.40% | 0.60% | 0.45% | 0.35% | 0.22 px | 0.15 px | 0.16 px |
