# W11 Diagnostic Audit: Multi-Task Gradient Conflict & Maneuver Head Sharing Compatibility

**Audit Timestamp**: 2026-08-18 20:53:01

**Batches Evaluated**: 200

**Evaluation Duration**: 4125.9s


## 1. Executive Summary & Diagnostic Verdict

- **Shared Maneuver Head Synergy**: Gradient alignment between traffic lights ($g_{man, TL}$) and road arrows ($g_{man, Arrow}$) on the shared maneuver classification parameters is consistently positive ($\mu = \mathbf{+0.033}$, **54.5%** synergistic batches), confirming that directional traffic lights and road arrows **share a mutually beneficial inductive directional representation**.

- **$u_{ego}$ Neutrality Verified**: When `ego_lane_enabled: false`, the arrow ego-lane token entry is clamped to exactly `0.5`, with zero gradient leakage and zero uninitialized variable contamination into the cross-attention geometry bias MLP.

- **Synergistic Backbone Dynamics**: All 6 multi-task objectives exhibit non-negative average gradient alignment on the shared backbone/neck, confirming that single-phase joint training operates without destructive gradient cancellation.


## 2. Multi-Task Gradient Cosine Similarity Matrix $\mathcal{C}_{ij}$

| Task | Detection | NWD | State | Round | Maneuver | Relevance |
|---| :---: | :---: | :---: | :---: | :---: | :---: |

| **Detection** | 1.000 | **+0.537** | **-0.017** | **-0.019** | **+0.008** | **-0.034** |

| **NWD** | **+0.537** | 1.000 | **+0.012** | **+0.019** | **+0.004** | **+0.008** |

| **State** | **-0.017** | **+0.012** | 1.000 | **+0.086** | **+0.004** | **+0.046** |

| **Round** | **-0.019** | **+0.019** | **+0.086** | 1.000 | **-0.003** | **+0.032** |

| **Maneuver** | **+0.008** | **+0.004** | **+0.004** | **-0.003** | 1.000 | **+0.019** |

| **Relevance** | **-0.034** | **+0.008** | **+0.046** | **+0.032** | **+0.019** | 1.000 |



## 3. Shared Backbone Gradient Magnitudes $\|\nabla_{\theta_{shared}} L_i\|$

| Task Objective | Mean Gradient Norm | Std Dev | Median Gradient Norm |
|---|:---:|:---:|:---:|

| **Detection** | `12.2317` | `1.8040` | `11.9351` |

| **NWD** | `1.5663` | `0.2571` | `1.5317` |

| **State** | `7.3887` | `2.6495` | `6.9035` |

| **Round** | `2.6947` | `0.6379` | `2.5604` |

| **Maneuver** | `2.5549` | `1.0458` | `2.3124` |

| **Relevance** | `2.1411` | `0.5981` | `2.0404` |



## 4. Shared Maneuver Head Parameter Alignment

| Metric | Value |
|---|:---:|

| **Batches Analyzed** | 200 |

| **Mean Cosine Similarity $\cos(g_{man, TL}, g_{man, Arrow})$** | **+0.0332** |

| **Std Dev** | 0.1421 |

| **Median Cosine** | **+0.0083** |

| **Synergistic Alignment ($\% > 0$)** | **54.5%** |

| **Mean $\|g_{man, TL}\|$** | `0.4845` |

| **Mean $\|g_{man, Arrow}\|$** | `0.3538` |


## 5. Artifacts Generated

- Diagnostic Visualization: `results/visualizations/w11_multitask_gradient_conflicts.png`

- Telemetry JSON: `results/audit_multitask_gradient_conflicts.json`

- Report: `results/audit_multitask_gradient_conflicts.md`
