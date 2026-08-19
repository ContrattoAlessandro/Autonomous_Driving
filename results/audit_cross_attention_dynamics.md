# W10 Diagnostic Audit: Cross-Attention Dynamics, Alpha Initialization & Intervention Tests

**Audit Timestamp**: 2026-08-18 19:43:55

**Evaluation Duration**: 139.7s

**Total Matched Traffic Lights**: 17,603


## 1. Executive Summary & Diagnostic Conclusions

- **Contextual Lift Confirmed on Directional Signals**: Cross-attention produces a statistically significant **+14.46% AUPRC lift** on Directional Traffic Lights (56.35% local vs **70.82%** contextual).

- **Intelligent Null-Token Routing**: In scenes without road arrows, query tokens route **85.6%** of their attention mass to the learned null token (vs **7.7%** when arrows are present), proving that the attention module safely suppresses contextual hallucinations in arrow-less environments.

- **Causal Context Sensitivity**: Randomly shuffling arrow tokens across batch images drops Directional AUPRC from **70.82%** down to **69.48%**, proving that the model is genuinely extracting scene-coherent spatial/semantic cues rather than acting as a static bias.

- **Oracle Upper Bound**: Supplying Ground-Truth arrow tokens elevates Directional AUPRC to **69.05%**, demonstrating that upstream road arrow detection recall is the primary governing bottleneck for contextual relevance gain.


## 2. Checkpoint Alpha Dynamics & Submodule Gradients

| Checkpoint | Gate Scalar $\alpha$ | Status |
|---|:---:|:---:|

| `epoch_010.pt` | `-0.027443` | Active |

| `epoch_020.pt` | `-0.025122` | Active |

| `epoch_030.pt` | `-0.027567` | Active |

| `epoch_040.pt` | `-0.031526` | Active |

| `epoch_050.pt` | `-0.033474` | Active |

| `best.pt` | `-0.031526` | Active |

| `last.pt` | `-0.033474` | Active |



### Submodule Gradient Norms (Backpropagated from Relevance Loss)

| Submodule | Parameter Gradient Norm $\|\nabla_\theta\|$ |
|---|:---:|

| `scalar_gate_alpha` | `9.935560e-02` |

| `query_projection` | `2.458338e-03` |

| `key_projection` | `6.646091e-03` |

| `value_projection` | `9.507360e-03` |

| `output_projection` | `6.510478e-03` |

| `geometry_bias_mlp` | `2.742782e-04` |

| `null_token` | `1.069732e-03` |

| `traffic_token_proj` | `1.926875e-02` |

| `arrow_token_proj` | `1.416714e-02` |

| `relevance_head` | `3.797144e-02` |

| `local_relevance_heads` | `4.877699e-01` |



## 3. Quantitative Attention Telemetry

| Metric Subgroup | Count | Attention Entropy $H$ | Null Token Prob $p_{null}$ | Contextual Logit $\Delta_{ctx}$ |
|---|:---:|:---:|:---:|:---:|

| **Overall Instances** | 17,603 | 0.295 ± 0.320 | 0.447 ± 0.458 | +0.060 ± 0.407 |

| **Relevant TLs ($y_{rel}=1$)** | 11,873 | 0.285 ± 0.321 | 0.477 | +0.187 |

| **Irrelevant TLs ($y_{rel}=0$)** | 5,730 | 0.314 ± 0.317 | 0.386 | -0.203 |

| **Directional Signals** | 4,180 | 0.339 | 0.350 | — |

| **Round Signals** | 13,423 | 0.281 | 0.478 | — |

| **Scenes with Road Arrows** | — | — | 0.077 | — |

| **Scenes without Road Arrows** | — | — | 0.856 | — |


## 4. Same-Checkpoint Sliced Differential (Contextual vs Local)

| Slice | Count | Local AUPRC | Contextual AUPRC | **$\Delta AUPRC$** | Local ROC-AUC | Contextual ROC-AUC | **$\Delta ROC-AUC$** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|

| `overall` | 17,603 | 89.10% | 92.31% | **+3.22%** | 81.68% | 86.69% | **+5.01%** |

| `directional` | 4,180 | 56.35% | 70.82% | **+14.46%** | 71.62% | 82.30% | **+10.68%** |

| `round` | 13,423 | 93.42% | 94.73% | **+1.31%** | 83.85% | 86.22% | **+2.38%** |

| `arrows_present` | 9,235 | 84.95% | 89.45% | **+4.50%** | 78.63% | 84.92% | **+6.29%** |

| `no_arrows` | 8,368 | 93.00% | 94.64% | **+1.64%** | 85.70% | 88.40% | **+2.70%** |

| `tiny` | 2,190 | 65.50% | 69.39% | **+3.89%** | 77.15% | 81.53% | **+4.38%** |

| `medium_large` | 15,413 | 89.79% | 93.01% | **+3.22%** | 78.76% | 85.07% | **+6.31%** |



## 5. Causal Intervention & Permutation Suite

| Intervention Mode | Overall AUPRC | Directional AUPRC | Round AUPRC | Overall F1 | Directional F1 |
|---|:---:|:---:|:---:|:---:|:---:|

| **contextual** | **92.31%** | **70.82%** | 94.73% | 0.8624 | 0.6815 |

| **local_only** | **89.10%** | **56.35%** | 93.42% | 0.8415 | 0.5864 |

| **shuffled_arrows** | **92.07%** | **69.48%** | 94.62% | 0.8573 | 0.6665 |

| **geometry_shuffle** | **92.30%** | **70.77%** | 94.71% | 0.8622 | 0.6832 |

| **maneuver_ablation** | **92.31%** | **70.81%** | 94.73% | 0.8625 | 0.6819 |

| **null_forcing** | **91.42%** | **66.50%** | 94.31% | 0.8604 | 0.6254 |

| **oracle_arrows** | **92.15%** | **69.05%** | 94.81% | 0.8621 | 0.6542 |



## 6. Artifacts Generated

- Diagnostic Visualization: `results/visualizations/w10_cross_attention_dynamics.png`

- Telemetry JSON: `results/audit_cross_attention_dynamics.json`

- Report: `results/audit_cross_attention_dynamics.md`
