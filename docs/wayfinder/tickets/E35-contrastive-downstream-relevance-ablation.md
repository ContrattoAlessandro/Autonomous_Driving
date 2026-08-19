---
title: "E35: TL <-> Road Arrow Contrastive Learning Downstream Relevance Ablation"
type: prototype
status: closed
blocked_by: ["E29-evaluation-contract-normalization.md"]
assignee: "@agent"
---

## Question

While ticket E26 proved that Supervised InfoNCE contrastive alignment structures the latent maneuver embedding space ($\cos^+=0.8467$ vs $\cos^-=0.1283$), does auxiliary contrastive supervision translate into statistically significant downstream gains in relevance AUPRC, directional reasoning, or Relevant Red safety recall?

---

## Experimental Protocol & Weight Sweep

Trained candidate models with varying auxiliary contrastive loss weights:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{det}} + \lambda_{\text{state}}\mathcal{L}_{\text{state}} + \lambda_{\text{rel}}\mathcal{L}_{\text{rel}} + \lambda_{\text{contrastive}}\mathcal{L}_{\text{contrastive}}$$

| Variant | $\lambda_{\text{contrastive}}$ | Latent Projection Dim | Objective | Status |
|---|:---:|:---:|---|:---:|
| **E35-A** | $0.00$ | - | Unregularized Multitask Baseline | Baseline $C_0$ |
| **E35-B** | $0.05$ | 64 | Mild semantic regularizer | Ablated |
| **E35-C** | $0.10$ | 64 | Canonical E26 formulation | Canonical E26 |
| **E35-D** | $0.25$ | 64 | Strong semantic enforcement | Ablated |

---

## Comprehensive 4-Way Downstream Ablation Matrix

Evaluated under the **Unified Evaluation Contract (E29 Standard)** across the full DTLD validation set (5,962 images, 25,344 GT TLs):

| Metric Dimension | E35-A ($\lambda=0.00$) | E35-B ($\lambda=0.05$) | E35-C ($\lambda=0.10$) | E35-D ($\lambda=0.25$) | Max Delta ($\text{E35-C} - \text{E35-A}$) | Significance Target | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Directional Relevance AUPRC** | **91.61%** | 91.65% | **91.70%** | 91.48% | **+0.09%** | $\ge +1.00\%$ | Not Significant |
| **Overall Relevance AUPRC** | **91.61%** | 91.64% | 91.68% | 91.52% | **+0.07%** | $\ge +0.50\%$ | Not Significant |
| **Relevance F1 ($\tau=0.50$)** | **84.22%** | 84.25% | 84.29% | 84.10% | **+0.07%** | $\ge +0.50\%$ | Not Significant |
| **Relevant Red Recall ($\tau=0.50$)** | **72.98%** | 73.04% | 73.12% | 72.75% | **+0.14%** | $\ge +1.00\%$ | Not Significant |
| **Relevant Red Recall ($\tau_{95}$)** | **94.85%** | 94.88% | **94.92%** | 94.70% | **+0.07%** | $\ge +1.00\%$ | Not Significant |
| **Calibrated Precision ($\tau_{95}$)** | **76.20%** | 76.25% | 76.32% | 75.95% | **+0.12%** | - | Invariant |
| **TL Maneuver Macro F1** | 88.12% | 88.45% | 89.05% | 89.20% | +0.93% | - | Minor Attribute Lift |
| **Arrow Maneuver Macro F1** | 94.30% | 94.62% | 95.10% | 95.25% | +0.80% | - | Minor Attribute Lift |
| **InfoNCE Loss** | 1.2450 | 0.5420 | 0.3124 | 0.1980 | -0.9326 | - | Latent Structured |
| **Latent Separation Margin** | +0.214 | +0.582 | +0.718 | +0.801 | +0.504 | - | Latent Structured |
| **Directional Shuffling Drop $\Delta$** | -0.07% | -0.09% | -0.08% | -0.10% | -0.01% | - | Shuffling Invariant |
| **Inference FPS (Batch=1)** | 50.6 FPS | 50.6 FPS | 50.6 FPS | 50.6 FPS | 0.0 FPS | $\ge 45\text{ FPS}$ | Zero Inference Penalty |
| **Training Step Time (ms)** | 112.4 ms | 114.8 ms | 116.5 ms | 121.8 ms | +4.1 ms (+3.6%) | - | Compute Overhead |

---

## Causal Attribution & Scientific Conclusions

1. **Failure of Latent Alignment to Propagate Downstream**:
   - Although Supervised InfoNCE successfully clusters maneuver tokens in latent space ($\cos^+=0.8467$ vs $\cos^-=0.1283$, separation margin $+0.7184$, loss dropping $1.2450 \to 0.3124$), downstream Directional Relevance AUPRC improves by only **$+0.09\%$** ($91.61\% \to 91.70\%$), far below the prespecified significance bound of $\ge +1.0\%$.
2. **Cross-Attention Inductive Bias Disentanglement**:
   - The cross-attention relevance reasoning module derives its primary predictive power from spatial geometric priors, bounding box relative topologies, and candidate visual token representations rather than the 3-class discrete maneuver embeddings.
   - Maneuver label shuffling at test time causes negligible degradation ($\Delta_{\text{shuffle}} = -0.08\%$), confirming that downstream relevance is essentially invariant to maneuver embedding alignment.
3. **Training Overhead vs Deployment Invariance**:
   - While the contrastive projection MLP is discarded at test time (maintaining exact $19.75\text{ ms}$ / $50.6\text{ FPS}$ latency), it introduces $+3.65\%$ to $+8.36\%$ backward compute overhead per training step and adds an extra hyperparameter ($\lambda_{\text{contrastive}}$) with no commensurate safety benefit.

---

## Formal Decision Verdict

- **Prespecified Decision Logic**:
  - If $\Delta \text{Directional AUPRC} \ge +1.0\%$: Retain contrastive head for training.
  - If downstream metrics are unchanged ($\Delta \le \pm 0.20\%$): **Formally reject contrastive loss** from active pipeline.
- **Observed Result**: $\Delta_{\text{Dir AUPRC}} = +0.09\% \le 0.20\%$.
- **Decision Verdict**: **FORMALLY REJECT CONTRASTIVE LOSS FROM PRODUCTION CHAMPION PIPELINE**.

**Action for Phase 4 Synthesis (E36)**: Contrastive loss is formally **excluded** from candidate configurations in Sequential Forward Selection ($C_0 \to C_5$). The locked champion model retains the clean unregularized multitask loss formulation.

**Status**: Resolved and Closed. Unblocks downstream champion synthesis in E36.

---

## Diagnostic Artifacts Produced

- **Source Code**: [tlr_yolo_mtl/training/losses.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/tlr_yolo_mtl/training/losses.py) (contrastive integration in `TLRMultiTaskCriterion` & `MultiTaskLossWeights`)
- **Audit Script**: [scripts/audit_e35_contrastive_downstream_ablation.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/scripts/audit_e35_contrastive_downstream_ablation.py)
- **Visualization Plot**: `results/visualizations/e35_contrastive_downstream_ablation.png`
- **Tabular Report**: [results/audit_e35_contrastive_downstream_ablation.md](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/results/audit_e35_contrastive_downstream_ablation.md)
- **JSON Telemetry**: [results/audit_e35_contrastive_downstream_ablation.json](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/results/audit_e35_contrastive_downstream_ablation.json)
- **Unit Tests**: [tests/test_contrastive_downstream_ablation.py](file:///c:/Users/alexa/Desktop/Tesi_Autonomous_Driving/tl_detection/tests/test_contrastive_downstream_ablation.py) (5/5 passing)
