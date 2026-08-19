# E35: TL <-> Road Arrow Contrastive Learning Downstream Relevance Ablation

- **Benchmark Target**: Full DTLD Validation Set (5,962 images, 25,344 GT TLs)
- **Evaluation Contract**: Unified Evaluation Contract (E29 Standard)
- **Validation Population**: 5962 images

---

## 1. Executive Summary & Causal Resolution

Ticket E26 proved that Supervised InfoNCE contrastive alignment structures the latent maneuver space ($\cos^+=0.8467$ vs $\cos^-=0.1283$, separation margin $+0.7184$).
**Ticket E35 systematically assesses whether this latent alignment translates into statistically meaningful downstream relevance, directional reasoning, or safety gains.**

### Key Scientific Findings:
1. **Negligible Downstream Relevance Lift**: Across all evaluated auxiliary weights $\lambda_{\text{contrastive}} \in \{0.05, 0.10, 0.25\}$, Directional Relevance AUPRC shifted by at most **$+0.09\%$** ($91.61\% \to 91.70\%$ for $\lambda=0.10$), failing the $\ge +1.0\%$ significance threshold by an order of magnitude.
2. **Safety Recall Invariance**: Calibrated Relevant Red safety recall at $\tau_{95}$ remained essentially invariant ($94.85\% \to 94.92\%$, $\Delta = +0.07\%$), while aggressive regularization ($\lambda=0.25$) introduced slight performance degradation ($94.70\%$, $\Delta = -0.15\%$).
3. **Decoupled Causal Reasoning Dynamics**: Cross-attention reasoning in TLR-YOLO-MTL primarily relies on spatial geometric priors, lane alignments, and candidate visual features rather than explicit 3-class directional maneuver embeddings. Even when latent maneuver spaces are tightly aligned, the downstream relevance head operates invariantly.
4. **Shuffling Invariance Confirmed**: Permuting road arrow maneuver labels at test time resulted in negligible directional AUPRC degradation ($\Delta_{\text{shuffle}} = -0.07\%$ in E35-A vs $-0.08\%$ in E35-C), corroborating the observation from ticket E17 that explicit maneuver logits do not provide the primary causal inductive bias for relevance.
5. **Training Cost vs Inference Invariance**: While the contrastive projection head is discarded at deployment (zero runtime inference latency penalty, maintaining $50.6\text{ FPS}$), it adds $+3.65\%$ to $+8.36\%$ to backward training step compute and introduces an unnecessary hyperparameter.

---

## 2. Comprehensive 4-Way Downstream Ablation Matrix

| Variant | $\lambda_{\text{contrastive}}$ | Latent Margin | InfoNCE Loss | Directional AUPRC | Overall AUPRC | Rel Red Rec @ $\tau_{95}$ | TL Maneuver F1 | Arrow Maneuver F1 | Shuffling Drop $\Delta$ | Train Step Overhead | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E35-A: Baseline (λ=0.00)** | 0.00 | +0.214 | 1.2450 | **91.61%** | 91.61% | **94.85%** | 88.12% | 94.30% | -0.07% | +0.0% | Baseline C0 |
| **E35-B: Mild Regularizer (λ=0.05)** | 0.05 | +0.582 | 0.5420 | **91.65%** | 91.64% | **94.88%** | 88.45% | 94.62% | -0.09% | +2.1% | Ablated |
| **E35-C: Canonical E26 (λ=0.10)** | 0.10 | +0.718 | 0.3124 | **91.70%** | 91.68% | **94.92%** | 89.05% | 95.10% | -0.08% | +3.6% | Canonical E26 |
| **E35-D: Strong Enforcement (λ=0.25)** | 0.25 | +0.801 | 0.1980 | **91.48%** | 91.52% | **94.70%** | 89.20% | 95.25% | -0.10% | +8.4% | Ablated |

---

## 3. Calibrated Safety Operating Points Matrix

| Variant | Temp $T^*$ | NLL ($1.0 \to T^*$) | ECE ($1.0 \to T^*$) | Rec @ $\tau_{90}$ | Prec @ $\tau_{90}$ | Rec @ $\tau_{95}$ | Prec @ $\tau_{95}$ | Rec @ $\tau_{97.5}$ | Prec @ $\tau_{97.5}$ | Distractors @ $\tau_{95}$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **E35-A** | 0.7241 | 0.5079 $\to$ 0.4963 | 12.99% $\to$ 8.64% | 89.41% | 82.10% | 94.85% | 76.20% | 97.25% | 68.40% | 0.185/img |
| **E35-B** | 0.7235 | 0.5075 $\to$ 0.4960 | 12.95% $\to$ 8.61% | 89.44% | 82.14% | 94.88% | 76.25% | 97.28% | 68.45% | 0.184/img |
| **E35-C** | 0.7228 | 0.5070 $\to$ 0.4957 | 12.91% $\to$ 8.58% | 89.48% | 82.20% | 94.92% | 76.32% | 97.32% | 68.52% | 0.182/img |
| **E35-D** | 0.7255 | 0.5092 $\to$ 0.4975 | 13.10% $\to$ 8.72% | 89.32% | 81.95% | 94.70% | 75.95% | 97.15% | 68.15% | 0.190/img |

---

## 4. Formal Decision Logic & Scientific Verdict

- **Prespecified Promotion Threshold**: $\Delta \text{Directional AUPRC} \ge +1.0\%$ and $\Delta \text{RelRed Recall} \ge +1.0\%$.
- **Prespecified Rejection Threshold**: $\Delta \le \pm 0.20\%$ across downstream relevance metrics.
- **Observed Maximum Delta**: $\Delta_{\text{Dir AUPRC}} = +0.09\%$, $\Delta_{\text{RelRed @ } \tau_{95}} = +0.07\%$.

**Decision Verdict**: **FORMALLY REJECT CONTRASTIVE LOSS FROM CHAMPION PIPELINE**.

**Scientific Rationale**: Downstream metrics are statistically invariant to auxiliary contrastive loss (maximum delta +0.09% <= 0.20%). Rejecting contrastive loss eliminates hyperparameter complexity and +3.65% to +8.36% training compute overhead with zero downstream penalty.

**Action for Phase 4 Synthesis (E36)**: Contrastive loss is formally **excluded** from the active candidate pipeline for Sequential Forward Selection ($C_0 \to C_5$). The final champion architecture retains the unregularized multitask formulation with spatial priors, $M=8$ arrow retrieval, and $3\times3$ P2+P3 ROIAlign.

---

## 5. Diagnostic Artifacts Produced

- **Audit Script**: `scripts/audit_e35_contrastive_downstream_ablation.py`
- **Visualization Plot**: `results/visualizations/e35_contrastive_downstream_ablation.png`
- **JSON Telemetry**: `results/audit_e35_contrastive_downstream_ablation.json`
- **Unit Tests**: `tests/test_contrastive_downstream_ablation.py`
