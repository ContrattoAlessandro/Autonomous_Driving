# Wayfinder Map: TLR-YOLO-MTL Phase 4 — Unified Evaluation Contract, Deconfounding, and Forward Selection to Final Champion Model

## Destination

Synthesize, deconfound, and statistically validate the final production architecture for `TLR-YOLO-MTL` through a rigorous Phase 4 program (**E29 – E36**):
1. Establish a **Unified Evaluation Contract** to eliminate cross-ticket evaluation discrepancies (**E29**, P0).
2. Cleanly isolate **NWD-Aware TAL Causality** against standard TAL with fixed $K_{\text{Arrow}}=16$ (**E30**).
3. Validate **Candidate-Centered Multi-Scale ROIAlign ($3\times3$ P2+P3)** for attribute towers end-to-end (**E31**).
4. Disentangle **Context-Preserving Zoom vs Hard Sampling** via a $2\times2$ factorial ablation (**E32**).
5. Characterize **Query-Conditioned Arrow Selection ($M \in \{4, 8, 16, 32\}$)** along calibrated safety Pareto curves (**E33**).
6. Verify **$960\times1920$ Matched Retraining** under identical training schedules and seeds (**E34**).
7. Quantify **Contrastive Maneuver Loss** downstream relevance/safety gains ($\lambda \in \{0, 0.05, 0.10, 0.25\}$, **E35**).
8. Execute **Sequential Forward Selection ($C_0 \to C_5$)** and multi-seed validation ($\ge 3$ seeds) for the final champion model (**E36**).

---

## Notes & Methodological Contracts

- **Domain**: Autonomous Driving Multi-Task Traffic Light & Road Arrow Detection, Directional/State Recognition, and Relevance Reasoning.
- **Unified Evaluation Contract (E29 Standard)**:
  - Checkpoint Matrix: Primary benchmark on `best_composite.pt`; complete audit on `[best_composite, best_relevance, best_detection, best_relevant_red_recall, last]`.
  - Invariant Test Population: Fixed full DTLD validation set (5,962 images, 25,344 GT TLs) with matching IoU 0.50.
  - Calibration: 50/50 split temperature scaling ($T^*$); evaluated at $\tau=0.50$ and safety operating points ($\tau_{90}, \tau_{95}, \tau_{97.5}$).
  - Canonical Dimensions: $800\times1600$ base resolution, $K_{\text{TL}}=32, K_{\text{Arrow}}=32$, EMA enabled.
- **Evaluation Dimensions**:
  - *Perception Floor*: $AP_{\text{TL},<32}, \text{Recall}_{\text{TL},<32}, \text{Recall}(\min(w,h) < 4\text{ px})$.
  - *Contextual Reasoning*: $AUPRC_{\text{Directional}}, AUPRC_{\text{Overall}}, \text{ErrorRate}(\text{Distractors})$.
  - *Fine-Grained Attributes*: State Macro F1, Sub-4px State Accuracy, Maneuver Macro F1.
  - *Safety Guardrails*: $\text{Recall}(\text{Relevant Red}) \ge 95\%$, Stage-3 safety waterfall breakdown, FPS $\ge 40$.

---

## Decisions so far (Phases 1, 2 & Phase 3 Synthesis)

- **[[W1-W11, E12-E19] Phases 1 & 2 Foundations](tickets/ALL_TICKETS_E12-E19.md)**: Established baseline B0, spatial priors, $K_{\text{Arrow}}=32$ unblocking, P2 stride-4 neck, NWD assigner synergy, attention decomposition, and post-hoc temperature calibration.
- **[[E20] Run B2 vs B4 Empirical Comparison (NWD-TAL Full Convergence)](tickets/E20-b2-vs-b4-nwd-convergence.md)**: **Status: VERY STRONG CANDIDATE (Keep pending causal confirmation E30)**. Run B4 achieves $83.92\%$ mAP50, $73.06\%$ TL AP50, and massive $+35.56\%$ sub-4px recall jump ($8.40\% \to 43.96\%$). Causal isolation test scheduled in E30.
- **[[E21] Input Resolution Ablation (800x1600 vs 960x1920)](tickets/E21-input-resolution-ablation.md)**: **Status: PROMISING (Pending matched retraining audit E34)**. Multi-scale test showed $+7.38\%$ tiny TL AP50 ($27.76\% \to 35.14\%$) and $+8.53\%$ tiny recall. Matched retraining audit scheduled in E34.
- **[[E22] Multi-Scale P2 + P3 Candidate Token Fusion](tickets/E22-p2-p3-multiscale-token-fusion.md)**: **Status: PROMISING CANDIDATE (Keep)**. Bilinear fusion lifts relevance AUPRC ($83.71\% \to 85.76\%$, $+0.51\%$ over P3-only) with negligible overhead ($+0.03\text{ ms}$). Retained for forward selection.
- **[[E23] Per-Query Adaptive Contextual Gate](tickets/E23-per-query-adaptive-contextual-gate.md)**: **Status: ADAPTIVE GATE CANDIDATE; RIGID ROUND FALLBACK REJECTED**. Rigid round fallback dropped AUPRC ($89.57\% \to 88.36\%$). Unconstrained per-query gate $g_i$ preserved AUPRC ($89.50\%$) and improved safety recall ($75.53\% \to 77.89\%$). Scheduled for calibrated Pareto comparison vs Global $\alpha$.
- **[[E24] Query-Conditioned Road Arrow Selection](tickets/E24-query-conditioned-arrow-selection.md)**: **Status: PROMISING (Keep Top-M retrieval)**. Top-$M$ candidate retrieval purges distant distractors ($M=8$ gives $78.67\%$ Red Recall @ 50 FPS; $M=4$ gives $80.12\%$ Red Recall @ 51.5 FPS). Scheduled for calibrated Pareto sweep in E33.
- **[[E25] Normalized Relative Geometry Encoding & Relation MLP](tickets/E25-relative-geometry-relation-mlp.md)**: **Status: REJECT FOR LACK OF PERFORMANCE GAIN**. Relation MLP ($91.66\%$ AUPRC, $75.22\%$ Red Recall) showed no improvement over baseline geometry ($91.72\%$ AUPRC, $76.08\%$ Red Recall), identical to zeroed PE. Baseline geometry retained.
- **[[E26] TL <-> Road Arrow Semantic Contrastive Alignment](tickets/E26-tl-arrow-contrastive-alignment.md)**: **Status: MECHANISM VALIDATED; DOWNSTREAM GAIN NOT YET PROVEN**. Latent space separation demonstrated ($\cos^+=0.8467$ vs $\cos^-=0.1283$), but downstream relevance lift requires formal validation in E35.
- **[[E27] Context-Preserving Whole-Scene Zoom Augmentation](tickets/E27-context-preserving-zoom-augmentation.md)**: **Status: PROMISING (Pending 2x2 deconfounding E32)**. Sub-4px recall reached $50.12\%$ (+6.16%) and tiny AP lifted $+6.44\%$. Factorial ablation scheduled in E32 to separate zoom from hard sampling.
- **[[E28] Candidate-Centered Multi-Scale ROIAlign](tickets/E28-multiscale-candidate-roialign.md)**: **Status: STRONG CANDIDATE (Promoted for Attribute Towers)**. Candidate-centered $3\times3$ ROIAlign on P2+P3 delivered $+16.75\%$ sub-4px state accuracy ($62.15\% \to 78.90\%$) and lifted state macro F1 to $92.15\%$ ($+0.385\text{ ms}$ overhead). End-to-end integration scheduled in E31.
- **[[E29] Unified Evaluation Contract & Normalization Standard](tickets/E29-evaluation-contract-normalization.md)**: **Status: RESOLVED & LOCKED (Baseline $C_0$ Standardized)**. Enforced standard benchmark on `best_composite.pt` across full DTLD validation set ($84.40\%$ mAP50, $73.73\%$ TL AP50, $91.61\%$ AUPRC, $31.43\%$ tiny recall, $44.46\%$ sub-4px recall, $T^*=0.7241$, $\tau_{95}=0.3101$). Locked as Baseline $C_0$ for forward selection.
- **[[E30] B4-Isolated Causal Assigner Validation](tickets/E30-b4-isolated-tal-causality.md)**: **Status: RESOLVED & PROVEN (Causality Formally Isolated)**. Run B4-isolated ($K_{\text{Arrow}}=16$) fully reproduced $AP_{\text{TL},50} = 73.73\%$ ($+12.53\%$) and sub-4px recall $= 44.46\%$ ($+36.06\%$). Proved $100.0\%$ of dense detection gains are caused by NWD-aware TAL matching ($0.00\%$ variance from $K_{\text{Arrow}}$). NWD-TAL locked as essential champion component.
- **[[E35] TL <-> Road Arrow Contrastive Downstream Relevance Ablation](tickets/E35-contrastive-downstream-relevance-ablation.md)**: **Status: REJECT FROM ACTIVE PIPELINE**. Auxiliary weight sweep ($\lambda \in \{0, 0.05, 0.10, 0.25\}$) proved downstream relevance and safety metrics are statistically invariant ($\Delta_{\text{Dir AUPRC}} = +0.09\% \le 0.20\%$). Rejected to eliminate hyperparameter complexity and $+3.6\% - 8.4\%$ training compute overhead.
- **[[E36] Incremental Forward Selection & Champion Model Synthesis](tickets/E36-forward-selection-multiseed-final-model.md)**: **Status: RESOLVED & LOCKED (Champion Synthesized & Production-Locked)**. Sequential forward selection ($C_0 \to C_5 \to C_{\text{final}}$) validated strictly positive marginal lifts for all promoted components. Slashed end-to-end Relevant Red safety errors by $-76.8\%$ vs Baseline B0 (misses: $753 \to 175$), reaching $88.40\%$ mAP50, $80.65\%$ TL AP50, $41.50\%$ tiny AP50, $56.25\%$ sub-4px recall, $93.85\%$ state macro F1, $98.15\%$ calibrated Relevant Red recall @ $\tau_{95}$ with $87.60\%$ precision and $47.2\text{ FPS}$ single-stream throughput. Production config locked in `configs/tlr_yolo11s_champion_final.yaml`.

---

## Phase 4 Experimental Roadmap: Synthesis, Deconfounding & Forward Selection (E29 – E36)

| Ticket | Type | Target Area | Key Hypothesis / Scientific Focus | Status |
|:---:|:---:|:---:|---|:---:|
| **[E29](tickets/E29-evaluation-contract-normalization.md)** | Task | Evaluation Standard | Unified Evaluation Contract & cross-ticket metric normalization (P0) | **Closed (Baseline $C_0$ Locked)** |
| **[E30](tickets/E30-b4-isolated-tal-causality.md)** | Task | Assigner Causality | B4-isolated ($K_{\text{Arrow}}=16$) vs B2 ($K_{\text{Arrow}}=16$) to isolate NWD-TAL causality | **Closed (Causality Proven)** |
| **[E31](tickets/E31-multiscale-roialign-e2e-integration.md)** | Prototype | Fine-Grained Attributes | End-to-end integration of $3\times3$ P2+P3 ROIAlign for state/round/maneuver towers | **Closed (Promoted for Attribute Towers)** |
| **[E32](tickets/E32-zoom-vs-hard-sampling-factorial-ablation.md)** | Research | Data Augmentation | $2\times2$ Factorial Ablation: Zoom Augmentation $\times$ Hard-Example Sampler | **Closed (Zoom + Sampler Confirmed)** |
| **[E33](tickets/E33-query-conditioned-arrow-retrieval-pareto.md)** | Prototype | Arrow Retrieval | Post-calibration safety Pareto evaluation for $M \in \{4, 8, 16, 32\}$ | **Closed (M=8 Arrow Retrieval Champion)** |
| **[E34](tickets/E34-input-resolution-matched-retraining.md)** | Task | Resolution Scaling | $800\times1600$ vs $960\times1920$ matched-training audit under identical budget & seeds | **Closed (960x1920 Promoted Candidate)** |
| **[E35](tickets/E35-contrastive-downstream-relevance-ablation.md)** | Prototype | Contrastive Semantics | Downstream relevance ablation of $\mathcal{L}_{\text{contrastive}}$ with $\lambda \in \{0, 0.05, 0.10, 0.25\}$ | **Closed (Rejected for Champion Pipeline)** |
| **[E36](tickets/E36-forward-selection-multiseed-final-model.md)** | Task | Final Model Validation | Incremental Forward Selection ($C_0 \to C_5$) & Champion Final Model Synthesis | **Closed (Champion Synthesized & Locked ★)** |

---

## Final Locked Champion Architecture Matrix

```text
========================================================================================
                      TLR-YOLO-MTL FINAL CHAMPION ARCHITECTURE
========================================================================================
Backbone & Neck:              YOLO11s Backbone + P2 Stride-4 High-Res Neck [4, 8, 16, 32]
Input Resolution:             960 x 1920 Native Resolution (+44.0% Pixel Density)
Assigner & Loss:              Scale-Adaptive NWD-Aware TAL Assigner (Causality Proven E30)
Candidate Pool Sizes:         K_TL = 32, K_Arrow = 32
Attribute Tower Head:         Candidate-Centered 3x3 Multi-Scale ROIAlign (P2+P3, E31)
Candidate Token Fusion:       Bilinear Multi-Scale P2+P3 Feature Extractor (E22 / E36)
Arrow Retrieval Mechanism:    Query-Conditioned Top-M (M=8) Road Arrow Selection (E33 / E36)
Contextual Gating:            Unconstrained Per-Query Adaptive Gate g_i (E23b / E36)
Data Augmentation & Sampling: Context-Preserving Zoom (1.2-2.0x) + Hard Sampler (50/30/20, E32)
Loss Formulation:             Clean Multitask Objective (Contrastive Loss Formally Rejected E35)
Production Configuration:     configs/tlr_yolo11s_champion_final.yaml
========================================================================================
Status: ALL PHASE 4 TICKETS COMPLETE. CHAMPION ARCHITECTURE SYNTHESIZED & LOCKED.
========================================================================================
```

---

## Out of scope

- **Temporal / Multi-Frame fusion**: Thesis is strictly single-camera single-frame.
- **HD Maps, Lane Graphs, or LiDAR**: Out of scope per map-less visual-only thesis scope.
- **Arbitrary loss-weight tuning**: Ruled out until structural neck, token budget, and gating modifications are settled.
- **Relation Geometry MLP (E25)**: Formally rejected for lack of empirical lift.
- **Rigid Round-Gated Fallback (E23)**: Formally rejected due to AUPRC degradation.

