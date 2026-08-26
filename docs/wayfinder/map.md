# Wayfinder Map: TLR-YOLO-MTL Phase 6 — Champion v3 Integration & Frontier Scaling (E47 – E54)

## Destination

Synthesize, validate, and audit the definitive single-checkpoint **Champion v3** production model (`tlr_yolo11s_champion_v3.yaml`) combining all verified Phase 5 interventions (**E47**), followed by resource-aware tiny-perception expansions (**E48 – E54**: Local-View Distillation, Sparse ROI Refinement, NWD-Quality Ranking, Scale Relay, Temporal Distillation, Scale-Conditioned Calibration, and Cross-Dataset Generalization on ATLAS/LISA) targeting $\text{AP}_{<8\text{px}} > 50\%$, $\text{State Macro-F1} \ge 92\%$, $\text{Relevance AUPRC} \ge 95\%$, and edge throughput $\ge 36\text{ FPS}$.

---

## Notes & Methodological Contracts

- **Domain**: Autonomous Driving Multi-Task Traffic Light & Road Arrow Detection, Directional/State Recognition, and Cross-Attention Ego-Lane Relevance Reasoning.
- **Architectural Philosophy**: Prioritize orthogonal inductive biases, training-only distillation ($0\text{ ms}$ inference overhead), sparse candidate-conditioned processing, and Gaussian geometric alignment over brute-force backbone scaling or global high-resolution feature maps.
- **Unified Evaluation Contract (E29/E37 Standard)**:
  - Evaluation PR curves generated with $\text{conf}_{\text{eval}} = 0.001$.
  - Operational deployment benchmark at $\text{conf}_{\text{deploy}} = 0.25, \text{IoU}_{\text{NMS}} = 0.45$, and Size-Adaptive NWD post-processing for $<64\text{ px}^2$.
  - Invariant validation set: Full DTLD validation set (5,962 images, 25,344 GT TLs, 6,108 GT Arrows).
- **Standing Hard Constraints**:
  - Edge Hardware: Single NVIDIA RTX 5070 (12GB VRAM).
  - Runtime Latency: Single-stream batch-1 FP16 latency $\le 27.5\text{ ms}$ ($\ge 36.0\text{ FPS}$).
  - Safety Floor: Relevant Red Recall @ $\tau_{95} \ge 96.0\%$.

---

## Decisions so far

- **[[W1-W11, E12-E19] Phases 1 & 2 Foundations](tickets/ALL_TICKETS_E12-E19.md)** — Established baseline B0, spatial priors, $K_{\text{Arrow}}=32$ unblocking, P2 stride-4 neck, NWD assigner synergy, attention decomposition, and post-hoc temperature calibration.
- **[[E20-E28] Phase 3 Component Exploration](tickets/ALL_TICKETS_E20-E28.md)** — Isolated candidate-centered ROIAlign, multi-scale P2+P3 candidate tokens, query-conditioned arrow retrieval ($M=8$), and unconstrained contextual gating ($g_i$).
- **[[E29] Unified Evaluation Contract & Normalization Standard](tickets/E29-evaluation-contract-normalization.md)** — Codified invariant evaluation benchmark on `best_composite.pt` and locked Baseline $C_0$.
- **[[E30] B4-Isolated Causal Assigner Validation](tickets/E30-b4-isolated-tal-causality.md)** — Proved $100\%$ of dense detection gains on tiny TLs originate causally from NWD-aware TAL matching.
- **[[E35] TL <-> Road Arrow Contrastive Downstream Relevance Ablation](tickets/E35-contrastive-downstream-relevance-ablation.md)** — Formally rejected auxiliary contrastive/association losses due to gradient antagonism with detection backbone.
- **[[E36] Incremental Forward Selection & Champion v1 Synthesis](tickets/E36-forward-selection-multiseed-final-model.md)** — Synthesized locked Champion v1 (`YOLO11s-P2 + NWD-TAL + 3x3 ROIAlign + M=8 + g_i Gate`), achieving $84.7\%$ mAP50, $71.2\%$ AP Tiny, $91.1\%$ Relevance AUPRC, $94.1\%$ State Acc at $37.3\text{ FPS}$.
- **[[E37] Rigorous Separation of Evaluation AP and Deployment Operating Points](tickets/E37-evaluation-vs-deployment-operating-points.md)** — Formally codified separation of evaluation PR curves ($\text{conf}_{\text{eval}}=0.001$) from operational deployment ($\text{conf}_{\text{deploy}}=0.25, \text{IoU}=0.45$). Established the uncorrupted sub-8px baseline floor ($AP_{<8\text{px}} = 29.53\%$).
- **[[E38] Distribution-Aware Scale-Matched & Paired Copy-Paste Augmentation](tickets/E38-scale-matched-paired-augmentation.md)** — Balanced tiny TL representation across scale quotas ($40\% <8\text{px}, 35\% 8\text{-}16\text{px}, 25\% >16\text{px}$) and paired context-preserving arrow copy-paste, boosting Sub-8px AP@50 from $29.53\%$ to $33.15\%$.
- **[[E39] Physics-Grounded Photometric Traffic Light Augmentation](tickets/E39-photometric-traffic-light-augmentation.md)** — Replaced generic HSV jitter with parametric Gaussian lamp bloom, exposure/gamma, and strict hue preservation, lifting State Macro-F1 from $84.2\%$ to $87.1\%$ with zero inference overhead.
- **[[E40] DySample Dynamic Upsampling in the P3 -> P2 Lateral Path](tickets/E40-dysample-p2-dynamic-upsampling.md)** — Replaced static interpolation with DySample point-sampling, boosting Sub-8px AP@50 to $36.15\%$ and Sub-4px Recall by $+3.35\%$ with zero latency penalty ($37.4\text{ FPS}$).
- **[[E41] Task-Specific P2/P3 Gated Feature Fusion & 5x5 State ROIAlign](tickets/E41-task-specific-gated-fusion-roialign5x5.md)** — Decoupled multi-task representations via learnable task gating ($\alpha_{\text{state}} \approx 77\% P2$) and expanded State ROIAlign to $5\times5$, lifting State Macro-F1 to $86.75\%$ ($+2.55\%$) with negligible latency ($+0.22\text{ ms}$).
- **[[E42] Geometry-Aware Cross-Attention with Explicit Relative Spatial Bias](tickets/E42-geometry-aware-cross-attention.md)** — Injected explicit 14D normalized spatial-geometric descriptors into the TL $\leftrightarrow$ Road Arrow attention matrix, boosting Relevance Precision to $88.10\%$ and slashing cross-lane false alarms by $-49.7\%$ relative.
- **[[E43] Counterfactual Hard-Negative Sampling for Ego-Lane Relevance](tickets/E43-counterfactual-hard-negative-sampling.md)** — Curated scene-coherent hard negatives (adjacent-lane TLs/arrows in same scene), lifting Relevance Precision to $91.30\%$, F1 to $90.34\%$, AUPRC to $0.9470$, and cutting cross-lane false alarms to $4.1\%$ with zero runtime overhead.
- **[[E44] Long-Tail State Head Loss Rebalancing](tickets/E44-long-tail-state-class-balanced-loss.md)** — Implemented Class-Balanced Focal Loss ($\beta = 0.9999$) and Balanced Softmax log-priors, lifting State Macro-F1 to **$91.28\%$**, Yellow F1 to $84.79\%$, Off F1 to $86.63\%$ ($+4.90\%$), preserving Red recall ($96.49\%$) with zero latency overhead.
- **[[E45] Size-Adaptive Gaussian NWD Suppression in Deployment Post-Processing](tickets/E45-size-adaptive-nwd-postprocessing.md)** — Slashed sub-8px duplicate detection rate from $18.42\%$ to $4.15\%$ ($-77.5\%$ relative) and lifted sub-8px AP to $46.10\%$ by branching NWD for $<64\text{ px}^2$ boxes at $+0.04\text{ ms}$ overhead ($37.15\text{ FPS}$).
- **[[E46] Multi-Task Gradient Conflict Diagnostics & Neck-Restricted Balancing](tickets/E46-multitask-gradient-conflict-balancing.md)** — Quantified pairwise gradient cosine similarities across all 6 loss heads; proved exceptional natural synergy ($\cos(\nabla\mathcal{L}_{\text{det}}, \nabla\mathcal{L}_{\text{nwd}}) = +0.775$) and confirmed static manual weights as optimal production default ($0.00\text{ ms}$ train overhead vs $+106\%$ for PCGrad).
- **[[E47] Cumulative Champion v3 Integration & Metric Lineage Audit](tickets/E47-cumulative-champion-v3-integration-lineage-audit.md)** — Formally synthesized and validated the unified Champion v3 production model (`tlr_yolo11s_champion_v3.yaml`) combining all Phase 5 modules, achieving $46.10\%$ Sub-8px AP, $91.28\%$ State Macro-F1, $91.30\%$ Relevance Precision, $0.9470$ Relevance AUPRC, and $37.15\text{ FPS}$ on RTX 5070.
- **[[E48] Local-View Tiny-TL High-Resolution Crop Distillation](tickets/E48-local-view-tiny-tl-distillation.md)** — Distilled high-resolution visual context from $64\times64$ crops into full-frame $P2/P3$ representations, lifting Sub-8px AP to $48.65\%$ and Sub-4px State Accuracy to $76.90\%$ with zero runtime overhead ($0.00\text{ ms}$).
- **[[E49] Sparse Candidate Refinement Head on Top-32 Sub-Grid Regions](tickets/E49-sparse-candidate-refinement-head.md)** — Introduced lightweight $7\times7$ ROIAlign + TinyConv refinement on Top-32 sub-16px candidates, achieving virtual P1 spatial fidelity, breaking the 50% milestone ($50.85\%$ Sub-8px AP), slashing sub-pixel jitter by $-31.6\%$ ($0.52\text{ px}$ RMSE), and lifting State Macro-F1 to $94.55\%$ at real-time edge throughput ($36.72\text{ FPS}$).
- **[[E50] NWD-Quality-Aware Confidence Head & Tiny-Aligned Ranking](tickets/E50-nwd-quality-aware-confidence-head.md)** — Implemented scale-adaptive classification $\times$ NWD localization quality scoring ($s_i = p_i^\alpha q_i^{1-\alpha}$), eliminating low-quality false-positive rank inversions by $-34.1\%$ and lifting Sub-8px AP to $52.45\%$ with zero runtime overhead ($0.00\text{ ms}$).
- **[[E51] Scale-Aware C2 -> P2 Feature Relay for Raw Texture Recovery](tickets/E51-scale-aware-c2-p2-feature-relay.md)** — Injected shallow $C2$ edge/chromatic gradients into high-res $P2$ neck via spatial-channel gating, reducing sub-8px center RMSE to $0.46\text{ px}$ and lifting Sub-8px AP to $53.85\%$ and Sub-4px State Acc to $82.45\%$ at $+0.09\text{ ms}$ latency ($36.60\text{ FPS}$).
- **[[E52] Temporal Sequence Teacher Distillation for Single-Frame Inference](tickets/E52-temporal-teacher-single-frame-student.md)** — Distilled multi-frame cross-attention temporal context from $(I_{t-1}, I_t, I_{t+1})$ triplets during training into the single-frame student, slashing inter-frame state flicker by **$-46.6\%$ relative** ($7.90\%$), boosting sub-8px trajectory recall to $85.30\%$, lifting Sub-8px AP to **$55.60\%$**, and preserving strictly single-frame zero-latency runtime inference ($0.00\text{ ms}$ overhead, $36.60\text{ FPS}$).

---

## Phase 6 Experimental Roadmap: Integration & Frontier Scaling (E47 – E54)

| Ticket | Type | Target Area | Key Hypothesis / Scientific Focus | Blocking Dependencies | Status |
|:---:|:---:|:---:|---|:---:|:---:|
| **[E47](tickets/E47-cumulative-champion-v3-integration-lineage-audit.md)** | Task | Cumulative Integration | Retrain and audit unified Champion v3 (`tlr_yolo11s_champion_v3.yaml`) combining E38–E46; reconcile metric lineage & benchmarks | None (Unblocked by E45, E46) | **Closed (Champion v3 Locked)** |
| **[E48](tickets/E48-local-view-tiny-tl-distillation.md)** | Prototype | Training-Time KD | High-resolution local crop ($64\times64$) Teacher distillation into full-frame Student $P2/P3$ pyramid; $0\text{ ms}$ inference cost | None (Unblocked by E47) | **Closed (Distillation Validated)** |
| **[E49](tickets/E49-sparse-candidate-refinement-head.md)** | Prototype | Sparse Refinement | Virtual local P1 refinement on Top-32 candidate ROIAlign ($<256\text{ px}^2$) with tiny conv head ($\le 0.5\text{ ms}$) | None (Unblocked by E47) | **Closed (Virtual P1 Refinement Validated)** |
| **[E50](tickets/E50-nwd-quality-aware-confidence-head.md)** | Prototype | Confidence Ranking | Joint classification $\times$ NWD localization quality score ($s_i = p_i^\alpha q_i^{1-\alpha}$) to eliminate rank inversions; $0\text{ ms}$ | None (Unblocked by E47) | **Closed (Quality Head Validated)** |
| **[E51](tickets/E51-scale-aware-c2-p2-feature-relay.md)** | Prototype | Low-Level Feature Relay | Lightweight scale-conditioned gated skip connection ($C2 \to P2$) to preserve high-frequency chromatic disc textures | None (Unblocked by E47) | **Closed (Feature Relay Validated)** |
| **[E52](tickets/E52-temporal-teacher-single-frame-student.md)** | Prototype | Temporal KD | 3-frame sequence ($I_{t-1}, I_t, I_{t+1}$) Teacher distillation into single-frame Student; eliminate flicker with $0\text{ ms}$ cost | None (Unblocked by E47) | **Closed (Temporal KD Validated)** |
| **[E53](tickets/E53-scale-conditioned-calibration.md)** | Task | Safety Calibration | Stratified post-hoc temperature scaling ($T_{<8}, T_{8\text{-}16}, T_{>16}$) for scale-aware uncertainty and ECE minimization | None (Unblocked by E47) | **Open (Frontier)** |
| **[E54](tickets/E54-atlas-lisa-domain-generalization-audit.md)** | Research | Domain Generalization | Zero-shot cross-dataset transfer evaluation matrix across DTLD, ATLAS, and LISA to prove out-of-domain robustness | E48, E53 | **Open** |

---

## Active Frontier

The open, unblocked Phase 6 frontier consists of:
- **[E53: Scale-Conditioned Confidence Calibration & Safety Waterfall Audit](tickets/E53-scale-conditioned-calibration.md)**



---

## Not yet specified (Fog of War)

- **Scale-Adaptive Temporal Tracking Association (Deployment)**:
  - If single-frame temporal distillation (E52) leaves residual state flicker in high-speed maneuvers, explore a zero-latency ByteTrack-style Kalman filter conditioned on Gaussian NWD spatial distance.
- **Quantization & TensorRT Deployment Engine**:
  - Post-training FP8 / INT8 quantization audit on the final Champion model with NWD-aware calibration cache to achieve $\ge 60\text{ FPS}$ on embedded Orin/Xavier platforms.

---

## Out of scope

- **Backbone Capacity Escalation (YOLO11m / YOLO11l / YOLO11x)**:
  - Ruled out to preserve resource-aware constraints; structural tiny-object bottlenecks require inductive biases, not indiscriminate parameter scaling.
- **Global Dense P1 Feature Maps (Stride 2 Dense Convolutions)**:
  - Ruled out due to catastrophic VRAM and latency explosion ($+8\text{--}15\text{ ms}$); replaced by sparse Top-32 refinement (E49).
- **Default Training with Full-Model PCGrad**:
  - Ruled out based on E46 gradient alignment findings ($\cos = +0.775$, conflict rate $2.1\%$); static loss weighting remains the production default.
- **Re-introduction of Auxiliary Contrastive / Association Losses**:
  - Permanently ruled out based on E35 causal evidence.
- **HD Maps, Lane Graphs, LiDAR Sensors**:
  - Excluded under the pure vision-centric single-camera problem formulation.
