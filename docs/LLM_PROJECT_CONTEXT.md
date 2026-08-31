# Project Context & Architectural Knowledge Base for LLM Agents

> **Repository**: `Tesi_Autonomous_Driving / tl_detection`  
> **Topic**: Resource-Aware Multi-Task Learning (MTL) for Distant/Tiny Traffic Light Perception, Fine-Grained Attribute Classification, and Geometry-Aware Cross-Attention Ego-Lane Relevance Reasoning in Autonomous Driving.  
> **Production Model**: **TLR-YOLO-MTL (YOLO11s-P2 + DySample + 14D Geometry Cross-Attention + Task-Gated ROIAlign)**.  
> **Evaluation Framework**: Unified Evaluation Contract (Canonical DTLD 5,962 validation/test images).  
> **Hardware Reference**: NVIDIA GeForce RTX 5070 12GB — Inference Throughput: **$36.6\text{ FPS}$** ($27.35\text{ ms/image}$).

---

## 1. Executive Summary & Thesis Problem Statement

Autonomous driving systems operating in complex urban environments without High-Definition (HD) maps face three fundamental, coupled perception and reasoning challenges:

1. **Extreme Scale Variation & Distant Sub-8px Perception**: Traffic lights (TLs) in high-resolution urban scenes ($960 \times 1920$) at distances $>100\text{ m}$ subtend fewer than $4\times 4$ to $8\times 8$ pixels. Standard downsampling backbones ($P3=8, P4=16, P5=32$) destroy high-frequency spatial gradients, causing high miss rates on distant signals.
2. **Fine-Grained Multi-Task Attribute Ambiguity**: Beyond bounding box detection, the autonomous stack must simultaneously resolve:
   - **State**: Red, Yellow, Green, Off (4 classes under heavy long-tail class imbalance).
   - **Signal Pictogram/Roundness**: Circular vs Arrow/Pictogram directional signal.
   - **Maneuver Direction**: Left, Straight, Right (multilabel classification).
   - **Paired Road Arrow Markings**: Detecting lane constraints and road surface arrows.
3. **Map-Less Ego-Lane Relevance Reasoning**: At multi-branch intersections, 10 to 30+ traffic lights may be simultaneously visible. The vehicle must autonomously determine **which specific light governs its own ego-lane corridor** versus lights governing adjacent turn-bays or cross-traffic, without relying on expensive HD maps or centimeter-accurate GPS.

---

## 2. Dataset Ecosystem, Taxonomy & Protocol

The system ingests and unifies datasets via a centralized JSON Lines schema (`datasets/tlr_mtl_dtld_paired/records.jsonl`):

| Dataset | Total Images | Primary Role in Project | Annotation Characteristics |
|:---|:---:|:---|:---|
| **DTLD** (DriveU Traffic Light Dataset) | 100,000+ | Primary Train, Validation & Benchmark | Dense German urban sequences ($1024 \times 2048$), high-precision TL boxes, 4-state labels, pictograms, and paired road arrows. |
| **ATLAS** | ~10,000 | External Generalization Test | Diverse weather and camera perspectives. |
| **LISA** | ~7,000 | External Generalization Test | US traffic lights, varied daylight/night conditions. |

### Splitting and Paired Filter Policy
- **Canonical Splits**: Deterministic split into `train`, `val` (5,962 benchmark images), and `test`.
- **Paired Source Requirement**: Multi-task cross-attention training requires paired traffic light + road arrow instances. The trainer strictly filters the **22,563 paired DTLD training instances** and evaluates on the canonical 5,962 validation holdout images.

---

## 3. End-to-End Architecture

```
                                  INPUT IMAGE I_t (960 x 1920 x 3)
                                                │
                      ┌─────────────────────────┴─────────────────────────┐
                      ▼                                                   ▼
           C2 (Stride 4, Raw Texture)                                 Backbone (YOLO11s C3-C5)
                      │                                                   │
                      │                                                   ▼
                      │                                           P3 Lateral Path (Stride 8)
                      │                                                   │
                      │                                                   ▼
                      │                                        DySample Dynamic Upsampler
                      │                                                   │
                      └───────────────────────────────────────────────────┤
                                                                          ▼
                                                            P2 High-Resolution (Stride 4)
                                                                          │
            ┌────────────────────────────────────┬────────────────────────┴────────────────────────┐
            ▼                                    ▼                                                 ▼
   Unified Detect (P2-P5)             Task-Gated Fusion (5x5 State)                     NWD-Quality Head
     - Traffic Lights (K=32)            - 5x5 ROIAlign State Head                         - s = p^α · q^(1-α)
     - Road Arrows (K=32)               - 3x3 ROIAlign Round/Maneuver                     - Continuous Gaussian NWD
            │                                    │                                                 │
            └────────────────────────────────────┴─────────────────────────────────────────────────┘
                                                 │
                                                 ▼
                               Geometry-Aware Cross-Attention
                                 - Top M=8 Spatial Arrow Retrieval
                                 - 14D Relative Spatial Geometry Bias
                                 - Adaptive Contextual Gate (Local Fallback)
```

### Multi-Task Loss Formulation

$$\mathcal{L}_{\text{total}} = \lambda_{\text{det}} \mathcal{L}_{\text{det}} + \lambda_{\text{state}} \mathcal{L}_{\text{state}} + \lambda_{\text{round}} \mathcal{L}_{\text{round}} + \lambda_{\text{man}} \mathcal{L}_{\text{man}} + \lambda_{\text{rel}} \mathcal{L}_{\text{rel}} + \lambda_{\text{nwd}} \mathcal{L}_{\text{nwd}}$$

Validated production loss weights:
- $\lambda_{\text{det}} = 1.00$, $\lambda_{\text{state}} = 0.75$, $\lambda_{\text{round}} = 0.50$, $\lambda_{\text{man}} = 1.00$, $\lambda_{\text{rel}} = 1.00$, $\lambda_{\text{nwd}} = 0.50$
- $\lambda_{\text{assoc}} = 0.00$, $\lambda_{\text{contrast}} = 0.00$ (Disabled to prevent gradient interference with the detection backbone).

---

## 4. Directory Structure & Key Files

```text
tl_detection/
├── configs/
│   ├── model/
│   │   ├── tlr_yolo11s_p2_dysample.yaml    # Official DySample P2 architecture
│   │   ├── tlr_yolo11s_p2.yaml             # Standard P2 architecture
│   │   └── tlr_yolo11n_p2.yaml             # Nano variant
│   ├── tlr_yolo11s_champion_v3.yaml        # Official production training configuration
│   └── tlr_yolo_mtl_data.yaml              # Dataset manifest configuration
├── datasets/
│   └── tlr_mtl_dtld_paired/                # Canonical JSONL manifests, records & splits
├── docs/
│   ├── LLM_PROJECT_CONTEXT.md              # Master knowledge base for LLM agents (this document)
│   └── metodologia_pipeline_attuale.md     # Official methodology and ablation plan
├── results/
│   ├── OFFICIAL_CHAMPION_MODEL_DECREE.md   # Official benchmark verification document
│   └── CHAMPION_MODEL_BENCHMARK_REFERENCE.md # Historical baseline reference
├── scripts/
│   ├── train_tlr_yolo_mtl.py               # Main training launcher
│   └── unified_evaluation_contract.py      # Standardized evaluation contract runner
├── tlr_yolo_mtl/
│   ├── deployment/postprocess.py           # Size-Adaptive NWD NMS & Multitask decoding
│   ├── evaluation/                         # Evaluator, greedy IoU matching, metrics
│   ├── model/
│   │   ├── dysample.py                     # DySample dynamic point upsampler
│   │   ├── geometry_attention.py           # Geometry-Aware Cross-Attention
│   │   ├── quality.py                      # NWD-Quality-Aware Confidence Head
│   │   ├── roialign_attributes.py          # Candidate-Centered 3x3/5x5 ROIAlign
│   │   ├── arrow_retrieval.py              # Query-Conditioned Arrow Selection
│   │   ├── adaptive_gate.py                # Adaptive Contextual Gate
│   │   └── unified.py                      # Unified detector module
│   └── training/
│       ├── data.py                         # Dataset, Sampler, Paired Augmentation
│       ├── engine.py                       # Training loop, AMP, TF32, EMA, Cosine LR
│       ├── losses.py                       # Multi-task criterion, Class-Balanced Softmax
│       └── tal.py                          # Scale-Adaptive NWD TAL Assigner
├── tests/                                  # 155/155 passing unit & integration tests
└── COMMANDS.md                             # CLI execution commands
```

---

## 5. Canonical Command Reference

All commands must be executed from inside `tl_detection/` using the `.venv` Python environment:

### 1. Training Champion Model
```powershell
.\.venv\Scripts\python.exe -B -m scripts.train_tlr_yolo_mtl `
  --config configs\tlr_yolo11s_champion_v3.yaml `
  --output-dir runs\tlr_yolo11s_champion_v3 `
  --overwrite
```

### 2. Unified Evaluation Contract Assessment
```powershell
.\.venv\Scripts\python.exe -u -B scripts\unified_evaluation_contract.py `
  --config configs\tlr_yolo11s_champion_v3.yaml `
  --weights runs\tlr_yolo11s_champion_v3\weights\best_composite.pt `
  --output-dir results\unified_evaluation_contract
```

### 3. Running Unit & Regression Tests
```powershell
.\.venv\Scripts\pytest.exe tests/ -v
```

---

## 6. Safety Constraints & Invariant Rules

Any future model modifications or loss tuning **MUST** strictly adhere to:

1. **Safety Floors**:
   - **Relevant Red Recall ($\tau_{95}$)**: Must remain $\ge 97.0\%$ to avoid catastrophic failure in autonomous braking.
   - **Sub-8px TL AP@50**: Must remain $\ge 38.0\%$ (preserve the P2 neck and DySample).
   - **Relevance AUPRC**: Must remain $\ge 0.9400$.
   - **Cross-Lane False Positives**: Must remain $\le 4.5\%$.
2. **Computational & Hardware Constraints**:
   - **Peak VRAM at Training**: Must stay $\le 10.5\text{ GB}$ (RTX 5070 12GB).
   - **Inference Latency**: Must remain $\le 27.5\text{ ms}$ ($\ge 36.0\text{ FPS}$).
