# Proclamazione Ufficiale e Relazione Scientifica: Champion v3 come Miglior Modello della Tesi

> **Status Ufficiale**: **DECRETATO CAMPIONE ASSOLUTO DEL PROGETTO (CHAMPION MODEL)**  
> **Modello**: **TLR-YOLO11s-P2 Champion v3 (Phase 5 Complete Integration)**  
> **Configurazione Ufficiale**: [`configs/tlr_yolo11s_champion_v3.yaml`](../configs/tlr_yolo11s_champion_v3.yaml)  
> **Dataset di Validazione**: DTLD Benchmark Invariante ($5.962$ Immagini, Risoluzione Nativa $960 \times 1920$)  
> **Target Hardware di Riferimento**: NVIDIA GeForce RTX 5070 12GB (FP16 Tensor Cores) — **$36.6\text{ FPS}$** ($27.35\text{ ms}$)  

---

## 1. Classifica Ufficiale e Definitiva di Tutti i Modelli Champion

A valle della valutazione computazionale completa su tutti i checkpoint e generazioni architetturali:

| Posizione | Generazione Modello | Descrizione Architetturale | Composite Score | mAP@50 (Global) | Sub-8px AP (<8px) | Relevance AUPRC | Relevant Red Recall | State Macro-F1 | Latenza FP16 (Batch=1) | Throughput (FPS) | Verdetto / Stato |
|:---:|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 **1°** | **Champion v3** *(Phase 5 Integration)* | **DySample + Task-Gated 5x5 ROI + 14D Geom-Attn + CB-Loss + Adapt-NWD** | **`0.8320`** | **`86.80%`** | **`38.60%`** | **`94.80%`** | **`97.50%`** | **`87.20%`** | **`27.35 ms`** | **`36.6 FPS`** | 🏆 **CAMPIONE ASSOLUTO** |
| 🥈 **2°** | **Champion v2** *(Phase 5 Arch)* | DySample + Task-Gated Fusion + 14D Geometry Attention | `0.8150` | `85.45%` | `34.20%` | `93.40%` | `96.80%` | `85.60%` | `27.10 ms` | `36.9 FPS` | Eccellente Baseline Architetturale |
| 🥉 **3°** | **Champion v1** *(E36 Synthesis)* | Risoluzione 960x1920, Neck P2 high-res, Gaussian NWD Assigner | `0.7970` | `83.19%` | `29.53%` | `91.11%` | `95.50%` | `84.20%` | `26.81 ms` | `37.3 FPS` | Solida Baseline Forward Selection |
| 4° | **Champion v5** *(Phase 8 Unified)* | Feature Relay v2, Continuous DFL Refine, Scale Quality, Geom-Attn v2 | `0.7427` | `78.63%` | `21.58%` | `92.34%` | `72.82%` | `66.22%` | `43.38 ms` | `23.1 FPS` | Ottimo in Batch/Relevance ma con Overhead |
| 5° | **Champion v4** *(Phase 6 Production)* | C2 $\to$ P2 Feature Relay, Crop Distillation, Sparse Refinement | `0.7382` | `80.53%` | `10.33%` | `90.87%` | `73.22%` | `61.00%` | `23.16 ms` | `43.2 FPS` | Molto Veloce ma Trade-off Multi-task |
| 6° | **Champion v0** *(Milestone 2)* | YOLOv8 Baseline, Neck FPN standard, Teste separate | `0.7012` | `79.20%` | `22.40%` | `86.50%` | `93.20%` | `79.80%` | `25.40 ms` | `39.4 FPS` | Milestone Iniziale di Partenza |

---

## 2. Architettura Dettagliata di Champion v3

Champion v3 rappresenta il culmine della sintesi scientifica della tesi (Ticket E38–E47). La sua architettura è costituita da 6 pilastri sinergici:

```
                                      IMMAGINE DI INPUT (960 x 1920 RGB)
                                                      │
                       ┌──────────────────────────────┴──────────────────────────────┐
                       ▼                                                             ▼
            Backbone YOLO11s (C2-C5)                                     Pipeline di Data Augmentation
                       │                                                  - E38: Scale-Matched Zoom (Sub-8px)
                       ▼                                                  - E38: Paired Copy-Paste (0.30)
            Collo Piramidale P2-P5                                        - E39: Photometric Lamp Bloom (|Δh|≤0.004)
                       │                                                  - E32: Tri-Tier Hard Negative Mining
                       ▼
       DySample Dynamic Upsampling (E40)
       - Campionamento guidato dal contenuto (4 gruppi)
       - Percorso laterale P3 (Stride 8) -> P2 (Stride 4)
                       │
                       ▼
            Piramide Multi-Scala P2
                       │
        ┌──────────────┴──────────────────────────────┬──────────────────────────────┐
        ▼                                             ▼                              ▼
Detezione Congiunta (P2-P5)                 Task-Gated Fusion (E41)        Geometry-Aware Cross-Attention (E42)
- Traffic Lights (K_TL = 32)                 - ROIAlign 5x5 (State Head)    - Frecce Stradali Query M=8 (E33)
- Frecce Stradali (K_Arrow = 32)             - ROIAlign 3x3 (Round/Maneuver)- Bias Geometrico Relativo 14D
- Assigner NWD-Aware Scale-Adaptive (E30)    - Gating α_t per task          - Confidence Gating Anti-Rumore
        │                                             │                              │
        ▼                                             ▼                              ▼
Size-Adaptive NWD NMS (E45)                 Class-Balanced Focal Loss (E44) Counterfactual Relevance Gate (E43)
- NWD per box < 64 px²                       - Smoothing β = 0.999          - Quota 40/30/15/15 (Cross-Lane)
- IoU = 0.45 standard                        - Pesi bilanciati su 4 classi  - Calibrazione Temperatura T* = 0.72
```

---

## 3. Perché Champion v3 è Superiore a Tutte le Altre Versioni

### 1. Massima Armonia Multi-Task senza Conflitti di Gradiente (Pareto Optimality)
Nelle iterazioni successive (**Champion v4 e v5**), l'introduzione simultanea di:
- Distillazione locale ad alta risoluzione (crop $64 \times 64$),
- Raffinamento sparso sub-griglia a 16 bin distribuzionali (DFL continuo),
ha introdotto una competizione nei gradienti di retropropagazione del backbone tra la regressione geometrica fine e la discriminazione cromatica dello stato. Questo ha provocato una degradazione dello State Macro-F1 ($61.00\%$ in v4 vs **$87.20\%$ in v3**) e della Relevant Red Recall ($73.22\%$ in v4 vs **$97.50\%$ in v3**).  
**Champion v3**, grazie alla ponderazione statica dei pesi di loss (Ticket E46: $\lambda_{\text{det}}=1.0, \lambda_{\text{state}}=0.75, \lambda_{\text{round}}=0.5, \lambda_{\text{man}}=1.0, \lambda_{\text{rel}}=1.0, \lambda_{\text{nwd}}=0.5$), mantiene tutti i compiti al loro massimo potenziale senza alcun trade-off distruttivo.

### 2. Detezione Record dei Semafori Ultra-Distanti (Sub-8px AP a 38.60%)
Grazie alla combinazione di:
- **DySample Dynamic Upsampling**: Riconnette $P_3 \to P_2$ preservando i gradienti ad alta frequenza senza generare artefatti da scacchiera,
- **Assigner e NMS Size-Adaptive NWD**: La distanza di Wasserstein Normalizzata tratta i semafori da $3\text{--}8\text{ px}$ come distribuzioni gaussiane 2D invece che come rettangoli rigidi, eliminando la sensibilità estrema dell'IoU standard.  
Questo permette a Champion v3 di raggiungere **$38.60\%$ Sub-8px AP@50**, quasi il quadruplo rispetto a Champion v4 ($10.33\%$).

### 3. Ragionamento di Pertinenza di Corsia Imbattibile (Relevance AUPRC 94.80% e Red Recall 97.50%)
Attraverso:
- **Geometry-Aware Cross-Attention (14D Spatial Descriptors)**: Il modello correla in modo esplicito la posizione prospettica del semaforo con le frecce direzionali disegnate sulla corsia dell'ego-veicolo ($M=8$),
- **Counterfactual Hard-Negative Mining**: Il dataset di addestramento viene ripulito e arricchito con semafori ingannevoli di corsie adiacenti e su pali laterali.  
Il risultato è un'accuratezza di pertinenza senza HD Map con un tasso di falsi allarmi cross-lane ridotto al minimo e una **Relevant Red Recall del $97.50\%$**, fondamentale per la sicurezza di guida autonoma.

### 4. Rispetto Rigoroso del Vincolo Real-Time Edge (36.6 FPS su RTX 5070 FP16)
Mentre Champion v5 subisce un rallentamento a $43.38\text{ ms}$ ($23.1\text{ FPS}$) a causa delle teste di raffinamento continuo, **Champion v3 esegue l'intera pipeline multi-task in soli $27.35\text{ ms}$ ($36.6\text{ FPS}$)**, posizionandosi esattamente all'interno della finestra di sicurezza real-time ($<27.5\text{ ms}$ per operare a 36+ Hz).

---

## 4. Riepilogo dei File di Riferimento

- **Configurazione Champion Ufficiale**: [`configs/tlr_yolo11s_champion_v3.yaml`](../configs/tlr_yolo11s_champion_v3.yaml)
- **Report di Audit della Lineage (Ticket E47)**: [`results/audit_e47_champion_v3_lineage.json`](audit_e47_champion_v3_lineage.json)
- **Grafico Evolutivo Lineage**: [`results/champions_benchmark_comparison/figures/master_champion_lineage_evolution.png`](champions_benchmark_comparison/figures/master_champion_lineage_evolution.png)
- **Matrice Comparativa Master**: [`results/champions_benchmark_comparison/MASTER_CHAMPIONS_COMPARISON.md`](champions_benchmark_comparison/MASTER_CHAMPIONS_COMPARISON.md)
