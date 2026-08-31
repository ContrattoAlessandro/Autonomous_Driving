# Metodologia Ufficiale della Pipeline TLR-YOLO-MTL: Detezione Multi-Task, Attributi Fini e Ragionamento Geometrico di Pertinenza

**Stato:** Documento Canonico Ufficiale e Relazione Metodologica Integrata  
**Modello di Riferimento Ufficiale:** **TLR-YOLO11s-P2 Multi-Task Model (Canonical Baseline Architecture)**  
**Configurazione Attiva di Riferimento:** [`configs/tlr_yolo11s_baseline.yaml`](../configs/tlr_yolo11s_baseline.yaml)  
**Decreto Ufficiale di Riferimento:** [`results/OFFICIAL_BASELINE_DECREE.md`](../results/OFFICIAL_BASELINE_DECREE.md)  
**Ultimo aggiornamento:** 31 Agosto 2026  

---

## 1. Ordine di Precedenza Contrattuale

In caso di discrepanze tra documenti, configurazioni o codice sorgente, l'ordine di precedenza vincolante è:

1. **File di Configurazione Ufficiali**:
   - [`configs/tlr_yolo11s_baseline.yaml`](../configs/tlr_yolo11s_baseline.yaml);
   - [`configs/model/tlr_yolo11s_p2_baseline.yaml`](../configs/model/tlr_yolo11s_p2_baseline.yaml);
   - [`configs/tlr_yolo11s_champion_v3.yaml`](../configs/tlr_yolo11s_champion_v3.yaml);
   - [`configs/tlr_yolo_mtl_data.yaml`](../configs/tlr_yolo_mtl_data.yaml);
   - [`configs/model/tlr_yolo11s_p2_dysample.yaml`](../configs/model/tlr_yolo11s_p2_dysample.yaml) e [`configs/model/tlr_yolo11s_p2.yaml`](../configs/model/tlr_yolo11s_p2.yaml).
2. **Codice Sorgente del Modello e dei Criteri di Addestramento**:
   - Architettura: [`tlr_yolo_mtl/model/`](../tlr_yolo_mtl/model/) (`unified.py`, `geometry_attention.py`, `roialign_attributes.py`, `arrow_retrieval.py`, `adaptive_gate.py`, `dysample.py`, `quality.py`, `refinement.py`);
   - Loss & Assigner: [`tlr_yolo_mtl/training/`](../tlr_yolo_mtl/training/) (`losses.py`, `tal.py`, `class_balanced_loss.py`, `engine.py`, `data.py`);
   - Pipeline di Dati e Augmentation: [`tlr_yolo_mtl/data/`](../tlr_yolo_mtl/data/) (`scale_matched_augmentation.py`, `photometric_augmentation.py`, `counterfactual_sampling.py`, `zoom_augmentation.py`, `taxonomy.py`).
3. **Contratto di Valutazione Standardizzato (Unified Evaluation Contract)**:
   - [`scripts/unified_evaluation_contract.py`](../scripts/unified_evaluation_contract.py) e [`tlr_yolo_mtl/evaluation/`](../tlr_yolo_mtl/evaluation/).
4. **Questo Documento Metodologico Canonico**.

---

## 2. Obiettivo Scientifico, Perimetro e Vincoli Invarianti

### 2.1 Obiettivo del Sistema
Il sistema **TLR-YOLO-MTL** (Traffic Light Recognition YOLO Multi-Task Learning) affronta in modo unificato e in tempo reale la percezione semaforica e il ragionamento spaziale di pertinenza di corsia per la guida autonoma in contesti urbani complessi.  
A partire da una **singola immagine RGB ad alta risoluzione ($960 \times 1920$)**, operando in modalità puramente visiva (**camera-only, map-less, single-frame**, senza mappe HD, senza LiDAR e senza informazioni future), la rete esegue contemporaneamente quattro compiti principali:

1. **Detezione Multi-Classe Simultanea**: Rilevamento accurato di semafori (inclusi semafori microscopici $<8\text{ px}$ o sub-4px a distanze $>100\text{ m}$) e di frecce segnaletiche orizzontali sulla superficie stradale (*road arrows*).
2. **Classificazione Fine-Grained degli Attributi Semaforici**:
   - **Stato luminoso (4 classi)**: `[Red, Yellow, Green, Off]`, supervisionato tramite *Class-Balanced Focal Softmax* con prior empirici per contrastare lo sbilanciamento estremo della coda lunga (Yellow e Off);
   - **Forma del segnale (binaria)**: `round = 1` per semafori circolari standard, `round = 0` per lanterne direzionali/pittogrammi;
   - **Manovra direzionale consentita (multi-label a 3 classi)**: `[Left, Straight, Right]`, condivisa nello stesso spazio semantico con le frecce stradali (supportando pittogrammi composti come `straight-left` $\implies [1, 1, 0]$).
3. **Ragionamento Geometrico di Pertinenza di Corsia (Ego-Lane Relevance Reasoning)**: Stima della probabilità che ciascun semaforo rilevato governi la corsia di marcia dell'ego-veicolo (*Relevant TL*) rispetto a semafori di svolta o di corsie adiacenti (*Irrelevant/Cross-Lane TLs*), mediante un modulo di *Geometry-Aware Cross-Attention* guidato dalle frecce stradali e da un descrittore spaziale-prospettico esplicito a 14 dimensioni.
4. **Calibrazione della Qualità e Post-Processing Size-Adaptive**: Decoppiamento tra probabilità di classe e qualità di localizzazione gaussiana continua (NWD Quality Score) e soppressione mirata dei duplicati per oggetti minuscoli tramite *Size-Adaptive NWD NMS*.

```
                                 IMMAGINE RGB DI INPUT (960 x 1920)
                                                 │
                      ┌──────────────────────────┴──────────────────────────┐
                      ▼                                                     ▼
           Backbone YOLO11s (C2-C5)                             Data Augmentation Suite
                      │                                          - Scale-Matched Zoom (40/35/25%)
                      ▼                                          - Paired Copy-Paste (p=0.30)
           Piramidi Neck P2-P5 (Stride 4/8/16/32)                - Photometric Lamp Bloom (p=0.50)
                      │                                          - Counterfactual 4-Tier Mining
                      ▼
        DySample Dynamic Point Upsampler (P3 -> P2, 4 gruppi)
                      │
                      ▼
         Percezione Unificata & Head Condivise (K_TL = 32, K_Arrow = 32)
                      │
       ┌──────────────┼──────────────────────────────┬──────────────────────────────┐
       ▼              ▼                              ▼                              ▼
  Detezione      Task-Gated Fusion              NWD Quality Head            State / Round /
  Unificata     ROIAlign 5x5 (State)            s = p^α(a) · q^(1-α(a))     Maneuver Heads
  (P2-P5)       ROIAlign 3x3 (Round/Maneuver)  (Score Calibrato)           (Multi-Task)
       │              │                              │                              │
       └──────────────┴──────────────────────────────┴──────────────────────────────┘
                                      │
                                      ▼
                   Geometry-Aware Cross-Attention (TL -> Arrow)
                   - Query-Conditioned Road Arrow Selection (Top M = 8)
                   - Bias Geometrico Relativo Prospettico 14D
                   - Adaptive Contextual Gate con Fallback Locale
                                      │
                                      ▼
                   Size-Adaptive NWD NMS & Multitask Post-Processing
```

### 2.2 Vincoli Invarianti e Soglie di Sicurezza Operativa
Qualsiasi iterazione sperimentale deve soddisfare i seguenti requisiti non negoziabili:
- **Relevant Red Recall ($\tau_{95}$)**: $\ge 97.0\%$ (sicurezza critica anti-tamponamento);
- **Sub-8px TL AP@50**: $\ge 38.0\%$ (mantenimento delle capacità di detezione su lunga distanza);
- **Relevance AUPRC**: $\ge 0.9400$ con tasso di falsi allarmi cross-lane $\le 4.5\%$;
- **Throughput e Latenza Real-Time**: Latenza inferenza FP16 $\le 27.5\text{ ms}$ ($\ge 36.0\text{ FPS}$) su NVIDIA RTX 5070 12GB;
- **Budget VRAM in Addestramento**: $\le 10.5\text{ GB}$ (riproducibilità su GPU consumer da 12GB).

---

## 3. Dataset, Tassonomia Fattorizzata e Mascheramento delle Loss

### 3.1 Dataset Ingestiti e Filtro "Paired"
La sorgente principale è il dataset **DTLD (DriveU Traffic Light Dataset)** ($1024 \times 2048$), arricchito con annotazioni di frecce orizzontali e pertinenza di corsia:

| Dataset | Record Totali | Ruolo nella Pipeline | Caratteristiche |
|:---|:---:|:---|:---|
| **DTLD Paired Train** | **22.563** | Addestramento Multi-Task | Record che soddisfano: `tl_detection=true`, `tl_relevance=true`, `arrow_detection=true`. |
| **DTLD Benchmark Val** | **5.962** | Selezione Checkpoint e Validazione | 25.344 TL GT, 6.108 Frecce GT. Set invariante per i benchmark comparativi. |
| **DTLD Official Test** | 12.453 | Valutazione Relevance su Immagini Singole | Immagini senza GT frecce (valutazione fallback locale). |
| **ATLAS & LISA** | ~17.000 | Generalizzazione Esterna | Test di robustezza di dominio cross-dataset (meteo avverso, layout USA). |

### 3.2 Tassonomia Fattorizzata e Mascheramento delle Loss

Ogni annotazione ground truth appartiene a una delle due macro-classi: `0 = traffic_light`, `1 = road_arrow`.

```
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
            TRAFFIC LIGHT (0)                                    ROAD ARROW (1)
        ┌───────────┼───────────┐                                       │
        ▼           ▼           ▼                                       ▼
    State (4)   Round (1)   Maneuver (3)                        Maneuver (3)
 [Red,Yel,Grn,Off] [0 / 1]  [L, S, R] (se direzionale)          [L, S, R] (multilabel)
        │
        ▼
  Relevance (1)
  (Ego-lane target)
```

- **Semaforo Circolare Standard**: `round = 1`, target manovra mascherato con `-1` (nessun gradiente generato, evitando il falso target `[1,1,1]`);
- **Semaforo Direzionale**: `round = 0`, target manovra multi-hot valido (es. `[1, 0, 0]` per solo sinistra, `[1, 1, 0]` per dritto+sinistra);
- **Freccia Stradale**: target stato, round e relevance mascherati con `-1`; target manovra multi-hot condiviso con i semafori direzionali;
- **Target `is_ego_lane`**: Supportato nel codice ma disabilitato (`ego_lane_enabled: false`) in assenza di annotazioni dense.

### 3.3 Protocollo Standardizzato (Evaluation vs Deployment)
Per garantire rigore scientifico ed eliminare bias nei confronti:

1. **Scientific Evaluation Contract ($\tau_{\text{eval}} = 0.001$)**:
   - Genera curve Precision-Recall complete a 101 punti di recall;
   - Calcola AUPRC, mAP@50, mAP@50-95 e metriche stratificate per dimensione senza troncamenti.
2. **Operational Edge Deployment ($\tau_{\text{deploy}} = 0.25, \tau_{\text{IoU}} = 0.45$)**:
   - Esegue la pipeline completa con NMS Class-Aware e Size-Adaptive NWD;
   - Misura latenza, throughput FPS, stabilità temporale e allocazione VRAM.

---

## 4. Architettura del Modello e Moduli Chiave

### 4.1 Backbone YOLO11s e Neck P2 con DySample
- **Risoluzione di Input**: $960 \times 1920$ RGB nativo;
- **Livello Piramidale P2 ad Alta Risoluzione**: Aggiunta del livello a stride 4 (mappa $240 \times 480$), fondamentale per rilevare oggetti con area inferiore a $64\text{ px}^2$;
- **DySample Dynamic Point-Based Upsampling (`tlr_yolo_mtl/model/dysample.py`)**: Nel percorso laterale $P_3 (\text{stride } 8) \to P_2 (\text{stride } 4)$, sostituisce l'interpolazione bilineare con un campionamento point-based a 4 gruppi (`groups=4`, `style='lp'`) guidato dal contenuto, generando offset 2D sub-pixel continui che preservano i contorni delle lanterne distanti.

### 4.2 Detezione Unificata e Scale-Adaptive NWD TAL Assigner
Una singola testa `Detect` a due classi (`nc=2`) supervisiona le predizioni su tutti i livelli piramidali P2–P5 (`tlr_yolo_mtl/training/tal.py`):
- **Assigner Scale-Adaptive NWD-TAL**:
  $$t = s^\alpha \cdot \left( \lambda_{\text{IoU}} \text{IoU} + (1 - \lambda_{\text{IoU}}) \text{NWD} \right)^\beta$$
  dove $\alpha = 0.5, \beta = 6.0, \text{topk} = 10, C = 12.0$.  
  Per oggetti con area $\le 64\text{ px}^2$, la transizione è continua:
  $$\text{weight} = \lambda_{\text{nwd}} \cdot \left(1 - \frac{\text{area}}{64.0}\right), \quad \text{Overlap} = (1 - \text{weight}) \cdot \text{IoU} + \text{weight} \cdot \text{NWD}$$
  La distanza di Wasserstein bidimensionale tra bounding box $B_1, B_2$ è calcolata come:
  $$W_2^2 = \|c_1 - c_2\|_2^2 + \frac{1}{4} \|s_1 - s_2\|_2^2, \quad \text{NWD} = \exp\left( - \frac{\sqrt{W_2^2}}{C} \right)$$
  eliminando il collasso a zero dei gradienti tipico dell'IoU rigido su oggetti minuscoli.

### 4.3 Estrazione Attributi tramite Task-Gated ROIAlign (`TaskGatedCandidateFeatureExtractor`)
Implementata in `tlr_yolo_mtl/model/roialign_attributes.py`:
- **State Head ROIAlign $5 \times 5$**: Campionamento bilineare espanso a $5 \times 5$ (25 punti di campionamento) per catturare la struttura verticale a 3 lenti della lanterna semaforica;
- **Round & Maneuver ROIAlign $3 \times 3$**: Campionamento compatto a $3 \times 3$ (9 punti);
- **Task-Gated Fusion ($\boldsymbol{\alpha}_t$)**: Vettori di gating apprendibili $\boldsymbol{\alpha}_t \in [0, 1]^D$ che modulano la fusione tra feature P2 (struttura fine) e P3 (contesto semantico):
  - $\text{raw\_gate}_{\text{state}} = 1.2 \implies \sigma(1.2) \approx 0.77$ (dominanza P2 per lo stato cromatico);
  - $\text{raw\_gate}_{\text{round}} = 0.5 \implies \sigma(0.5) \approx 0.62$ (P2 favorito per la forma circolare);
  - $\text{raw\_gate}_{\text{man}} = 0.0 \implies \sigma(0.0) = 0.50$ (bilanciamento P2/P3);
  - $\text{raw\_gate}_{\text{rel}} = -0.85 \implies \sigma(-0.85) \approx 0.30$ (dominanza P3 per la pertinenza di corsia).

### 4.4 Geometry-Aware Cross-Attention (TL $\to$ Arrow)
Implementata in `tlr_yolo_mtl/model/geometry_attention.py` e `arrow_retrieval.py`:
1. **Query-Conditioned Arrow Retrieval ($M=8$, `QueryConditionedArrowMatcher`)**: Per ciascun semaforo candidato ($K_{\text{TL}}=32$), seleziona le $M=8$ frecce stradali con maggiore compatibilità geometrica tramite un MLP a 10 parametri di input;
2. **Bias Geometrico Relativo 14D (`ExplicitRelativeGeometryEncoder`)**:
   Nella matrice di attenzione viene iniettato un bias calcolato da un MLP a 2 layer (`GeometryAttentionBiasMLP`, $14 \to 32 \to 4\text{ heads}$) a partire dal descrittore $\boldsymbol{\phi}_{ij}$:
   - $\Delta x / W, \Delta y / H$ (offset relativi normalizzati);
   - $\Delta x / w_{\text{TL}}, \Delta y / h_{\text{TL}}$ (offset normalizzati sulla scala del semaforo);
   - $\log(w_{\text{TL}}/w_{\text{arr}}), \log(h_{\text{TL}}/h_{\text{arr}}), \log(\text{Area}_{\text{TL}}/\text{Area}_{\text{arr}})$ (rapporti logaritmici di scala e area);
   - $y_{\text{TL}} / H, y_{\text{arr}} / H$ (posizioni verticali e profondità prospettica);
   - $(x_{\text{arr}} - x_{\text{ego}}) / W$ (offset laterale della freccia rispetto all'asse dell'ego-veicolo);
   - $\text{Directional Affinity}$ (prodotto scalare tra vettori manovra del semaforo e della freccia);
   - $p(\text{round})$ (probabilità di segnale circolare);
   - $s_{\text{TL}}, s_{\text{arr}}$ (score di confidenza di detezione dei candidati).
3. **Adaptive Contextual Gate con Null Token (`AdaptiveContextualGate`)**:
   Il gating dinamico $g_i \in [0, 1]$ è calcolato su $z_i = [f_{\text{TL}}, p(\text{round}), H(a_i), m_{\text{null}}, s_{\text{arr}}^{\max}, N_{\text{valid}}, |\text{conflict}|]$:
   $$R_i = \text{logit}_{\text{local}, i} + g_i \cdot (\Delta_{\text{ctx}, i} - \Delta_{\text{null}, i})$$
   assicurando stabilità e transizione continua al fallback locale quando non vi sono frecce rilevanti.

---

## 5. Pipeline di Data Augmentation Specializzata

Implementata in `tlr_yolo_mtl/data/`:
1. **Distribution-Aware Scale-Matched Zoom (`scale_matched_augmentation.py`)**:
   Campionamento bilanciato delle quote di scala dei semafori durante il training: $40\%$ sub-8px ($<8\text{ px}$ o $<64\text{ px}^2$), $35\%$ 8–16px, $25\%$ $>16\text{ px}$, con fattore di zoom $\in [1.20, 2.00]$ e probabilità $0.50$;
2. **Context-Preserving Paired Copy-Paste (`scale_matched_augmentation.py`)**:
   Trapianto simultaneo di semafori e delle associate frecce stradali con probabilità $0.30$, preservando la corrispondenza geometrica e la validità della label di pertinenza;
3. **Physics-Grounded Photometric Lamp Bloom (`photometric_augmentation.py`)**:
   Sintesi di alone luminoso gaussiano $\mathcal{N}(\mu_{\text{lamp}}, \sigma_{\text{bloom}}^2)$ con spettri emissivi calibrati:
   - Red: $(255, 35, 30)$, Yellow: $(255, 210, 25)$, Green: $(30, 255, 130)$;
   - Vincolo di conservazione stretta del tono di colore: $|\Delta h_{\text{HSV}}| \le 0.004$ per evitare corruzioni sintetiche delle etichette cromatiche;
4. **Counterfactual Hard-Negative Mining (`counterfactual_sampling.py`)**:
   Bilanciamento mirato delle coppie semaforo-freccia con quote 40/30/15/15:
   - $40\%$ Coppie Positive;
   - $30\%$ Negativi Standard / Facili;
   - $15\%$ Cross-Lane Confusers (frecce e semafori di corsie adiacenti nello stesso incrocio);
   - $15\%$ Spatial Neighbor Confusers (semafori montati sullo stesso braccio a sbalzo / mast-arm con $\Delta x < 100\text{ px}$).

---

## 6. Regime di Addestramento e Funzione di Loss

### 6.1 Funzione di Loss Multi-Task
Il criterio di ottimizzazione congiunto è formulato come:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{det}} \mathcal{L}_{\text{det}} + \lambda_{\text{state}} \mathcal{L}_{\text{state}} + \lambda_{\text{round}} \mathcal{L}_{\text{round}} + \lambda_{\text{man}} \mathcal{L}_{\text{man}} + \lambda_{\text{rel}} \mathcal{L}_{\text{rel}} + \lambda_{\text{nwd}} \mathcal{L}_{\text{nwd}}$$

Pesi di loss configurati in `configs/tlr_yolo11s_champion_v3.yaml`:
- $\lambda_{\text{det}} = 1.00$ (DFL + CIoU + classificazione 2-classi);
- $\lambda_{\text{state}} = 0.75$ (*Class-Balanced Focal Softmax*, $\beta = 0.9999$, prior DTLD $[0.390, 0.044, 0.482, 0.085]$ con pesi inversi al numero efficace di campioni $E_n = \frac{1 - \beta^n}{1 - \beta}$);
- $\lambda_{\text{round}} = 0.50$ (Focal BCE, $\gamma = 1.5$);
- $\lambda_{\text{man}} = 1.00$ (Focal BCE Multi-Label, $\gamma = 2.0$);
- $\lambda_{\text{rel}} = 1.00$ (Focal BCE con campionamento controfattuale 40/30/15/15, $\gamma = 2.0$);
- $\lambda_{\text{nwd}} = 0.50$ (NWD Loss sui soli semafori, costante $C = 12.0$);
- $\lambda_{\text{assoc}} = 0.00, \lambda_{\text{contrast}} = 0.00$ (Loss contrastive disabilitate per evitare conflitti di gradiente con il backbone di detezione).

### 6.2 Strategia di Addestramento a Singola Fase Congiunta
L'addestramento si articola in **50 epoche** (100 optimizer step/epoca, $3.200$ immagini/epoca):

| Iperparametro | Valore | Descrizione |
|:---|---:|:---|
| **Ottimizzatore** | AdamW | Weight decay $0.01$, gradient clip norm $10.0$ |
| **Learning Rate Backbone** | $1 \times 10^{-4}$ | Warm-start COCO (`yolo11s.pt`) |
| **Learning Rate Head / Attention** | $1 \times 10^{-3}$ | Attive fin dall'epoca 0 |
| **LR Scheduler** | Coseno Continuo | $\eta_{\min} = 1 \times 10^{-6}$ su tutte le 50 epoche |
| **Warmup Gradiente Relevance $\to$ Perception** | $0.0 \to 1.0$ | Warmup lineare continuo per stabilizzare i token iniziali |
| **Batch Effettivo** | 32 Immagini DTLD | Micro-batch 4 con 8 step di accumulo gradienti (VRAM $<9.2\text{ GB}$) |
| **Precisione** | AMP FP16 + TF32 | Scaler automatico PyTorch e casting sicuro |
| **EMA** | Decay = 0.9999 | Pesi EMA utilizzati per checkpointing e test |

---

## 7. Post-Processing e Size-Adaptive Gaussian NWD NMS

Implementato in `tlr_yolo_mtl/deployment/postprocess.py`:
$$\text{Overlap Metric}(B_1, B_2) = \begin{cases} \text{NWD}(B_1, B_2; C=12.0) & \text{se } \min(\text{area}_1, \text{area}_2) < 64.0\text{ px}^2 \\ \text{IoU}(B_1, B_2) & \text{altrimenti} \end{cases}$$
Soglia di soppressione: $\tau_{\text{NWD}} = 0.50$ per semafori minuscoli, $\tau_{\text{IoU}} = 0.45$ per frecce e semafori standard.

---

## 8. Limiti Aperti e Sviluppi Futuri

1. **Risoluzione Ottica Estrema ($<3\text{ px}$)**: Semafori a distanze estreme ($>150\text{ m}$) sottendono pochissimi pixel sul sensore. L'esplorazione di crop patch locali ad alta risoluzione o architetture multi-scala avanzate rimane una direzione aperta.
2. **Annotazione Esplicita di Corsia**: La pertinenza è attualmente inferita correlando le frecce orizzontali e la geometria prospettica. L'aggiunta di maschere poligonali di corsia o label dirette `is_ego_lane` consentirà una supervisione ancora più diretta.
3. **Robustezza a Domini Esterni**: Estendere l'addestramento con strategie di data augmentation fotometrica mirata per migliorare la confidenza in scenari notturni, pioggia battente o riflessi intensi.
