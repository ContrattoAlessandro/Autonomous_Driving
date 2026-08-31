# Relazione Scientifica Ufficiale: Modello Baseline Canonico di Tesi

**Status Ufficiale:** **MODELLO BASELINE DI RIFERIMENTO CANONICO (LOCKED BASELINE)**  
**Architettura:** **TLR-YOLO11s-P2 Baseline (DySample + Task-Gated 5x5 ROI + 18D Geometry Attention)**  
**Configurazione Ufficiale:** [`configs/tlr_yolo11s_baseline.yaml`](../configs/tlr_yolo11s_baseline.yaml)  
**Grafo Modello:** [`configs/model/tlr_yolo11s_p2_baseline.yaml`](../configs/model/tlr_yolo11s_p2_baseline.yaml)  
**Dataset di Riferimento:** DTLD Benchmark Invariante ($5.962$ Immagini, Risoluzione $960 \times 1920$)  
**Target Hardware di Riferimento:** NVIDIA GeForce RTX 5070 12GB — Latenza FP16: **`26.92 ms`** (**`37.15 FPS`**)  
**Data di Chiusura:** 31 Agosto 2026  

---

## 1. Quadro Comparativo Ufficiale e Benchmark di Riferimento

Il Modello Baseline costituisce il riferimento immutabile e verificato per la Tesi di Laurea Magistrale, stabilito a seguito della sintesi scientifica completa dei moduli multi-task.

### Tabella Prestazionale Ufficiale (Validazione $5.962$ Immagini DTLD)

| Dimensione di Valutazione | Valore Baseline Canonico | Descrizione / Standard di Sicurezza |
| :--- | :---: | :--- |
| **Selection Composite Score** | **`0.8320`** | Metrica composita multi-task standard di riferimento |
| **mAP@50 Globale (Joint)** | **`85.16%`** | Detezione congiunta semafori e frecce direzionali |
| **AP@50 Traffic Light** | **`75.48%`** | Precisione di detezione semaforica complessa |
| **AP@50 Road Arrow** | **`94.85%`** | Precisione di detezione frecce a terra |
| **Sub-8px AP@50 ($< 64\text{ px}^2$)** | **`46.10%`** | Detezione ad altissima risoluzione per semafori lontani |
| **Relevance AUPRC** | **`94.70%`** | Area under precision-recall per semafori di corsia ego |
| **Relevant Red Recall ($\tau_{95}$)** | **`96.80%`** | Safety Floor garantito anti-tamponamento |
| **State Accuracy Globale** | **`95.42%`** | Accuratezza colore stato semaforico |
| **State Macro-F1** | **`91.28%`** | Macro F1 bilanciato tra tutti i 4 stati (Red, Green, Yellow, Off) |
| **Sub-4px State Accuracy** | **`52.18%`** | Riconoscimento dello stato su semafori $< 4\text{ px}$ |
| **Latenza FP16 (Batch=1)** | **`26.92 ms`** (**`37.15 FPS`**) | Conformità al vincolo edge real-time ($\le 27.5\text{ ms}$) |

---

## 2. Architettura Consolidata della Baseline

1. **Backbone e Neck P2 High-Resolution**:
   - YOLO11s con 4 livelli di piramide ($P_2, P_3, P_4, P_5$), risoluzione $960 \times 1920$ e stride $4$.
2. **DySample Dynamic Point-Sampling**:
   - Upsampling point-based a 4 gruppi sul percorso laterale $P_3 \to P_2$, eliminando le sfocature dell'interpolazione bilineare.
3. **Task-Specific Gated Fusion & 5x5 ROIAlign**:
   - Estrazione feature a griglia $5 \times 5$ per la State Head con pesi $\alpha_t$ specifici per compito.
4. **18D Vanishing Point Geometry Attention**:
   - Cross-attention geometrica tra semafori e frecce con descrittore prospettico $[\Delta x_{\text{vp}}, \Delta y_{\text{vp}}, \text{dist}_{\text{horizon}}, \theta_{\text{road}}]$.
5. **Class-Balanced Focal Loss**:
   - Ponderazione $\beta=0.9999$ sui conteggi effettivi per mitigare lo sbilanciamento delle classi rare (giallo, spento).
6. **Size-Adaptive NWD Post-Processing**:
   - NMS selettivo con metrica di Wasserstein gaussiana per oggetti piccoli ($< 64\text{ px}^2$) e IoU per oggetti grandi.

---

## 3. Artefatti Canonici

* **Configurazione Canonica**: [`configs/tlr_yolo11s_baseline.yaml`](../configs/tlr_yolo11s_baseline.yaml)
* **Grafo Modello**: [`configs/model/tlr_yolo11s_p2_baseline.yaml`](../configs/model/tlr_yolo11s_p2_baseline.yaml)
* **Contratto di Valutazione**: [`scripts/unified_evaluation_contract.py`](../scripts/unified_evaluation_contract.py)
