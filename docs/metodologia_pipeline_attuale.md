# Metodologia corrente: detector unificato e cross-attention TL→arrow

**Stato:** contratto canonico implementato  
**Ultimo aggiornamento:** 18 agosto 2026

In caso di conflitto prevalgono, nell’ordine:

1. `configs/tlr_yolo_mtl_train.yaml` e `configs/tlr_yolo_mtl_data.yaml`;
2. il codice in `tlr_yolo_mtl/model/unified.py` e
   `tlr_yolo_mtl/training/losses.py`;
3. questo documento;
4. le milestone precedenti, conservate come storia del progetto.

## 1. Obiettivo e perimetro

Da una singola immagine RGB il sistema deve:

1. rilevare semafori e frecce stradali;
2. stimare stato, forma round e manovre direzionali;
3. stimare la relevance binaria di ogni semaforo;
4. rendere ispezionabile l’associazione TL→arrow senza presentare
   l’attention come prova causale del ragionamento.

La pipeline è camera-only, map-less e single-frame. Non usa HD map, LiDAR,
lane graph, dati futuri o route token. Il codice supporta un target opzionale
`is_ego_lane` per le frecce, ma la configurazione attiva lo disabilita finché
queste label non saranno annotate.

## 2. Dati e split

Il manifest locale contiene DTLD, ATLAS e LISA, ma il dataloader attivo usa
esclusivamente record DTLD train che soddisfano contemporaneamente:

```text
traffic_light_detection = true
traffic_light_relevance = true
arrow_detection = true
```

Sono quindi utilizzate 22.563 immagini train paired. Le 5.962 immagini DTLD
validation restano per selezione del checkpoint e calibrazione. Le 12.453
immagini official-test non entrano mai nel training; non avendo GT frecce,
servono per la relevance finale ma non chiudono da sole l’analisi causale del
contesto. ATLAS e LISA sono disponibili soltanto come test esterni del
detector.

Il reader accetta il manifest storico schema 2.1 e lo migra in memoria al
contratto 3.0. Non è quindi obbligatorio rigenerare subito il corpus.

## 3. Tassonomia fattorizzata

Ogni ground truth appartiene a uno dei due tipi:

```text
0 = traffic_light
1 = road_arrow
```

Le uscite condizionali sono:

| Uscita | Traffic light | Road arrow |
|---|---:|---:|
| box + tipo | sì | sì |
| stato `[red,yellow,green,off]` | sì | mascherato |
| `round` binario | sì | mascherato |
| manovra `[left,straight,right]` multi-label | direzionale | sì |
| `is_ego_lane` binario | mascherato | opzionale |
| relevance | sì | mascherato |

I pittogrammi TL composti `straight-left`, `straight-right` e `left-right`
sono target multi-hot validi. Un TL round ha `round=1` e loss manovra
mascherata: non viene creato il falso target `[1,1,1]`.

## 4. Modello

### 4.1 Perception condivisa

YOLO11n usa backbone e neck P3–P5, stride 8/16/32 e warm-start COCO. La testa
`Detect` ha `nc=2` e produce un solo tensore denso `[B,6,A]`: quattro parametri
box e due score di tipo. Non esiste una seconda istanza di `Detect` per le
frecce.

Una sola assegnazione `TaskAlignedAssigner` allinea box, tipo e attributi. Le
loss condizionali leggono lo stesso `foreground_mask` e `target_gt_idx`; i
target `-1` eliminano ogni gradiente per attributi non applicabili.

La testa manovra è condivisa dai due tipi, così semafori direzionali e frecce
vivono nello stesso spazio semantico.

### 4.2 Set di candidati

Dalle predizioni decodificate vengono selezionati set a dimensione fissa:

```text
K_TL    = 32
K_arrow = 16
```

La top-k conserva gli indici nel tensore denso e produce una mask di validità.
La selezione degli indici non è differenziabile; il sistema va quindi definito
“jointly trainable”, non end-to-end differenziabile in senso stretto.

Per evitare che l’avvio casuale della top-k privi la relevance di supervisione,
esiste anche una testa locale densa. Essa riceve la relevance loss su tutti gli
anchor positivi TL fin dal primo passo.

### 4.3 Token

Ogni token usa feature locali proiettate, posizione/scala normalizzata della
box, attributi semantici e confidence di detection:

```text
q_TL = W_TL [feature | PE(box) | p(round) | p(maneuver) | score_TL]
a_AR = W_AR [feature | PE(box) | p(maneuver) | u_ego | score_AR]
```

La dimensione attiva è `D=128`, con quattro attention head. `u_ego` vale 0,5
neutro quando `ego_lane_enabled=false`; non vengono introdotti lane token.

### 4.4 Cross-attention lane-aware

Una sola cross-attention usa i TL come query e le frecce come key/value. A ogni
coppia viene aggiunto un bias prodotto da un MLP sui sei valori:

```text
[Δx, Δy, log(w_TL/w_AR), log(h_TL/h_AR), u_ego, compatibilità]
```

La compatibilità dei segnali direzionali deriva dalla sovrapposizione delle
probabilità multi-label. Per i segnali round viene usato un parametro wildcard
appreso. L’ultimo layer del bias è zero-inizializzato.

Un null token è sempre valido, anche quando nessuna freccia supera la soglia.
Il residual è:

```text
z = LayerNorm(q + alpha * Attention(q, arrows + null))
```

con `alpha=0` all’inizializzazione. La relevance finale è la relevance locale
più una correzione contestuale costruita da `MLP([q,z])`; la correzione viene
sottratta rispetto allo stesso MLP nel caso locale. Perciò, quando `alpha=0`,
il risultato è numericamente identico al fallback locale.

## 5. Loss

Il criterio attivo è:

```text
1.00 * detection unificata (DFL + CIoU + classificazione a due tipi)
0.75 * stato
0.50 * round
1.00 * manovra multi-label
0.50 * ego-lane opzionale
1.00 * relevance locale/contestuale
0.50 * NWD sui soli semafori
0.00 * association supervision, predisposta ma disabilitata
```

Stato usa focal cross-entropy; round, manovra, ego-lane e relevance usano focal
BCE. Le ignore region continuano a rimuovere dalla classificazione gli anchor
background che cadono in aree non supervisionabili.

## 6. Controllo dei gradienti e training

Il training canonico dura 130 epoche in una **singola fase congiunta end-to-end** (`joint_training_single_phase`):

| Configurazione | Valore | Note |
|---|---:|---|
| Epoche totali | 130 | 100 optimizer step/epoca (3.200 img campionate/epoca) |
| Perception (Backbone + Neck) | Trainabile | Warm-start COCO, LR iniziale `1e-4` |
| Head & Cross-Attention | Trainabili | Attive fin dall'epoca 0, LR iniziale `1e-3` |
| LR Scheduler | Coseno | `eta_min = 1e-6` unico e continuo su tutte le 130 epoche |
| Gradiente relevance→perception | $0.0 \to 1.0$ | Warmup lineare continuo per stabilizzare i token iniziali |
| Batch effettivo | 32 DTLD | Micro-batch 16 (accumulo 2) o micro-batch 4 (accumulo 8) |

Sulle feature freccia il gradiente contestuale è inoltre moltiplicato per 0,25
solo per record realmente paired. Il valore forward non cambia. Il batch
effettivo è 32 DTLD; non esiste più una quota ATLAS/LISA.

> [!NOTE]
> **Superamento del training a 3 fasi**: La precedente strategia a 3 fasi
> (`perception_and_local_relevance` $\to$ `cross_attention` $\to$ `joint_finetuning`)
> introduceva congelamenti artificiali della perception (20 epoche) e discontinuità
> nel learning rate scheduler. I confronti sperimentali hanno dimostrato che il
> training congiunto a fase singola con warmup lineare dello scaling del gradiente
> ($0.0 \to 1.0$) ottiene prestazioni equivalenti eliminando l'overhead e la complessità
> multi-fase, risultando nella scelta metodologica ufficiale più pulita e riproducibile.

Il trainer conserva AdamW, AMP FP16, gradient clipping (norm 10.0), EMA (0.9999),
accumulo e checkpoint atomici. Lo schema checkpoint v3 impedisce il resume
accidentale dei checkpoint FiLM/doppio-detector.

## 7. Inferenza, post-processing ed export

L’export ONNX fisso espone 11 tensori:

1. detection unificata;
2. stato;
3. round;
4. manovra;
5. ego-lane;
6. indici candidati TL;
7. mask candidati TL;
8. indici candidati frecce;
9. mask candidati frecce;
10. relevance dei 32 candidati TL;
11. pesi di attention `[B,4,32,17]`, incluso il null token.

Il post-processing esegue NMS class-aware sui set selezionati e conserva sia
l’indice denso sia lo slot top-k. Detection score, relevance probability e
`joint_score = detection × relevance` rimangono separati.

La parità PyTorch/ONNX Runtime viene confrontata dopo canonicalizzazione degli
slot top-k, perché elementi con score identico possono essere restituiti in
ordine diverso senza cambiare il risultato semantico.

## 8. Valutazione e ablation study

Le metriche minime sono AP TL tiny, AP frecce, macro-F1 stato, F1 round,
macro-F1 manovra e AUPRC relevance. Il confronto sperimentale deve isolare:

1. detector separati + local-only, baseline storica;
2. detector unificato + local-only;
3. FiLM storico;
4. cross-attention frecce;
5. cross-attention + ego-lane bias, solo dopo annotazione;
6. oracle con frecce/ego-lane ground truth.

Controlli negativi obbligatori:

- permutare le frecce fra immagini;
- rimuovere le frecce più attenzionate;
- confrontare attention e perturbazioni senza chiamare l’attention una prova
  causale o “XAI inconfutabile”.

## 9. Limiti ancora aperti

- Il 5° percentile dei TL DTLD è di pochi pixel: la cross-attention non può
  recuperare oggetti non rilevati. P2/high-resolution resta un’ablation
  separata, non è attiva nella mainline.
- DTLD relevance dipende dalla rotta pianificata, che una singola immagine può
  rendere ambigua. Il sistema corrente dichiara esplicitamente il vincolo
  camera-only e non usa informazioni future.
- `is_ego_lane` richiede nuove annotazioni. Finché mancano, il relativo bias è
  neutro e disabilitato nella configurazione.
- Il test ufficiale non possiede GT frecce: serve annotarlo o congelare un
  holdout paired per sequenza prima degli esperimenti finali.
- Latenza TensorRT e memoria a risoluzione piena vanno misurate, non assunte.
