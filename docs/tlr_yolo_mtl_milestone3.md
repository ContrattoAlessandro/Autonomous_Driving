# TLR-YOLO-MTL — Milestone 3: stato e pittogramma

**Stato corrente:** implementato e attivo sul modello YOLO11n P3-P5.

## Architettura

La detection dei semafori resta a classe unica (`vehicle_traffic_light`). Su
ciascuna feature P3, P4 e P5 sono aggiunte due torri leggere indipendenti:

```text
feature del livello
-> depthwise convolution 3x3
-> BatchNorm + SiLU
-> pointwise convolution 1x1
-> 4 logits
```

Tassonomie:

- stato: `red`, `yellow`, `green`, `off`;
- pittogramma: `round`, `left`, `straight`, `right`.

Il modello attivo è YOLO11n senza P2; il modello multi-task completo, dopo
l'aggiunta di tutte le milestone, contiene circa 3,12 milioni di parametri.

## Associazione e loss masking

I logits seguono lo stesso ordine delle 26.250 candidate dense del detector.
La loss riusa `foreground_mask` e `target_gt_idx` del `TaskAlignedAssigner`:

- soltanto i match positivi ricevono supervisione attributo;
- ogni positivo eredita il target della ground truth assegnata alla box;
- target `-1` indica label assente, sconosciuta o non rappresentabile;
- i target mascherati producono loss connessa zero e gradiente zero;
- gli indici mantenuti dalla NMS raccolgono attributi dalla stessa candidate.

DTLD, ATLAS e LISA contribuiscono secondo le rispettive validity mask. Classi
composte non rappresentabili non vengono forzate in una classe errata.

## Verifica

```powershell
.\.venv\Scripts\python.exe -B -m scripts.check_tlr_yolo_mtl_attributes
```

Il controllo esegue forward FP16 a 800x1600, verifica tassonomie, masking e
backward tramite il vero assegnatore YOLO. Il report viene scritto in
`results/tlr_yolo_mtl/milestone3_attributes.json`.

Il report attualmente salvato è una verifica storica YOLO11l. Lo script usa ora
YOLO11n per default e deve essere rieseguito per aggiornare l'artefatto della
mainline.

## Stato sperimentale

Architettura e loss sono complete. Le metriche finali di stato e pittogramma
richiedono il training paired completo e la successiva inferenza su validation
e test.
