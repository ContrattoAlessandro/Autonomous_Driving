# TLR-YOLO-MTL

Pipeline camera-only e map-less per rilevare semafori e frecce stradali e
stimare la relevance di ciascun semaforo per il veicolo ego.

## Architettura attiva

La mainline usa YOLO11n P3–P5 a input `800×1600` e viene addestrata soltanto
sul dominio DTLD paired. Il modello contiene:

- un unico detector a due tipi: `traffic_light` e `road_arrow`;
- attributi condizionali fattorizzati: stato, `round`, manovra multi-label e
  appartenenza ego-lane opzionale;
- una testa di manovra `[left, straight, right]` condivisa fra semafori
  direzionali e frecce;
- una relevance locale densa usata anche come fallback;
- una sola gated multi-head cross-attention TL→arrow con bias geometrico,
  compatibilità semantica, null token e set padded `K_TL=32`, `K_arrow=16`.

Il gate e l’ultimo strato del bias geometrico partono da zero. Al primo forward
la relevance contestuale coincide quindi esattamente con quella locale.
`round` è una wildcard appresa e non viene convertito in `[1,1,1]`.

Il campo opzionale `is_ego_lane` è già supportato da schema, loss e modello e
agisce soltanto nel bias dell’attenzione. Nella configurazione corrente resta
disabilitato perché il corpus locale non contiene ancora queste label; non
vengono usate predizioni ego-lane non supervisionate.

ATLAS e LISA rimangono nel manifest e possono essere usati come test esterni
del detector, ma non partecipano al training principale. Le 12.453 immagini
DTLD official-test restano riservate alla valutazione finale.

Il contratto completo è in
[`docs/metodologia_pipeline_attuale.md`](docs/metodologia_pipeline_attuale.md).

## Verifica rapida

Da questa directory:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests

.\.venv\Scripts\python.exe -B -m scripts.check_tlr_yolo_mtl_unified `
  --device cuda

.\.venv\Scripts\python.exe -B -m scripts.check_tlr_yolo_mtl_training `
  --device cuda --image-size 320
```

Avvio del training:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.train_tlr_yolo_mtl `
  --config configs\tlr_yolo_mtl_train.yaml --batch 4 `
  --output-dir runs\tlr_yolo_mtl_unified_dtld_seed42
```

Il batch fisico 4 viene accumulato fino al batch effettivo 32. I checkpoint
della precedente architettura FiLM/doppio detector non sono compatibili e non
devono essere ripresi con `--resume`.

## Stato

Implementati e verificati:

- schema 3.0 retrocompatibile in lettura con i manifest 2.1;
- detector unificato, attributi fattorizzati e target composti multi-hot;
- cross-attention lane-aware, fallback locale e controllo dei gradienti;
- loss multi-task con una sola assegnazione YOLO;
- training in tre fasi, post-processing class-aware ed export ONNX a 11 output;
- test unitari, forward reale e parità PyTorch/ONNX Runtime su smoke input.

Restano attività sperimentali, non di implementazione: annotare `is_ego_lane`
se si vuole attivare quel bias, eseguire training multi-seed e ablation study,
congelare un paired-test indipendente e profilare TensorRT sull’hardware target.

La pipeline YOLOv8/YOLO26 nativa ATLAS è conservata come baseline storica in
[`README_ambiente.md`](README_ambiente.md).
