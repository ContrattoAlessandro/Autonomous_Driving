# TLR-YOLO-MTL — Milestone 5: relevance locale

**Stato corrente:** implementato come baseline locale della relevance.

## Architettura

La testa multi-task P3-P5 produce un logit binario di relevance per ogni
candidate semaforo:

```text
feature del livello
-> depthwise convolution 3x3
-> BatchNorm + SiLU
-> pointwise convolution 1x1
-> 1 logit relevance
```

Ogni logit usa una sigmoid indipendente. Più semafori possono quindi essere
rilevanti nella stessa immagine; non viene applicato un softmax fra istanze.

## Supervisione e score

La loss usa i match positivi del `TaskAlignedAssigner`. Un target contribuisce
solo quando il task relevance è valido per l'immagine e l'istanza ha target
binario 0/1. Le altre istanze ricevono `-1` e gradiente zero. La formulazione è
focal BCE con `gamma=2`.

Il post-processing mantiene separati:

- detection score;
- probabilità di relevance;
- `joint_score = detection_score * relevance_score`.

## Ruolo nell'esperimento

Questa branch è la baseline `local-only`. La metodologia deve confrontarla con
la relevance condizionata dalle frecce del Milestone 6, mantenendo invariati
split, sampler, seed e budget di training.

## Verifica

```powershell
.\.venv\Scripts\python.exe -B -m scripts.check_tlr_yolo_mtl_relevance
```

Il test controlla forward, target masking, assegnazione e possibilità di avere
più semafori rilevanti. Report:
`results/tlr_yolo_mtl/milestone5_relevance.json`.

Il report attualmente salvato è una verifica storica YOLO11l. Lo script usa ora
YOLO11n per default e deve essere rieseguito per la mainline.

## Stato sperimentale

La branch e la loss sono complete. Il confronto local-only versus contesto non
è ancora stato eseguito con training paired completo.
