# TLR-YOLO-MTL — Milestone 2: detector YOLO11 P3–P5

**Stato corrente:** YOLO11n P3–P5 è il detector della mainline.

**Riferimento canonico:** `metodologia_pipeline_attuale.md`.

## Decisione architetturale

La fase di prototipazione attiva usa YOLO11n standard con livelli P3, P4 e P5.
YOLO11l resta disponibile come baseline per una fase successiva. La testa P2 è
esclusa dalla pipeline corrente per contenere memoria, numero di candidate e
costo di training/deployment.

Il confronto di sviluppo ha misurato:

| Configurazione | Celle dense a 1600×800 | Picco forward FP16 | Esito micro-overfit |
|---|---:|---:|---|
| YOLO11n P3–P5 prototipo attivo | 26.250 | 78.599.680 byte | smoke multi-task superato |
| YOLO11l P3–P5 baseline | 26.250 | 245.435.904 byte | superato |
| YOLO11l P2–P5 sperimentale | 106.250 | 383.801.856 byte | non superato |

La configurazione attiva è in `configs/model/tlr_yolo11n.yaml`; quella large resta
in `configs/model/yolo11l.yaml`. Non esiste una configurazione P2 attiva.

## Ambiente validato

- Windows, Python 3.12.5;
- PyTorch 2.13.0+cu130;
- Ultralytics 8.4.117;
- NVIDIA GeForce RTX 5070;
- ONNX 1.22.0, opset 17.

## Warm-start COCO

Il trasferimento attivo da `yolo11n.pt` mantiene gli indici ufficiali P3–P5 e
carica 448 tensori: 2.546.000 parametri su 2.590.035 (98,30%). Restano casuali
le convoluzioni finali della classe unica e le successive teste multi-task.
I numeri YOLO11l originari restano disponibili nei report storici.

La pipeline marca il wrapper come checkpoint non resumable dopo il
trasferimento. Questo passaggio è necessario perché, altrimenti,
`YOLO.train()` ricostruisce il modello dal YAML e scarta silenziosamente i
pesi caricati in memoria.

## Forward a shape finale

Forward GPU FP16 verificato con input `[1, 3, 800, 1600]`:

| Livello | Stride | Feature |
|---|---:|---|
| P3 | 8 | `[1, 64, 100, 200]` |
| P4 | 16 | `[1, 128, 50, 100]` |
| P5 | 32 | `[1, 256, 25, 50]` |

L’output detection-only è `[1, 5, 26250]`; il nano detection-only contiene
2.590.035 parametri. Con tutte le teste MTL contiene 3.118.696 parametri.

## Micro-overfit

I risultati seguenti appartengono alla baseline YOLO11l e restano un controllo
storico; per il nano è già superato lo smoke backward multi-task, mentre il
micro-overfit detection-only non è stato ripetuto.

Il fixture usa quattro immagini DTLD pulite, da quattro sequenze diverse, con
un semaforo interno e ben visibile per immagine. Le immagini di train e
validation coincidono intenzionalmente: questo è un test di memorizzazione,
non una stima di generalizzazione.

Per limitare il costo del controllo, il micro-overfit usa lato lungo 800 e
batch 4; forward ed export restano alla shape finale 1600×800.

Risultati dopo 100 epoche:

- rapporto loss finale/iniziale: 0,06378;
- precision: 0,98837;
- recall: 1,00000;
- mAP50: 0,99500;
- mAP50–95: 0,97000;
- criterio `overfit_ok`: superato.

## Export ONNX

L’export detection-only FP16 riportato qui è quello storico YOLO11l, verificato
con checker ONNX. Il binario di prova è stato rimosso durante la pulizia perché
riproducibile e non appartiene al modello addestrato finale:

- input fisso `[1, 3, 800, 1600]`;
- output fisso `[1, 5, 26250]`;
- opset 17;
- 653 nodi;
- dimensione 50.953.869 byte.

## Verifica ancora riproducibile

```powershell
.\.venv\Scripts\python.exe -B -m scripts.check_tlr_yolo_mtl_model `
  --config configs\model\tlr_yolo11n.yaml --weights yolo11n.pt `
  --output results\tlr_yolo_mtl\prototype_nano_model_smoke_fullres.json
```

Il fixture e gli script monouso del vecchio micro-overfit sono stati rimossi;
il relativo report rimane come documentazione storica.

Report:

- `results/tlr_yolo_mtl/prototype_nano_model_smoke_fullres.json`;
- `results/tlr_yolo_mtl/milestone2_overfit.json`.

## Stato

YOLO11n P3–P5 è il detector attivo per la prototipazione; YOLO11l resta la
baseline successiva. P2 non fa parte della pipeline attiva.
