# Baseline storica Ultralytics su ATLAS

> Questo documento descrive la pipeline preliminare YOLOv8/YOLO26 su ATLAS.
> Non è la mainline della tesi. La pipeline attiva è TLR-YOLO-MTL ed è
> descritta in `README.md` e `docs/metodologia_pipeline_attuale.md`.

Questa baseline usa il trainer ufficiale Ultralytics per la detection e la
classificazione delle 25 classi native ATLAS. È conservata per confronti di
architettura, risoluzione e P2, ma non predice relevance e non usa il contesto
delle frecce DTLD paired.

## Dataset e split ATLAS

ATLAS mantiene la propria ontologia a 25 classi pittogramma-stato. Le label
native YOLO non vengono riscritte. Il test ufficiale rimane separato; la
validation è selezionata dal train per blocchi temporali condivisi tra le
camere, con deduplicazione e controllo SHA-256.

```powershell
python scripts/convert_atlas.py --raw ../dataset_ATLAS/ATLAS --val-frac 0.1
```

## Training delle baseline

```powershell
# YOLOv8 nano, mainline della baseline storica
python scripts/check_env.py --scale n
python scripts/train.py --data atlas --model yolov8n.yaml --init coco `
  --imgsz 1280 --epochs 300 --batch 8

# Confronto YOLO26 nano
python scripts/check_env.py --family yolo26 --scale n
python scripts/train.py --data atlas --model yolo26n.yaml --init coco `
  --imgsz 1280 --epochs 300 --batch 8 --name yolo26n_atlas_coco
```

Il batching aspect-ratio-aware (`rect=true`) e le loss restano quelli
Ultralytics. Il resume richiede un checkpoint esplicito:

```powershell
python scripts/train.py --resume runs/yolov8n_atlas_coco/weights/last.pt
```

## Ablazione P2 storica

P2 è disabilitata sia nella pipeline TLR-YOLO-MTL sia nella baseline standard.
Per l'ablazione controllata ATLAS deve essere abilitata esplicitamente:

```powershell
python scripts/train.py --data atlas --init coco `
  --model yolov8n-p2.yaml --allow-p2 --imgsz 1280 --epochs 50 --batch 8 `
  --name yolov8n-p2_atlas_coco
```

Queste run misurano la detection ATLAS e non costituiscono un'ablazione della
relevance. L'esperimento scientifico principale richiede invece il confronto
local-only/context-stopgrad/paired-context descritto nella metodologia corrente.
