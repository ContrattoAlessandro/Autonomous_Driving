# tl_detection — YOLOv8 native-ATLAS traffic-light detection

Riproducible benchmark Ultralytics per la detection di semafori su immagini di
guida autonoma. La configurazione principale usa **YOLOv8 senza P2**, il
trainer ufficiale Ultralytics e input quadrati ad alta risoluzione.

## Dataset ATLAS

ATLAS mantiene la propria ontologia: **25 classi pictogramma–stato**, comprese
`off`, `red-yellow` e le frecce direzionali. Le annotazioni sono già nel formato
YOLO (`class cx cy width height`) e non vengono riscritte dal progetto.

Il paper di riferimento è Polley et al., *The ATLAS of Traffic Lights: A
Reliable Perception Framework for Autonomous Driving* (IEEE IV 2025,
[preprint](https://arxiv.org/abs/2504.19722)).

```text
dataset_ATLAS/ATLAS/
├── ATLAS_classes.yaml
├── train/{front_medium,front_tele,front_wide}/{images,labels}/
└── test/{front_medium,front_tele,front_wide}/{images,labels}/
```

Preparazione degli split:

```powershell
python scripts/convert_atlas.py --raw ../dataset_ATLAS/ATLAS --val-frac 0.1
```

Il comando conserva il test nativo e ricava validation dal train. I file
`datasets/yolo/atlas/{train,val,test}.txt` contengono percorsi assoluti per
evitare errori di working directory.

## Training

```powershell
python scripts/check_env.py --scale n
python scripts/train.py --data atlas --model yolov8n.yaml --init coco --imgsz 1280 --epochs 300 --batch 8
```

Il numero di epoche si può modificare dal comando, per esempio con
`--epochs 10`; se l'opzione viene omessa resta il valore definito in
`configs/hyp_base.yaml` (attualmente 300).

Il training usa il batching aspect-ratio-aware ufficiale di Ultralytics
(`rect=true`) come impostazione predefinita:

```powershell
python scripts/train.py --data atlas --model yolov8n.yaml --init coco --imgsz 1280 --epochs 300 --batch 8
```

Sono rifiutati esplicitamente i modelli `*-p2`. Il resume richiede il checkpoint
esplicito:

```powershell
python scripts/train.py --resume runs/yolov8n_atlas_coco/weights/last.pt
```

Prima del training `train.py` controlla percorsi, split, classi e formato delle
label. Loss, DataLoader, mosaic, letterbox e NMS restano quelli ufficiali
Ultralytics; le sole modifiche sono i valori YAML conservativi per oggetti
piccoli e classi colore/pictogramma.

## Layout

```text
configs/model/yolov8n.yaml       modello principale senza P2
configs/hyp_base.yaml            override ufficiali per il training
configs/data_atlas.yaml          nc=25, nomi ATLAS nativi
scripts/convert_atlas.py         validazione + split, senza remap label
scripts/train.py                 entry point YOLOv8
scripts/eval.py                  mAP ufficiale sul test separato
runs/                             output Ultralytics
results/                          metriche e tabelle
```
