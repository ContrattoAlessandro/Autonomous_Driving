# Quick reference — common commands

All commands run from `tl_detection/`. Assumes a venv with `pip install -r requirements.txt`.

## 0. Setup
```bash
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
python scripts/check_env.py            # verify ultralytics + GPU + YOLOv8 config + weights
```

## 1. Download + convert datasets
Datasets go under `datasets/raw/<source>/`:
- **DTLD**: https://www.traffic-light-data.com/  → `datasets/raw/dtld/`
- **Bosch STL**: https://hci.iwr.uni-heidelberg.de/node/6132  → `datasets/raw/bosch/`
- **LISA TL**: http://cvrr.ucsd.edu/LISA/lisa-traffic-light-dataset.html  → `datasets/raw/lisa/`
- **Open Images V7**: fetched automatically by fiftyone (no manual download).

```bash
python scripts/convert_dtld.py   --raw datasets/raw/dtld
python scripts/convert_bosch.py  --raw datasets/raw/bosch
python scripts/convert_lisa.py   --raw datasets/raw/lisa
python scripts/convert_oi.py     --max-images 6000
# ATLAS is already YOLO; preserve native test and build a temporal, deduplicated val split
python scripts/convert_atlas.py  --raw ../dataset_ATLAS/ATLAS --val-frac 0.1
```
Each writes `datasets/yolo/<source>/{tierA,tierB,tierC}/labels/<split>/*.txt` +
`datasets/yolo/<source>/images/<split>/*.<ext>`. `--dry-run` scans only.

## 2. Harmonize splits + EDA
```bash
python scripts/harmonize_labels.py            # builds tierA/B/C train.txt/val.txt/test.txt
python scripts/eda.py --tier all               # class + bbox-size histograms, % small objects
```
The EDA headline number (`% objects < 32 px`) justifies the high input resolution.

## 3. Train
```bash
# primary: native ATLAS 25-class model, official YOLOv8 nano without P2
python scripts/train.py --data atlas --init coco --model yolov8n.yaml --imgsz 1280 --epochs 300 --batch 8

# larger official YOLOv8 baseline, if VRAM permits
python scripts/train.py --data atlas --init coco --model yolov8s.yaml --imgsz 1280 --epochs 300 --batch 2

# short smoke/test run: change the epoch count from the command
python scripts/train.py --data atlas --init coco --model yolov8n.yaml --imgsz 1280 --epochs 10 --batch 8

# alternative project tiers (their native project taxonomy is unchanged)
python scripts/train.py --tier B --init coco

# Tier-A detection baseline (all 4 datasets)
python scripts/train.py --tier A --init coco
```
Outputs → `runs/<model-stem>_<dataset>_<init>/` (weights in `weights/best.pt`).
The mainline rejects P2 model names and uses official YOLOv8 COCO weights matched
to the selected scale. Verify a scale with `python scripts/check_env.py --scale n`.

> **Windows pagefile**: if you hit `WinError 1455` ("file di paging troppo
> piccolo"), enlarge the Windows pagefile and/or lower `--workers`. See
> `docs/fix_winerror1455.md`.

## 4. Hyperparameter search (optional, before the final run)
```bash
python scripts/tune.py --data atlas --init coco --iters 30 --epochs 40 --imgsz 960
cp runs/tune/yolov8_data_atlas_coco/best_hyperparameters.yaml configs/hyp_tuned.yaml
python scripts/train.py --data atlas --init coco --hyp configs/hyp_tuned.yaml
```

## 5. Evaluate
```bash
python scripts/eval.py --weights runs/yolov8n_atlas_coco/weights/best.pt \
       --data atlas --size --speed --onnx
```
Writes `results/eval/<run>/{ultralytics_metrics,size_stratified,speed}.json`.

## 6. Build thesis tables
```bash
python scripts/make_tables.py                # all runs under results/eval/
python scripts/make_tables.py --runs yolov8n_atlas_coco
```
Writes LaTeX `.tex` + PNG plots to `results/tables/`.

## Memory note
RTX 5070 = 12 GB. Choose the batch explicitly with `--batch N`; use
`--batch -1` only when you want Ultralytics auto-batch. If you hit CUDA OOM,
lower `--batch` or `--imgsz`.
If you hit `WinError 1455` (pagefile too small — a Windows memory
issue, NOT OOM and NOT a data/labels problem), enlarge the pagefile and/or lower
`--workers`. See `docs/fix_winerror1455.md`.
