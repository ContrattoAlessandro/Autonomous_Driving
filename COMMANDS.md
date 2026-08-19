# Comandi riproducibili

Eseguire i comandi da `tl_detection/`. La metodologia canonica è in
`docs/metodologia_pipeline_attuale.md`.

## 1. Ambiente

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

L'ambiente validato usa Python 3.12, PyTorch CUDA, Ultralytics, una RTX 5070 da
12 GB e ONNX opset 17.

## 2. Preparazione DTLD paired

I JPEG puliti esistenti sono in `../DTLD_jpg_plain`. Rigenerarli soltanto in
una nuova directory vuota; lo script rifiuta destinazioni non vuote e non
disegna annotazioni nei pixel.

```powershell
.\.venv\Scripts\python.exe -B -m scripts.prepare_dtld_images `
  --data-path ..\DTLD `
  --label-path ..\DTLD\v2.0 `
  --target-path ..\DTLD_jpg_plain_new
```

Rigenerazione del manifest canonico, degli split e del QA:

```powershell
.\.venv\Scripts\python.exe -B -m tlr_yolo_mtl prepare `
  --output datasets\tlr_mtl_dtld_paired --skip-overlays
```

Output atteso:

```text
datasets/tlr_mtl_dtld_paired/
├── manifest.json
├── records.jsonl
├── splits.json
└── qa_report.json
```

Il manifest può continuare a contenere i 117.038 record DTLD/ATLAS/LISA. Il
trainer filtra però soltanto i 22.563 DTLD train paired; ATLAS e LISA sono
riservati a valutazioni esterne.

QA aggiuntivo con hash o overlay:

```powershell
.\.venv\Scripts\python.exe -B -m tlr_yolo_mtl qa `
  --input datasets\tlr_mtl_dtld_paired\records.jsonl `
  --output datasets\tlr_mtl_dtld_paired\qa_hash_report.json `
  --hash-images

.\.venv\Scripts\python.exe -B -m tlr_yolo_mtl qa `
  --input datasets\tlr_mtl_dtld_paired\records.jsonl `
  --output datasets\tlr_mtl_dtld_paired\qa_overlay_report.json `
  --overlays datasets\tlr_mtl_dtld_paired\overlays `
  --overlay-fraction 0.01
```

## 3. Verifiche del modello

```powershell
.\.venv\Scripts\python.exe -B -m scripts.check_tlr_yolo_mtl_model
.\.venv\Scripts\python.exe -B -m scripts.check_tlr_yolo_mtl_unified
.\.venv\Scripts\python.exe -B -m scripts.check_tlr_yolo_mtl_training
```

I vecchi check `attributes/arrows/relevance/context` appartengono alla
pipeline storica con doppio detector e FiLM e non descrivono la mainline.

Prima del training finale sul corpus paired eseguire almeno training smoke e
probe di memoria alla risoluzione finale:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.check_tlr_yolo_mtl_training_memory `
  --batch 4 `
  --output results\tlr_yolo_mtl\paired_memory_probe_batch4.json
```

## 4. Training TLR-YOLO-MTL

Il training canonico è a **fase singola congiunta end-to-end** (`joint_training_single_phase`, 130 epoche).

Prova di un optimizer step:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.train_tlr_yolo_mtl `
  --config configs\tlr_yolo_mtl_train.yaml --batch 4 `
  --phase joint_training_single_phase --max-optimizer-steps 1 `
  --output-dir runs\tlr_yolo_mtl_unified_dtld_trial
```

Training completo (130 epoche congiunte):

### Training Finale Champion (Configurazione E36)

Training definitivo ad alta risoluzione nativa (960×1920), 130 epoche congiunte end-to-end, batch micro 2 × accumulazione 16 = effective batch 32:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.train_tlr_yolo_mtl `
  --config configs\tlr_yolo11s_champion_final.yaml `
  --output-dir runs\tlr_yolo11s_champion_final
```

Ripresa dall'ultima epoca completa del champion:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.train_tlr_yolo_mtl `
  --config configs\tlr_yolo11s_champion_final.yaml `
  --resume runs\tlr_yolo11s_champion_final\weights\last.pt
```

Ripresa dall'ultima epoca completa:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.train_tlr_yolo_mtl `
  --resume runs\tlr_yolo_mtl_unified_dtld_seed42\weights\last.pt
```

Non usare `--resume` con checkpoint FiLM/doppio-detector: il trainer richiede
lo schema unified-attention v3.

## 5. Valutazione e calibrazione

Le primitive di matching, metriche, score di selezione e temperature scaling
sono in `tlr_yolo_mtl/evaluation/`. La pipeline end-to-end di inferenza su
validation e test deve essere completata prima della selezione di `best.pt`.
Il test non deve essere usato per scegliere checkpoint, soglie o temperature.

Il confronto minimo è:

1. detector separati + relevance locale, baseline storica;
2. detector unificato + relevance locale;
3. FiLM storico;
4. cross-attention con stop-gradient;
5. cross-attention jointly trained;
6. ego-lane bias soltanto dopo avere annotato `is_ego_lane`.

## 6. Export e deployment

Verifica architetturale corrente:

```powershell
.\.venv\Scripts\python.exe -B -m scripts.check_tlr_yolo_mtl_deployment
```

L'ONNX espone 11 output, inclusi set padded, relevance e attention. Quello
prodotto da questo controllo usa il warm-start COCO e teste non
addestrate. Dopo la selezione finale deve essere rigenerato da `best.pt`, poi
validato in ONNX Runtime/TensorRT e profilato end-to-end.

## 7. Test automatici

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests
```

## 8. Baseline storiche ATLAS

La pipeline YOLOv8/YOLO26 nativa ATLAS è conservata per confronti di detection,
ma non è la mainline multi-task. Comando principale:

```powershell
python scripts/train.py --data atlas --init coco --model yolov8n.yaml `
  --imgsz 1280 --epochs 300 --batch 8
```

Per dettagli e ablazione P2 vedere `README_ambiente.md`.

## Nota memoria Windows

`WinError 1455` indica un pagefile Windows insufficiente, non un CUDA OOM e non
un errore nelle label. Ridurre `--workers` o aumentare il pagefile; vedere
`docs/fix_winerror1455.md`.
