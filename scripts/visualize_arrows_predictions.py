"""Find and visualize images containing road arrows and traffic lights."""

import json
from pathlib import Path
import cv2
import numpy as np
import torch
from tlr_yolo_mtl.deployment.export import build_full_model
from scripts.visualize_tlr_model_predictions import draw_inference_result

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = Path("runs/tlr_yolo_mtl_baseline_fast/weights/best.pt")
wrapper, _ = build_full_model(checkpoint=checkpoint_path)
model = wrapper.model.to(device).eval()

records_file = Path("datasets/tlr_mtl_dtld_paired/records.jsonl")
out_dir = Path("results/visualizations/baseline_best")
out_dir.mkdir(parents=True, exist_ok=True)

arrow_samples = []
with open(records_file, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if len(r.get("road_arrows", [])) > 0:
            arrow_samples.append(r)
            if len(arrow_samples) >= 6:
                break

print(f"Found {len(arrow_samples)} images with road arrows. Processing...")

for i, rec in enumerate(arrow_samples):
    img_path = Path(rec["image_path"])
    if not img_path.exists():
        continue
    raw_bgr = cv2.imread(str(img_path))
    if raw_bgr is None:
        continue
    input_img = cv2.resize(raw_bgr, (1600, 800))
    rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    with torch.no_grad():
        output = model(tensor)
        raw_pred = output[1] if isinstance(output, tuple) else output

    vis = draw_inference_result(input_img, rec, raw_pred, score_thresh_tl=0.20, score_thresh_arrow=0.20)
    out_file = out_dir / f"arrow_sample_{i+1}_{img_path.stem}.jpg"
    cv2.imwrite(str(out_file), vis)
    print(f"Saved arrow visualization: {out_file}")
