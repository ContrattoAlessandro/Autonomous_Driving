"""Evaluate true arrow maneuver classification on the validation set."""

import json
from pathlib import Path
import cv2
import numpy as np
import torch
from tlr_yolo_mtl.deployment.export import build_full_model
from tlr_yolo_mtl.evaluation.matching import greedy_iou_match

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = Path("runs/tlr_yolo_mtl_baseline_fast/weights/best.pt")
wrapper, _ = build_full_model(checkpoint=checkpoint_path)
model = wrapper.model.to(device).eval()

records_file = Path("datasets/tlr_mtl_dtld_paired/records.jsonl")

# Direction vector channels: [LEFT, STRAIGHT, RIGHT]
CHANNELS = ["LEFT", "STRAIGHT", "RIGHT"]

all_gt_man = []
all_pred_man = []
matched_count = 0
total_gt_arrows = 0

with open(records_file, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("split") != "val":
            continue
        arrows = r.get("road_arrows", [])
        if not arrows:
            continue

        img_path = Path(r["image_path"])
        if not img_path.exists():
            continue
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            continue

        h_orig, w_orig, _ = raw_bgr.shape
        input_img = cv2.resize(raw_bgr, (1600, 800))
        rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        with torch.no_grad():
            output = model(tensor)
            raw_pred = output[1] if isinstance(output, tuple) else output

        arr_boxes = raw_pred["arrow_candidate_boxes"][0].cpu().numpy()
        arr_scores = raw_pred["arrow_candidate_scores"][0].cpu().numpy()
        arr_valid = raw_pred["arrow_candidate_valid"][0].cpu().numpy().astype(bool)
        arr_indices = raw_pred["arrow_candidate_indices"][0]
        
        man_logits = raw_pred["maneuver_logits"][0] # (3, anchors)
        # gather arrow candidates: (3, 16) -> (16, 3)
        safe_idx = arr_indices.clamp(0, man_logits.shape[-1] - 1)
        cand_man_logits = man_logits[:, safe_idx].permute(1, 0).cpu().numpy()
        cand_man_probs = 1.0 / (1.0 + np.exp(-cand_man_logits)) # (16, 3)

        # Convert arrow pred boxes to xyxy
        p_boxes = []
        p_scores = []
        p_mans = []
        for idx in range(len(arr_boxes)):
            if arr_valid[idx] and arr_scores[idx] >= 0.15:
                xc, yc, bw, bh = arr_boxes[idx]
                p_boxes.append([xc - bw/2, yc - bh/2, xc + bw/2, yc + bh/2])
                p_scores.append(arr_scores[idx])
                p_mans.append(cand_man_probs[idx])

        gt_boxes = []
        gt_mans = []
        for a in arrows:
            box = a["bbox_xyxy"]
            gt_boxes.append([box[0]/w_orig, box[1]/h_orig, box[2]/w_orig, box[3]/h_orig])
            gt_mans.append(a["direction_multihot"]) # [left, straight, right]

        total_gt_arrows += len(gt_boxes)

        if p_boxes and gt_boxes:
            p_boxes_np = np.array(p_boxes)
            p_scores_np = np.array(p_scores)
            gt_boxes_np = np.array(gt_boxes)
            matches, _, _ = greedy_iou_match(p_boxes_np, p_scores_np, gt_boxes_np, iou_threshold=0.40)
            for m in matches:
                matched_count += 1
                all_gt_man.append(gt_mans[m.target_index])
                all_pred_man.append(p_mans[m.prediction_index])

        if total_gt_arrows >= 200:
            break

print(f"\nTotal GT arrows evaluated: {total_gt_arrows}")
print(f"Matched detected arrows: {matched_count}")

if all_gt_man:
    gt_arr = np.array(all_gt_man)
    pred_arr = np.array(all_pred_man)
    pred_bin = (pred_arr >= 0.50).astype(int)

    for c_idx, c_name in enumerate(CHANNELS):
        tp = np.sum((gt_arr[:, c_idx] == 1) & (pred_bin[:, c_idx] == 1))
        fp = np.sum((gt_arr[:, c_idx] == 0) & (pred_bin[:, c_idx] == 1))
        fn = np.sum((gt_arr[:, c_idx] == 1) & (pred_bin[:, c_idx] == 0))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-6)
        print(f"Channel {c_name:<10}: Prec={prec:.3f} | Rec={rec:.3f} | Real F1={f1:.3f} (Positives: {np.sum(gt_arr[:, c_idx]==1)})")
