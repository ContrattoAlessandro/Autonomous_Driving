"""Print detailed detection table for the visualized images."""

import json
from pathlib import Path
import cv2
import numpy as np
import torch
from tlr_yolo_mtl.deployment.export import build_full_model

STATE_NAMES = ["red", "yellow", "green", "off"]
MANEUVER_NAMES = ["straight", "left", "right", "straight_left", "straight_right", "left_right", "uturn", "other"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = Path("runs/tlr_yolo_mtl_baseline_fast/weights/best.pt")
wrapper, _ = build_full_model(checkpoint=checkpoint_path)
model = wrapper.model.to(device).eval()

records_file = Path("datasets/tlr_mtl_dtld_paired/records.jsonl")

samples = []
with open(records_file, "r", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        if r.get("split") == "val":
            tls = r.get("traffic_lights", [])
            arrows = r.get("road_arrows", [])
            if any(tl.get("relevance") == 1 for tl in tls) and any(tl.get("relevance") == 0 for tl in tls):
                samples.append(r)
                if len(samples) >= 3:
                    break

for i, rec in enumerate(samples):
    img_path = Path(rec["image_path"])
    raw_bgr = cv2.imread(str(img_path))
    input_img = cv2.resize(raw_bgr, (1600, 800))
    rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

    with torch.no_grad():
        output = model(tensor)
        raw_pred = output[1] if isinstance(output, tuple) else output

    tl_boxes = raw_pred["traffic_candidate_boxes"][0].cpu().numpy()
    tl_scores = raw_pred["traffic_candidate_scores"][0].cpu().numpy()
    tl_valid = raw_pred["traffic_candidate_valid"][0].cpu().numpy().astype(bool)
    tl_indices = raw_pred["traffic_candidate_indices"][0]

    st_logits = raw_pred["state_logits"][0]
    dense_states = st_logits.argmax(dim=0).cpu().numpy()
    tl_states = dense_states[tl_indices.clamp(0, st_logits.shape[-1] - 1).cpu().numpy()]

    rel_logits = raw_pred["relevance_logits"][0, 0].cpu().numpy()
    rel_probs = 1.0 / (1.0 + np.exp(-rel_logits))

    arr_boxes = raw_pred["arrow_candidate_boxes"][0].cpu().numpy()
    arr_scores = raw_pred["arrow_candidate_scores"][0].cpu().numpy()
    arr_valid = raw_pred["arrow_candidate_valid"][0].cpu().numpy().astype(bool)
    arr_indices = raw_pred["arrow_candidate_indices"][0]
    man_logits = raw_pred["maneuver_logits"][0]
    dense_man = man_logits.argmax(dim=0).cpu().numpy()
    arr_man = dense_man[arr_indices.clamp(0, man_logits.shape[-1] - 1).cpu().numpy()]

    print(f"\n==================== SAMPLE {i+1}: {img_path.name} ====================")
    print("--- GROUND TRUTH TRAFFIC LIGHTS ---")
    for tl in rec.get("traffic_lights", []):
        st = tl.get("state")
        rel = "RELEVANT" if tl.get("relevance") == 1 else "IRRELEVANT"
        box = [round(b, 1) for b in tl.get("bbox_xyxy", [])]
        print(f"  GT TL -> State: {st:<7} | Relevance: {rel:<10} | Box: {box}")

    print("--- MODEL PREDICTIONS (Confidence > 0.25) ---")
    pred_count = 0
    for idx in range(len(tl_boxes)):
        if tl_valid[idx] and tl_scores[idx] >= 0.25:
            pred_count += 1
            st_name = STATE_NAMES[tl_states[idx]] if 0 <= tl_states[idx] < 4 else "unk"
            p_rel = rel_probs[idx]
            rel_str = "RELEVANT" if p_rel >= 0.50 else "IRRELEVANT"
            xc, yc, bw, bh = tl_boxes[idx]
            print(f"  PRED TL #{pred_count} -> Conf: {tl_scores[idx]:.2f} | State: {st_name:<6} | Rel: {rel_str:<10} ({p_rel:.1%}) | Box norm: [{xc:.3f}, {yc:.3f}, {bw:.3f}, {bh:.3f}]")

    print("--- MODEL ROAD ARROWS PREDICTIONS (Confidence > 0.20) ---")
    arr_count = 0
    for idx in range(len(arr_boxes)):
        if arr_valid[idx] and arr_scores[idx] >= 0.20:
            arr_count += 1
            m_name = MANEUVER_NAMES[arr_man[idx]] if 0 <= arr_man[idx] < 8 else "unk"
            print(f"  PRED ARROW #{arr_count} -> Conf: {arr_scores[idx]:.2f} | Maneuver: {m_name.upper()}")
