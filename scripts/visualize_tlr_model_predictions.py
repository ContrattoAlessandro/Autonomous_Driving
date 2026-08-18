"""Visual inference and classification demonstration for TLR-YOLO-MTL."""

from __future__ import annotations

import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch

from tlr_yolo_mtl.deployment.export import build_full_model

STATE_NAMES = ["red", "yellow", "green", "off"]
STATE_COLORS = {
    "red": (0, 0, 255),      # BGR Red
    "yellow": (0, 215, 255),  # BGR Yellow
    "green": (0, 255, 0),    # BGR Green
    "off": (128, 128, 128),  # BGR Gray
}

MANEUVER_NAMES = [
    "straight", "left", "right", "straight_left", 
    "straight_right", "left_right", "uturn", "other"
]

def load_inference_model(checkpoint_path: Path, device: torch.device):
    wrapper, report = build_full_model(checkpoint=checkpoint_path)
    model = wrapper.model.to(device).eval()
    return model

def draw_inference_result(
    img_bgr: np.ndarray,
    record: dict,
    raw_pred: dict,
    score_thresh_tl: float = 0.25,
    score_thresh_arrow: float = 0.25,
) -> np.ndarray:
    h, w, _ = img_bgr.shape
    vis = img_bgr.copy()

    # Predictions
    tl_boxes = raw_pred["traffic_candidate_boxes"][0].cpu().numpy()  # (32, 4) (xc, yc, w, h)
    tl_scores = raw_pred["traffic_candidate_scores"][0].cpu().numpy()
    tl_valid = raw_pred["traffic_candidate_valid"][0].cpu().numpy().astype(bool)
    tl_indices = raw_pred["traffic_candidate_indices"][0]

    arr_boxes = raw_pred["arrow_candidate_boxes"][0].cpu().numpy()  # (16, 4)
    arr_scores = raw_pred["arrow_candidate_scores"][0].cpu().numpy()
    arr_valid = raw_pred["arrow_candidate_valid"][0].cpu().numpy().astype(bool)
    arr_indices = raw_pred["arrow_candidate_indices"][0]

    # State predictions
    st_logits = raw_pred["state_logits"][0]  # (4, anchors)
    dense_states = st_logits.argmax(dim=0).cpu().numpy()
    tl_states = dense_states[tl_indices.clamp(0, st_logits.shape[-1] - 1).cpu().numpy()]

    # Maneuver predictions for arrows (3 channels: Left, Straight, Right)
    man_logits = raw_pred["maneuver_logits"][0]  # (3, anchors)

    # Relevance predictions
    rel_logits = raw_pred["relevance_logits"][0, 0].cpu().numpy()  # (32,)
    rel_probs = 1.0 / (1.0 + np.exp(-rel_logits))

    # 1. Draw Road Arrows
    for idx in range(len(arr_boxes)):
        if not arr_valid[idx] or arr_scores[idx] < score_thresh_arrow:
            continue
        xc, yc, bw, bh = arr_boxes[idx]
        x1 = int(max(0, (xc - bw / 2) * w))
        y1 = int(max(0, (yc - bh / 2) * h))
        x2 = int(min(w - 1, (xc + bw / 2) * w))
        y2 = int(min(h - 1, (yc + bh / 2) * h))

        safe_a_idx = arr_indices[idx].clamp(0, man_logits.shape[-1] - 1)
        a_logits = man_logits[:, safe_a_idx].cpu().numpy()
        a_probs = 1.0 / (1.0 + np.exp(-a_logits))

        active_dirs = []
        if a_probs[0] >= 0.40:
            active_dirs.append(f"LEFT ({a_probs[0]:.0%})")
        if a_probs[1] >= 0.40:
            active_dirs.append(f"STRAIGHT ({a_probs[1]:.0%})")
        if a_probs[2] >= 0.40:
            active_dirs.append(f"RIGHT ({a_probs[2]:.0%})")
        if not active_dirs:
            best_c = int(np.argmax(a_probs))
            c_name = ["LEFT", "STRAIGHT", "RIGHT"][best_c]
            active_dirs.append(f"{c_name} ({a_probs[best_c]:.0%})")

        m_name = " + ".join(active_dirs)
        conf = arr_scores[idx]

        # Draw arrow box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 180, 0), 2)  # Bright Blue/Cyan border
        label = f"Arrow: {m_name} [Conf: {conf:.2f}]"
        
        # Solid black badge for crisp text readability
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        badge_y1 = max(0, y1 - th - 10)
        badge_y2 = y1
        cv2.rectangle(vis, (x1, badge_y1), (x1 + tw + 10, badge_y2), (0, 0, 0), -1)
        cv2.rectangle(vis, (x1, badge_y1), (x1 + tw + 10, badge_y2), (255, 180, 0), 1)
        cv2.putText(vis, label, (x1 + 5, badge_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Draw Traffic Lights
    for idx in range(len(tl_boxes)):
        if not tl_valid[idx] or tl_scores[idx] < score_thresh_tl:
            continue
        xc, yc, bw, bh = tl_boxes[idx]
        x1 = int(max(0, (xc - bw / 2) * w))
        y1 = int(max(0, (yc - bh / 2) * h))
        x2 = int(min(w - 1, (xc + bw / 2) * w))
        y2 = int(min(h - 1, (yc + bh / 2) * h))

        st_idx = tl_states[idx]
        st_name = STATE_NAMES[st_idx] if 0 <= st_idx < len(STATE_NAMES) else "unknown"
        color = STATE_COLORS.get(st_name, (0, 255, 0))
        conf = tl_scores[idx]
        p_rel = rel_probs[idx]
        is_rel = p_rel >= 0.50

        # Thickness and border: Thick border if relevant
        thickness = 3 if is_rel else 1
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

        rel_tag = "RELEVANT" if is_rel else "IRRELEVANT"
        rel_color = (50, 255, 50) if is_rel else (200, 200, 200)
        label = f"TL: {st_name.upper()} ({conf:.2f}) | {rel_tag} ({p_rel:.0%})"
        
        # Solid black badge for crisp text readability
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        badge_y1 = max(0, y1 - th - 8)
        badge_y2 = y1
        cv2.rectangle(vis, (x1, badge_y1), (x1 + tw + 8, badge_y2), (0, 0, 0), -1)
        cv2.rectangle(vis, (x1, badge_y1), (x1 + tw + 8, badge_y2), color, 1)
        cv2.putText(vis, label, (x1 + 4, badge_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.48, rel_color, 1, cv2.LINE_AA)

    # Top banner with metadata
    banner = np.zeros((45, w, 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)
    title = f"IMAGE: {record.get('image_id', '')} | City: {record.get('metadata', {}).get('city', 'N/A')}"
    cv2.putText(banner, title, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return np.vstack((banner, vis))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path("runs/tlr_yolo_mtl_baseline_fast/weights/best.pt")
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading model from {checkpoint_path} on {device}...")
    model = load_inference_model(checkpoint_path, device)

    records_file = Path("datasets/tlr_mtl_dtld_paired/records.jsonl")
    out_dir = Path("results/visualizations/baseline_best")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Selecting interesting validation samples...")
    selected_records = []
    with open(records_file, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != "val":
                continue
            tls = r.get("traffic_lights", [])
            arrows = r.get("road_arrows", [])
            
            # Select samples with both relevant and irrelevant lights, or arrows + lights
            has_rel = any(tl.get("relevance") == 1 for tl in tls)
            has_irrel = any(tl.get("relevance") == 0 for tl in tls)
            has_arrows = len(arrows) > 0
            
            if (has_rel and has_irrel) or (has_rel and has_arrows):
                selected_records.append(r)
                if len(selected_records) >= 8:
                    break

    print(f"Found {len(selected_records)} representative validation images. Running inference...")

    for idx, rec in enumerate(selected_records):
        img_path = Path(rec["image_path"])
        if not img_path.exists():
            continue
        
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            continue
        
        # Preprocess to 800x1600
        input_img = cv2.resize(raw_bgr, (1600, 800))
        rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(device)

        with torch.no_grad():
            output = model(tensor)
            if isinstance(output, tuple) and len(output) >= 2:
                raw_pred = output[1]
            else:
                raw_pred = output

        vis = draw_inference_result(input_img, rec, raw_pred)
        out_file = out_dir / f"sample_{idx+1}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_file), vis)
        print(f"Saved visualization: {out_file}")

    print("\nVisual inference completed successfully!")

if __name__ == "__main__":
    main()
