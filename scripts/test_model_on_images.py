"""Inference and visual evaluation on validation images with accurate label encoding and NMS."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str((PROJECT_ROOT / ".ultralytics").resolve()))

import cv2
import numpy as np
import torch
import yaml

from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.unified import UnifiedHeadConfig, attach_unified_relevance_head
from tlr_yolo_mtl.deployment.postprocess import postprocess_multitask_outputs

# --- LABEL TAXONOMY & ENCODINGS ---
STATE_NAMES = ["red", "yellow", "green", "off"]
STATE_COLORS = {
    "red": (0, 0, 255),       # BGR Red
    "yellow": (0, 215, 255),  # BGR Yellow / Amber
    "green": (0, 255, 0),     # BGR Green
    "off": (140, 140, 140),   # BGR Gray
}
DIRECTION_NAMES = ["LEFT", "STRAIGHT", "RIGHT"]


def load_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload.get("config", {})
    if not cfg:
        with open("configs/tlr_yolo_mtl_train.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    wrapper = build_detection_model(cfg["model_config"])
    arch_cfg = cfg.get("architecture", {})
    attach_unified_relevance_head(wrapper, config=UnifiedHeadConfig(**arch_cfg))
    
    state_dict = payload.get("model", payload)
    wrapper.model.load_state_dict(state_dict, strict=True)
    model = wrapper.model.to(device).eval()
    return model, cfg


def decode_arrow_maneuver(maneuver_probs: np.ndarray, thresh: float = 0.35) -> str:
    active = []
    for idx, name in enumerate(DIRECTION_NAMES):
        if maneuver_probs[idx] >= thresh:
            active.append(f"{name} ({maneuver_probs[idx]:.0%})")
    if not active:
        best = int(np.argmax(maneuver_probs))
        active.append(f"{DIRECTION_NAMES[best]} ({maneuver_probs[best]:.0%})")
    return " + ".join(active)


def run_postprocessed_inference(model: torch.nn.Module, tensor: torch.Tensor, tl_conf: float = 0.25, arr_conf: float = 0.25, iou_thresh: float = 0.45):
    with torch.no_grad():
        out0, out1 = model(tensor)

    eleven_tensors = (
        out0,
        out1["state_logits"],
        out1["round_logits"],
        out1["maneuver_logits"],
        out1["ego_lane_logits"],
        out1["traffic_candidate_indices"],
        out1["traffic_candidate_valid"],
        out1["arrow_candidate_indices"],
        out1["arrow_candidate_valid"],
        out1["relevance_logits"],
        out1["attention_weights"],
    )

    post = postprocess_multitask_outputs(
        eleven_tensors,
        traffic_confidence=tl_conf,
        arrow_confidence=arr_conf,
        iou_threshold=iou_thresh,
    )
    return post


def draw_side_by_side_evaluation(
    img_bgr: np.ndarray,
    record: dict,
    post: dict,
) -> np.ndarray:
    orig_h, orig_w, _ = img_bgr.shape
    vis_w, vis_h = 1600, 800
    base_img = cv2.resize(img_bgr, (vis_w, vis_h))

    # --- LEFT PANEL: GROUND TRUTH ---
    gt_vis = base_img.copy()
    
    # Draw GT Traffic Lights
    for tl in record.get("traffic_lights", []):
        x1_o, y1_o, x2_o, y2_o = tl["bbox_xyxy"]
        x1 = int(max(0, (x1_o / orig_w) * vis_w))
        y1 = int(max(0, (y1_o / orig_h) * vis_h))
        x2 = int(min(vis_w - 1, (x2_o / orig_w) * vis_w))
        y2 = int(min(vis_h - 1, (y2_o / orig_h) * vis_h))

        st = str(tl.get("state") or "unknown")
        is_rel = tl.get("relevance") == 1
        color = STATE_COLORS.get(st, (200, 200, 200))
        thickness = 3 if is_rel else 1
        cv2.rectangle(gt_vis, (x1, y1), (x2, y2), color, thickness)

        rel_text = "RELEVANT" if is_rel else "IRRELEVANT"
        badge_text = f"GT: {st.upper()} | {rel_text}"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        by1 = max(0, y1 - th - 6)
        cv2.rectangle(gt_vis, (x1, by1), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.rectangle(gt_vis, (x1, by1), (x1 + tw + 6, y1), color, 1)
        cv2.putText(gt_vis, badge_text, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw GT Road Arrows
    for arr in record.get("road_arrows", []):
        x1_o, y1_o, x2_o, y2_o = arr["bbox_xyxy"]
        x1 = int(max(0, (x1_o / orig_w) * vis_w))
        y1 = int(max(0, (y1_o / orig_h) * vis_h))
        x2 = int(min(vis_w - 1, (x2_o / orig_w) * vis_w))
        y2 = int(min(vis_h - 1, (y2_o / orig_h) * vis_h))

        m_vec = arr.get("direction_multihot", [0, 0, 0])
        dirs = [DIRECTION_NAMES[i] for i, v in enumerate(m_vec) if v == 1]
        m_str = "+".join(dirs) if dirs else "ARROW"
        ego_str = " (EGO)" if arr.get("ego_lane") == 1 else ""

        cv2.rectangle(gt_vis, (x1, y1), (x2, y2), (255, 165, 0), 2)
        badge_text = f"GT: {m_str}{ego_str}"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        by1 = max(0, y1 - th - 6)
        cv2.rectangle(gt_vis, (x1, by1), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.rectangle(gt_vis, (x1, by1), (x1 + tw + 6, y1), (255, 165, 0), 1)
        cv2.putText(gt_vis, badge_text, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # GT Banner
    gt_banner = np.zeros((36, vis_w, 3), dtype=np.uint8)
    gt_banner[:] = (45, 45, 45)
    cv2.putText(gt_banner, "--- GROUND TRUTH ANNOTATIONS ---", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    gt_panel = np.vstack((gt_banner, gt_vis))

    # --- RIGHT PANEL: MODEL PREDICTIONS (NMS-DEDUPLICATED) ---
    pred_vis = base_img.copy()

    tl_data = post["traffic_lights"]
    tl_boxes = tl_data["boxes_xyxy"][0].cpu().numpy()
    tl_scores = tl_data["detection_scores"][0].cpu().numpy()
    tl_valid = tl_data["valid"][0].cpu().numpy().astype(bool)
    tl_state_idx = tl_data["state_indices"][0].cpu().numpy()
    tl_state_probs = tl_data["state_probabilities"][0].cpu().numpy()
    tl_rel_probs = tl_data["relevance_probabilities"][0].cpu().numpy()
    tl_round_probs = tl_data["round_probabilities"][0, 0].cpu().numpy()

    arr_data = post["road_arrows"]
    arr_boxes = arr_data["boxes_xyxy"][0].cpu().numpy()
    arr_scores = arr_data["detection_scores"][0].cpu().numpy()
    arr_valid = arr_data["valid"][0].cpu().numpy().astype(bool)
    arr_man_probs = arr_data["maneuver_probabilities"][0].cpu().numpy()
    arr_ego_probs = arr_data["ego_lane_probabilities"][0, 0].cpu().numpy()

    # 1. Draw Predicted Road Arrows
    for idx in range(len(arr_boxes)):
        if not arr_valid[idx]:
            continue
        x1, y1, x2, y2 = [int(v) for v in arr_boxes[idx]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(vis_w - 1, x2), min(vis_h - 1, y2)

        a_probs = arr_man_probs[:, idx]
        m_str = decode_arrow_maneuver(a_probs)
        is_ego = arr_ego_probs[idx] >= 0.50
        ego_str = " (EGO)" if is_ego else ""
        conf = arr_scores[idx]

        cv2.rectangle(pred_vis, (x1, y1), (x2, y2), (255, 180, 0), 2)
        badge_text = f"Arr: {m_str}{ego_str} [{conf:.2f}]"
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        by1 = max(0, y1 - th - 6)
        cv2.rectangle(pred_vis, (x1, by1), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.rectangle(pred_vis, (x1, by1), (x1 + tw + 6, y1), (255, 180, 0), 1)
        cv2.putText(pred_vis, badge_text, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 2. Draw Predicted Traffic Lights
    for idx in range(len(tl_boxes)):
        if not tl_valid[idx]:
            continue
        x1, y1, x2, y2 = [int(v) for v in tl_boxes[idx]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(vis_w - 1, x2), min(vis_h - 1, y2)

        st_i = tl_state_idx[idx]
        st_prob = tl_state_probs[st_i, idx]
        st_name = STATE_NAMES[st_i] if 0 <= st_i < 4 else "unknown"
        color = STATE_COLORS.get(st_name, (0, 255, 0))

        conf = tl_scores[idx]
        p_rel = tl_rel_probs[idx]
        is_rel = p_rel >= 0.50

        thickness = 3 if is_rel else 1
        cv2.rectangle(pred_vis, (x1, y1), (x2, y2), color, thickness)

        rel_label = "RELEVANT" if is_rel else "IRRELEVANT"
        rel_color = (0, 255, 0) if is_rel else (180, 180, 180)
        badge_text = f"TL: {st_name.upper()} ({st_prob:.0%}) | {rel_label} ({p_rel:.0%}) [{conf:.2f}]"

        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        by1 = max(0, y1 - th - 6)
        cv2.rectangle(pred_vis, (x1, by1), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.rectangle(pred_vis, (x1, by1), (x1 + tw + 6, y1), color, 1)
        cv2.putText(pred_vis, badge_text, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, rel_color, 1, cv2.LINE_AA)

    # Pred Banner
    pred_banner = np.zeros((36, vis_w, 3), dtype=np.uint8)
    pred_banner[:] = (45, 45, 45)
    cv2.putText(pred_banner, "--- TLR-YOLO-MTL PREDICTIONS (NMS-Decoded, Seed 42, Epoch 50) ---", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    pred_panel = np.vstack((pred_banner, pred_vis))

    # Master top header with sample metadata
    header = np.zeros((45, vis_w, 3), dtype=np.uint8)
    header[:] = (25, 25, 25)
    img_id = record.get("image_id", "N/A")
    city = record.get("metadata", {}).get("city", "N/A")
    title = f"DTLD Validation Sample: {img_id} | City: {city} | Split: {record.get('split')}"
    cv2.putText(header, title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    combined = np.vstack((header, gt_panel, pred_panel))
    return combined


def print_detailed_evaluation_table(sample_num: int, record: dict, post: dict):
    img_id = record.get("image_id", "N/A")
    print("\n" + "=" * 110)
    print(f"SAMPLE #{sample_num}: {img_id} (City: {record.get('metadata', {}).get('city', 'N/A')})")
    print("=" * 110)

    # Ground Truth Traffic Lights
    gt_tls = record.get("traffic_lights", [])
    print(f"GROUND TRUTH TRAFFIC LIGHTS ({len(gt_tls)}):")
    if not gt_tls:
        print("  (None)")
    for i, tl in enumerate(gt_tls):
        st = str(tl.get("state") or "unknown")
        rel = "RELEVANT" if tl.get("relevance") == 1 else "IRRELEVANT"
        round_t = "Round" if tl.get("round") == 1 else ("Arrow" if tl.get("round") == 0 else "Unknown")
        man_t = str(tl.get("maneuver") if tl.get("maneuver") is not None else "N/A")
        box = [round(b, 1) for b in tl.get("bbox_xyxy", [])]
        print(f"  GT TL #{i+1:02d} | State: {st:<7} | Rel: {rel:<10} | Type: {round_t:<6} | Dir: {man_t:<15} | Box: {box}")


    # Ground Truth Road Arrows
    gt_arrows = record.get("road_arrows", [])
    print(f"\nGROUND TRUTH ROAD ARROWS ({len(gt_arrows)}):")
    if not gt_arrows:
        print("  (None)")
    for i, arr in enumerate(gt_arrows):
        m_vec = arr.get("direction_multihot", [0, 0, 0])
        dirs = [DIRECTION_NAMES[j] for j, v in enumerate(m_vec) if v == 1]
        m_str = "+".join(dirs) if dirs else "ARROW"
        ego_str = "EGO_LANE" if arr.get("ego_lane") == 1 else "OTHER_LANE"
        box = [round(b, 1) for b in arr.get("bbox_xyxy", [])]
        print(f"  GT ARR #{i+1:02d} | Maneuver: {m_str:<12} | Lane: {ego_str:<10} | Box: {box}")

    # Predictions
    tl_data = post["traffic_lights"]
    tl_boxes = tl_data["boxes_xyxy"][0].cpu().numpy()
    tl_scores = tl_data["detection_scores"][0].cpu().numpy()
    tl_valid = tl_data["valid"][0].cpu().numpy().astype(bool)
    tl_state_idx = tl_data["state_indices"][0].cpu().numpy()
    tl_state_probs = tl_data["state_probabilities"][0].cpu().numpy()
    tl_rel_probs = tl_data["relevance_probabilities"][0].cpu().numpy()
    tl_round_probs = tl_data["round_probabilities"][0, 0].cpu().numpy()

    arr_data = post["road_arrows"]
    arr_boxes = arr_data["boxes_xyxy"][0].cpu().numpy()
    arr_scores = arr_data["detection_scores"][0].cpu().numpy()
    arr_valid = arr_data["valid"][0].cpu().numpy().astype(bool)
    arr_man_probs = arr_data["maneuver_probabilities"][0].cpu().numpy()
    arr_ego_probs = arr_data["ego_lane_probabilities"][0, 0].cpu().numpy()

    print(f"\nPREDICTED TRAFFIC LIGHTS (NMS Deduplicated, Conf >= 0.25):")
    pred_tl_cnt = 0
    for idx in range(len(tl_boxes)):
        if tl_valid[idx]:
            pred_tl_cnt += 1
            st_i = tl_state_idx[idx]
            st_prob = tl_state_probs[st_i, idx]
            st_name = STATE_NAMES[st_i] if 0 <= st_i < 4 else "unknown"
            
            p_rel = tl_rel_probs[idx]
            rel_str = "RELEVANT" if p_rel >= 0.50 else "IRRELEVANT"
            
            p_round = tl_round_probs[idx]
            round_str = f"Round ({p_round:.0%})" if p_round >= 0.50 else f"Arrow ({(1-p_round):.0%})"
            
            x1, y1, x2, y2 = [round(float(v), 1) for v in tl_boxes[idx]]
            print(f"  PRED TL #{pred_tl_cnt:02d} | Conf: {tl_scores[idx]:.3f} | State: {st_name:<6} ({st_prob:.1%}) | Rel: {rel_str:<10} ({p_rel:.1%}) | Type: {round_str:<12} | Box: [{x1}, {y1}, {x2}, {y2}]")
    if pred_tl_cnt == 0:
        print("  (No detections above threshold)")

    print(f"\nPREDICTED ROAD ARROWS (NMS Deduplicated, Conf >= 0.25):")
    pred_arr_cnt = 0
    for idx in range(len(arr_boxes)):
        if arr_valid[idx]:
            pred_arr_cnt += 1
            a_probs = arr_man_probs[:, idx]
            m_str = decode_arrow_maneuver(a_probs)
            is_ego = arr_ego_probs[idx] >= 0.50
            ego_str = "EGO_LANE" if is_ego else "OTHER_LANE"
            x1, y1, x2, y2 = [round(float(v), 1) for v in arr_boxes[idx]]
            print(f"  PRED ARR #{pred_arr_cnt:02d} | Conf: {arr_scores[idx]:.3f} | Maneuver: {m_str:<22} | Lane: {ego_str} ({arr_ego_probs[idx]:.0%}) | Box: [{x1}, {y1}, {x2}, {y2}]")
    if pred_arr_cnt == 0:
        print("  (No detections above threshold)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path("runs/tlr_yolo_mtl_single_phase_seed42/weights/best.pt")
    if not checkpoint_path.exists():
        checkpoint_path = Path("runs/tlr_yolo_mtl_single_phase_seed42/weights/epoch_050.pt")

    print(f"Loading checkpoint from: {checkpoint_path} on {device}...")
    model, cfg = load_model(checkpoint_path, device)

    records_file = Path("datasets/tlr_mtl_dtld_paired/records.jsonl")
    out_dir = Path("results/visualizations/seed42_epoch50_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter diverse validation scenarios:
    # 1. Samples with both Road Arrows and Traffic Lights
    # 2. Samples with Red Traffic Lights
    # 3. Samples with Green Traffic Lights and mixed Relevance
    # 4. Samples with Off / Yellow Traffic Lights
    selected = []
    seen_ids = set()

    with open(records_file, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") != "val":
                continue
            img_id = r.get("image_id")
            if img_id in seen_ids:
                continue

            tls = r.get("traffic_lights", [])
            arrows = r.get("road_arrows", [])
            has_arrows = len(arrows) > 0
            has_tls = len(tls) > 0
            has_rel = any(tl.get("relevance") == 1 for tl in tls)
            has_irrel = any(tl.get("relevance") == 0 for tl in tls)
            has_red = any(tl.get("state") == "red" for tl in tls)
            has_green = any(tl.get("state") == "green" for tl in tls)

            # Category 1: Arrows + TLs (high priority)
            if has_arrows and has_tls and len(selected) < 4:
                selected.append(r)
                seen_ids.add(img_id)
                continue

            # Category 2: Mixed Relevance (Relevant + Irrelevant)
            if has_rel and has_irrel and len(selected) < 6:
                selected.append(r)
                seen_ids.add(img_id)
                continue

            # Category 3: Red lights
            if has_red and has_rel and len(selected) < 8:
                selected.append(r)
                seen_ids.add(img_id)
                continue

            if len(selected) >= 8:
                break


    print(f"Selected {len(selected)} rich validation test cases for evaluation.")


    for idx, rec in enumerate(selected):
        img_path = Path(rec["image_path"])
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            continue

        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            continue

        # Resize to model input size (1600, 800)
        input_img = cv2.resize(raw_bgr, (1600, 800))
        rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        # Run Postprocessed inference with NMS
        post = run_postprocessed_inference(model, tensor, tl_conf=0.25, arr_conf=0.25, iou_thresh=0.45)

        # Print detailed formatted table comparing GT vs Prediction
        print_detailed_evaluation_table(idx + 1, rec, post)

        # Generate and save high-resolution side-by-side visualization
        vis_image = draw_side_by_side_evaluation(raw_bgr, rec, post)
        out_path = out_dir / f"eval_sample_{idx+1:02d}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), vis_image)
        print(f"--> Saved evaluation overlay to: {out_path}")

    print("\n" + "=" * 110)
    print(f"Evaluation completed successfully! All overlays saved to {out_dir}")
    print("=" * 110)


if __name__ == "__main__":
    main()
