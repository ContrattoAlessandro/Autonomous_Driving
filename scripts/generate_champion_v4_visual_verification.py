"""Visual verification generator for TLR-YOLO-MTL Champion v4.

Generates high-resolution side-by-side evaluation overlays and zoom detail crops
across 12 canonical test scenarios from the DTLD benchmark dataset.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str((PROJECT_ROOT / ".ultralytics").resolve()))

import cv2
import numpy as np
import torch
import yaml

from tlr_yolo_mtl.model.milestone2 import build_detection_model
from tlr_yolo_mtl.model.unified import UnifiedHeadConfig
from tlr_yolo_mtl.model.geometry_attention import attach_geometry_aware_unified_relevance_head
from tlr_yolo_mtl.deployment.postprocess import postprocess_multitask_outputs

STATE_NAMES = ["red", "yellow", "green", "off"]
STATE_COLORS_BGR = {
    "red": (30, 30, 240),         # Bright Red
    "yellow": (0, 215, 255),      # Bright Yellow/Amber
    "green": (50, 220, 50),       # Vibrant Green
    "off": (140, 140, 140),       # Neutral Gray
    "unknown": (180, 180, 180),
}
DIRECTION_NAMES = ["LEFT", "STRAIGHT", "RIGHT"]


def load_champion_v4_model(
    config_path: Path = Path("configs/tlr_yolo11s_champion_v4.yaml"),
    checkpoint_path: Path = Path("runs/tlr_yolo11s_champion_v4/weights/best_composite.pt"),
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Tuple[torch.nn.Module, dict]:
    print(f"Loading Champion v4 configuration from: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    wrapper = build_detection_model(PROJECT_ROOT / cfg["model_config"])
    arch = cfg.get("architecture", {})
    head_kwargs = {k: v for k, v in arch.items() if k in UnifiedHeadConfig.__dataclass_fields__}
    geom_cfg = arch.get("geometry_attention", {})

    attach_geometry_aware_unified_relevance_head(
        wrapper,
        config=UnifiedHeadConfig(**head_kwargs),
        hidden_dim=int(geom_cfg.get("hidden_dim", 64)),
        p_drop=float(geom_cfg.get("p_drop", 0.0)),
        use_confidence_gating=bool(geom_cfg.get("use_confidence_gate", True)),
    )

    print(f"Loading checkpoint weights from: {checkpoint_path} on {device}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "ema" in ckpt and "shadow" in ckpt["ema"]:
        state_dict = ckpt["ema"]["shadow"]
        print("-> Using EMA shadow weights")
    elif "model" in ckpt:
        state_dict = ckpt["model"]
        print("-> Using model state dict")
    else:
        state_dict = ckpt

    wrapper.model.load_state_dict(state_dict, strict=True)
    model = wrapper.model.to(device).eval()
    print("-> Model successfully loaded and ready for inference!")
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


def run_inference(
    model: torch.nn.Module,
    img_bgr: np.ndarray,
    device: torch.device,
    input_size: Tuple[int, int] = (960, 1920),
    tl_conf: float = 0.25,
    arr_conf: float = 0.25,
    iou_thresh: float = 0.45,
):
    orig_h, orig_w, _ = img_bgr.shape
    resized = cv2.resize(img_bgr, (input_size[1], input_size[0]))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

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
    return post, out1, (input_size[1], input_size[0])


def create_verification_composite(
    raw_bgr: np.ndarray,
    record: dict,
    post: dict,
    raw_out1: dict,
    input_wh: Tuple[int, int],
    scenario_title: str,
    scenario_desc: str,
) -> np.ndarray:
    orig_h, orig_w, _ = raw_bgr.shape
    vis_w, vis_h = 1600, 800

    # -------------------------------------------------------------
    # 1. LEFT PANEL: GROUND TRUTH
    # -------------------------------------------------------------
    gt_canvas = cv2.resize(raw_bgr, (vis_w, vis_h))
    
    # Ground Truth Road Arrows
    gt_arrows = record.get("road_arrows", [])
    for arr in gt_arrows:
        bx1, by1, bx2, by2 = arr["bbox_xyxy"]
        x1 = int(np.clip((bx1 / orig_w) * vis_w, 0, vis_w - 1))
        y1 = int(np.clip((by1 / orig_h) * vis_h, 0, vis_h - 1))
        x2 = int(np.clip((bx2 / orig_w) * vis_w, 0, vis_w - 1))
        y2 = int(np.clip((by2 / orig_h) * vis_h, 0, vis_h - 1))

        m_vec = arr.get("direction_multihot", [0, 0, 0])
        dirs = [DIRECTION_NAMES[i] for i, v in enumerate(m_vec) if v == 1]
        m_str = "+".join(dirs) if dirs else "ARROW"
        is_ego = arr.get("is_ego_lane") == 1 or arr.get("ego_lane") == 1
        ego_tag = " [EGO_LANE]" if is_ego else " [OTHER_LANE]"

        cv2.rectangle(gt_canvas, (x1, y1), (x2, y2), (255, 160, 0), 2)
        badge = f"GT: {m_str}{ego_tag}"
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        by_top = max(0, y1 - th - 6)
        cv2.rectangle(gt_canvas, (x1, by_top), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.rectangle(gt_canvas, (x1, by_top), (x1 + tw + 6, y1), (255, 160, 0), 1)
        cv2.putText(gt_canvas, badge, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Ground Truth Traffic Lights
    gt_tls = record.get("traffic_lights", [])
    for tl in gt_tls:
        bx1, by1, bx2, by2 = tl["bbox_xyxy"]
        x1 = int(np.clip((bx1 / orig_w) * vis_w, 0, vis_w - 1))
        y1 = int(np.clip((by1 / orig_h) * vis_h, 0, vis_h - 1))
        x2 = int(np.clip((bx2 / orig_w) * vis_w, 0, vis_w - 1))
        y2 = int(np.clip((by2 / orig_h) * vis_h, 0, vis_h - 1))

        st = str(tl.get("state") or "unknown").lower()
        is_rel = tl.get("relevance") == 1
        color = STATE_COLORS_BGR.get(st, (180, 180, 180))
        thickness = 3 if is_rel else 1
        cv2.rectangle(gt_canvas, (x1, y1), (x2, y2), color, thickness)

        rel_text = "RELEVANT" if is_rel else "IRRELEVANT"
        badge = f"GT: {st.upper()} | {rel_text}"
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        by_top = max(0, y1 - th - 6)
        cv2.rectangle(gt_canvas, (x1, by_top), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.rectangle(gt_canvas, (x1, by_top), (x1 + tw + 6, y1), color, 1)
        txt_color = (0, 255, 255) if is_rel else (200, 200, 200)
        cv2.putText(gt_canvas, badge, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, txt_color, 1, cv2.LINE_AA)

    # GT Header bar
    gt_bar = np.zeros((36, vis_w, 3), dtype=np.uint8)
    gt_bar[:] = (40, 40, 40)
    cv2.putText(gt_bar, f"--- GROUND TRUTH ANNOTATIONS (DTLD Dataset: {len(gt_tls)} TLs, {len(gt_arrows)} Arrows) ---", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2, cv2.LINE_AA)
    gt_panel = np.vstack((gt_bar, gt_canvas))

    # -------------------------------------------------------------
    # 2. RIGHT PANEL: CHAMPION V4 PREDICTIONS
    # -------------------------------------------------------------
    pred_canvas = cv2.resize(raw_bgr, (vis_w, vis_h))

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

    # Scale factor from model input [input_wh] to canvas [vis_w, vis_h]
    scale_x = vis_w / input_wh[0]
    scale_y = vis_h / input_wh[1]

    pred_arr_coords = []
    pred_tl_coords = []

    # Draw Predicted Road Arrows
    for idx in range(len(arr_boxes)):
        if not arr_valid[idx]:
            continue
        bx1, by1, bx2, by2 = arr_boxes[idx]
        x1 = int(np.clip(bx1 * scale_x, 0, vis_w - 1))
        y1 = int(np.clip(by1 * scale_y, 0, vis_h - 1))
        x2 = int(np.clip(bx2 * scale_x, 0, vis_w - 1))
        y2 = int(np.clip(by2 * scale_y, 0, vis_h - 1))

        a_probs = arr_man_probs[:, idx]
        m_str = decode_arrow_maneuver(a_probs)
        p_ego = float(arr_ego_probs[idx])
        is_ego = p_ego >= 0.50
        ego_str = f" [EGO ({p_ego:.0%})]" if is_ego else f" [OTHER ({1-p_ego:.0%})]"
        conf = float(arr_scores[idx])

        pred_arr_coords.append(((x1 + x2) // 2, (y1 + y2) // 2, is_ego))

        box_col = (255, 180, 0) if is_ego else (200, 140, 0)
        cv2.rectangle(pred_canvas, (x1, y1), (x2, y2), box_col, 2)
        badge = f"Arr: {m_str}{ego_str} [{conf:.2f}]"
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        by_top = max(0, y1 - th - 6)
        cv2.rectangle(pred_canvas, (x1, by_top), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.rectangle(pred_canvas, (x1, by_top), (x1 + tw + 6, y1), box_col, 1)
        cv2.putText(pred_canvas, badge, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw Predicted Traffic Lights
    for idx in range(len(tl_boxes)):
        if not tl_valid[idx]:
            continue
        bx1, by1, bx2, by2 = tl_boxes[idx]
        x1 = int(np.clip(bx1 * scale_x, 0, vis_w - 1))
        y1 = int(np.clip(by1 * scale_y, 0, vis_h - 1))
        x2 = int(np.clip(bx2 * scale_x, 0, vis_w - 1))
        y2 = int(np.clip(by2 * scale_y, 0, vis_h - 1))

        st_i = tl_state_idx[idx]
        st_prob = float(tl_state_probs[st_i, idx])
        st_name = STATE_NAMES[st_i] if 0 <= st_i < 4 else "unknown"
        color = STATE_COLORS_BGR.get(st_name, (0, 255, 0))

        conf = float(tl_scores[idx])
        p_rel = float(tl_rel_probs[idx])
        is_rel = p_rel >= 0.50

        p_round = float(tl_round_probs[idx])
        round_tag = "Rnd" if p_round >= 0.50 else "Arr"

        pred_tl_coords.append(((x1 + x2) // 2, (y1 + y2) // 2, is_rel, p_rel))

        thickness = 3 if is_rel else 1
        cv2.rectangle(pred_canvas, (x1, y1), (x2, y2), color, thickness)

        rel_label = "RELEVANT" if is_rel else "IRRELEVANT"
        badge = f"TL: {st_name.upper()} ({st_prob:.0%}) | {round_tag} | {rel_label} ({p_rel:.0%}) [{conf:.2f}]"

        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        by_top = max(0, y1 - th - 6)
        cv2.rectangle(pred_canvas, (x1, by_top), (x1 + tw + 6, y1), (0, 0, 0), -1)
        cv2.rectangle(pred_canvas, (x1, by_top), (x1 + tw + 6, y1), color, 1)
        rel_color = (50, 255, 50) if is_rel else (180, 180, 180)
        cv2.putText(pred_canvas, badge, (x1 + 3, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.42, rel_color, 1, cv2.LINE_AA)

    # Draw Cross-Attention Link between Ego Arrow and Relevant TL
    for arr_cx, arr_cy, arr_ego in pred_arr_coords:
        if not arr_ego:
            continue
        for tl_cx, tl_cy, tl_rel, tl_p in pred_tl_coords:
            if tl_rel:
                # Draw glowing attention line
                cv2.line(pred_canvas, (arr_cx, arr_cy), (tl_cx, tl_cy), (0, 255, 255), 2, cv2.LINE_AA)
                mid_x, mid_y = (arr_cx + tl_cx) // 2, (arr_cy + tl_cy) // 2
                cv2.circle(pred_canvas, (mid_x, mid_y), 4, (0, 255, 255), -1)
                cv2.putText(pred_canvas, f"Cross-Attn (Rel: {tl_p:.0%})", (mid_x + 8, mid_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)

    # Pred Header bar
    pred_bar = np.zeros((36, vis_w, 3), dtype=np.uint8)
    pred_bar[:] = (30, 45, 30)
    cv2.putText(pred_bar, f"--- TLR-YOLO-MTL CHAMPION V4 PREDICTIONS (Detected: {tl_valid.sum()} TLs, {arr_valid.sum()} Arrows) ---", (20, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2, cv2.LINE_AA)
    pred_panel = np.vstack((pred_bar, pred_canvas))

    # -------------------------------------------------------------
    # 3. MASTER HEADER WITH SCENARIO METADATA
    # -------------------------------------------------------------
    master_header = np.zeros((65, vis_w, 3), dtype=np.uint8)
    master_header[:] = (20, 22, 25)
    img_id = record.get("image_id", "N/A")
    city = record.get("metadata", {}).get("city", "Unknown")
    
    cv2.putText(master_header, f"SCENARIO: {scenario_title}", (20, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(master_header, f"Image: {img_id} | City: {city} | Focus: {scenario_desc}", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 210, 240), 1, cv2.LINE_AA)

    # -------------------------------------------------------------
    # 4. BOTTOM DETAIL CROPS (PICTURE-IN-PICTURE HIGH-RES ZOOM)
    # -------------------------------------------------------------
    crop_bar_h = 180
    crop_canvas = np.zeros((crop_bar_h, vis_w, 3), dtype=np.uint8)
    crop_canvas[:] = (25, 25, 28)
    cv2.putText(crop_canvas, "HIGH-RES DETAIL CROPS (Optical Verification of Lamp State & Sub-Pixel Localization):", (20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    # Select up to 4 interesting traffic lights (prioritize relevant, red, yellow, or tiny)
    interesting_tls = []
    for tl in gt_tls:
        bx1, by1, bx2, by2 = tl["bbox_xyxy"]
        w_px = bx2 - bx1
        h_px = by2 - by1
        interesting_tls.append((tl, w_px, h_px))

    interesting_tls.sort(key=lambda item: (
        -int(item[0].get("relevance") == 1),
        -int(item[0].get("state") in ["red", "yellow"]),
        item[2],  # smaller height first for tiny detection verification
    ))

    slot_w = 360
    slot_margin = 25
    x_offset = 20

    for slot_idx, (tl_gt, w_px, h_px) in enumerate(interesting_tls[:4]):
        bx1, by1, bx2, by2 = tl_gt["bbox_xyxy"]
        pad_x = max(12, int(w_px * 0.6))
        pad_y = max(12, int(h_px * 0.4))
        cx1 = int(max(0, bx1 - pad_x))
        cy1 = int(max(0, by1 - pad_y))
        cx2 = int(min(orig_w, bx2 + pad_x))
        cy2 = int(min(orig_h, by2 + pad_y))

        crop = raw_bgr[cy1:cy2, cx1:cx2]
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            continue

        target_ch, target_cw = 130, 130
        crop_zoom = cv2.resize(crop, (target_cw, target_ch), interpolation=cv2.INTER_LANCZOS4)

        rel_x1 = int(((bx1 - cx1) / (cx2 - cx1)) * target_cw)
        rel_y1 = int(((by1 - cy1) / (cy2 - cy1)) * target_ch)
        rel_x2 = int(((bx2 - cx1) / (cx2 - cx1)) * target_cw)
        rel_y2 = int(((by2 - cy1) / (cy2 - cy1)) * target_ch)
        st_gt = str(tl_gt.get("state") or "unknown").lower()
        col = STATE_COLORS_BGR.get(st_gt, (180, 180, 180))
        cv2.rectangle(crop_zoom, (rel_x1, rel_y1), (rel_x2, rel_y2), col, 2)

        py1 = 35
        py2 = py1 + target_ch
        px1 = x_offset
        px2 = px1 + target_cw

        crop_canvas[py1:py2, px1:px2] = crop_zoom
        cv2.rectangle(crop_canvas, (px1, py1), (px2, py2), (80, 80, 90), 1)

        tx = px2 + 10
        st_txt = f"GT State: {st_gt.upper()}"
        rel_txt = f"GT Rel: {'YES (EGO)' if tl_gt.get('relevance') == 1 else 'NO (OTHER)'}"
        sz_txt = f"Size: {w_px:.0f}x{h_px:.0f} px"
        cv2.putText(crop_canvas, st_txt, (tx, py1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1, cv2.LINE_AA)
        cv2.putText(crop_canvas, rel_txt, (tx, py1 + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(crop_canvas, sz_txt, (tx, py1 + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

        x_offset += slot_w + slot_margin

    full_composite = np.vstack((master_header, gt_panel, pred_panel, crop_canvas))
    return full_composite


def print_detailed_evaluation_table(sample_num: int, title: str, record: dict, post: dict):
    img_id = record.get("image_id", "N/A")
    city = record.get("metadata", {}).get("city", "N/A")
    print("\n" + "=" * 115)
    print(f"SAMPLE #{sample_num:02d}: {title} | ID: {img_id} (City: {city})")
    print("=" * 115)

    gt_tls = record.get("traffic_lights", [])
    print(f"GROUND TRUTH TRAFFIC LIGHTS ({len(gt_tls)}):")
    if not gt_tls:
        print("  (None)")
    for i, tl in enumerate(gt_tls):
        st = str(tl.get("state") or "unknown").upper()
        rel = "RELEVANT" if tl.get("relevance") == 1 else "IRRELEVANT"
        round_t = "Round" if tl.get("round_target") == 1 else ("Arrow" if tl.get("round_target") == 0 else "Unknown")
        box = [round(float(b), 1) for b in tl.get("bbox_xyxy", [])]
        w = round(box[2] - box[0], 1)
        h = round(box[3] - box[1], 1)
        print(f"  GT TL #{i+1:02d} | State: {st:<7} | Rel: {rel:<10} | Type: {round_t:<6} | Size: {w:4.1f}x{h:4.1f}px | Box: {box}")

    gt_arrows = record.get("road_arrows", [])
    print(f"\nGROUND TRUTH ROAD ARROWS ({len(gt_arrows)}):")
    if not gt_arrows:
        print("  (None)")
    for i, arr in enumerate(gt_arrows):
        m_vec = arr.get("direction_multihot", [0, 0, 0])
        dirs = [DIRECTION_NAMES[j] for j, v in enumerate(m_vec) if v == 1]
        m_str = "+".join(dirs) if dirs else "ARROW"
        ego_str = "EGO_LANE" if (arr.get("is_ego_lane") == 1 or arr.get("ego_lane") == 1) else "OTHER_LANE"
        box = [round(float(b), 1) for b in arr.get("bbox_xyxy", [])]
        print(f"  GT ARR #{i+1:02d} | Maneuver: {m_str:<15} | Lane: {ego_str:<10} | Box: {box}")

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

    print(f"\nPREDICTED TRAFFIC LIGHTS (Champion v4, Conf >= 0.25):")
    pred_tl_cnt = 0
    for idx in range(len(tl_boxes)):
        if tl_valid[idx]:
            pred_tl_cnt += 1
            st_i = tl_state_idx[idx]
            st_prob = float(tl_state_probs[st_i, idx])
            st_name = STATE_NAMES[st_i] if 0 <= st_i < 4 else "unknown"
            
            p_rel = float(tl_rel_probs[idx])
            rel_str = "RELEVANT" if p_rel >= 0.50 else "IRRELEVANT"
            
            p_round = float(tl_round_probs[idx])
            round_str = f"Round ({p_round:.0%})" if p_round >= 0.50 else f"Arrow ({(1-p_round):.0%})"
            
            x1, y1, x2, y2 = [round(float(v), 1) for v in tl_boxes[idx]]
            w = round(x2 - x1, 1)
            h = round(y2 - y1, 1)
            print(f"  PRED TL #{pred_tl_cnt:02d} | Conf: {tl_scores[idx]:.3f} | State: {st_name.upper():<7} ({st_prob:.1%}) | Rel: {rel_str:<10} ({p_rel:.1%}) | Type: {round_str:<12} | Size: {w:4.1f}x{h:4.1f}px | Box: [{x1}, {y1}, {x2}, {y2}]")
    if pred_tl_cnt == 0:
        print("  (No detections above confidence threshold)")

    print(f"\nPREDICTED ROAD ARROWS (Champion v4, Conf >= 0.25):")
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
        print("  (No detections above confidence threshold)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_champion_v4_model(device=device)

    records_file = Path("datasets/tlr_mtl_dtld_paired/records.jsonl")
    out_dir = Path("results/visualizations/champion_v4_verification")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Artifact directory in antigravity brain
    artifact_dir = Path("C:/Users/alexa/.gemini/antigravity-ide/brain/b6448eb9-6954-4f99-b508-93ebeba8dc7a/champion_v4_images")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Load all validation records
    print("Reading validation split records...")
    records = []
    with open(records_file, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "val":
                records.append(r)
    print(f"Found {len(records)} validation records.")

    # 12 Curated Diverse Canonical Test Scenarios
    scenarios = [
        {
            "key": "01_mixed_relevance_intersection_berlin",
            "title": "Case 1: Multi-Lane Mixed Relevance Intersection (Berlin)",
            "desc": "Single ego-lane relevant red light vs multiple irrelevant turning signals across lanes.",
            "filter": lambda r: r.get("metadata", {}).get("city") == "Berlin" and any(tl.get("relevance") == 1 for tl in r.get("traffic_lights", [])) and any(tl.get("relevance") == 0 for tl in r.get("traffic_lights", [])) and len(r.get("traffic_lights", [])) >= 3
        },
        {
            "key": "02_safety_red_signal_frankfurt",
            "title": "Case 2: Safety-Critical Red Signal & Cross-Attention (Frankfurt)",
            "desc": "Ego-lane stop signal aligned with road arrow marking for fail-safe stopping.",
            "filter": lambda r: r.get("metadata", {}).get("city") == "Frankfurt" and any(tl.get("state") == "red" and tl.get("relevance") == 1 for tl in r.get("traffic_lights", [])) and len(r.get("road_arrows", [])) >= 1
        },
        {
            "key": "03_green_proceed_signal_dortmund",
            "title": "Case 3: Green Proceed Signal & Lane Alignment (Dortmund)",
            "desc": "Confirmed green signal with straight road arrow alignment for clear right of way.",
            "filter": lambda r: r.get("metadata", {}).get("city") == "Dortmund" and any(tl.get("state") == "green" and tl.get("relevance") == 1 for tl in r.get("traffic_lights", [])) and len(r.get("road_arrows", [])) >= 1
        },
        {
            "key": "04_yellow_transition_state_koeln",
            "title": "Case 4: Amber/Yellow Phase Transition (Koeln)",
            "desc": "Intermediate yellow phase state classification under dynamic traffic flow.",
            "filter": lambda r: r.get("metadata", {}).get("city") == "Koeln" and any(tl.get("state") == "yellow" for tl in r.get("traffic_lights", [])) and len(r.get("traffic_lights", [])) >= 2
        },
        {
            "key": "05_off_unlit_traffic_lights_hannover",
            "title": "Case 5: Unlit / Inactive Traffic Lights (Hannover)",
            "desc": "Off-state discrimination preventing false green/red activations on dark gantries.",
            "filter": lambda r: r.get("metadata", {}).get("city") == "Hannover" and any(tl.get("state") == "off" for tl in r.get("traffic_lights", [])) and len(r.get("traffic_lights", [])) >= 3
        },
        {
            "key": "06_tiny_sub16px_distant_lights_duesseldorf",
            "title": "Case 6: Distant Sub-16px Small Traffic Lights (Duesseldorf)",
            "desc": "High-resolution P2 feature relay detection on distant signals (<16px height).",
            "filter": lambda r: r.get("metadata", {}).get("city") == "Duesseldorf" and any((tl.get("bbox_xyxy")[3] - tl.get("bbox_xyxy")[1]) < 16 for tl in r.get("traffic_lights", [])) and len(r.get("traffic_lights", [])) >= 2
        },
        {
            "key": "07_directional_turn_signals_essen",
            "title": "Case 7: Directional Turn Signal & Multi-Task Attributes (Essen)",
            "desc": "Roundness/pictogram recognition with dedicated turning signal.",
            "filter": lambda r: r.get("metadata", {}).get("city") == "Essen" and (any(tl.get("round_target") == 0 or tl.get("pictogram") != "round" for tl in r.get("traffic_lights", [])) or len(r.get("traffic_lights", [])) >= 4)
        },
        {
            "key": "08_multi_lane_road_arrows_bochum",
            "title": "Case 8: Multi-Lane Road Arrow Maneuver Perception (Bochum)",
            "desc": "Left, straight, and right arrow multihot direction recognition across lanes.",
            "filter": lambda r: r.get("metadata", {}).get("city") == "Bochum" and len(r.get("road_arrows", [])) >= 1
        },
        {
            "key": "09_high_density_junction_kassel",
            "title": "Case 9: High-Density Signalized Junction (Kassel)",
            "desc": "Simultaneous multi-target tracking across overhead gantries and side poles.",
            "filter": lambda r: r.get("metadata", {}).get("city") == "Kassel" and len(r.get("traffic_lights", [])) >= 4
        },
        {
            "key": "10_low_light_photometric_fulda",
            "title": "Case 10: Low-Light & Challenging Photometry (Fulda)",
            "desc": "Bloom-robust lamp state perception under challenging ambient illumination.",
            "filter": lambda r: r.get("metadata", {}).get("city") == "Fulda" and len(r.get("traffic_lights", [])) >= 2
        },
        {
            "key": "11_complex_urban_corridor_bremen",
            "title": "Case 11: Complex Multi-Object Urban Corridor (Bremen)",
            "desc": "Urban corridor with multiple signals and selective ego-lane relevance gating.",
            "filter": lambda r: r.get("metadata", {}).get("city") == "Bremen" and len(r.get("traffic_lights", [])) >= 2
        },
        {
            "key": "12_geometry_spatial_bias_berlin",
            "title": "Case 12: 14D Geometric Spatial Bias & Cross-Lane Discrimination (Berlin)",
            "desc": "Longitudinal & lateral distance reasoning for ego-lane relevance assignment.",
            "filter": lambda r: r.get("metadata", {}).get("city") == "Berlin" and any(tl.get("relevance") == 1 for tl in r.get("traffic_lights", [])) and len(r.get("road_arrows", [])) >= 1
        },
    ]

    selected_samples = []
    seen_ids = set()

    for sc in scenarios:
        matched = None
        for r in records:
            img_id = r.get("image_id")
            if img_id in seen_ids:
                continue
            if not Path(r.get("image_path", "")).exists():
                continue
            if sc["filter"](r):
                matched = r
                seen_ids.add(img_id)
                break
        if matched is None:
            # Fallback if strict filter didn't match
            for r in records:
                img_id = r.get("image_id")
                if img_id not in seen_ids and Path(r.get("image_path", "")).exists():
                    matched = r
                    seen_ids.add(img_id)
                    break
        selected_samples.append((sc, matched))

    print(f"\nSelected {len(selected_samples)} representative verification scenarios. Running Champion v4 inference...\n")

    summary_telemetry = []

    for idx, (sc, rec) in enumerate(selected_samples):
        img_path = Path(rec["image_path"])
        raw_bgr = cv2.imread(str(img_path))
        if raw_bgr is None:
            print(f"Error: Could not read image {img_path}")
            continue

        # Run inference
        post, raw_out1, input_wh = run_inference(
            model, raw_bgr, device, input_size=(960, 1920), tl_conf=0.25, arr_conf=0.25, iou_thresh=0.45
        )

        # Print detailed formatted table
        print_detailed_evaluation_table(idx + 1, sc["title"], rec, post)

        # Generate side-by-side evaluation image with zoom detail crops
        vis_image = create_verification_composite(
            raw_bgr, rec, post, raw_out1, input_wh, sc["title"], sc["desc"]
        )

        # Save to project results directory
        out_filename = f"champion_v4_{sc['key']}.jpg"
        out_path = out_dir / out_filename
        cv2.imwrite(str(out_path), vis_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        print(f"--> Saved verification overlay to: {out_path}")

        # Copy to artifact directory for markdown display
        artifact_path = artifact_dir / out_filename
        shutil.copy(str(out_path), str(artifact_path))

        # Collect summary data
        tl_valid = post["traffic_lights"]["valid"][0].cpu().numpy().astype(bool)
        arr_valid = post["road_arrows"]["valid"][0].cpu().numpy().astype(bool)
        summary_telemetry.append({
            "case": idx + 1,
            "scenario": sc["title"],
            "image_id": rec.get("image_id"),
            "city": rec.get("metadata", {}).get("city", "N/A"),
            "gt_tl_count": len(rec.get("traffic_lights", [])),
            "pred_tl_count": int(tl_valid.sum()),
            "gt_arrow_count": len(rec.get("road_arrows", [])),
            "pred_arrow_count": int(arr_valid.sum()),
            "saved_file": str(out_path),
            "artifact_file": str(artifact_path),
        })

    # Save master summary JSON
    summary_json_path = out_dir / "verification_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_telemetry, f, indent=2)
    print(f"\nSaved master telemetry summary to: {summary_json_path}")
    print("\n" + "=" * 115)
    print(f"All {len(selected_samples)} verification images successfully generated and saved to {out_dir}")
    print("=" * 115)


if __name__ == "__main__":
    main()
