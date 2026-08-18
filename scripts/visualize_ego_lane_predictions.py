"""Generate visual inspection overlays for Ego-Lane pseudo-labels using SegFormer Lane Lines.

Draws:
1. Translucent green polygon for the Ego-Corridor (P_ego)
2. Yellow lines for physical lane markings
3. Green bounding boxes for arrows predicted as is_ego_lane = 1 (with u_j score)
4. Red bounding boxes for arrows predicted as is_ego_lane = 0 (with u_j score)
5. Traffic light bounding boxes for context

Saves rendered images into tl_detection/results/visualizations/ego_lane_samples/
"""

from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

from tlr_yolo_mtl.data.io import read_records
from tlr_yolo_mtl.data.schema import ImageRecord


def extract_ego_corridor(lane_mask_small: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
    mask_h, mask_w = lane_mask_small.shape
    center_x = mask_w / 2.0
    horizon_y = int(mask_h * 0.45)
    bottom_y = int(mask_h * 0.98)

    left_boundary = []
    right_boundary = []
    scale_x = orig_w / float(mask_w)
    scale_y = orig_h / float(mask_h)

    for y in range(bottom_y, horizon_y, -4):
        row_lanes = np.where(lane_mask_small[y, :] > 0)[0]
        left_pts = row_lanes[row_lanes < center_x]
        right_pts = row_lanes[row_lanes > center_x]

        t = (y - horizon_y) / max(bottom_y - horizon_y, 1)

        if len(left_pts) > 0:
            lx = int(left_pts.max())
        else:
            lx = int(center_x - (mask_w * 0.28) * t)

        if len(right_pts) > 0:
            rx = int(right_pts.min())
        else:
            rx = int(center_x + (mask_w * 0.28) * t)

        if rx - lx < int(mask_w * 0.08):
            lx = int(center_x - (mask_w * 0.14) * t)
            rx = int(center_x + (mask_w * 0.14) * t)

        left_boundary.append((int(round(lx * scale_x)), int(round(y * scale_y))))
        right_boundary.append((int(round(rx * scale_x)), int(round(y * scale_y))))

    polygon_pts = left_boundary + right_boundary[::-1]
    corridor_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    if polygon_pts:
        poly_np = np.array(polygon_pts, dtype=np.int32)
        cv2.fillPoly(corridor_mask, [poly_np], 1)

    return corridor_mask


def draw_overlay(
    record: ImageRecord,
    model: torch.nn.Module,
    device: torch.device,
    output_path: Path,
) -> None:
    img_path = Path(record.image_path)
    if not img_path.is_absolute():
        img_path = (WORKSPACE_ROOT / img_path).resolve()

    image = cv2.imread(str(img_path))
    if image is None:
        return
    h0, w0, _ = image.shape

    # Preprocessing
    resized = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    tensor = torch.from_numpy((rgb.transpose(2, 0, 1) - mean) / std).float().unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        logits = outputs.logits
        upsampled = torch.nn.functional.interpolate(logits, size=(256, 512), mode="bilinear", align_corners=False)
        probs = torch.softmax(upsampled, dim=1)
        lane_probs = (probs[:, 1] + probs[:, 2]).cpu().numpy()[0]
        lane_mask_small = (lane_probs > 0.30).astype(np.uint8)

    # Reconstruct corridor and lane line mask at full resolution
    corridor_mask = extract_ego_corridor(lane_mask_small, w0, h0)
    lane_mask_full = cv2.resize(lane_mask_small, (w0, h0), interpolation=cv2.INTER_NEAREST)

    overlay = image.copy()
    # Green tint for Ego Corridor
    overlay[corridor_mask > 0] = (
        overlay[corridor_mask > 0] * 0.60 + np.array([0, 220, 0], dtype=np.uint8) * 0.40
    ).astype(np.uint8)
    # Yellow for detected lane demarcation lines
    overlay[lane_mask_full > 0] = (
        overlay[lane_mask_full > 0] * 0.20 + np.array([0, 255, 255], dtype=np.uint8) * 0.80
    ).astype(np.uint8)

    # Draw arrows
    for arrow in record.road_arrows:
        x1, y1, x2, y2 = map(int, map(round, arrow.bbox_xyxy))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w0, x2), min(h0, y2)

        crop = corridor_mask[y1:y2, x1:x2]
        overlap = float(crop.mean()) if crop.size > 0 else 0.0
        is_ego = overlap >= 0.40

        m_vec = arrow.direction_multihot
        m_str = []
        if m_vec[0]: m_str.append("L")
        if m_vec[1]: m_str.append("S")
        if m_vec[2]: m_str.append("R")
        dir_label = "-".join(m_str) if m_str else "arrow"

        if is_ego:
            color = (0, 255, 0)
            tag = f"EGO ({dir_label}) u={overlap:.2f}"
        else:
            color = (0, 0, 255)
            tag = f"OTHER_LANE ({dir_label}) u={overlap:.2f}"

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
        label_size, _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        ty = max(y1 - 8, label_size[1] + 8)
        cv2.rectangle(
            overlay,
            (x1, ty - label_size[1] - 4),
            (x1 + label_size[0] + 6, ty + 4),
            color,
            -1,
        )
        cv2.putText(
            overlay,
            tag,
            (x1 + 3, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0) if is_ego else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Draw Traffic Lights
    for tl in record.traffic_lights:
        x1, y1, x2, y2 = map(int, map(round, tl.bbox_xyxy))
        tl_color = (0, 255, 255) if tl.relevance == 1 else (180, 180, 180)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), tl_color, 2)

    # Top banner legend
    banner = np.zeros((45, w0, 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)
    legend_text = (
        f"{record.image_id} | "
        f"[GREEN CORRIDOR = Ego Lane Polygon] | "
        f"[YELLOW = Lane Lines] | "
        f"[GREEN BOX = Ego Arrow] | "
        f"[RED BOX = Adjacent Lane Arrow]"
    )
    cv2.putText(banner, legend_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    final_img = np.vstack([banner, overlay])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), final_img)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "tlr_mtl_dtld_paired" / "records.jsonl",
    )
    parser.add_argument(
        "--model-id",
        default="Efferbach/segformer-finetuned-lane-10k-steps",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "visualizations" / "ego_lane_samples",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=30,
    )
    args = parser.parse_args()

    print(f"Loading records from {args.records}...")
    records = list(read_records(args.records))

    multi_lane_records = []
    for r in records:
        if r.source_dataset == "DTLD" and len(r.road_arrows) >= 2:
            xs = [(a.bbox_xyxy[0] + a.bbox_xyxy[2]) / 2 for a in r.road_arrows]
            if max(xs) - min(xs) > 200:
                multi_lane_records.append(r)

    standard_records = [r for r in records if r.source_dataset == "DTLD" and r.road_arrows and r not in multi_lane_records]

    selected = multi_lane_records[: min(20, len(multi_lane_records))] + standard_records[: min(10, len(standard_records))]
    selected = selected[: args.n_samples]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading SegFormer Lane model on {device}...")
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_id).to(device).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rendering overlays to {args.output_dir}...")

    for idx, record in enumerate(selected):
        out_file = args.output_dir / f"sample_{idx+1:02d}_{record.image_id.replace('/', '_')}.jpg"
        draw_overlay(record, model, device, out_file)
        if (idx + 1) % 5 == 0 or idx + 1 == len(selected):
            print(f"  [{idx+1}/{len(selected)}] Saved: {out_file.name}")

    print(f"\n[Done] Rendered {len(selected)} sample overlays to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
