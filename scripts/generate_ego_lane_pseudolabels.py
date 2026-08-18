"""Generate high-precision Ego-Lane pseudo-labels on DTLD using Lane Line + Ego Corridor Polygon.

This script:
1. Runs SegFormer Lane Line model on DTLD images to detect physical lane demarcation lines.
2. Reconstructs the exact Ego-Corridor Polygon (P_ego) between the left and right ego-boundary lines.
3. Pools P_ego over each annotated road arrow to calculate:
     u_j = Overlap(b_j, P_ego) in [0.0, 1.0]
     is_ego_lane = 1 if u_j >= 0.40 else 0
     valid_ego_lane = True
     arrow_ego_lane = True in TaskValidity
4. Validates the entire corpus schema and writes updated records.jsonl and qa_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tlr_yolo_mtl.data.io import read_records, write_records, write_json
from tlr_yolo_mtl.data.qa import build_qa_report
from tlr_yolo_mtl.data.schema import ImageRecord, validate_records
from tlr_yolo_mtl.data.splits import assert_no_split_leakage


class DTLDLaneDataset(Dataset):
    """Dataset for batch image loading and pre-processing for SegFormer."""

    def __init__(
        self,
        records: Sequence[ImageRecord],
        target_size: tuple[int, int] = (1024, 512),  # (width, height)
    ) -> None:
        self.records = records
        self.target_w, self.target_h = target_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        img_path = Path(record.image_path)
        if not img_path.is_absolute():
            img_path = (WORKSPACE_ROOT / img_path).resolve()

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"cannot read image at {img_path}")

        resized = cv2.resize(image, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        # Transpose to [3, H, W] and normalize
        tensor_np = (rgb.transpose(2, 0, 1) - self.mean) / self.std
        tensor = torch.from_numpy(tensor_np).float()

        return tensor, index


def extract_ego_corridor_polygon(
    lane_mask_small: np.ndarray,
    orig_w: int,
    orig_h: int,
) -> np.ndarray:
    """Build the binary ego-corridor polygon mask at original image resolution."""
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

        # Left ego line is the max x < center_x
        if len(left_pts) > 0:
            lx = int(left_pts.max())
        else:
            # Perspective corridor linear fallback for dashed/missing lines
            lx = int(center_x - (mask_w * 0.28) * t)

        # Right ego line is the min x > center_x
        if len(right_pts) > 0:
            rx = int(right_pts.min())
        else:
            rx = int(center_x + (mask_w * 0.28) * t)

        # Ensure corridor has reasonable minimum width
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-records",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "tlr_mtl_dtld_paired" / "records.jsonl",
    )
    parser.add_argument(
        "--output-records",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "tlr_mtl_dtld_paired" / "records.jsonl",
    )
    parser.add_argument(
        "--model-id",
        default="Efferbach/segformer-finetuned-lane-10k-steps",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="batch size for GPU inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="overlap threshold for is_ego_lane classification",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="optional limit of images for dry run",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    print(f"Reading records from {args.input_records}...")
    all_records = list(read_records(args.input_records))
    print(f"Loaded {len(all_records)} total records.")

    dtld_arrow_indices = []
    dtld_arrow_records = []
    for idx, record in enumerate(all_records):
        if record.source_dataset == "DTLD" and record.task_valid.arrow_detection:
            dtld_arrow_indices.append(idx)
            dtld_arrow_records.append(record)

    print(f"Found {len(dtld_arrow_records)} DTLD paired records with arrow detection.")
    if args.limit:
        dtld_arrow_indices = dtld_arrow_indices[: args.limit]
        dtld_arrow_records = dtld_arrow_records[: args.limit]
        print(f"Limited processing to {len(dtld_arrow_records)} records.")

    device = torch.device(args.device)
    print(f"Loading SegFormer Lane model ({args.model_id}) on {device}...")
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_id).to(device).eval()
    print("SegFormer loaded successfully.")

    dataset = DTLDLaneDataset(dtld_arrow_records, target_size=(1024, 512))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    total_arrows = 0
    ego_arrows = 0
    non_ego_arrows = 0
    scores_list = []
    start_time = time.time()

    print(f"Starting lane inference across {len(dtld_arrow_records)} images...")

    with torch.no_grad():
        for batch_tensors, batch_indices in tqdm(loader, desc="Lane Ego-Corridor Inference"):
            batch_tensors = batch_tensors.to(device, non_blocking=True)
            outputs = model(batch_tensors)
            logits = outputs.logits  # [B, 3, 128, 256]

            # Upsample to [B, 3, 256, 512]
            upsampled = torch.nn.functional.interpolate(
                logits, size=(256, 512), mode="bilinear", align_corners=False
            )
            probs = torch.softmax(upsampled, dim=1)
            # Lane lines = class 1 (left) + class 2 (right) or lane probabilities > 0.30
            lane_probs = (probs[:, 1] + probs[:, 2]).cpu().numpy()  # [B, 256, 512]
            lane_masks = (lane_probs > 0.30).astype(np.uint8)

            for i in range(len(batch_indices)):
                local_idx = int(batch_indices[i].item())
                orig_record_idx = dtld_arrow_indices[local_idx]
                record = all_records[orig_record_idx]
                lane_mask_small = lane_masks[i]  # [256, 512]

                orig_w = record.original_width
                orig_h = record.original_height

                corridor_mask = extract_ego_corridor_polygon(lane_mask_small, orig_w, orig_h)

                for arrow in record.road_arrows:
                    x1, y1, x2, y2 = map(int, map(round, arrow.bbox_xyxy))
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(orig_w, x2), min(orig_h, y2)

                    arrow_crop = corridor_mask[y1:y2, x1:x2]
                    overlap = float(arrow_crop.mean()) if arrow_crop.size > 0 else 0.0
                    is_ego = 1 if overlap >= args.threshold else 0

                    arrow.is_ego_lane = is_ego
                    arrow.valid_ego_lane = True
                    arrow.source_attributes["ego_lane_score"] = round(float(overlap), 4)

                    total_arrows += 1
                    if is_ego:
                        ego_arrows += 1
                    else:
                        non_ego_arrows += 1
                    scores_list.append(overlap)

                record.task_valid.arrow_ego_lane = True

    elapsed = time.time() - start_time
    fps = len(dtld_arrow_records) / max(elapsed, 0.001)
    print(f"\n[Completed in {elapsed:.2f}s | Speed: {fps:.1f} FPS]")
    print(f"Total arrows evaluated: {total_arrows}")
    print(f"  - Ego-lane arrows (is_ego_lane=1): {ego_arrows} ({ego_arrows / max(total_arrows, 1) * 100:.2f}%)")
    print(f"  - Non-ego arrows  (is_ego_lane=0): {non_ego_arrows} ({non_ego_arrows / max(total_arrows, 1) * 100:.2f}%)")
    if scores_list:
        scores_arr = np.array(scores_list)
        print(f"Score u_j percentiles: p10={np.percentile(scores_arr, 10):.3f}, p50={np.percentile(scores_arr, 50):.3f}, p90={np.percentile(scores_arr, 90):.3f}, mean={np.mean(scores_arr):.3f}")

    print(f"\nValidating all {len(all_records)} records...")
    validate_records(all_records)
    assert_no_split_leakage(all_records, hash_images=False)
    print("Validation passed successfully!")

    print(f"Writing updated records to {args.output_records}...")
    written_count = write_records(args.output_records, all_records)
    print(f"Successfully wrote {written_count} records.")

    qa = build_qa_report(all_records, hash_images=False)
    qa_path = args.output_records.parent / "qa_report.json"
    write_json(qa_path, qa)
    print(f"Updated QA report at {qa_path}")


if __name__ == "__main__":
    main()
