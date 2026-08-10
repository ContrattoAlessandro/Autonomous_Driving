"""harmonize_labels.py — assemble unified train/val/test splits for each tier.

Inputs (produced by convert_{dtld,bosch,lisa,oi}.py):
    datasets/yolo/<source>/<tier>/labels/{train,val,test}/*.txt
    datasets/yolo/<source>/images/{train,val,test}/*.<ext>

This script:
  1. Indexes every (image, label) pair per source+tier.
  2. Builds stratified train/val/test lists:
       - Sources with a native test split (Bosch 'test', LISA 'test',
         DTLD-only-images-in-test) keep their test split for cross-dataset eval.
       - DTLD (no native split) is stratified into train/val by a hash of the
         image stem (90/10), test set held out separately if requested.
       - The remaining 'train' pool is split 90/10 into train/val (seeded).
  3. Writes the list files the data_tier*.yaml point to:
       datasets/yolo/tier{A,B,C}/{train,val,test}.txt
     Each line = an absolute path to an IMAGE, so ``path:`` cannot be
     accidentally prepended twice by Ultralytics.
  4. Updates configs/data_tierC.yaml's nc/names from the observed Tier-C classes.

Usage:
    python scripts/harmonize_labels.py --val-frac 0.1 --seed 42
    python scripts/harmonize_labels.py --tier A --sources dtld,bosch,lisa,openimages
"""
from __future__ import annotations
import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from _common import yolo_root, STATE2ID, ID2STATE

TIERS = ("tierA", "tierB", "tierC")
# Sources contributing to each tier:
#   Tier A: all four (incl. Open Images, which is state-free).
#   Tier B/C: the three state-rich sources only.
TIER_SOURCES = {
    "tierA": ["dtld", "bosch", "lisa", "openimages"],
    "tierB": ["dtld", "bosch", "lisa"],
    "tierC": ["dtld"],
}
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def find_label_image_pairs(source: str, tier: str, split: str) -> list[tuple[Path, Path]]:
    """Return [(image_path, label_path), ...] for a source/tier/split.
    Images are searched under source/images/<split> and matched to labels by stem."""
    img_dir = yolo_root() / source / "images" / split
    lbl_dir = yolo_root() / source / tier / "labels" / split
    if not lbl_dir.exists():
        return []
    pairs = []
    for lbl in lbl_dir.glob("*.txt"):
        # find the image with the same stem anywhere under the source's image dir
        img = None
        for ext in IMG_EXTS:
            cand = img_dir / f"{lbl.stem}{ext}"
            if cand.exists():
                img = cand
                break
        if img is None:
            # search all image subdirs of this source
            for cand in (yolo_root() / source).rglob(f"{lbl.stem}*"):
                if cand.suffix.lower() in IMG_EXTS:
                    img = cand
                    break
        if img is not None:
            pairs.append((img, lbl))
    return pairs


def stratified_split(pairs: list[tuple[Path, Path]], val_frac: float, seed: int) -> tuple[list, list]:
    """Split into train/val stratified by the MAJORITY class of each image's labels.
    Keeps class balance approximately constant across splits."""
    rng = random.Random(seed)
    by_class: dict[int, list[tuple[Path, Path]]] = defaultdict(list)
    for img, lbl in pairs:
        classes = []
        for line in lbl.read_text().splitlines():
            line = line.strip()
            if line:
                classes.append(int(float(line.split()[0])))
        # majority class (or 0 if empty) -> stratification key
        key = Counter(classes).most_common(1)[0][0] if classes else -1
        by_class[key].append((img, lbl))
    train, val = [], []
    for c, items in by_class.items():
        rng.shuffle(items)
        k = max(1, int(round(len(items) * val_frac)))
        val.extend(items[:k])
        train.extend(items[k:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_list_file(out_dir: Path, name: str, items: list[tuple[Path, Path]]) -> Path:
    """Write <out_dir>/<name>.txt listing image paths (one per line)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.txt"
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for img, _ in items:
            # Absolute paths are robust when train.py is launched from a
            # directory other than tl_detection and avoid path double-joins.
            f.write(str(img.resolve()).replace("\\", "/") + "\n")
    return p


def tierC_classes(tierC_pairs: list[tuple[Path, Path]]) -> tuple[int, dict[int, str]]:
    """Scan tierC labels to discover the actual pictogram classes used, and return
    (nc, id2name). DTLD-only; ids >= 6 are pictograms (see convert_dtld.py)."""
    seen = set()
    for _, lbl in tierC_pairs:
        for line in lbl.read_text().splitlines():
            if line.strip():
                seen.add(int(float(line.split()[0])))
    base = {0: "red", 1: "yellow", 2: "green", 3: "red_yellow", 4: "off", 5: "unknown"}
    picto = {6: "arrow_left", 7: "arrow_straight", 8: "arrow_right",
             9: "pedestrian", 10: "bicycle", 11: "tram",
             12: "arrow_back_left", 13: "arrow_back_right"}
    id2name = {**base, **picto}
    # Preserve native IDs.  Re-indexing names without rewriting every label
    # changes the class meaning (e.g. a native class 8 would still be written
    # as 8 while the YAML would call it class 2).
    if not seen:
        return 0, {}
    max_id = max(seen)
    final = {i: id2name.get(i, f"class_{i}") for i in range(max_id + 1)}
    return max_id + 1, final


def assert_disjoint_splits(
    train_pairs: list[tuple[Path, Path]],
    val_pairs: list[tuple[Path, Path]],
    test_pairs: list[tuple[Path, Path]],
) -> None:
    """Fail before writing if an image appears in multiple splits."""
    sets = {
        "train": {img.resolve() for img, _ in train_pairs},
        "val": {img.resolve() for img, _ in val_pairs},
        "test": {img.resolve() for img, _ in test_pairs},
    }
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[first] & sets[second]
        if overlap:
            raise RuntimeError(
                f"split leakage between {first}/{second}: {len(overlap)} shared images; "
                f"example={next(iter(overlap))}"
            )


def main() -> None:
    p = argparse.ArgumentParser(description="Assemble unified YOLO splits per tier")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tier", choices=TIERS + ["all"], default="all")
    p.add_argument("--sources", type=str, default=None,
                   help="override source list for tierA (comma-separated)")
    args = p.parse_args()

    tiers = TIERS if args.tier == "all" else (args.tier,)
    summary_rows = []

    for tier in tiers:
        sources = TIER_SOURCES[tier]
        if args.sources and tier == "tierA":
            sources = [s.strip() for s in args.sources.split(",")]
        out_dir = yolo_root() / tier

        # collect per-split pools
        test_pairs, trainval_pairs = [], []
        per_source_counts = {}
        for src in sources:
            # native test splits (Bosch/LISA 'test'); DTLD/openimages have none
            tp = find_label_image_pairs(src, tier, "test")
            trp = find_label_image_pairs(src, tier, "train") + find_label_image_pairs(src, tier, "val")
            test_pairs.extend(tp)
            trainval_pairs.extend(trp)
            per_source_counts[src] = {"trainval": len(trp), "test": len(tp)}

        # stratified train/val split of the non-test pool
        train_pairs, val_pairs = stratified_split(trainval_pairs, args.val_frac, args.seed)

        assert_disjoint_splits(train_pairs, val_pairs, test_pairs)

        write_list_file(out_dir, "train", train_pairs)
        write_list_file(out_dir, "val", val_pairs)
        write_list_file(out_dir, "test", test_pairs)

        n_train, n_val, n_test = len(train_pairs), len(val_pairs), len(test_pairs)
        print(f"[{tier}] train={n_train} val={n_val} test={n_test}  "
              f"per-source={per_source_counts}")
        summary_rows.append((tier, n_train, n_val, n_test, per_source_counts))

        # Tier C: refresh configs/data_tierC.yaml nc/names from observed classes
        if tier == "tierC" and train_pairs + val_pairs:
            nc, id2name = tierC_classes(train_pairs + val_pairs + test_pairs)
            update_tierC_config(nc, id2name)
            print(f"[tierC] refreshed nc={nc} names={id2name}")

    # write a small JSON summary for the thesis
    import json
    out = yolo_root().parent / "results" / "splits_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(
            [{"tier": t, "train": tr, "val": v, "te": te, "per_source": ps}
             for t, tr, v, te, ps in summary_rows],
            f, indent=2,
        )
    print(f"\nwrote {out}")


def update_tierC_config(nc: int, id2name: dict[int, str]) -> None:
    """Rewrite configs/data_tierC.yaml nc + names block in place."""
    import yaml
    cfg = yolo_root().parent / "configs" / "data_tierC.yaml"
    data = yaml.safe_load(cfg.read_text())
    data["nc"] = nc
    data["names"] = {i: id2name[i] for i in range(nc)}
    cfg.write_text(yaml.safe_dump(data, sort_keys=False))


if __name__ == "__main__":
    main()
