"""make_tables.py — turn eval outputs into LaTeX-ready tables + plots for the thesis.

Reads results/eval/<run>/ultralytics_metrics.json (+ size_stratified.json if present)
and writes, to results/tables/:
  * main_results.tex        mAP50/50-95 + P + R across runs (one row per run)
  * per_class_<run>.tex     per-class AP for a single run
  * size_recall_<run>.tex   recall-per-size-bin table
  * size_recall_plot_<run>.png   recall-vs-size curve
  * main_results_plot.png   grouped bar chart comparing runs

Usage:
    python scripts/make_tables.py
    python scripts/make_tables.py --runs yolo26x-p2_tierB_obj365,yolo26x-p2_tierB_coco
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL_ROOT = ROOT / "results" / "eval"
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

KEY_MAP = {
    "mAP@50": ("box.map50", "{:.3f}"),
    "mAP@50-95": ("box.map", "{:.3f}"),
    "P": ("box.mp", "{:.3f}"),
    "R": ("box.mr", "{:.3f}"),
}


def get(d: dict, dotted: str):
    cur = d
    for p in dotted.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


def collect_runs(runs: list[str]) -> list[dict]:
    out = []
    for r in runs:
        f = EVAL_ROOT / r / "ultralytics_metrics.json"
        if not f.exists():
            print(f"[warn] no metrics for run '{r}' ({f}); skipping")
            continue
        out.append({"name": r, "metrics": json.loads(f.read_text()),
                    "size": _load_size(r)})
    return out


def _load_size(run: str) -> dict | None:
    f = EVAL_ROOT / run / "size_stratified.json"
    return json.loads(f.read_text()) if f.exists() else None


def main_table(runs: list[dict]) -> str:
    cols = "l" + "c" * len(KEY_MAP)
    head = " & ".join(["Run", *KEY_MAP.keys()]) + r" \\"
    lines = [r"\begin{tabular}{" + cols + r"}", r"\toprule", head, r"\midrule"]
    for r in runs:
        cells = [r["name"].replace("_", r"\_")]
        for _, (key, fmt) in KEY_MAP.items():
            v = get(r["metrics"], key)
            cells.append(fmt.format(v) if v is not None else "--")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def per_class_table(run: dict) -> str:
    pc = run["metrics"].get("per_class") or {}
    lines = [r"\begin{tabular}{lcccc}", r"\toprule",
             r"Class & AP@50 & AP@50-95 & P & R \\", r"\midrule"]
    for cls, m in sorted(pc.items()):
        lines.append(f"{cls.replace('_', r'\_')} & {m['ap50']:.3f} & {m['ap']:.3f} & "
                     f"{m['p']:.3f} & {m['r']:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def size_table(run: dict) -> str:
    sz = run.get("size") or {}
    if not sz:
        return ""
    lines = [r"\begin{tabular}{lcc}", r"\toprule", r"Size bin (px) & \# objects & Recall \\", r"\midrule"]
    for b in ["8-16", "16-32", "32-96", ">96"]:
        v = sz.get(b, {})
        rec = v.get("recall")
        lines.append(f"{b} & {v.get('n', 0)} & {rec:.3f} \\\\" if rec is not None else f"{b} & {v.get('n', 0)} & -- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def size_plot(run: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sz = run.get("size") or {}
    if not sz:
        return
    bins = ["8-16", "16-32", "32-96", ">96"]
    rec = [sz.get(b, {}).get("recall") or 0.0 for b in bins]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(bins, rec, "o-", lw=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("object size bin  √(w·h) [px]")
    ax.set_ylabel("recall @ IoU 0.5")
    ax.set_title(f"{run['name']} — recall vs object size")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"size_recall_plot_{run['name']}.png", dpi=130)
    plt.close(fig)


def main_bar_plot(runs: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    names = [r["name"] for r in runs]
    series = {k: [get(r["metrics"], key) or 0.0 for r in runs] for k, (key, _) in KEY_MAP.items()}
    x = np.arange(len(names))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names)), 3.5))
    for i, (label, vals) in enumerate(series.items()):
        ax.bar(x + i * w, vals, w, label=label)
    ax.set_xticks(x + w * (len(series) - 1) / 2)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("score")
    ax.set_title("Detection performance comparison")
    ax.legend(ncol=len(series))
    fig.tight_layout()
    fig.savefig(OUT / "main_results_plot.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build LaTeX tables + plots from eval outputs")
    ap.add_argument("--runs", type=str, default=None,
                    help="comma-separated run names under results/eval/ (default: all)")
    args = ap.parse_args()

    if args.runs:
        names = [r.strip() for r in args.runs.split(",")]
    else:
        names = sorted(p.name for p in EVAL_ROOT.glob("*") if p.is_dir())

    runs = collect_runs(names)
    if not runs:
        print(f"no eval runs found under {EVAL_ROOT}. Run scripts/eval.py first.")
        return

    (OUT / "main_results.tex").write_text(main_table(runs))
    for r in runs:
        (OUT / f"per_class_{r['name']}.tex").write_text(per_class_table(r))
        if r.get("size"):
            (OUT / f"size_recall_{r['name']}.tex").write_text(size_table(r))
            size_plot(r)
    main_bar_plot(runs)
    print(f"wrote tables + plots to {OUT}")


if __name__ == "__main__":
    main()
