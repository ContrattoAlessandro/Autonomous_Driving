"""_bench_square.py — benchmark throughput a imgsz INTERO (quadrato) con AMP.

IMPORTANTE: Ultralytics NON supporta imgsz=[h,w] per train()/val(). Qualsiasi
lista viene collassata al valore massimo (check_imgsz max_dim=1). Quindi
[736,1280] diventa silenziosamente 1280x1280. Per ridurre VRAM usare un intero
piu' piccolo: 960 (56% dei pixel di 1280) o 1024 (64%).

Testa imgsz=960 (raccomandato) e opzionalmente 1024 per confronto.

Lancia:   python scripts/_bench_square.py
          python scripts/_bench_square.py --variants n,s,m --imgsz 1024
Output:   results/_bench_square.json + tabella a video
"""
from __future__ import annotations
import argparse, json, statistics, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent

# (variante, batch da provare in ordine — si ferma al primo OK)
DEFAULT_JOBS = [
    ("yolo26n-p2", [16, 8]),
    ("yolo26s-p2", [8, 4]),
    ("yolo26m-p2", [4, 2]),
    ("yolo26l-p2", [4, 2]),
    ("yolo26x-p2", [2, 1]),
]


def make_dataset():
    import numpy as np
    from PIL import Image
    for split, n in (("train", 80), ("val", 10)):
        di = ROOT / "_bench" / "images" / split
        dl = ROOT / "_bench" / "labels" / split
        di.mkdir(parents=True, exist_ok=True)
        dl.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            Image.fromarray((np.random.rand(1080, 1920, 3) * 255).astype("uint8")).save(di / f"fr{i:04d}.jpg")
            (dl / f"fr{i:04d}.txt").write_text("0 0.5 0.2 0.01 0.02\n")


def measure(variant, imgsz, batch, iters=8, warmup=2):
    from ultralytics import YOLO
    torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    times = []; last = {"t": None}

    def cb(tr):
        now = time.perf_counter()
        if last["t"] is not None:
            times.append(now - last["t"])
        last["t"] = now

    model = YOLO(f"{variant}.yaml")
    model.add_callback("on_train_batch_end", cb)
    run = ROOT / "runs" / "_bench" / f"sq{imgsz}_{variant}_b{batch}"
    model.train(
        data=str(ROOT / "_bench_data.yaml"), epochs=1, imgsz=imgsz, batch=batch,
        workers=4, device=0, project=str(run.parent), name=run.name, exist_ok=True,
        verbose=False, mosaic=0.0, mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
        copy_paste=0.0, erasing=0.0, fliplr=0.0, scale=0.0, translate=0.0,
        optimizer="SGD", lr0=0.01, close_mosaic=0, seed=42, deterministic=False,
        amp=True, val=False, plots=False, save=False,
    )
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e9
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    del model; torch.cuda.empty_cache()

    clean = times[warmup:]
    if not clean:
        return {"variant": variant, "batch": batch, "status": "no_data"}
    ms_iter = statistics.median(clean) * 1000.0
    ms_img = ms_iter / batch
    status = "thrash" if ms_img > 600 else "ok"
    return {
        "variant": variant, "batch": batch, "imgsz": imgsz, "status": status,
        "ms_per_iter": round(ms_iter, 1), "ms_per_image": round(ms_img, 2),
        "img_per_sec": round(1000.0 * batch / ms_iter, 1),
        "vram_peak_gb": round(peak, 2), "params_M": round(params, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="n,s,m,l,x", help="lista varianti (n,s,m,l,x)")
    ap.add_argument("--imgsz", type=int, default=960, help="imgsz quadrato (default 960)")
    ap.add_argument("--keep-thrash", action="store_true")
    args = ap.parse_args()

    sel = {c[0]: f"yolo26{c[0]}-p2" for c in [("n",), ("s",), ("m",), ("l",), ("x",)]}
    wanted = [sel[c.strip()] for c in args.variants.split(",")]
    jobs = [j for j in DEFAULT_JOBS if j[0] in wanted]

    make_dataset()
    results = []
    print(f"imgsz={args.imgsz}x{args.imgsz} (square)  AMP=on  GPU={torch.cuda.get_device_name(0)}\n")
    print(f"{'variant':14s} {'batch':>5} {'params':>8} {'VRAM':>7} {'ms/iter':>9} {'ms/img':>8} {'img/s':>7}  status")
    print("-" * 80)
    for variant, batches in jobs:
        for b in batches:
            try:
                r = measure(variant, args.imgsz, b)
            except torch.cuda.OutOfMemoryError:
                r = {"variant": variant, "batch": b, "status": "oom"}
                torch.cuda.empty_cache()
            results.append(r)
            if r["status"] == "ok":
                print(f"{variant:14s} {b:>5} {r['params_M']:>6.1f}M {r['vram_peak_gb']:>5.2f}G "
                      f"{r['ms_per_iter']:>9.1f} {r['ms_per_image']:>8.2f} {r['img_per_sec']:>7.1f}  {r['status']}")
                if not args.keep_thrash:
                    break
            elif r["status"] == "thrash":
                print(f"{variant:14s} {b:>5} {'':>8} {r.get('vram_peak_gb',''):>5}G "
                      f"{r.get('ms_per_iter',''):>9} {r.get('ms_per_image',''):>8} {'':>7}  THRASH (riduci batch)")
            else:
                print(f"{variant:14s} {b:>5} {'':>8} {'':>7} {'':>9} {'':>8} {'':>7}  {r['status']}")

    out = ROOT / "results" / "_bench_square.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
