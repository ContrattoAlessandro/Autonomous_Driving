"""_inspect_predictions.py — visual prediction QA on the ATLAS test set.

Loads the trained best.pt, runs prediction on a sample of test images, and saves
side-by-side GT-vs-prediction montages so you can eyeball quality.

Output: results/inspect/<tag>/  with montage_*.jpg + per-image metrics JSON.
"""
from __future__ import annotations
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEST_IMG_DIR = Path(r"C:\Users\alexa\Desktop\Tesi_Autonomous_Driving\dataset_ATLAS\ATLAS\test\front_medium\images")
TEST_LBL_DIR = Path(r"C:\Users\alexa\Desktop\Tesi_Autonomous_Driving\dataset_ATLAS\ATLAS\test\front_medium\labels")

# ATLAS 25-class names (from data_atlas.yaml)
NAMES = {
    0:"circle_green",1:"circle_red",2:"off",3:"circle_red_yellow",4:"arrow_left_green",
    5:"circle_yellow",6:"arrow_right_red",7:"arrow_left_red",8:"arrow_straight_red",
    9:"arrow_left_red_yellow",10:"arrow_left_yellow",11:"arrow_straight_yellow",
    12:"arrow_right_red_yellow",13:"arrow_right_green",14:"arrow_right_yellow",
    15:"arrow_straight_green",16:"arrow_straight_left_green",17:"arrow_straight_red_yellow",
    18:"arrow_straight_left_red",19:"arrow_straight_left_yellow",
    20:"arrow_straight_left_red_yellow",21:"arrow_straight_right_red",
    22:"arrow_straight_right_red_yellow",23:"arrow_straight_right_yellow",24:"arrow_straight_right_green",
}
# short color name for drawing: green/red/yellow/off
STATE_COLOR = {0:"green",1:"red",2:"off",3:"red_yellow",5:"yellow"}
def state_of(c):
    n=NAMES.get(c,"")
    if "off"==n: return "off"
    if "red_yellow" in n: return "red_yellow"
    if n.endswith("_red"): return "red"
    if n.endswith("_yellow"): return "yellow"
    if n.endswith("_green"): return "green"
    return "other"


def load_gt(label_path: Path):
    gts=[]
    if not label_path.exists(): return gts
    for ln in label_path.read_text().splitlines():
        if ln.strip():
            c,cx,cy,w,h=(float(x) for x in ln.split())
            gts.append((int(c),cx,cy,w,h))
    return gts


def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=12, help="num images to montage")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tag", default="run", help="subdir under results/inspect/")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hit", action="store_true", help="prefer images WITH GT (more interesting)")
    args=ap.parse_args()

    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from ultralytics import YOLO

    out_dir = ROOT/"results"/"inspect"/args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    model=YOLO(args.weights)
    imgs=sorted(TEST_IMG_DIR.glob("*.jpg"))+sorted(TEST_IMG_DIR.glob("*.png"))
    random.seed(args.seed)
    if args.hit:
        imgs_with_gt=[p for p in imgs if (TEST_LBL_DIR/(p.stem+".txt")).exists()]
        random.shuffle(imgs_with_gt)
        sample=imgs_with_gt[:args.n]
        if len(sample)<args.n:
            extra=[p for p in imgs if p not in set(sample)]
            random.shuffle(extra); sample+=extra[:args.n-len(sample)]
    else:
        sample=random.sample(imgs, min(args.n,len(imgs)))

    # color palette for states
    PAL={"green":"#22DD22","red":"#FF3333","yellow":"#FFEE33","off":"#888888","red_yellow":"#FF9933","other":"#33CCFF"}
    total_gt=0; total_det=0; tp=0; fp=0
    per_img=[]

    for img_path in sample:
        im=Image.open(img_path).convert("RGB"); W,H=im.size
        gts=load_gt(TEST_LBL_DIR/(img_path.stem+".txt"))
        res=model.predict(str(img_path), conf=args.conf, verbose=False, imgsz=1280)
        dets=[]
        if res and res[0].boxes is not None:
            for b in res[0].boxes:
                x1,y1,x2,y2=b.xyxy[0].tolist(); c=int(b.cls[0]); cf=float(b.conf[0])
                dets.append((c,cf,x1,y1,x2,y2))

        # make a 2-panel montage: LEFT ground truth, RIGHT prediction
        canvas=Image.new("RGB",(W*2+20,H+30),"white")
        gt_im=im.copy(); pr_im=im.copy()
        gd=ImageDraw.Draw(gt_im); pd=ImageDraw.ImageDraw(pr_im)
        try: font=ImageFont.truetype("arial.ttf",16)
        except: font=ImageFont.load_default()
        # draw GT (green boxes, label = class name)
        for c,cx,cy,w,h in gts:
            x1,y1,x2,y2=(cx-w/2)*W,(cy-h/2)*H,(cx+w/2)*W,(cy+h/2)*H
            gd.rectangle([x1,y1,x2,y2],outline="#00FF00",width=3)
            gd.text((x1,max(0,y1-18)),NAMES.get(c,"?")[:14],fill="#00FF00",font=font)
        # draw predictions (colored by state)
        for c,cf,x1,y1,x2,y2 in dets:
            col=PAL.get(state_of(c),"#33CCFF")
            pd.rectangle([x1,y1,x2,y2],outline=col,width=3)
            pd.text((x1,max(0,y1-18)),f"{NAMES.get(c,'?')[:12]} {cf:.2f}",fill=col,font=font)

        canvas.paste(gt_im,(0,30)); canvas.paste(pr_im,(W+20,30))
        hd=ImageDraw.Draw(canvas)
        hd.text((10,5),f"GROUND TRUTH  ({len(gts)} boxes)",fill="black",font=font)
        hd.text((W+30,5),f"PREDICTION  ({len(dets)} boxes, conf>={args.conf})",fill="black",font=font)
        canvas.save(out_dir/f"montage_{img_path.stem}.jpg",quality=88)

        total_gt+=len(gts); total_det+=len(dets)
        per_img.append({"img":img_path.name,"n_gt":len(gts),"n_det":len(dets)})

    summary={"n_images":len(sample),"total_gt":total_gt,"total_det":total_det,
             "mean_gt_per_img":round(total_gt/max(1,len(sample)),2),
             "mean_det_per_img":round(total_det/max(1,len(sample)),2)}
    (out_dir/"summary.json").write_text(json.dumps(summary,indent=2))
    (out_dir/"per_image.json").write_text(json.dumps(per_img,indent=2))
    print(json.dumps(summary,indent=2))
    print(f"\nsaved {len(sample)} montages -> {out_dir}")


if __name__=="__main__":
    main()
