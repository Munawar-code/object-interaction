#!/usr/bin/env python3
"""
Photo Analysis (single-file CLI)
- Pure Python (no Colab/Drive/magics)
- VS Code friendly
- Stages as subcommands: detect, parse, pair, actions, stats, preview (preview requires opencv & matplotlib).

Example:
  python photo_analysis.py detect  --img_dir ./data/preproc --out_dir ./out/hoi_json --model yolov8x.pt
  python photo_analysis.py parse   --in_dir  ./out/hoi_json   --out_dir ./out/parsed_json
  python photo_analysis.py pair    --in_dir  ./out/parsed_json --out_dir ./out/pairs_json_relaxed --obj_conf_min 0.15 --topk 10
  python photo_analysis.py actions --in_dir  ./out/pairs_json_relaxed --out_dir ./out/hoi_actions_json
  python photo_analysis.py stats   --det_dir ./out/hoi_json
  python photo_analysis.py preview --pairs_dir ./out/pairs_json --img_root ./data/preproc
"""
from __future__ import annotations
import argparse, json, math, sys, logging, random
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple

# ---------- Constants & helpers ----------
COCO_NAMES: List[str] = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}

def list_images(root: Path) -> Iterable[Path]:
    return (p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)

def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))

def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())

def iou(boxA, boxB) -> float:
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA); interH = max(0.0, yB - yA)
    inter  = interW * interH
    areaA  = max(0.0, (boxA[2]-boxA[0])) * max(0.0, (boxA[3]-boxA[1]))
    areaB  = max(0.0, (boxB[2]-boxB[0])) * max(0.0, (boxB[3]-boxB[1]))
    union  = areaA + areaB - inter + 1e-9
    return float(inter / union)

def centroid(b): return ((b[0]+b[2]) * 0.5, (b[1]+b[3]) * 0.5)
def l2(p, q): return math.hypot(p[0]-q[0], p[1]-q[1])
def union_box(a, b): return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]

# ---------- Subcommand: detect ----------
def cmd_detect(args):
    from ultralytics import YOLO
    try:
        import torch
        device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    except Exception:
        device = args.device or "cpu"

    logging.info("Using device: %s", device)
    model = YOLO(args.model)

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    images = list(list_images(img_dir))
    logging.info("Found %d images in %s", len(images), img_dir)

    from tqdm import tqdm
    for img_path in tqdm(images, desc="Detecting"):
        res = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device,
            verbose=False
        )
        dets = res[0].boxes
        objs = []
        for b in dets:
            cls = int(b.cls.item())
            conf = float(b.conf.item())
            box = list(map(float, b.xyxy[0].tolist()))
            objs.append({"cls": cls, "conf": conf, "box": box})

        rel = str(img_path.relative_to(img_dir))
        out_path = out_dir / (Path(rel).with_suffix(".json").name)
        save_json(out_path, {"image": rel, "detections": objs, "meta": {
            "model": args.model, "imgsz": args.imgsz, "conf": args.conf, "iou": args.iou
        }})

# ---------- Subcommand: parse ----------
def cmd_parse(args):
    in_dir = Path(args.in_dir); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.json"))
    logging.info("Found %d detection JSONs", len(files))
    from tqdm import tqdm
    for jf in tqdm(files, desc="Parsing"):
        data = load_json(jf)
        dets = data.get("detections", [])
        humans, objects = [], []
        for d in dets:
            cls = int(d["cls"])
            item = {
                "box": d["box"],
                "conf": float(d["conf"]),
                "cls": cls,
                "label": COCO_NAMES[cls] if 0 <= cls < len(COCO_NAMES) else f"class_{cls}",
            }
            (humans if cls == 0 else objects).append(item)
        save_json(out_dir / jf.name, {
            "image": data.get("image"),
            "humans": humans,
            "objects": objects,
            "meta": {"source_json": jf.name}
        })

# ---------- Subcommand: pair ----------
def cmd_pair(args):
    in_dir = Path(args.in_dir); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.json"))
    logging.info("Found %d parsed files", len(files))
    from tqdm import tqdm
    for jf in tqdm(files, desc="Pairing"):
        data = load_json(jf)
        humans  = data.get("humans", [])
        objects = [o for o in data.get("objects", []) if float(o.get("conf", 0.0)) >= args.obj_conf_min]

        interactions = []
        for hi, h in enumerate(humans):
            hc = centroid(h["box"])
            ranked: List[Tuple[float, int, Dict[str, Any]]] = []
            for oi, o in enumerate(objects):
                oc = centroid(o["box"])
                ranked.append((l2(hc, oc), oi, o))
            ranked.sort(key=lambda x: x[0])
            for dist, oi, o in ranked[:args.topk]:
                interactions.append({
                    "human_idx": hi,
                    "object_idx": oi,
                    "human_box": h["box"],
                    "object_box": o["box"],
                    "object_label": o.get("label", f"class_{o.get('cls', -1)}"),
                    "object_cls": int(o.get("cls", -1)),
                    "scores": {
                        "human_conf": float(h.get("conf", 1.0)),
                        "object_conf": float(o.get("conf", 0.0)),
                        "centroid_dist": float(dist),
                        "iou": float(iou(h["box"], o["box"])),
                    },
                    "union_box": union_box(h["box"], o["box"])
                })

        save_json(out_dir / jf.name, {
            "image": data.get("image"),
            "interactions": interactions,
            "counts": {"humans": len(humans), "objects": len(objects), "pairs": len(interactions)},
            "meta": {"source": data.get("meta", {}).get("source_json", jf.name)}
        })

# ---------- Subcommand: actions ----------
HOLDABLE = {
    "cup","bottle","cell phone","wine glass","bowl","fork","knife","spoon","banana","apple",
    "book","remote","mouse","keyboard","tennis racket","baseball bat","baseball glove",
    "sports ball","skateboard","umbrella","teddy bear","scissors","vase"
}
SCREENLIKE = {"laptop","tv","cell phone","book","clock","microwave","oven","refrigerator"}
SEATS = {"chair","couch","bed","bench","dining table","toilet"}
RIDABLE = {"bicycle","motorcycle","horse","skateboard","surfboard","snowboard","boat"}

def zones(h):
    x1,y1,x2,y2 = h
    w = max(1.0, x2-x1); hgt = max(1.0, y2-y1)
    head = [x1 + 0.20*w, y1,            x2 - 0.20*w, y1 + 0.30*hgt]
    hands_left  = [x1,          y1 + 0.30*hgt, x1 + 0.25*w, y1 + 0.70*hgt]
    hands_right = [x2 - 0.25*w, y1 + 0.30*hgt, x2,          y1 + 0.70*hgt]
    lap  = [x1 + 0.20*w, y1 + 0.55*hgt, x2 - 0.20*w, y2]
    return head, hands_left, hands_right, lap

def overlaps(a,b,thr=0.05): return iou(a,b) >= thr

def centroid2(b): return ((b[0]+b[2]) * 0.5, (b[1]+b[3]) * 0.5)

def action_rules(human_box, obj_box, obj_label):
    actions = []
    hc = centroid2(human_box); oc = centroid2(obj_box)
    dist = l2(hc, oc)
    head, lh, rh, lap = zones(human_box)
    aH = max(1.0, (human_box[2]-human_box[0])*(human_box[3]-human_box[1]))
    aO = max(1e-9, (obj_box[2]-obj_box[0])*(obj_box[3]-obj_box[1]))
    rel_area = aO / aH
    ov = iou(human_box, obj_box)

    if obj_label in HOLDABLE:
        near_hands = overlaps(obj_box, lh, 0.01) or overlaps(obj_box, rh, 0.01)
        inside = ov > 0.02
        if near_hands or inside:
            score = 0.6 + min(0.3, 0.6 * rel_area)
            actions.append({"label":"holding","score":round(score,3)})
    if obj_label in SEATS and (overlaps(obj_box, lap, 0.02) or (oc[1] > hc[1] and ov > 0.01)):
        actions.append({"label":"sitting_on","score":0.75})
    if obj_label in RIDABLE and (oc[1] > hc[1] and ov > 0.01):
        actions.append({"label":"riding","score":0.75})
    if obj_label in SCREENLIKE and (overlaps(obj_box, head, 0.01) or dist < 0.35 * aH**0.5):
        actions.append({"label":"looking_at","score":0.6})
    if obj_label == "backpack" and ov > 0.02:
        actions.append({"label":"wearing","score":0.7})
    if obj_label == "tie" and ov > 0.02:
        actions.append({"label":"wearing","score":0.7})
    if obj_label == "handbag" and (overlaps(obj_box, lh, 0.01) or overlaps(obj_box, rh, 0.01)):
        actions.append({"label":"carrying","score":0.65})
    return actions

def cmd_actions(args):
    in_dir = Path(args.in_dir); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.json"))
    logging.info("Found %d pair files", len(files))
    from tqdm import tqdm
    for jf in tqdm(files, desc="Actions"):
        d = load_json(jf)
        inters = d.get("interactions", []); enriched = []
        for it in inters:
            acts = action_rules(it["human_box"], it["object_box"], it.get("object_label",""))
            if acts:
                e = dict(it); e["actions"] = acts; enriched.append(e)
        save_json(out_dir / jf.name, {
            "image": d.get("image"),
            "interactions": enriched,
            "counts": {
                "humans": d.get("counts",{}).get("humans", 0),
                "objects": d.get("counts",{}).get("objects", 0),
                "pairs": len(enriched)
            },
            "meta": {"source": jf.name, "method":"rule_based_v1"}
        })

# ---------- Subcommand: stats ----------
def cmd_stats(args):
    from collections import Counter
    det_dir = Path(args.det_dir)
    files = sorted(det_dir.glob("*.json"))
    n_imgs = len(files)
    imgs_with_person = imgs_with_object = 0
    obj_counter = Counter()
    person_counts: List[int] = []
    object_counts: List[int] = []
    low_conf_objects = 0
    total_objects = 0

    for f in files:
        d = load_json(f)
        dets = d.get("detections", [])
        persons = [x for x in dets if int(x["cls"]) == 0]
        objects = [x for x in dets if int(x["cls"]) != 0]
        person_counts.append(len(persons))
        object_counts.append(len(objects))
        if persons: imgs_with_person += 1
        if objects: imgs_with_object += 1
        for o in objects:
            total_objects += 1
            if float(o.get("conf", 0.0)) < args.conf_thresh: low_conf_objects += 1
            cid = int(o["cls"])
            label = COCO_NAMES[cid] if 0 <= cid < len(COCO_NAMES) else f"class_{cid}"
            obj_counter[label] += 1

    print(f"Images: {n_imgs}")
    if n_imgs:
        print(f"% images with ≥1 person: {imgs_with_person/n_imgs*100:.1f}%")
        print(f"% images with ≥1 non-person object: {imgs_with_object/n_imgs*100:.1f}%")
        print(f"Avg persons/image: {sum(person_counts)/n_imgs:.2f}")
        print(f"Avg objects/image: {sum(object_counts)/n_imgs:.2f}")
    print(f"{low_conf_objects}/{max(total_objects,1)} ({(low_conf_objects/max(total_objects,1))*100:.1f}%) objects have conf < {args.conf_thresh}")
    print("\nTop 20 object classes:")
    for k, v in obj_counter.most_common(20):
        print(f"{k:15s} {v}")

# ---------- Subcommand: preview (optional) ----------
def cmd_preview(args):
    try:
        import cv2, matplotlib.pyplot as plt
    except Exception as e:
        print("preview requires opencv-python and matplotlib installed.", file=sys.stderr)
        sys.exit(2)

    pairs_dir = Path(args.pairs_dir)
    img_root = Path(args.img_root)
    files = list(pairs_dir.glob("*.json"))
    if not files:
        print("No pair JSONs found.")
        return
    sample = random.sample(files, min(args.max_images, len(files)))

    def draw_box(img, box, color, text=None):
        x1,y1,x2,y2 = map(int, box)
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        if text:
            cv2.putText(img, text, (x1, max(0, y1-6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    for jf in sample:
        d = load_json(jf)
        img_path = img_root / d["image"]
        img = cv2.imread(str(img_path))
        if img is None:
            print("Missing image:", img_path)
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for inter in d.get("interactions", [])[:10]:
            draw_box(img, inter["human_box"], (255,0,0), "person")
            lbl = inter.get("object_label", f"cls_{inter.get('object_cls', -1)}")
            draw_box(img, inter["object_box"], (0,255,0), lbl)
        plt.figure(figsize=(7,7))
        plt.title(f"{d['image']} | pairs={len(d.get('interactions', []))}")
        plt.imshow(img); plt.axis("off"); plt.show()

# ---------- CLI ----------
def build_parser():
    p = argparse.ArgumentParser(prog="photo_analysis.py", description="Human–Object Interaction pipeline (no Colab).")
    p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = p.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("detect", help="Run YOLOv8 detection over images")
    pd.add_argument("--img_dir", required=True)
    pd.add_argument("--out_dir", required=True)
    pd.add_argument("--model", default="yolov8x.pt")
    pd.add_argument("--imgsz", type=int, default=640)
    pd.add_argument("--conf", type=float, default=0.25)
    pd.add_argument("--iou", type=float, default=0.45)
    pd.add_argument("--device", default=None, help="cuda:0 or cpu (default: auto)")
    pd.set_defaults(func=cmd_detect)

    pp = sub.add_parser("parse", help="Parse detection JSONs into humans/objects")
    pp.add_argument("--in_dir", required=True)
    pp.add_argument("--out_dir", required=True)
    pp.set_defaults(func=cmd_parse)

    pr = sub.add_parser("pair", help="Build relaxed human–object pairs")
    pr.add_argument("--in_dir", required=True)
    pr.add_argument("--out_dir", required=True)
    pr.add_argument("--obj_conf_min", type=float, default=0.15)
    pr.add_argument("--topk", type=int, default=10)
    pr.set_defaults(func=cmd_pair)

    pa = sub.add_parser("actions", help="Infer HOI actions from pairs")
    pa.add_argument("--in_dir", required=True)
    pa.add_argument("--out_dir", required=True)
    pa.set_defaults(func=cmd_actions)

    ps = sub.add_parser("stats", help="Basic dataset stats from detection JSONs")
    ps.add_argument("--det_dir", required=True)
    ps.add_argument("--conf_thresh", type=float, default=0.25)
    ps.set_defaults(func=cmd_stats)

    pv = sub.add_parser("preview", help="(Optional) Visualize pairs on images [needs opencv + matplotlib]")
    pv.add_argument("--pairs_dir", required=True)
    pv.add_argument("--img_root", required=True)
    pv.add_argument("--max_images", type=int, default=5)
    pv.set_defaults(func=cmd_preview)
    return p

def main(argv=None):
    argv = argv or sys.argv[1:]
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=(logging.DEBUG if args.verbose else logging.INFO), format="[%(levelname)s] %(message)s")
    return args.func(args)

if __name__ == "__main__":
    raise SystemExit(main())
