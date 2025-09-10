import argparse, logging, json, math
from pathlib import Path
from tqdm import tqdm
from photo_analysis.utils import iou, centroid, l2

HOLDABLE = {
    "cup","bottle","cell phone","wine glass","bowl","fork","knife","spoon","banana","apple",
    "book","remote","mouse","keyboard","tennis racket","baseball bat","baseball glove",
    "sports ball","skateboard","umbrella","teddy bear","scissors","vase"
}
SCREENLIKE = {"laptop","tv","cell phone","book","clock","microwave","oven","refrigerator"}
SEATS = {"chair","couch","bed","bench","dining table","toilet"}
RIDABLE = {"bicycle","motorcycle","horse","skateboard","surfboard","snowboard","boat"}

def area(b):     return max(0.0, (b[2]-b[0])) * max(0.0, (b[3]-b[1]))

def zones(h):
    x1,y1,x2,y2 = h
    w = max(1.0, x2-x1); hgt = max(1.0, y2-y1)
    head = [x1 + 0.20*w, y1,            x2 - 0.20*w, y1 + 0.30*hgt]
    hands_left  = [x1,          y1 + 0.30*hgt, x1 + 0.25*w, y1 + 0.70*hgt]
    hands_right = [x2 - 0.25*w, y1 + 0.30*hgt, x2,          y1 + 0.70*hgt]
    lap  = [x1 + 0.20*w, y1 + 0.55*hgt, x2 - 0.20*w, y2]
    return head, hands_left, hands_right, lap

def overlaps(a,b,thr=0.05): return iou(a,b) >= thr

def action_rules(human_box, obj_box, obj_label):
    actions = []
    hc = centroid(human_box); oc = centroid(obj_box)
    dist = l2(hc, oc)
    head, lh, rh, lap = zones(human_box)
    aH, aO = area(human_box), area(obj_box)
    rel_area = aO / (aH + 1e-9)
    ov = iou(human_box, obj_box)

    # holding/carrying
    if obj_label in HOLDABLE:
        near_hands = overlaps(obj_box, lh, 0.01) or overlaps(obj_box, rh, 0.01)
        inside = ov > 0.02
        if near_hands or inside:
            score = 0.6 + min(0.3, 0.6 * rel_area)
            actions.append({"label":"holding","score":round(score,3)})

    if obj_label in SEATS:
        if overlaps(obj_box, lap, 0.02) or (oc[1] > hc[1] and ov > 0.01):
            actions.append({"label":"sitting_on","score":0.75})

    if obj_label in RIDABLE:
        if oc[1] > hc[1] and ov > 0.01:
            actions.append({"label":"riding","score":0.75})

    if obj_label in SCREENLIKE:
        if overlaps(obj_box, head, 0.01) or dist < 0.35 * math.sqrt(aH):
            actions.append({"label":"looking_at","score":0.6})

    if obj_label == "backpack" and ov > 0.02:
        actions.append({"label":"wearing","score":0.7})
    if obj_label == "tie" and ov > 0.02:
        actions.append({"label":"wearing","score":0.7})
    if obj_label == "handbag" and (overlaps(obj_box, lh, 0.01) or overlaps(obj_box, rh, 0.01)):
        actions.append({"label":"carrying","score":0.65})

    return actions

def main():
    ap = argparse.ArgumentParser(description="Infer HOI actions from pairs_json_relaxed.")
    ap.add_argument("--in_dir", required=True, help="Input: pairs_json_relaxed")
    ap.add_argument("--out_dir", required=True, help="Output: hoi_actions_json")
    args = ap.parse_args()

    in_dir = Path(args.in_dir); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.json"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("Found %d pair files", len(files))

    for jf in tqdm(files, desc="Actions"):
        d = json.loads(jf.read_text())
        inters = d.get("interactions", [])
        enriched = []
        for it in inters:
            actions = action_rules(it["human_box"], it["object_box"], it.get("object_label",""))
            if actions:
                e = dict(it); e["actions"] = actions
                enriched.append(e)

        out = {
            "image": d.get("image"),
            "interactions": enriched,
            "counts": {
                "humans": d.get("counts",{}).get("humans", 0),
                "objects": d.get("counts",{}).get("objects", 0),
                "pairs": len(enriched)
            },
            "meta": {"source": jf.name, "method":"rule_based_v1"}
        }
        (out_dir / jf.name).write_text(json.dumps(out))

if __name__ == "__main__":
    main()
