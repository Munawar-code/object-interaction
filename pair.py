import argparse, logging, json, math
from pathlib import Path
from tqdm import tqdm
from photo_analysis.utils import centroid, l2, iou, union_box

def main():
    ap = argparse.ArgumentParser(description="Build relaxed human–object pairs from parsed_json.")
    ap.add_argument("--in_dir", required=True, help="Input: parsed_json")
    ap.add_argument("--out_dir", required=True, help="Output: pairs_json_relaxed")
    ap.add_argument("--obj_conf_min", type=float, default=0.15)
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()

    in_dir = Path(args.in_dir); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.json"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("Found %d parsed files", len(files))

    for jf in tqdm(files, desc="Pairing"):
        data = json.loads(jf.read_text())
        humans  = data.get("humans", [])
        objects = [o for o in data.get("objects", []) if float(o.get("conf", 0.0)) >= args.obj_conf_min]

        interactions = []
        for hi, h in enumerate(humans):
            hc = centroid(h["box"])
            ranked = []
            for oi, o in enumerate(objects):
                oc = centroid(o["box"])
                dist = l2(hc, oc)
                ranked.append((dist, oi, o))
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

        out = {
            "image": data.get("image"),
            "interactions": interactions,
            "counts": {"humans": len(humans), "objects": len(objects), "pairs": len(interactions)},
            "meta": {"source": data.get("meta", {}).get("source_json", jf.name)}
        }
        (out_dir / jf.name).write_text(json.dumps(out))

if __name__ == "__main__":
    main()
