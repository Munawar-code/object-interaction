import argparse, logging, json
from pathlib import Path
from tqdm import tqdm
from photo_analysis.utils import COCO_NAMES

def main():
    ap = argparse.ArgumentParser(description="Parse YOLO detection JSONs into humans/objects.")
    ap.add_argument("--in_dir", required=True, help="Input: hoi_json")
    ap.add_argument("--out_dir", required=True, help="Output: parsed_json")
    args = ap.parse_args()

    in_dir = Path(args.in_dir); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.json"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("Found %d detection files", len(files))

    for jf in tqdm(files, desc="Parsing"):
        data = json.loads(jf.read_text())
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

        parsed = {
            "image": data.get("image"),
            "humans": humans,
            "objects": objects,
            "meta": {"source_json": jf.name}
        }
        (out_dir / jf.name).write_text(json.dumps(parsed))

if __name__ == "__main__":
    main()
