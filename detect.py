import argparse, logging, json
from pathlib import Path
from ultralytics import YOLO
import torch
from tqdm import tqdm
from photo_analysis.utils import list_images, save_json

def main():
    ap = argparse.ArgumentParser(description="Run YOLOv8 detection over a folder of images.")
    ap.add_argument("--img_dir", required=True, help="Folder with input images")
    ap.add_argument("--out_dir", required=True, help="Output folder for hoi_json")
    ap.add_argument("--model", default="yolov8x.pt", help="YOLO model weights")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--device", default=None, help="e.g., cuda:0 or cpu (default: auto)")
    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("Using device: %s", device)

    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    images = list(list_images(img_dir))
    logging.info("Found %d images under %s", len(images), img_dir)

    # You can also do: results = model.predict(source=str(img_dir), ...)
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

if __name__ == "__main__":
    main()
