from pathlib import Path
import json, math
from typing import List, Dict, Any, Iterable

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
