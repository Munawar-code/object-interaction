# Photo Analysis (CLI, VS Code-Friendly)

End-to-end HOI (Human–Object Interaction) pipeline built around YOLOv8.
No Colab-specific code — runs locally via CLI scripts. Tested in VS Code.

## Install

```bash
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quickstart

Assume your images live in `./data/preproc` and outputs go to `./out`:

```bash
# 1) Detection -> hoi_json
python scripts/detect.py --img_dir ./data/preproc --out_dir ./out/hoi_json --model yolov8x.pt --imgsz 640 --conf 0.25

# 2) Parse -> parsed_json
python scripts/parse.py --in_dir ./out/hoi_json --out_dir ./out/parsed_json

# 3) Build pairs -> pairs_json_relaxed
python scripts/pair.py --in_dir ./out/parsed_json --out_dir ./out/pairs_json_relaxed --obj_conf_min 0.15 --topk 10

# 4) Infer actions -> hoi_actions_json
python scripts/actions.py --in_dir ./out/pairs_json_relaxed --out_dir ./out/hoi_actions_json
```

## Notes
- Auto-detects CPU/GPU; override with `--device cuda:0` or `--device cpu` if needed.
- JSON files include a `meta` section for reproducibility.
- For large folders, Ultralytics will internally batch predictions.

## Repo layout

```
photo-analysis/
├─ requirements.txt
├─ pyproject.toml
├─ README.md
├─ LICENSE
├─ .gitignore
├─ src/photo_analysis/
│  ├─ __init__.py
│  ├─ utils.py
│  └─ schemas.py
└─ scripts/
   ├─ detect.py
   ├─ parse.py
   ├─ pair.py
   └─ actions.py
```

## License
MIT
