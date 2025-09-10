Photo Analysis: Human–Object Interaction
   This project explores how people interact with objects in images.
   It combines object detection with simple rule-based reasoning to go beyond just “what’s in the picture” and ask “what is happening between people and objects?”

Why?
   Most computer vision pipelines stop at detecting and labeling objects:
   “Here’s a person, here’s a chair, here’s a phone.”

But in real life, the interesting part is the interaction:
   1. Is the person holding the phone?
   2. Are they sitting on the chair?
   3. Are they riding the bicycle?
   4. Are they looking at the laptop?

   Those relational cues give us context that’s useful in areas like surveillance, retail analytics, healthcare monitoring, sports analysis, and even social media insights.

What the project does
   Detects objects and humans in images using YOLOv8. Parses and separates humans from non-human objects. Pairs humans with nearby objects using geometry and confidence rules. Infers likely actions (holding, sitting, riding, looking, wearing, carrying) with a rule-based model.

Outcomes
   JSON files that don’t just list objects, but also describe who interacts with what, and how. A lightweight framework that can be extended with ML classifiers for more advanced HOI recognition. A step toward making computer vision outputs more interpretable and human-centric.

Key Features
   Based on open-source YOLOv8 detection. No heavy training required — the action reasoning is rule-based. Works on any set of images, from everyday photos to specialized datasets. Modular design: detection → parsing → pairing → action inference.

Repo layout

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
