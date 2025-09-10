from pydantic import BaseModel, Field
from typing import List, Optional

class Detection(BaseModel):
    cls: int
    conf: float
    box: List[float]

class DetectionFile(BaseModel):
    image: str
    detections: List[Detection]

class ParsedItem(BaseModel):
    box: List[float]
    conf: float
    cls: int
    label: str

class ParsedFile(BaseModel):
    image: str
    humans: List[ParsedItem] = Field(default_factory=list)
    objects: List[ParsedItem] = Field(default_factory=list)
    meta: dict = Field(default_factory=dict)

class Pair(BaseModel):
    human_idx: int
    object_idx: int
    human_box: List[float]
    object_box: List[float]
    object_label: str
    object_cls: int
    scores: dict
    union_box: List[float]

class PairFile(BaseModel):
    image: str
    interactions: List[Pair] = Field(default_factory=list)
    counts: dict
    meta: dict

class Action(BaseModel):
    label: str
    score: float

class ActionPair(Pair):
    actions: List[Action]

class ActionFile(BaseModel):
    image: str
    interactions: List[ActionPair] = Field(default_factory=list)
    counts: dict
    meta: dict
