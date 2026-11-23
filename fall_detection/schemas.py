from dataclasses import dataclass
from typing import Optional

STAGE_A_CLASSES = ['normal', 'hard_negative_motion', 'fall']
STAGE_B_CLASSES = ['slip_fall', 'stair_fall', 'unknown_fall']

@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int
    area: int
    confidence: float = 0.0

@dataclass
class EventRecord:
    event_id: str
    camera_id: str
    track_id: int
    start_time: str
    end_time: str
    stage_a: str
    stage_a_confidence: float
    stage_b: Optional[str] = None
    stage_b_confidence: Optional[float] = None
    scene_context: str = 'unknown'
    model_version: str = 'research-baseline-0.1'
