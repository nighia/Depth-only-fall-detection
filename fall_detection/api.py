
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional, Literal

app = FastAPI(title='Depth-only Fall Detection Research API', version='0.1.0')

class EventPayload(BaseModel):
    event_id: str
    camera_id: str
    track_id: int
    start_time: str
    end_time: str
    stage_a: Literal['normal','hard_negative_motion','fall']
    stage_a_confidence: float = Field(ge=0, le=1)
    stage_b: Optional[Literal['slip_fall','stair_fall','unknown_fall']] = None
    stage_b_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    scene_context: Literal['flat_ground','stairs','unknown'] = 'unknown'
    model_version: str = 'research-baseline-0.1'

@app.get('/health')
def health():
    return {'status':'ok','model_loaded':False,'pipeline':'depth-only-fall-detection'}

@app.post('/events')
def receive_event(event: EventPayload):
    # Persistence, authentication and audit logging are not implemented in this scaffold.
    return {'accepted':True,'event':event.model_dump()}
