"""Pydantic models — these define the JSON contract between the backend and
the AR helmet client."""
from typing import List, Optional
from pydantic import BaseModel, Field


class Detection(BaseModel):
    class_id: int = Field(..., description="0=personnel 1=two_wheeler 2=light_vehicle "
                                             "3=heavy_vehicle -1=moving_object (coherent motion "
                                             "the trained classes didn't recognize, e.g. a UAV)")
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    # Normalised xyxy so the AR client can scale to its own viewport
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: Optional[int] = Field(None, description="Stable across frames when tracking")
    moving: Optional[bool] = Field(None, description="True if centroid displaced beyond threshold")


class DetectionResponse(BaseModel):
    source_id: str
    frame_width: int
    frame_height: int
    inference_ms: float
    detections: List[Detection]


class TrainRequest(BaseModel):
    model: str = "yolo26s.pt"
    data: str = "data/battlesight.yaml"
    epochs: int = Field(60, ge=1, le=1000)
    imgsz: int = Field(640, ge=320, le=1536)
    batch: int = Field(8, ge=1, le=64)
    name: str = "battlesight_v2"


class TrainJob(BaseModel):
    job_id: str
    status: str               # queued | running | completed | failed | cancelled
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    run_name: str
    save_dir: Optional[str] = Field(
        None, description="Actual output dir; ultralytics increments a reused name"
    )
    current_epoch: Optional[int] = None
    total_epochs: int
    log_tail: List[str] = []
    best_weights: Optional[str] = None
