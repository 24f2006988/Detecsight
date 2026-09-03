from fastapi import APIRouter, HTTPException

from app.schemas import TrainJob, TrainRequest
from app.trainer import trainer

router = APIRouter(prefix="/train", tags=["training"])


@router.post("", response_model=TrainJob, status_code=202)
async def start_training(req: TrainRequest):
    """Kick off a fine-tuning run as a background subprocess.
    Returns immediately with a job_id: poll /train/{job_id} for progress."""
    try:
        return trainer.start(req)
    except RuntimeError as e:
        raise HTTPException(409, str(e))


@router.get("", response_model=list[TrainJob])
async def list_training_jobs():
    return trainer.list_jobs()


@router.get("/{job_id}", response_model=TrainJob)
async def get_training_job(job_id: str):
    job = trainer.get(job_id)
    if job is None:
        raise HTTPException(404, f"No job {job_id}")
    return job


@router.post("/{job_id}/cancel")
async def cancel_training_job(job_id: str):
    if not trainer.cancel(job_id):
        raise HTTPException(409, "Job is not running")
    return {"cancelled": job_id}
