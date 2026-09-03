import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.exclusion import exclusion_store

router = APIRouter(prefix="/exclude", tags=["exclusion"])


@router.post("")
async def add_exclusion(
    name: str,
    file: UploadFile = File(..., description="Reference image of the object to stop detecting"),
):
    """Upload a reference image; any future detection whose crop looks like it
    gets dropped from results, regardless of what class it would otherwise be."""
    raw = await file.read()
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image: is it a valid JPEG/PNG?")
    exclusion_store.add(name, frame)
    return {"added": name, "total": len(exclusion_store.list())}


@router.get("")
async def list_exclusions():
    return {"exclusions": exclusion_store.list()}


@router.delete("/{name}")
async def remove_exclusion(name: str):
    if not exclusion_store.remove(name):
        raise HTTPException(404, f"No exclusion named '{name}'")
    return {"removed": name}
