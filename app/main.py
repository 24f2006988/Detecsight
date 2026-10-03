from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app import config
from app.detector import detector
from app.exclusion import exclusion_store
from app.routers import detect, exclude, track, train
from app.trainer import trainer

STATIC_DIR = Path(__file__).resolve().parent / "static"

# aiortc is optional. If it isn't installed the server still starts and
# /ws/track/{source_id} works, only /webrtc/offer is missing.
try:
    from app.routers import webrtc as webrtc_router
    WEBRTC_AVAILABLE = True
except ImportError as e:
    webrtc_router = None
    WEBRTC_AVAILABLE = False
    print(f"[startup] WebRTC transport unavailable ({e!r}); install `aiortc` "
          f"to enable /webrtc/offer/{{source_id}}. /ws/track/{{source_id}} is unaffected.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    detector.load()
    yield
    if WEBRTC_AVAILABLE:
        await webrtc_router.close_all()
    detector.unload()


app = FastAPI(
    title="DetecSight",
    description="Person and vehicle detection and tracking for drone and helmet-camera video.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router)
app.include_router(track.router)
# training and exclusion change server state, so a public demo leaves them out
if not config.PUBLIC_DEMO:
    app.include_router(train.router)
    app.include_router(exclude.router)
if WEBRTC_AVAILABLE:
    app.include_router(webrtc_router.router)


@app.get("/", include_in_schema=False)
async def demo_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["system"])
async def health():
    return {
        "status": "ok",
        "model_loaded": detector.model is not None,
        "model_path": config.MODEL_PATH,
        # "ground" is always there. "drone" only shows up if a separate
        # drone checkpoint actually loaded, see Detector.load().
        "views_loaded": ["ground"] + list(detector._view_models.keys()),
        "device": config.DEVICE,
        "classes": config.CLASS_NAMES,
        "training_active": trainer.is_training(),
        "tracked_per_feed": detector.stats(),
        "exclusions": exclusion_store.list(),
        "webrtc_available": WEBRTC_AVAILABLE,
        "public_demo": config.PUBLIC_DEMO,
    }
