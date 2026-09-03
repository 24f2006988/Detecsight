from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.detector import detector
from app.exclusion import exclusion_store
from app.routers import detect, exclude, track, train
from app.trainer import trainer

# aiortc is an optional dependency of the WebRTC transport (pip install
# aiortc). Guard the import so a server that hasn't installed it yet still
# starts with /ws/track/{source_id} fully working -- only /webrtc/offer is
# unavailable.
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
    # --- startup ---
    detector.load()
    yield
    # --- shutdown ---
    if WEBRTC_AVAILABLE:
        await webrtc_router.close_all()
    detector.unload()


app = FastAPI(
    title="BattleSight AR - Detection Service",
    description="Drone-view tactical target detection, tracking, and fine-tuning control.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten before anything leaves your laptop
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router)
app.include_router(track.router)
app.include_router(train.router)
app.include_router(exclude.router)
if WEBRTC_AVAILABLE:
    app.include_router(webrtc_router.router)


@app.get("/health", tags=["system"])
async def health():
    return {
        "status": "ok",
        "model_loaded": detector.model is not None,
        "model_path": config.MODEL_PATH,
        # "ground" always resolves (it's self.model); "drone" only appears
        # once weights/drone_best.pt (or BATTLESIGHT_DRONE_MODEL) actually
        # loaded -- see Detector.load(). Callers can check this before
        # requesting ?view=drone to know whether it's really a separate
        # checkpoint or a silent fallback to the default model.
        "views_loaded": ["ground"] + list(detector._view_models.keys()),
        "device": config.DEVICE,
        "classes": config.CLASS_NAMES,
        "training_active": trainer.is_training(),
        # tracked targets currently held per feed, to spot a runaway feed
        "tracked_per_feed": detector.stats(),
        "exclusions": exclusion_store.list(),
        "webrtc_available": WEBRTC_AVAILABLE,
    }
