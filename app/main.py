import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.core.config import settings
from app.core.logging import configure_logging
from app.routes import analyse, detect

configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

_DEMO_DIR = Path(__file__).parent.parent / "demo" / "map"


# ── No-cache middleware for demo static files ─────────────────────────────────
class NoCacheDemoMiddleware(BaseHTTPMiddleware):
    """Prevent browsers from caching demo static assets (JS/CSS/HTML)."""

    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        if request.url.path.startswith("/demo"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Palimpsest API — loading ChangeDetector ...")
    from app.services.detector import ChangeDetector

    app.state.detector = ChangeDetector()
    logger.info("ChangeDetector ready on %s", app.state.detector.device)
    yield
    logger.info("Palimpsest API shutting down")


app = FastAPI(title="Palimpsest", version="0.1.0", lifespan=lifespan)
app.add_middleware(NoCacheDemoMiddleware)
app.include_router(detect.router,  prefix="/api/v1")
app.include_router(analyse.router, prefix="/api/v1")

# Serve the interactive map demo at /demo
if _DEMO_DIR.exists():
    app.mount("/demo", StaticFiles(directory=str(_DEMO_DIR), html=True), name="demo")


@app.get("/health")
def health(request: Request):
    detector = getattr(request.app.state, "detector", None)
    return {
        "status":       "ok",
        "model_loaded": detector is not None,
        "model_version": detector.MODEL_VERSION if detector else None,
    }
