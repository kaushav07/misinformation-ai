"""
FastAPI application entry point.
All routers registered here with versioned prefix /api/v1.
Startup event initialises Qdrant collections.
"""
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.routers import analyze, fact_check, counter, multimodal, trending, segmentation
from app.config import settings


# ─── Startup / shutdown ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Misinformation Detector API…")
    try:
        from app.services.qdrant_service import init_all_collections
        init_all_collections()
        logger.info("✅ Qdrant collections ready")
    except Exception as e:
        logger.warning(f"Qdrant init skipped: {e}")
    yield
    logger.info("🛑 Shutting down…")


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Against Misinformation",
    description="""
## Real-time multimodal, multilingual misinformation detection & countering platform.

### Features
- **Text / Image / Audio / Video** analysis
- **Hindi & Tamil** language support
- **Semantic claim deduplication** via Qdrant vector DB
- **Active counter-narrative** generation (Claude-powered)
- **Trending misinformation tracker**
- **Virality prediction** (unique feature)
- **Explainability** — LIME-style feature attribution
- **Special track**: Offroad semantic segmentation (Duality AI)
    """,
    version="2.0.0",
    lifespan=lifespan,
)

# ─── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Request timing middleware ────────────────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(round((time.time() - start) * 1000, 2))
    return response


# ─── Routers ─────────────────────────────────────────────────────────────────
PREFIX = "/api/v1"

app.include_router(analyze.router,       prefix=PREFIX, tags=["Analysis"])
app.include_router(fact_check.router,    prefix=PREFIX, tags=["Fact Check"])
app.include_router(counter.router,       prefix=PREFIX, tags=["Counter Narrative"])
app.include_router(multimodal.router,    prefix=PREFIX, tags=["Multimodal"])
app.include_router(trending.router,      prefix=PREFIX, tags=["Trending"])
app.include_router(segmentation.router,  prefix=PREFIX, tags=["Segmentation (Special Track)"])


# ─── Root & health ────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    return {
        "message": "AI Against Misinformation API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=dict, tags=["System"])
def health():
    from app.services.qdrant_service import _get_client
    qdrant_ok = False
    try:
        _get_client().get_collections()
        qdrant_ok = True
    except Exception:
        pass

    return {
        "status": "ok",
        "version": "2.0.0",
        "services": {
            "qdrant": "connected" if qdrant_ok else "unavailable",
            "nli_model": settings.nli_model,
            "embed_model": settings.embed_model,
            "claude": "configured" if (settings.anthropic_api_key and settings.anthropic_api_key != "your_anthropic_api_key_here") else "not configured (rule-based fallback active)",
            "google_fc": "configured" if (settings.google_fact_check_api_key and settings.google_fact_check_api_key != "your_google_fact_check_api_key_here") else "not configured",
        },
    }
