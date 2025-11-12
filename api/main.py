# api/main.py
from __future__ import annotations

import time
import logging
from typing import Dict, Tuple, List

import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

# local imports
from model.predict_week import predict_week
from .settings import settings
from .db import init_db, save_predictions, list_predictions, has_week

# ---------------- App & CORS ----------------
app = FastAPI(title="NFL Model API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Logging ----------------
logger = logging.getLogger("uvicorn.error")

@app.middleware("http")
async def timing_middleware(request, call_next):
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"{request.method} {request.url.path} -> {getattr(response, 'status_code', 0)} [{dt_ms:.1f} ms]")

# ---------------- Startup ----------------
@app.on_event("startup")
def _startup():
    # initialize DB (creates tables if needed)
    init_db()
    logger.info("DB initialized. Service is ready.")

# ---------------- Cache ----------------
# key: (season, week) -> (timestamp, payload_json_list)
_cache: Dict[Tuple[int, int], Tuple[float, List[dict]]] = {}

# ---------------- Helper ----------------
async def _run_predict(season: int, week: int) -> pd.DataFrame:
    """Run the synchronous model in a thread so the server stays responsive."""
    return await run_in_threadpool(predict_week, season, week)

# ---------------- Endpoints ----------------
@app.get("/health")
def health():
    return {
        "ok": True,
        "cache_size": len(_cache),
        "cache_ttl": settings.cache_ttl_seconds,
        "defaults": {"season": settings.default_season, "week": settings.default_week},
    }

@app.get("/predict")
async def predict(
    season: int = Query(settings.default_season, ge=1999),
    week:   int = Query(settings.default_week,   ge=1, le=22),
    force_snapshot: bool = Query(False, description="If true, save to DB even on cache hit"),
):
    key = (season, week)
    now = time.time()

    # Serve from cache if fresh
    hit = _cache.get(key)
    if hit and (now - hit[0] < settings.cache_ttl_seconds):
        payload = hit[1]
        # NEW: save to DB if forced OR week not yet in DB
        if force_snapshot or not has_week(season, week):
            try:
                n = save_predictions(season, week, payload)
                logger.info(f"/predict cache=HIT -> snapshot rows={n} season={season} week={week}")
            except Exception:
                logger.exception("auto snapshot on HIT failed (non-fatal)")
        else:
            logger.info(f"/predict cache=HIT key={key} (already in DB)")
        return payload

    logger.info(f"/predict cache=MISS key={key}")

    # Compute fresh
    try:
        df = await _run_predict(season, week)
    except Exception:
        logger.exception("predict failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

    payload = df.to_dict(orient="records")
    _cache[key] = (now, payload)

    # Save on miss (same as before)
    try:
        n = save_predictions(season, week, payload)
        logger.info(f"/predict cache=MISS -> snapshot rows={n} season={season} week={week}")
    except Exception:
        logger.exception("auto snapshot on MISS failed (non-fatal)")

    return payload

@app.post("/predict/snapshot")
async def snapshot(
    season: int = Query(settings.default_season, ge=1999),
    week:   int = Query(settings.default_week,   ge=1, le=22),
):
    """Compute current predictions and persist them to the DB."""
    try:
        df = await _run_predict(season, week)
        rows = df.to_dict(orient="records")
        n = save_predictions(season, week, rows)
        return {"ok": True, "saved": n, "season": season, "week": week}
    except Exception:
        logger.exception("snapshot failed")
        raise HTTPException(status_code=500, detail="Snapshot failed")

@app.get("/history")
def history(
    season: int | None = Query(None, ge=1999),
    week:   int | None = Query(None, ge=1, le=22),
    limit:  int = Query(100, ge=1, le=1000),
):
    """Return saved snapshots (flattened payloads)."""
    try:
        return list_predictions(season, week, limit)
    except Exception:
        logger.exception("history failed")
        raise HTTPException(status_code=500, detail="History fetch failed")
