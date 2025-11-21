# api/main.py
from __future__ import annotations

import time
import logging
from typing import Dict, List, Tuple, Any

from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .settings import settings
from .db import init_db, list_predictions, save_predictions
from .odds import fetch_draftkings_spreads

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="NFL Model API", version="0.3.0")

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Simple cache: (season, week) -> (timestamp, rows) ----
_cache: Dict[Tuple[int, int], Tuple[float, List[Dict[str, Any]]]] = {}


@app.on_event("startup")
def _startup() -> None:
    db_url = init_db()
    logger.info("DB initialized at %s", db_url)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/predict")
async def get_predictions(
    season: int = Query(settings.default_season, ge=1999),
    week: int = Query(settings.default_week, ge=1, le=22),
):
    """
    Return predictions for (season, week) *from the database only*.

    These rows must have been uploaded beforehand via POST /predict/snapshot
    (e.g. from your local machine using the heavy model).
    """
    key = (season, week)
    now = time.time()

    # cache check
    hit = _cache.get(key)
    if hit and now - hit[0] < settings.cache_ttl_seconds:
        logger.info("/predict cache=HIT key=%s rows=%d", key, len(hit[1]))
        return hit[1]

    # load from DB
    rows = list_predictions(season=season, week=week, limit=256)
    if not rows:
        logger.info("/predict cache=MISS key=%s but no rows in DB", key)
        raise HTTPException(
            status_code=404,
            detail="No predictions stored for that season/week yet.",
        )

    _cache[key] = (now, rows)
    logger.info("/predict cache=MISS key=%s rows=%d", key, len(rows))
    return rows


@app.post("/predict/snapshot")
async def save_snapshot(
    season: int = Query(..., ge=1999),
    week: int = Query(..., ge=1, le=22),
    games: List[Dict[str, Any]] = Body(..., description="List of per-game prediction rows"),
):
    """
    Upload a full week's predictions (from your local model) and store them in the DB.
    This is called from your local script, NOT from the frontend.
    """
    if not games:
        raise HTTPException(status_code=400, detail="Empty payload")

    n = save_predictions(season, week, games)
    _cache.pop((season, week), None)  # invalidate cache
    logger.info("/predict/snapshot saved rows=%d season=%d week=%d", n, season, week)
    return {"saved": n, "season": season, "week": week}


@app.get("/history")
async def history(
    season: int | None = Query(None),
    week: int | None = Query(None),
    limit: int = Query(500, ge=1, le=2000),
):
    """
    Return saved predictions from the DB (most recent first).
    Used by your History page.
    """
    rows = list_predictions(season=season, week=week, limit=limit)
    return rows


@app.get("/odds/draftkings")
async def draftkings_odds():
    """
    Proxy endpoint to fetch DraftKings spreads via odds API.
    """
    return await fetch_draftkings_spreads()
