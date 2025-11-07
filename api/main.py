# api/main.py
import time
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool   # <-- NEW
from model.predict_week import predict_week
from .settings import settings                     # <-- NEW

app = FastAPI(title="NFL Model API", version="0.1.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- SIMPLE CACHE ----------
_cache: dict[tuple[int, int], tuple[float, list[dict]]] = {}   # <-- NEW cache storage

# ---------- HELPER ----------
async def _run_predict(season: int, week: int) -> pd.DataFrame:  # <-- NEW helper
    # Runs your CPU-bound model in a thread so FastAPI stays async
    return await run_in_threadpool(predict_week, season, week)

# ---------- HEALTH ENDPOINT ----------
@app.get("/health")
def health():
    return {"ok": True, "cache_size": len(_cache), "cache_ttl": settings.cache_ttl_seconds}

# ---------- PREDICT ENDPOINT ----------
@app.get("/predict")          # <--- replace your old /predict route with this
async def predict(
    season: int = Query(settings.default_season, ge=1999),
    week: int   = Query(settings.default_week, ge=1, le=22),
):
    key = (season, week)
    now = time.time()

    # serve from cache if fresh
    hit = _cache.get(key)
    if hit and now - hit[0] < settings.cache_ttl_seconds:
        return hit[1]

    # compute fresh
    try:
        df = await _run_predict(season, week)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    payload = df.to_dict(orient="records")
    _cache[key] = (now, payload)
    return payload
