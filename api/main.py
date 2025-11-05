# api/main.py
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from model.predict_week import predict_week  # imports your function from predict_week.py

app = FastAPI(title="NFL Model API", version="0.1.0")

# Dev CORS (ok locally; tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/predict")
def predict(
    season: int = Query(..., ge=1999),
    week: int = Query(..., ge=1, le=22),
):
    try:
        df: pd.DataFrame = predict_week(season, week)
        return df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
