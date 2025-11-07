# api/db.py
import os, json, datetime as dt
from typing import List, Dict, Any
from sqlalchemy import create_engine, text

# SQLite now (file beside your code). Later you can set DATABASE_URL to Postgres.
DB_URL = os.getenv("DATABASE_URL", "sqlite:///./nfl.db")
engine = create_engine(DB_URL, future=True)

DDL = """
CREATE TABLE IF NOT EXISTS model_predictions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  season INTEGER NOT NULL,
  week   INTEGER NOT NULL,
  game_id TEXT NOT NULL,
  payload TEXT NOT NULL,              -- JSON as text (portable)
  created_at TIMESTAMP NOT NULL
);
"""

def init_db():
    with engine.begin() as conn:
        conn.execute(text(DDL))

def save_predictions(season:int, week:int, rows:List[Dict[str, Any]]) -> int:
    now = dt.datetime.utcnow()
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO model_predictions (season, week, game_id, payload, created_at) VALUES (:season,:week,:game_id,:payload,:ts)"),
            [
                {
                    "season": season,
                    "week": week,
                    "game_id": str(r.get("game_id")),
                    "payload": json.dumps(r),
                    "ts": now,
                }
                for r in rows
            ],
        )
    return len(rows)

def list_predictions(season:int|None=None, week:int|None=None, limit:int=100) -> list[dict]:
    q = "SELECT id, season, week, game_id, payload, created_at FROM model_predictions"
    where, params = [], {}
    if season is not None:
        where.append("season = :season")
        params["season"] = season
    if week is not None:
        where.append("week = :week")
        params["week"] = week
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY created_at DESC, id DESC LIMIT :limit"
    params["limit"] = limit
    with engine.begin() as conn:
        rows = conn.execute(text(q), params).mappings().all()
    # return parsed payloads
    out = []
    for r in rows:
        item = dict(r)
        item["payload"] = json.loads(item["payload"])
        return_item = {
            "id": item["id"],
            "season": item["season"],
            "week": item["week"],
            "game_id": item["game_id"],
            "created_at": item["created_at"].isoformat() if hasattr(item["created_at"], "isoformat") else str(item["created_at"]),
            **item["payload"],
        }
        out.append(return_item)
    return out
