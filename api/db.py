# api/db.py
from __future__ import annotations

import os
import json
import datetime as dt
from typing import List, Dict, Any, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from .settings import settings

# ---------- DB URL (absolute path for SQLite) ----------
DEFAULT_SQLITE_PATH = os.path.abspath(os.getenv("SQLITE_FILE", "nfl.db"))

DB_URL = (
    settings.database_url
    or os.getenv("DATABASE_URL")
    or f"sqlite:///{DEFAULT_SQLITE_PATH}"
)

engine = create_engine(DB_URL, future=True)
DIALECT = engine.url.get_backend_name()

# ---------- Schema statements (execute separately) ----------
if DIALECT == "sqlite":
    DDL_TABLE = """
    CREATE TABLE IF NOT EXISTS model_predictions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      season INTEGER NOT NULL,
      week   INTEGER NOT NULL,
      game_id TEXT NOT NULL,
      payload TEXT NOT NULL,
      created_at TIMESTAMP NOT NULL
    );
    """
    # SQLite-only dedupe using rowid
    DEDUP_SQL = """
    DELETE FROM model_predictions
    WHERE rowid NOT IN (
      SELECT MAX(rowid)
      FROM model_predictions
      GROUP BY season, week, game_id
    );
    """
else:
    # Postgres / others (Supabase)
    DDL_TABLE = """
    CREATE TABLE IF NOT EXISTS model_predictions (
      id BIGSERIAL PRIMARY KEY,
      season INTEGER NOT NULL,
      week   INTEGER NOT NULL,
      game_id TEXT NOT NULL,
      payload TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL
    );
    """
    # We don't need a rowid-style dedupe in Postgres for now
    DEDUP_SQL = None

DDL_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS ux_model_predictions_unique
  ON model_predictions (season, week, game_id);
"""

# ---------- Lifecycle ----------
def init_db() -> str:
    """
    Initialize DB schema.
    - Creates table.
    - Creates unique index (in SQLite we dedupe first if needed).
    Returns the DB URL for logging.
    """
    with engine.begin() as conn:
        conn.exec_driver_sql(DDL_TABLE)
        try:
            conn.exec_driver_sql(DDL_INDEX)
        except SQLAlchemyError:
            # This path is only for SQLite duplicates; skip for Postgres
            if DIALECT == "sqlite" and DEDUP_SQL is not None:
                conn.exec_driver_sql(DEDUP_SQL)
                conn.exec_driver_sql(DDL_INDEX)
            else:
                raise
    return DB_URL

# ---------- Writes ----------
def save_predictions(season: int, week: int, rows: List[Dict[str, Any]]) -> int:
    """
    Upsert each game's payload for (season, week).
    Also stores pred_margin and home_win_prob in dedicated columns
    for easier querying/inspection in the DB.
    """
    if not rows:
        return 0

    now = dt.datetime.utcnow()
    params = []
    for r in rows:
        params.append(
            {
                "season": season,
                "week": week,
                "game_id": str(r.get("game_id")),
                "payload": json.dumps(r),
                "ts": now,
                "pred_margin": float(r.get("pred_margin")) if r.get("pred_margin") is not None else None,
                "home_win_prob": float(r.get("home_win_prob")) if r.get("home_win_prob") is not None else None,
            }
        )

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO model_predictions (
                    season,
                    week,
                    game_id,
                    payload,
                    created_at,
                    pred_margin,
                    home_win_prob
                )
                VALUES (
                    :season,
                    :week,
                    :game_id,
                    :payload,
                    :ts,
                    :pred_margin,
                    :home_win_prob
                )
                ON CONFLICT(season, week, game_id) DO UPDATE SET
                  payload       = excluded.payload,
                  created_at    = excluded.created_at,
                  pred_margin   = excluded.pred_margin,
                  home_win_prob = excluded.home_win_prob
                """
            ),
            params,
        )
    return len(rows)

# ---------- Reads ----------
def list_predictions(
    season: Optional[int] = None,
    week: Optional[int] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Return flattened saved rows (payload merged with basic fields).
    Most-recent first.
    """
    q = "SELECT id, season, week, game_id, payload, created_at FROM model_predictions"
    where, p = [], {}
    if season is not None:
        where.append("season = :season")
        p["season"] = season
    if week is not None:
        where.append("week = :week")
        p["week"] = week
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY created_at DESC, id DESC LIMIT :limit"
    p["limit"] = limit

    with engine.begin() as conn:
        rows = conn.execute(text(q), p).mappings().all()

    out: List[Dict[str, Any]] = []
    for r in rows:
        payload = json.loads(r["payload"])
        out.append(
            {
                "id": r["id"],
                "season": r["season"],
                "week": r["week"],
                "game_id": r["game_id"],
                "created_at": (
                    r["created_at"].isoformat()
                    if hasattr(r["created_at"], "isoformat")
                    else str(r["created_at"])
                ),
                **payload,
            }
        )
    return out

def count_predictions() -> int:
    with engine.begin() as conn:
        n = conn.execute(text("SELECT COUNT(*) AS n FROM model_predictions")).scalar_one()
    return int(n)

def dedupe_keep_newest() -> int:
    if DIALECT != "sqlite" or DEDUP_SQL is None:
        return 0
    with engine.begin() as conn:
        res = conn.execute(text(DEDUP_SQL))
        return res.rowcount or 0
    clear

def has_week(season: int, week: int) -> bool:
    """Return True if there is at least one row for (season, week)."""
    with engine.begin() as conn:
        res = conn.execute(
            text("SELECT 1 FROM model_predictions WHERE season=:season AND week=:week LIMIT 1"),
            {"season": season, "week": week},
        ).first()
    return res is not None