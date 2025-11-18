# api/odds.py
from __future__ import annotations

import logging
from typing import List, Dict, Any

import httpx
from fastapi import HTTPException

from .settings import settings

logger = logging.getLogger("uvicorn.error")

SPORT = "americanfootball_nfl"
REGION = "us"
MARKETS = "spreads"
ODDS_API_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"


async def fetch_draftkings_spreads() -> List[Dict[str, Any]]:
    """
    Fetch NFL spreads from The Odds API, filter to DraftKings,
    and normalize to a simple structure the frontend expects.
    """
    api_key = settings.odds_api_key
    if not api_key:
        logger.error("ODDS_API_KEY is not configured in .env")
        raise HTTPException(status_code=500, detail="ODDS_API_KEY is not configured")

    params = {
        "apiKey": api_key,
        "regions": REGION,
        "markets": MARKETS,
        "oddsFormat": "decimal",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(ODDS_API_URL, params=params)
    except httpx.RequestError as exc:
        logger.exception(f"Error calling odds API: {exc}")
        raise HTTPException(status_code=502, detail="Failed to reach odds provider")

    if resp.status_code != 200:
        # log the body so we can see the real error (rate limit, bad key, etc.)
        logger.error(
            "Odds API error %s: %s",
            resp.status_code,
            resp.text[:500],
        )
        raise HTTPException(
            status_code=502,
            detail=f"Odds API error {resp.status_code}",
        )

    data = resp.json()

    dk_lines: List[Dict[str, Any]] = []
    for game in data:
        # Each game has a list of bookmakers; we only want DraftKings
        for book in game.get("bookmakers", []):
            if book.get("key") != "draftkings":
                continue

            markets = book.get("markets", [])
            if not markets:
                continue

            # spreads market is the first (only) one we requested
            spreads = markets[0].get("outcomes", [])
            dk_lines.append(
                {
                    "home_team": game.get("home_team"),
                    "away_team": game.get("away_team"),
                    "commence_time": game.get("commence_time"),
                    "spread": [
                        {
                            "team": o.get("name"),
                            "line": o.get("point"),
                            "price": o.get("price"),
                        }
                        for o in spreads
                    ],
                }
            )

    logger.info("Fetched %d DraftKings spread games", len(dk_lines))
    return dk_lines
