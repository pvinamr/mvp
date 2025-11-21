# upload_predictions.py
"""
Run your heavy model locally and upload a week's predictions to your live API.

Usage (from project root, venv active):

    python upload_predictions.py --season 2025 --week 8
"""

import argparse
import os
import sys
from typing import Any, Dict, List

import requests
from predict_week import predict_week  # your existing function


# Default to your Render URL; can be overridden with API_BASE_URL env var
API_BASE = os.environ.get("API_BASE_URL", "https://mvp-x60l.onrender.com")


def main(season: int, week: int) -> None:
    print(f"Running local model for {season} Week {week}...")
    df = predict_week(season, week)
    rows: List[Dict[str, Any]] = df.to_dict(orient="records")
    print(f"Got {len(rows)} games, uploading to {API_BASE}...")

    url = f"{API_BASE}/predict/snapshot"
    params = {"season": season, "week": week}

    try:
        resp = requests.post(url, params=params, json=rows, timeout=60)
    except Exception as exc:
        print("ERROR: Failed to POST snapshot:", exc, file=sys.stderr)
        sys.exit(1)

    if resp.status_code != 200:
        print("Server responded with error:", resp.status_code, resp.text, file=sys.stderr)
        sys.exit(1)

    print("Upload successful:", resp.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    args = parser.parse_args()

    main(args.season, args.week)
