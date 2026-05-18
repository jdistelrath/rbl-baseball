"""
api_draft_route.py — Flask Blueprint exposing the UD Draft endpoints.

Wires draft_engine.project_all_players_ml + build_draft_cheat_sheet into
the existing app.

To register in app.py, add these three lines:

    from api_draft_route import draft_bp
    # ... after `app = Flask(__name__)` ...
    app.register_blueprint(draft_bp)

Endpoints:
    GET /api/underdog-draft   — full slate projection + cheat sheet
    GET /api/draft-status     — model freshness / engine version info
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, jsonify

import draft_engine

draft_bp = Blueprint("draft", __name__, url_prefix="/api")

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
BATTER_MODEL = MODELS / "ud_batter_lgbm_latest.pkl"
PITCHER_MODEL = MODELS / "ud_pitcher_lgbm_latest.pkl"

_last_run = {"timestamp": None, "duration_ms": None, "player_count": 0, "ok": False}


def _file_mtime_iso(p: Path) -> str | None:
    if not p.exists():
        return None
    try:
        ts = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
        return ts.isoformat()
    except OSError:
        return None


def _load_slate():
    """
    Pull today's lineups + season stat frames using whatever data layer the
    rest of the app uses. Imports are deferred so a missing module doesn't
    break import-time blueprint registration.
    """
    try:
        from data_fetcher import get_today_schedule, get_confirmed_lineups
    except Exception:
        get_today_schedule = get_confirmed_lineups = None
    try:
        from data_fetcher import get_batter_statcast, get_pitcher_statcast
    except Exception:
        get_batter_statcast = get_pitcher_statcast = None

    schedule = get_today_schedule() if get_today_schedule else []

    # Merge lineups into each game dict — format draft_engine expects
    games = []
    for game in schedule:
        g = dict(game)
        try:
            g["lineups"] = get_confirmed_lineups(game["game_id"]) if get_confirmed_lineups else {}
        except Exception:
            g["lineups"] = {}
        games.append(g)

    batter_df = get_batter_statcast() if get_batter_statcast else None
    pitcher_df = get_pitcher_statcast() if get_pitcher_statcast else None
    return games, batter_df, pitcher_df


@draft_bp.route("/underdog-draft", methods=["GET"])
def underdog_draft():
    started = time.perf_counter()
    try:
        games, batter_df, pitcher_df = _load_slate()
        projections = draft_engine.project_all_players_ml(games, batter_df, pitcher_df)
        cheat_sheet = draft_engine.build_draft_cheat_sheet(projections)
    except Exception as exc:
        _last_run.update({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "player_count": 0,
            "ok": False,
            "error": str(exc),
        })
        return jsonify({"status": "error", "error": str(exc)}), 500

    duration_ms = int((time.perf_counter() - started) * 1000)
    _last_run.update({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": duration_ms,
        "player_count": len(projections),
        "ok": True,
        "error": None,
    })

    engine = "ml" if BATTER_MODEL.exists() and PITCHER_MODEL.exists() else "heuristic"
    return jsonify({
        "status": "ok",
        "engine": engine,
        "duration_ms": duration_ms,
        "player_count": len(projections),
        "projections": projections,
        "cheat_sheet": cheat_sheet,
    })


@draft_bp.route("/draft-status", methods=["GET"])
def draft_status():
    batter_mtime = _file_mtime_iso(BATTER_MODEL)
    pitcher_mtime = _file_mtime_iso(PITCHER_MODEL)
    ml_active = batter_mtime is not None and pitcher_mtime is not None

    return jsonify({
        "status": "ok",
        "engine_active": "ml" if ml_active else "heuristic",
        "models": {
            "batter": {
                "path": str(BATTER_MODEL),
                "exists": BATTER_MODEL.exists(),
                "trained_at": batter_mtime,
                "size_bytes": BATTER_MODEL.stat().st_size if BATTER_MODEL.exists() else 0,
            },
            "pitcher": {
                "path": str(PITCHER_MODEL),
                "exists": PITCHER_MODEL.exists(),
                "trained_at": pitcher_mtime,
                "size_bytes": PITCHER_MODEL.stat().st_size if PITCHER_MODEL.exists() else 0,
            },
        },
        "last_run": _last_run,
    })


# ---------------------------------------------------------------------------
# Integration into app.py — add these THREE lines:
#
#     from api_draft_route import draft_bp           # 1) import
#     # ...existing `app = Flask(__name__)` line...
#     app.register_blueprint(draft_bp)               # 2) register
#     # 3) (no third change required — index.html already has the Draft tab,
#     #     but make sure your existing `@app.route("/")` renders index.html)
#
# After registration:
#     curl http://localhost:5050/api/underdog-draft
#     curl http://localhost:5050/api/draft-status
# ---------------------------------------------------------------------------
