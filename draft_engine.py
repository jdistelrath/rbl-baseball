"""
draft_engine.py — Underdog Draft inference module.

Replaces the heuristic projections in market_underdog_draft.py with a
LightGBM-based projection system. Loads ud_batter_lgbm_latest.pkl /
ud_pitcher_lgbm_latest.pkl from models/ and produces projection dicts
matching the contract:

    {name, team, opponent, position, lineup_spot, projected_fp,
     fp_breakdown, tier, key_edge}

If the model files are missing, falls back to a season-stat heuristic so
the app keeps working. The fallback is intentionally simple — its job is
to never raise, not to compete with the model.

Public entry points:
    project_all_players_ml(lineups, batter_df, pitcher_df) -> list[dict]
    build_draft_cheat_sheet(projections) -> dict
"""

from __future__ import annotations

import glob
import pickle
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

try:
    import lightgbm as lgb  # noqa: F401
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DATA = ROOT / "data"

BATTER_MODEL_PATH = MODELS / "ud_batter_lgbm_latest.pkl"
PITCHER_MODEL_PATH = MODELS / "ud_pitcher_lgbm_latest.pkl"

ROLLING_WINDOWS = [7, 14, 30]


# ---------------------------------------------------------------------------
# UD scoring constants (used by the heuristic fallback)
# ---------------------------------------------------------------------------

HITTER_SCORING = {
    "single": 3.0, "double": 6.0, "triple": 8.0, "home_run": 10.0,
    "walk": 3.0, "hbp": 3.0, "rbi": 2.0, "run": 2.0, "stolen_base": 4.0,
}

PITCHER_SCORING = {
    "win": 2.0, "quality_start": 3.0, "strikeout": 1.0,
    "inning_pitched": 1.0, "earned_run": -1.0,
}


# ---------------------------------------------------------------------------
# Position classification — DH counts as OF, C folded into IF.
# ---------------------------------------------------------------------------

IF_POSITIONS = {"1B", "2B", "3B", "SS", "C", "IF"}
OF_POSITIONS = {"LF", "CF", "RF", "DH", "OF"}
P_POSITIONS = {"P", "SP", "RP"}


def classify_position(pos):
    if not pos:
        return None
    p = str(pos).upper().strip()
    if p in P_POSITIONS:
        return "P"
    if p in IF_POSITIONS:
        return "IF"
    if p in OF_POSITIONS:
        return "OF"
    return None


# Tier thresholds per Frank's spec §6.2
def _tier(fp: float) -> str:
    if fp > 30:
        return "ELITE"
    if fp >= 20:
        return "SOLID"
    if fp >= 12:
        return "VALUE"
    return "AVOID"


# ---------------------------------------------------------------------------
# Safe coercion
# ---------------------------------------------------------------------------

def _safe_int(v, default=0):
    try:
        if v is None:
            return default
        return int(v)
    except (TypeError, ValueError):
        return default


def _safe_float(v, default=0.0):
    try:
        if v is None:
            return default
        f = float(v)
        if f != f:  # NaN
            return default
        return f
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Lazy module-level cache for models / game logs
# ---------------------------------------------------------------------------

_state = {
    "batter_bundle": None,
    "pitcher_bundle": None,
    "batter_logs": None,
    "pitcher_logs": None,
    "models_attempted": False,
    "logs_attempted": False,
}


def _load_pickle(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:
        print(f"[draft_engine] failed to load {path.name}: {exc}")
        return None


def _load_models():
    if _state["models_attempted"]:
        return _state["batter_bundle"], _state["pitcher_bundle"]
    _state["models_attempted"] = True
    if not _HAS_LGB:
        print("[draft_engine] lightgbm not installed — using heuristic fallback")
        return None, None
    _state["batter_bundle"] = _load_pickle(BATTER_MODEL_PATH)
    _state["pitcher_bundle"] = _load_pickle(PITCHER_MODEL_PATH)
    return _state["batter_bundle"], _state["pitcher_bundle"]


def _load_game_logs():
    """Load combined batter / pitcher game-log frames from data/."""
    if _state["logs_attempted"]:
        return _state["batter_logs"], _state["pitcher_logs"]
    _state["logs_attempted"] = True

    def _concat(pattern):
        paths = sorted(glob.glob(str(DATA / pattern)))
        if not paths:
            return None
        frames = []
        for p in paths:
            try:
                frames.append(pd.read_pickle(p))
            except Exception as exc:
                print(f"[draft_engine] failed to read {p}: {exc}")
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)
        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
        return df

    _state["batter_logs"] = _concat("game_logs_batters_*.pkl")
    _state["pitcher_logs"] = _concat("game_logs_pitchers_*.pkl")
    return _state["batter_logs"], _state["pitcher_logs"]


# ---------------------------------------------------------------------------
# Feature schema (must match training schema in scripts/build_features.py).
# Order is fixed because some saved models pickle the column list explicitly.
# ---------------------------------------------------------------------------

BATTER_SEASON_COLS = [
    "season_BA", "season_OBP", "season_SLG", "season_OPS", "season_ISO",
    "season_wRC_plus", "season_hr_fb_ratio", "season_k_rate",
    "season_barrel_rate", "season_hard_hit_pct", "season_PA", "season_G",
]

BATTER_ROLLING_BASES = [
    "avg_fp", "games_played", "hr_rate", "bb_rate",
    "sb_per_game", "singles_rate", "k_rate",
]

BATTER_CONTEXT_COLS = [
    "lineup_spot", "home_away", "month", "year", "days_since_last_game",
]

PITCHER_SEASON_COLS = [
    "season_ERA", "season_k_per_9", "season_ip_per_gs", "season_whip",
    "season_bb_pct", "season_k_pct", "season_hr_fb_ratio_allowed",
]

PITCHER_ROLLING_BASES = [
    "avg_fp", "starts", "k_per_ip", "er_per_ip",
    "qs_rate", "ip_per_start", "win_rate",
]

PITCHER_CONTEXT_COLS = ["home_away", "month", "year", "days_rest"]


def _batter_feature_columns():
    cols = list(BATTER_SEASON_COLS)
    for w in ROLLING_WINDOWS:
        for base in BATTER_ROLLING_BASES:
            cols.append(f"{base}_{w}d")
    cols.extend(BATTER_CONTEXT_COLS)
    return cols


def _pitcher_feature_columns():
    cols = list(PITCHER_SEASON_COLS)
    for w in ROLLING_WINDOWS:
        for base in PITCHER_ROLLING_BASES:
            cols.append(f"{base}_{w}d")
    cols.extend(PITCHER_CONTEXT_COLS)
    return cols


BATTER_FEATURE_COLS = _batter_feature_columns()
PITCHER_FEATURE_COLS = _pitcher_feature_columns()


# ---------------------------------------------------------------------------
# Season stat row lookup
# ---------------------------------------------------------------------------

def _norm_name(s) -> str:
    return str(s or "").strip().lower()


def _find_row_by_name(name: str, df) -> dict | None:
    if df is None or len(df) == 0:
        return None
    n = _norm_name(name)
    if not n:
        return None
    name_col = None
    for c in ("Name", "name", "player_name", "Player"):
        if c in df.columns:
            name_col = c
            break
    if name_col is None:
        return None
    norm = df[name_col].astype(str).str.strip().str.lower()
    hits = df[norm == n]
    if len(hits) == 0:
        last = n.split()[-1] if n.split() else ""
        if last:
            hits = df[norm.str.endswith(" " + last)]
    if len(hits) == 0:
        return None
    return hits.iloc[0].to_dict()


def _find_mlb_id(name: str, df) -> int | None:
    row = _find_row_by_name(name, df)
    if not row:
        return None
    for k in ("mlbID", "mlb_id", "MLBID", "mlbid"):
        if k in row and row[k] is not None:
            try:
                return int(row[k])
            except (TypeError, ValueError):
                pass
    return None


# ---------------------------------------------------------------------------
# Rolling feature computation
# ---------------------------------------------------------------------------

def _rolling_batter_window(player_logs: pd.DataFrame, asof: date, window_days: int) -> dict:
    cutoff = pd.Timestamp(asof) - pd.Timedelta(days=window_days)
    asof_ts = pd.Timestamp(asof)
    sub = player_logs[
        (player_logs["game_date"] > cutoff) & (player_logs["game_date"] < asof_ts)
    ]
    out = {f"{base}_{window_days}d": float("nan") for base in BATTER_ROLLING_BASES}
    out[f"games_played_{window_days}d"] = float(len(sub))
    if len(sub) == 0:
        out[f"hr_rate_{window_days}d"] = 0.0
        out[f"bb_rate_{window_days}d"] = 0.0
        out[f"sb_per_game_{window_days}d"] = 0.0
        out[f"singles_rate_{window_days}d"] = 0.0
        out[f"k_rate_{window_days}d"] = 0.0
        return out

    ab = sub.get("AB", pd.Series([0] * len(sub))).fillna(0).sum()
    bb = sub.get("BB", pd.Series([0] * len(sub))).fillna(0).sum()
    hbp = sub.get("HBP", pd.Series([0] * len(sub))).fillna(0).sum()
    hr = sub.get("HR", pd.Series([0] * len(sub))).fillna(0).sum()
    h = sub.get("H", pd.Series([0] * len(sub))).fillna(0).sum()
    doubles = sub.get("doubles", pd.Series([0] * len(sub))).fillna(0).sum()
    triples = sub.get("triples", pd.Series([0] * len(sub))).fillna(0).sum()
    so = sub.get("SO", pd.Series([0] * len(sub))).fillna(0).sum()
    sb = sub.get("SB", pd.Series([0] * len(sub))).fillna(0).sum()
    singles = max(0, h - doubles - triples - hr)
    games = max(1, len(sub))

    out[f"avg_fp_{window_days}d"] = (
        float(sub["actual_fp"].mean()) if "actual_fp" in sub.columns and len(sub) else float("nan")
    )
    out[f"hr_rate_{window_days}d"] = float(hr / ab) if ab > 0 else 0.0
    pa_proxy = ab + bb + hbp
    out[f"bb_rate_{window_days}d"] = float(bb / pa_proxy) if pa_proxy > 0 else 0.0
    out[f"sb_per_game_{window_days}d"] = float(sb / games)
    out[f"singles_rate_{window_days}d"] = float(singles / ab) if ab > 0 else 0.0
    out[f"k_rate_{window_days}d"] = float(so / ab) if ab > 0 else 0.0
    return out


def _rolling_pitcher_window(player_logs: pd.DataFrame, asof: date, window_days: int) -> dict:
    cutoff = pd.Timestamp(asof) - pd.Timedelta(days=window_days)
    asof_ts = pd.Timestamp(asof)
    sub = player_logs[
        (player_logs["game_date"] > cutoff) & (player_logs["game_date"] < asof_ts)
    ]
    out = {f"{base}_{window_days}d": float("nan") for base in PITCHER_ROLLING_BASES}
    starts = int(len(sub))
    out[f"starts_{window_days}d"] = float(starts)
    if starts == 0:
        out[f"k_per_ip_{window_days}d"] = 0.0
        out[f"er_per_ip_{window_days}d"] = 0.0
        out[f"qs_rate_{window_days}d"] = 0.0
        out[f"ip_per_start_{window_days}d"] = 0.0
        out[f"win_rate_{window_days}d"] = 0.0
        return out

    ip = sub.get("IP", pd.Series([0.0] * starts)).fillna(0).sum()
    so = sub.get("SO", pd.Series([0] * starts)).fillna(0).sum()
    er = sub.get("ER", pd.Series([0] * starts)).fillna(0).sum()
    win = sub.get("win", pd.Series([0] * starts)).fillna(0).sum()
    qs = sub.get("qs_flag", pd.Series([0] * starts)).fillna(0).sum()

    out[f"avg_fp_{window_days}d"] = (
        float(sub["actual_fp"].mean()) if "actual_fp" in sub.columns else float("nan")
    )
    out[f"k_per_ip_{window_days}d"] = float(so / ip) if ip > 0 else 0.0
    out[f"er_per_ip_{window_days}d"] = float(er / ip) if ip > 0 else 0.0
    out[f"qs_rate_{window_days}d"] = float(qs / starts)
    out[f"ip_per_start_{window_days}d"] = float(ip / starts)
    out[f"win_rate_{window_days}d"] = float(win / starts)
    return out


def _days_since_last(player_logs: pd.DataFrame, asof: date, cap: int = 7) -> float:
    if player_logs is None or len(player_logs) == 0:
        return float(cap)
    prior = player_logs[player_logs["game_date"] < pd.Timestamp(asof)]
    if len(prior) == 0:
        return float(cap)
    last = prior["game_date"].max()
    delta = (pd.Timestamp(asof) - last).days
    return float(min(max(delta, 0), cap))


# ---------------------------------------------------------------------------
# Feature builders (one row each, ordered to match training schema)
# ---------------------------------------------------------------------------

def _build_batter_features(season_row: dict, mlb_id, lineup_spot: int,
                           home_away: int, asof: date,
                           batter_logs: pd.DataFrame | None) -> pd.DataFrame:
    feats: dict = {}

    # Season aggregates — try the columns the spec names, fall back to common
    # FanGraphs / pybaseball field names.
    aliases = {
        "season_BA": ("BA", "AVG"),
        "season_OBP": ("OBP",),
        "season_SLG": ("SLG",),
        "season_OPS": ("OPS",),
        "season_ISO": ("ISO",),
        "season_wRC_plus": ("wRC+", "wRC_plus", "wRCplus"),
        "season_hr_fb_ratio": ("HR/FB", "hr_fb"),
        "season_k_rate": ("K%", "k_rate"),
        "season_barrel_rate": ("Barrel%", "barrel_rate"),
        "season_hard_hit_pct": ("HardHit%", "hard_hit_pct"),
        "season_PA": ("PA",),
        "season_G": ("G",),
    }
    for col in BATTER_SEASON_COLS:
        val = float("nan")
        for k in aliases.get(col, ()):
            if season_row and k in season_row and season_row[k] is not None:
                val = _safe_float(season_row[k], float("nan"))
                if val == val:
                    break
        feats[col] = val

    # Rolling
    if batter_logs is not None and mlb_id is not None and "mlb_id" in batter_logs.columns:
        plogs = batter_logs[batter_logs["mlb_id"] == mlb_id]
    else:
        plogs = pd.DataFrame(columns=["game_date"])
    for w in ROLLING_WINDOWS:
        feats.update(_rolling_batter_window(plogs, asof, w))

    # Context
    feats["lineup_spot"] = int(lineup_spot or 0)
    feats["home_away"] = int(bool(home_away))
    feats["month"] = asof.month
    feats["year"] = asof.year
    feats["days_since_last_game"] = _days_since_last(plogs, asof, cap=7)

    return pd.DataFrame([[feats[c] for c in BATTER_FEATURE_COLS]], columns=BATTER_FEATURE_COLS)


def _build_pitcher_features(season_row: dict, mlb_id, home_away: int,
                            asof: date, pitcher_logs: pd.DataFrame | None) -> pd.DataFrame:
    feats: dict = {}
    aliases = {
        "season_ERA": ("ERA",),
        "season_k_per_9": ("K9", "K/9"),
        "season_ip_per_gs": ("IP_per_GS", "ip_per_gs"),
        "season_whip": ("WHIP",),
        "season_bb_pct": ("BB%", "bb_pct"),
        "season_k_pct": ("K%", "k_pct"),
        "season_hr_fb_ratio_allowed": ("HR/FB", "hr_fb_allowed"),
    }
    for col in PITCHER_SEASON_COLS:
        val = float("nan")
        for k in aliases.get(col, ()):
            if season_row and k in season_row and season_row[k] is not None:
                val = _safe_float(season_row[k], float("nan"))
                if val == val:
                    break
        feats[col] = val

    # Special case: ip_per_gs derivable from IP/GS
    if feats["season_ip_per_gs"] != feats["season_ip_per_gs"] and season_row:
        ip = _safe_float(season_row.get("IP", 0))
        gs = _safe_int(season_row.get("GS", 0))
        if gs > 0:
            feats["season_ip_per_gs"] = ip / gs

    if pitcher_logs is not None and mlb_id is not None and "mlb_id" in pitcher_logs.columns:
        plogs = pitcher_logs[pitcher_logs["mlb_id"] == mlb_id]
    else:
        plogs = pd.DataFrame(columns=["game_date"])
    for w in ROLLING_WINDOWS:
        feats.update(_rolling_pitcher_window(plogs, asof, w))

    feats["home_away"] = int(bool(home_away))
    feats["month"] = asof.month
    feats["year"] = asof.year
    feats["days_rest"] = _days_since_last(plogs, asof, cap=10)

    return pd.DataFrame(
        [[feats[c] for c in PITCHER_FEATURE_COLS]], columns=PITCHER_FEATURE_COLS
    )


# ---------------------------------------------------------------------------
# Bundle prediction — supports both raw model and {"model": ..., "features": [...]}
# ---------------------------------------------------------------------------

def _predict(bundle, X: pd.DataFrame, default_cols: list[str]) -> float:
    if bundle is None:
        return float("nan")
    model = bundle
    cols = default_cols
    if isinstance(bundle, dict):
        model = bundle.get("model")
        cols = bundle.get("features") or bundle.get("feature_names") or default_cols
    if model is None:
        return float("nan")
    # Reorder / restrict columns to whatever the model was trained on.
    X2 = X.copy()
    for c in cols:
        if c not in X2.columns:
            X2[c] = float("nan")
    X2 = X2[cols]
    try:
        pred = model.predict(X2)
        return float(pred[0])
    except Exception as exc:
        print(f"[draft_engine] predict failed: {exc}")
        return float("nan")


# ---------------------------------------------------------------------------
# Heuristic fallback (kept simple — guarantees the app never crashes)
# ---------------------------------------------------------------------------

# League per-PA baselines + lineup-spot multipliers (from existing module).
RUN_MULT = [0, 1.20, 1.15, 1.05, 1.00, 0.95, 0.90, 0.85, 0.80, 0.85]
RBI_MULT = [0, 0.65, 0.85, 1.20, 1.30, 1.20, 1.05, 0.95, 0.85, 0.75]
LEAGUE_RBI_PER_PA = 0.12
LEAGUE_R_PER_PA = 0.11


def _heuristic_hitter(name, team, opponent, position, lineup_spot, row) -> dict | None:
    if not row:
        return None
    pa = _safe_int(row.get("PA", 0))
    g = _safe_int(row.get("G", 0))
    if pa < 30 or g < 5:
        return None
    h = _safe_int(row.get("H", 0))
    hr = _safe_int(row.get("HR", 0))
    doubles = _safe_int(row.get("2B", 0))
    triples = _safe_int(row.get("3B", 0))
    obp = _safe_float(row.get("OBP", 0))
    ba = _safe_float(row.get("BA", 0))
    singles = max(0, h - doubles - triples - hr)
    h_per_pa = h / pa if pa else 0
    bb_hbp_per_pa = max(0.0, obp - h_per_pa)

    spot = lineup_spot if 1 <= (lineup_spot or 0) <= 9 else 5
    proj_pa = 4.3 if spot <= 5 else 3.8

    hits_fp = (
        (singles / pa) * HITTER_SCORING["single"]
        + (doubles / pa) * HITTER_SCORING["double"]
        + (triples / pa) * HITTER_SCORING["triple"]
    ) * proj_pa
    hr_fp = (hr / pa) * HITTER_SCORING["home_run"] * proj_pa
    walk_fp = bb_hbp_per_pa * HITTER_SCORING["walk"] * proj_pa
    rbi_fp = LEAGUE_RBI_PER_PA * RBI_MULT[spot] * proj_pa * HITTER_SCORING["rbi"]
    run_fp = LEAGUE_R_PER_PA * RUN_MULT[spot] * proj_pa * HITTER_SCORING["run"]
    total = hits_fp + hr_fp + walk_fp + rbi_fp + run_fp

    edge = []
    if (hr / pa if pa else 0) >= 0.05:
        edge.append(f"{hr/pa*100:.1f}% HR/PA")
    if obp >= 0.360:
        edge.append(f"{obp:.3f} OBP")
    if ba >= 0.290:
        edge.append(f"{ba:.3f} BA")
    if spot <= 3:
        edge.append(f"hits {spot}")
    elif spot in (4, 5):
        edge.append("RBI spot")
    return {
        "name": name, "team": team, "opponent": opponent,
        "position": classify_position(position) or "OF",
        "raw_position": position, "lineup_spot": spot,
        "projected_fp": round(total, 1),
        "fp_breakdown": {
            "hits": round(hits_fp, 1), "hr": round(hr_fp, 1),
            "walks": round(walk_fp, 1), "rbi": round(rbi_fp, 1),
            "runs": round(run_fp, 1), "sb": 0.0,
        },
        "tier": _tier(total),
        "key_edge": " · ".join(edge) if edge else "—",
    }


def _heuristic_pitcher(name, team, opponent, row) -> dict | None:
    if not row:
        return None
    gs = _safe_int(row.get("GS", 0))
    ip_total = _safe_float(row.get("IP", 0))
    k9 = _safe_float(row.get("K9", row.get("K/9", 0)))
    era = _safe_float(row.get("ERA", 0))
    if gs >= 1 and ip_total > 0:
        ip_per_start = min(ip_total / gs, 6.0)
    else:
        ip_per_start = 5.0
    if ip_per_start < 3.0 or k9 <= 0:
        return None
    proj_ip = ip_per_start
    proj_k = k9 * proj_ip / 9.0
    proj_er = era * proj_ip / 9.0 if era > 0 else 2.5
    win_prob = 0.5
    if proj_ip >= 6.0 and 0 < era <= 3.50:
        qs_prob = 0.45
    elif proj_ip >= 6.0 and 0 < era <= 4.20:
        qs_prob = 0.30
    elif proj_ip >= 6.0:
        qs_prob = 0.15
    else:
        qs_prob = 0.10
    ip_fp = proj_ip * PITCHER_SCORING["inning_pitched"]
    k_fp = proj_k * PITCHER_SCORING["strikeout"]
    er_fp = proj_er * PITCHER_SCORING["earned_run"]
    win_fp = win_prob * PITCHER_SCORING["win"]
    qs_fp = qs_prob * PITCHER_SCORING["quality_start"]
    total = ip_fp + k_fp + er_fp + win_fp + qs_fp

    edge = []
    if k9 >= 9.5:
        edge.append(f"{k9:.1f} K/9")
    elif k9 >= 8.0:
        edge.append(f"solid {k9:.1f} K/9")
    if 0 < era <= 3.50:
        edge.append(f"{era:.2f} ERA")
    elif era >= 5.0:
        edge.append(f"shaky {era:.2f} ERA")
    if proj_ip >= 6.0:
        edge.append("QS upside")
    return {
        "name": name, "team": team, "opponent": opponent,
        "position": "P", "raw_position": "SP", "lineup_spot": 0,
        "projected_fp": round(total, 1),
        "fp_breakdown": {
            "ip": round(ip_fp, 1), "k": round(k_fp, 1),
            "er": round(er_fp, 1), "win": round(win_fp, 1),
            "qs": round(qs_fp, 1),
        },
        "tier": _tier(total),
        "key_edge": " · ".join(edge) if edge else "—",
    }


# ---------------------------------------------------------------------------
# Public: project_all_players_ml
# ---------------------------------------------------------------------------

def project_all_players_ml(lineups, batter_df, pitcher_df) -> list[dict]:
    """
    Build projection dicts for every player on today's slate.

    `lineups` is a list of game dicts: each has home_team, away_team,
    home_pitcher_name, away_pitcher_name, and 'lineups' = {home: [...], away: [...]}.
    `batter_df` / `pitcher_df` are the season aggregate frames already cached
    by the existing data layer (FanGraphs / pybaseball pull).
    """
    asof = date.today()
    batter_bundle, pitcher_bundle = _load_models()
    have_models = batter_bundle is not None and pitcher_bundle is not None
    if have_models:
        batter_logs, pitcher_logs = _load_game_logs()
    else:
        batter_logs = pitcher_logs = None

    projections: list[dict] = []

    for game in lineups or []:
        home = game.get("home_team", "")
        away = game.get("away_team", "")

        # ---- Pitchers ----
        for role, opp_team, p_team in [
            ("home_pitcher_name", away, home),
            ("away_pitcher_name", home, away),
        ]:
            pname = game.get(role)
            if not pname or pname == "TBD":
                continue
            row = _find_row_by_name(pname, pitcher_df)
            if row is None:
                continue
            home_away = 1 if role == "home_pitcher_name" else 0
            proj = None
            if have_models:
                mlb_id = _find_mlb_id(pname, pitcher_df)
                X = _build_pitcher_features(row, mlb_id, home_away, asof, pitcher_logs)
                yhat = _predict(pitcher_bundle, X, PITCHER_FEATURE_COLS)
                if yhat == yhat:
                    edge_bits = []
                    k9 = _safe_float(row.get("K9", row.get("K/9", 0)))
                    era = _safe_float(row.get("ERA", 0))
                    if k9 >= 9.5:
                        edge_bits.append(f"{k9:.1f} K/9")
                    if 0 < era <= 3.50:
                        edge_bits.append(f"{era:.2f} ERA")
                    proj = {
                        "name": pname, "team": p_team, "opponent": opp_team,
                        "position": "P", "raw_position": "SP", "lineup_spot": 0,
                        "projected_fp": round(yhat, 1),
                        "fp_breakdown": {"model": round(yhat, 1)},
                        "tier": _tier(yhat),
                        "key_edge": " · ".join(edge_bits) if edge_bits else "model projection",
                    }
            if proj is None:
                proj = _heuristic_pitcher(pname, p_team, opp_team, row)
            if proj:
                projections.append(proj)

        # ---- Hitters ----
        lineup = game.get("lineups") or {}
        for side, opp_team in [("home", away), ("away", home)]:
            bteam = home if side == "home" else away
            home_away = 1 if side == "home" else 0
            for batter in lineup.get(side, []) or []:
                name = batter.get("name", "")
                if not name:
                    continue
                pos = batter.get("position", "")
                pos_class = classify_position(pos)
                if pos_class is None or pos_class == "P":
                    continue
                row = _find_row_by_name(name, batter_df)
                if row is None:
                    continue
                spot = batter.get("batting_order") or batter.get("lineup_spot") or 0
                proj = None
                if have_models:
                    mlb_id = _find_mlb_id(name, batter_df)
                    X = _build_batter_features(
                        row, mlb_id, int(spot or 0), home_away, asof, batter_logs
                    )
                    yhat = _predict(batter_bundle, X, BATTER_FEATURE_COLS)
                    if yhat == yhat:
                        edge_bits = []
                        ba = _safe_float(row.get("BA", 0))
                        obp = _safe_float(row.get("OBP", 0))
                        if obp >= 0.360:
                            edge_bits.append(f"{obp:.3f} OBP")
                        if ba >= 0.290:
                            edge_bits.append(f"{ba:.3f} BA")
                        if 1 <= int(spot or 0) <= 3:
                            edge_bits.append(f"hits {int(spot)}")
                        elif int(spot or 0) in (4, 5):
                            edge_bits.append("RBI spot")
                        proj = {
                            "name": name, "team": bteam, "opponent": opp_team,
                            "position": pos_class, "raw_position": pos,
                            "lineup_spot": int(spot or 0),
                            "projected_fp": round(yhat, 1),
                            "fp_breakdown": {"model": round(yhat, 1)},
                            "tier": _tier(yhat),
                            "key_edge": " · ".join(edge_bits) if edge_bits else "model projection",
                        }
                if proj is None:
                    proj = _heuristic_hitter(name, bteam, opp_team, pos, int(spot or 0), row)
                if proj:
                    projections.append(proj)

    return projections


# ---------------------------------------------------------------------------
# Public: build_draft_cheat_sheet
# ---------------------------------------------------------------------------

def build_draft_cheat_sheet(projections: list[dict]) -> dict:
    """
    Rank the projections and bucket them into the cheat-sheet sections.

    Returns:
        top_picks               — overall top 12 by projected_fp
        pitcher_picks           — top 8 pitchers
        stack_recommendations   — top 5 team stacks (>=3 hitters / team, summed FP)
        avoid_list              — players in AVOID tier (sorted by lineup_spot, capped 12)
    """
    if not projections:
        return {
            "top_picks": [], "pitcher_picks": [],
            "stack_recommendations": [], "avoid_list": [],
        }

    overall = sorted(projections, key=lambda p: p.get("projected_fp", 0), reverse=True)
    pitchers = [p for p in overall if p.get("position") == "P"]
    hitters = [p for p in overall if p.get("position") in ("IF", "OF")]

    # Stack recommendations: for each team, sum top 4 hitter projections.
    by_team: dict[str, list[dict]] = {}
    for h in hitters:
        by_team.setdefault(h.get("team") or "—", []).append(h)
    stacks = []
    for team, players in by_team.items():
        if len(players) < 3:
            continue
        top4 = sorted(players, key=lambda p: p["projected_fp"], reverse=True)[:4]
        stack_fp = round(sum(p["projected_fp"] for p in top4), 1)
        stacks.append({
            "team": team,
            "stack_fp": stack_fp,
            "hitter_count": len(top4),
            "players": [
                {"name": p["name"], "lineup_spot": p["lineup_spot"],
                 "projected_fp": p["projected_fp"], "tier": p["tier"]}
                for p in top4
            ],
        })
    stacks.sort(key=lambda s: s["stack_fp"], reverse=True)

    avoid = [p for p in overall if p.get("tier") == "AVOID"]
    avoid.sort(key=lambda p: (p.get("lineup_spot") or 99, -p.get("projected_fp", 0)))

    return {
        "top_picks": overall[:12],
        "pitcher_picks": pitchers[:8],
        "stack_recommendations": stacks[:5],
        "avoid_list": avoid[:12],
    }


__all__ = ["project_all_players_ml", "build_draft_cheat_sheet"]
