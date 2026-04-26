"""
Prop Projection Accuracy Backtest.
Validates market_sweep model projections against actual MLB game outcomes.

Usage: python main.py backtest_props
"""

import math
from datetime import date

import numpy as np
import pandas as pd
import requests

from config import CFG


# ---------------------------------------------------------------------------
# Cache helpers (reuse data_fetcher pattern)
# ---------------------------------------------------------------------------

def _cache_path(name):
    return CFG.cache_dir / f"{name}.pkl"


def _load_cache(name):
    import pickle
    p = _cache_path(name)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(name, obj):
    import pickle
    p = _cache_path(name)
    with open(p, "wb") as f:
        pickle.dump(obj, f)


# ---------------------------------------------------------------------------
# Fetch game logs from MLB Stats API
# ---------------------------------------------------------------------------

def _fetch_pitcher_game_logs(player_id, season):
    """Fetch pitcher game log for a season. Returns list of start dicts."""
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
        f"?stats=gameLog&group=pitching&season={season}&gameType=R"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        splits = data.get("stats", [{}])[0].get("splits", [])
        starts = []
        for s in splits:
            stat = s.get("stat", {})
            if int(stat.get("gamesStarted", 0)) == 0:
                continue  # relief appearance
            ip_str = stat.get("inningsPitched", "0")
            ip = float(ip_str)
            whole = int(ip)
            frac = ip - whole
            ip_actual = whole + (frac * 10 / 3.0)
            starts.append({
                "date": s.get("date", ""),
                "opponent": s.get("opponent", {}).get("name", ""),
                "actual_k": int(stat.get("strikeOuts", 0)),
                "ip": ip_actual,
            })
        return starts
    except Exception:
        return []


def _fetch_batter_game_logs(player_id, season):
    """Fetch batter game log for a season. Returns list of game dicts."""
    url = (
        f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
        f"?stats=gameLog&group=hitting&season={season}&gameType=R"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        splits = data.get("stats", [{}])[0].get("splits", [])
        games = []
        for s in splits:
            stat = s.get("stat", {})
            ab = int(stat.get("atBats", 0))
            if ab == 0:
                continue  # didn't bat (pinch runner, etc.)
            games.append({
                "date": s.get("date", ""),
                "actual_h": int(stat.get("hits", 0)),
                "actual_tb": int(stat.get("totalBases", 0)),
                "actual_hr": int(stat.get("homeRuns", 0)),
                "ab": ab,
            })
        return games
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Build projection models (same logic as market_sweep.py)
# ---------------------------------------------------------------------------

def _project_pitcher_ks(pitcher_row, opp_k_rate=0.225):
    """Project Ks for a start using K/9 * IP/start * opp K-rate adjustment."""
    try:
        k9 = float(pitcher_row.get("K9", pitcher_row.get("K/9", 0)))
        ip = float(pitcher_row.get("IP", 0))
        gs = float(pitcher_row.get("GS", 1))
    except (TypeError, ValueError):
        return None
    if gs < 3 or k9 < 1:
        return None
    ip_per_start = min(ip / max(gs, 1), 6.5)
    k_rate_adj = 1.0 + (opp_k_rate - 0.225) * 0.5
    return (k9 / 9.0) * ip_per_start * k_rate_adj


def _project_batter_rates(batter_row):
    """Project per-game hits and total bases."""
    try:
        g = int(batter_row.get("G", 0))
        h = int(batter_row.get("H", 0))
        tb = int(batter_row.get("TB", 0))
    except (TypeError, ValueError):
        return None, None
    if g < 10:
        return None, None
    return h / g, tb / g


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------

def run_props_backtest(years=None):
    """
    Backtest prop projections against actual game outcomes.
    Uses prior-year season stats for projections (no lookahead).
    """
    if years is None:
        years = [2022, 2023, 2024, 2025]

    from data_fetcher import get_batter_statcast, get_pitcher_statcast

    print("[backtest_props] === Prop Projection Accuracy Backtest ===")

    for year in years:
        print(f"\n{'='*60}")
        print(f"  {year} SEASON")
        print(f"{'='*60}")

        # Load PRIOR year stats for model projections (no lookahead)
        print(f"  Loading {year-1} season stats for projections...")
        batter_df = get_batter_statcast(season=year - 1)
        pitcher_df = get_pitcher_statcast(season=year - 1)

        if pitcher_df.empty or batter_df.empty:
            print(f"  No stats available for {year-1}, skipping.")
            continue

        # --- Pitcher Ks ---
        _backtest_pitcher_ks(year, pitcher_df)

        # --- Batter Hits ---
        _backtest_batter_hits(year, batter_df)

        # --- Batter Total Bases ---
        _backtest_batter_tb(year, batter_df)


def _backtest_pitcher_ks(year, pitcher_df):
    """Backtest K projections for all qualified starters."""
    cache_key = f"props_bt_pitcher_ks_{year}"
    cached = _load_cache(cache_key)

    if cached is not None:
        results = cached
        print(f"  [Ks] Using cached results ({len(results)} starts)")
    else:
        # Get starters with GS >= 10 and K/9 > 0
        starters = pitcher_df[
            (pitcher_df["GS"] >= 10) &
            (pitcher_df["K9"] > 0)
        ]
        print(f"  [Ks] Testing {len(starters)} pitchers...")

        results = []
        for _, row in starters.iterrows():
            pid = int(row["mlbID"])
            name = row["Name"]
            proj_k = _project_pitcher_ks(row)
            if proj_k is None:
                continue

            game_logs = _fetch_pitcher_game_logs(pid, year)
            if not game_logs:
                continue

            for game in game_logs:
                results.append({
                    "name": name,
                    "projected": round(proj_k, 2),
                    "actual": game["actual_k"],
                    "ip": game["ip"],
                    "date": game["date"],
                })

        _save_cache(cache_key, results)
        print(f"  [Ks] Fetched {len(results)} starts")

    if not results:
        print("  [Ks] No data.")
        return

    df = pd.DataFrame(results)
    df["error"] = df["projected"] - df["actual"]
    df["abs_error"] = df["error"].abs()
    df["within_1_5"] = df["abs_error"] <= 1.5

    # Directional accuracy vs common lines
    for line in [4.5, 5.5, 6.5]:
        subset = df[df["projected"] != line].copy()
        if len(subset) < 50:
            continue
        subset["model_over"] = subset["projected"] > line
        subset["actual_over"] = subset["actual"] > line
        correct = (subset["model_over"] == subset["actual_over"]).sum()
        total = len(subset)
        pct = correct / total * 100
        print(f"  [Ks] vs {line} line: {correct}/{total} = {pct:.1f}% directional accuracy")

    mae = df["abs_error"].mean()
    within = df["within_1_5"].mean() * 100
    bias = df["error"].mean()
    print(f"  [Ks] MAE: {mae:.2f} | Within +/-1.5: {within:.1f}% | Bias: {bias:+.2f}")
    print(f"  [Ks] Starts analyzed: {len(df)}")


def _backtest_batter_hits(year, batter_df):
    """Backtest hits/game projections."""
    cache_key = f"props_bt_batter_hits_{year}"
    cached = _load_cache(cache_key)

    if cached is not None:
        results = cached
        print(f"  [Hits] Using cached results ({len(results)} games)")
    else:
        # Qualified batters
        if "H" not in batter_df.columns or "G" not in batter_df.columns:
            print("  [Hits] H or G column missing from stats. Skipping.")
            return

        qualified = batter_df[batter_df["G"] >= 50].copy()
        print(f"  [Hits] Testing {len(qualified)} batters...")

        results = []
        for _, row in qualified.iterrows():
            pid = int(row["mlbID"])
            name = row["Name"]
            h_per_g, _ = _project_batter_rates(row)
            if h_per_g is None:
                continue

            game_logs = _fetch_batter_game_logs(pid, year)
            if not game_logs:
                continue

            for game in game_logs:
                results.append({
                    "name": name,
                    "projected": round(h_per_g, 2),
                    "actual": game["actual_h"],
                    "date": game["date"],
                })

        _save_cache(cache_key, results)
        print(f"  [Hits] Fetched {len(results)} games")

    if not results:
        print("  [Hits] No data.")
        return

    df = pd.DataFrame(results)
    df["error"] = df["projected"] - df["actual"]
    df["abs_error"] = df["error"].abs()
    df["within_0_5"] = df["abs_error"] <= 0.5

    # Directional accuracy vs common lines
    for line in [0.5, 1.5]:
        subset = df.copy()
        subset["model_over"] = subset["projected"] > line
        subset["actual_over"] = subset["actual"] > line
        correct = (subset["model_over"] == subset["actual_over"]).sum()
        total = len(subset)
        pct = correct / total * 100
        print(f"  [Hits] vs {line} line: {correct}/{total} = {pct:.1f}% directional accuracy")

    mae = df["abs_error"].mean()
    within = df["within_0_5"].mean() * 100
    bias = df["error"].mean()
    print(f"  [Hits] MAE: {mae:.2f} | Within +/-0.5: {within:.1f}% | Bias: {bias:+.2f}")
    print(f"  [Hits] Games analyzed: {len(df)}")


def _backtest_batter_tb(year, batter_df):
    """Backtest total bases/game projections."""
    cache_key = f"props_bt_batter_tb_{year}"
    cached = _load_cache(cache_key)

    if cached is not None:
        results = cached
        print(f"  [TB] Using cached results ({len(results)} games)")
    else:
        if "TB" not in batter_df.columns or "G" not in batter_df.columns:
            print("  [TB] TB or G column missing from stats. Skipping.")
            return

        qualified = batter_df[batter_df["G"] >= 50].copy()
        print(f"  [TB] Testing {len(qualified)} batters...")

        results = []
        for _, row in qualified.iterrows():
            pid = int(row["mlbID"])
            name = row["Name"]
            _, tb_per_g = _project_batter_rates(row)
            if tb_per_g is None:
                continue

            game_logs = _fetch_batter_game_logs(pid, year)
            if not game_logs:
                continue

            for game in game_logs:
                results.append({
                    "name": name,
                    "projected": round(tb_per_g, 2),
                    "actual": game["actual_tb"],
                    "date": game["date"],
                })

        _save_cache(cache_key, results)
        print(f"  [TB] Fetched {len(results)} games")

    if not results:
        print("  [TB] No data.")
        return

    df = pd.DataFrame(results)
    df["error"] = df["projected"] - df["actual"]
    df["abs_error"] = df["error"].abs()
    df["within_0_5"] = df["abs_error"] <= 0.5

    # Directional accuracy vs common lines
    for line in [0.5, 1.5, 2.5]:
        subset = df.copy()
        subset["model_over"] = subset["projected"] > line
        subset["actual_over"] = subset["actual"] > line
        correct = (subset["model_over"] == subset["actual_over"]).sum()
        total = len(subset)
        pct = correct / total * 100
        print(f"  [TB] vs {line} line: {correct}/{total} = {pct:.1f}% directional accuracy")

    mae = df["abs_error"].mean()
    within = df["within_0_5"].mean() * 100
    bias = df["error"].mean()
    print(f"  [TB] MAE: {mae:.2f} | Within +/-0.5: {within:.1f}% | Bias: {bias:+.2f}")
    print(f"  [TB] Games analyzed: {len(df)}")
