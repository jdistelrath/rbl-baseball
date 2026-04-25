"""
Module 6: Backtest.
Walk-forward validation harness. Run manually only -- not on cron.

Usage:
    python main.py backtest            # season-level (fast)
    python main.py backtest_gamelevel  # game-level via Statcast (slow, ~15-30 min)
    python main.py backtest_totals     # totals model calibration (20-40 min)
"""

import json
import math
import os
from datetime import date, datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss

from config import CFG
from scorer import FEATURE_COLS

# ---------------------------------------------------------------------------
# Data Loading (historical)
# ---------------------------------------------------------------------------

def _load_historical_batting(year):
    """Load batting stats for a given year via MLB Stats API."""
    from data_fetcher import get_batter_statcast
    return get_batter_statcast(season=year)


def _load_historical_pitching(year):
    """Load pitching stats for a given year via MLB Stats API."""
    from data_fetcher import get_pitcher_statcast
    return get_pitcher_statcast(season=year)


def _load_hr_actuals(year):
    """
    Load actual HR totals per player for a season.
    Returns dict {player_name: total_HR}.
    """
    df = _load_historical_batting(year)
    if df.empty:
        return {}
    name_col = "Name" if "Name" in df.columns else df.columns[0]
    hr_col = "HR" if "HR" in df.columns else None
    if hr_col is None:
        return {}
    return dict(zip(df[name_col], df[hr_col]))


# ---------------------------------------------------------------------------
# Feature extraction for backtest (simplified vs live pipeline)
# ---------------------------------------------------------------------------

def _safe_float(val, default=0.0):
    try:
        v = float(val)
        return v if not (math.isnan(v) or math.isinf(v)) else default
    except (TypeError, ValueError):
        return default


def _get_stat(row, col_names, default=0.0):
    if row is None:
        return default
    for col in col_names:
        if col in row.index:
            v = _safe_float(row[col], default)
            return v
    return default


def _extract_batter_features(batter_row):
    """Extract feature vector from a batting_stats_bref row."""
    # barrel_rate not in BRef -- use position average
    barrel = _get_stat(batter_row, ["barrel_rate", "Barrel%"], 0.065)
    if barrel > 1:
        barrel /= 100.0

    # hr_fb_ratio: derived column from bref, or FanGraphs column name
    hr_fb = _get_stat(batter_row, ["hr_fb_ratio", "HR/FB", "HR_FB"], 0.12)
    if hr_fb > 1:
        hr_fb /= 100.0

    iso = _get_stat(batter_row, ["ISO"], 0.155)

    # hard_hit_pct not in BRef -- use position average
    hard_hit = _get_stat(batter_row, ["hard_hit_pct", "Hard%", "HardHit%"], 0.35)
    if hard_hit > 1:
        hard_hit /= 100.0

    return {
        "barrel_rate": barrel,
        "hr_fb_ratio": hr_fb,
        "iso": iso,
        "hard_hit_pct": hard_hit,
    }


def _build_backtest_dataset(train_years, batting_dfs, pitching_dfs, hr_actuals):
    """
    Build a dataset of (feature_vector, did_homer_binary) for walk-forward.

    Since we don't have game-level historical matchups easily available,
    we use season-level stats and binary: did this batter hit >= median HR rate?
    This is a proxy for the daily model.
    """
    rows = []
    for year in train_years:
        b_df = batting_dfs.get(year, pd.DataFrame())
        p_df = pitching_dfs.get(year, pd.DataFrame())
        actuals = hr_actuals.get(year, {})

        if b_df.empty:
            continue

        name_col = "Name" if "Name" in b_df.columns else b_df.columns[0]

        for _, brow in b_df.iterrows():
            bname = brow[name_col] if name_col in brow.index else ""
            feats = _extract_batter_features(brow)

            # Use league-average pitcher features (since we don't have matchups)
            feats["pitcher_hr_fb"] = 0.11
            feats["pitcher_fly_ball_rate"] = 0.34
            feats["pitcher_xfip"] = 4.20
            feats["pitcher_hard_hit_allowed"] = 0.35

            # Neutral context
            feats["park_hr_factor"] = 1.0
            feats["wind_bonus"] = 0.0
            feats["temp_bonus"] = 0.0
            feats["batting_order_position"] = 0.0
            feats["platoon_factor"] = 1.0

            # Target: above-median HR rate for the season
            hr_count = actuals.get(bname, 0)
            ab = _safe_float(_get_stat(brow, ["AB", "PA"], 400), 400)
            hr_rate = hr_count / max(ab, 1)

            rows.append({**feats, "hr_rate": hr_rate, "name": bname, "year": year})

    if not rows:
        return pd.DataFrame(), np.array([])

    df = pd.DataFrame(rows)
    # Binary target: top 30% HR rate -> 1, else 0 (proxy for "likely to homer on any day")
    threshold = df["hr_rate"].quantile(0.70)
    y = (df["hr_rate"] >= threshold).astype(int).values
    return df, y


def _score_with_weights(features_df, weights_dict):
    """Apply weights to features, z-score normalize, sigmoid -> probabilities."""
    n = len(features_df)
    composite = np.zeros(n)

    for col in FEATURE_COLS:
        if col not in features_df.columns:
            continue
        vals = features_df[col].values.astype(float)
        mean = np.mean(vals)
        std = np.std(vals)
        if std < 1e-9:
            z = np.zeros(n)
        else:
            z = (vals - mean) / std
        w = weights_dict.get(col, 1.0)
        composite += w * z

    probs = 1.0 / (1.0 + np.exp(-composite))
    return probs


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def _optimize_weights(train_df, train_y):
    """Optimize feature weights to minimize Brier score on training data."""
    initial = np.ones(len(FEATURE_COLS))

    def objective(w):
        w_dict = {col: w[i] for i, col in enumerate(FEATURE_COLS)}
        probs = _score_with_weights(train_df, w_dict)
        return brier_score_loss(train_y, probs)

    result = minimize(objective, initial, method="Nelder-Mead",
                      options={"maxiter": 2000, "xatol": 1e-4})
    optimized = {col: float(result.x[i]) for i, col in enumerate(FEATURE_COLS)}
    return optimized, result.fun


# ---------------------------------------------------------------------------
# Simulated PrizePicks ROI
# ---------------------------------------------------------------------------

def _simulate_roi(probs, actuals, tiers_thresholds):
    """
    Simulate PrizePicks ROI at each confidence tier.
    3-leg = 5x, 6-leg = 40x.
    """
    strong_t, standard_t = tiers_thresholds
    results = {"STRONG": {"bets": 0, "wins": 0, "wagered": 0, "returned": 0},
               "STANDARD": {"bets": 0, "wins": 0, "wagered": 0, "returned": 0},
               "SPECULATIVE": {"bets": 0, "wins": 0, "wagered": 0, "returned": 0}}

    for prob, actual in zip(probs, actuals):
        if prob >= strong_t:
            tier = "STRONG"
            bet = CFG.bet_high
        elif prob >= standard_t:
            tier = "STANDARD"
            bet = CFG.bet_standard
        else:
            tier = "SPECULATIVE"
            bet = CFG.bet_lottery

        results[tier]["bets"] += 1
        results[tier]["wagered"] += bet
        if actual == 1:
            results[tier]["wins"] += 1
            results[tier]["returned"] += bet * 5  # simplified: treat as single-leg 5x

    for tier in results:
        w = results[tier]["wagered"]
        r = results[tier]["returned"]
        results[tier]["roi"] = ((r - w) / w * 100) if w > 0 else 0.0

    return results


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------

def run_backtest():
    """
    Walk-forward backtest:
    - Train on start_year to end_year-2
    - Validate on end_year-1
    - Test on end_year
    - Holdout year not touched

    Outputs:
    - outputs/backtest/backtest_results_{date}.json
    - weights.json (calibrated weights at project root)
    """
    start = CFG.bt_start_year
    end = CFG.bt_end_year

    print(f"[backtest] Loading historical data {start}-{end}...")

    batting_dfs = {}
    pitching_dfs = {}
    hr_actuals = {}

    for year in range(start, end + 1):
        print(f"  Loading {year}...")
        batting_dfs[year] = _load_historical_batting(year)
        pitching_dfs[year] = _load_historical_pitching(year)
        hr_actuals[year] = _load_hr_actuals(year)

    # Walk-forward splits
    train_years = list(range(start, end - 1))  # 2021-2022
    val_years = [end - 1]                       # 2023
    test_years = [end]                          # 2024

    print(f"[backtest] Train: {train_years}, Val: {val_years}, Test: {test_years}")

    # Build datasets
    train_df, train_y = _build_backtest_dataset(train_years, batting_dfs, pitching_dfs, hr_actuals)
    val_df, val_y = _build_backtest_dataset(val_years, batting_dfs, pitching_dfs, hr_actuals)
    test_df, test_y = _build_backtest_dataset(test_years, batting_dfs, pitching_dfs, hr_actuals)

    if train_df.empty:
        print("[backtest] No training data available. Aborting.")
        return

    print(f"[backtest] Dataset sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # Optimize weights on training data
    print("[backtest] Optimizing weights...")
    optimized_weights, train_brier = _optimize_weights(train_df, train_y)
    print(f"[backtest] Train Brier score: {train_brier:.4f}")

    # Validate
    val_results = {}
    if not val_df.empty:
        val_probs = _score_with_weights(val_df, optimized_weights)
        val_brier = brier_score_loss(val_y, val_probs)
        print(f"[backtest] Validation Brier score: {val_brier:.4f}")

        sorted_vp = np.sort(val_probs)[::-1]
        strong_t = sorted_vp[max(0, int(len(val_probs) * 0.15) - 1)]
        standard_t = sorted_vp[max(0, int(len(val_probs) * 0.50) - 1)]
        val_roi = _simulate_roi(val_probs, val_y, (strong_t, standard_t))
        val_results = {"brier": val_brier, "roi": val_roi}

        print("[backtest] Validation ROI by tier:")
        for tier, data in val_roi.items():
            print(f"  {tier}: {data['bets']} bets, {data['wins']} wins, ROI={data['roi']:.1f}%")

    # Test
    test_results = {}
    if not test_df.empty:
        test_probs = _score_with_weights(test_df, optimized_weights)
        test_brier = brier_score_loss(test_y, test_probs)
        print(f"[backtest] Test Brier score: {test_brier:.4f}")

        sorted_tp = np.sort(test_probs)[::-1]
        strong_t = sorted_tp[max(0, int(len(test_probs) * 0.15) - 1)]
        standard_t = sorted_tp[max(0, int(len(test_probs) * 0.50) - 1)]
        test_roi = _simulate_roi(test_probs, test_y, (strong_t, standard_t))
        test_results = {"brier": test_brier, "roi": test_roi}

        print("[backtest] Test ROI by tier:")
        for tier, data in test_roi.items():
            print(f"  {tier}: {data['bets']} bets, {data['wins']} wins, ROI={data['roi']:.1f}%")

    # Save weights
    weights_path = CFG.base_dir / "weights.json"
    with open(weights_path, "w") as f:
        json.dump(optimized_weights, f, indent=2)
    print(f"[backtest] Saved calibrated weights to {weights_path}")

    # Save full results
    results = {
        "date": date.today().isoformat(),
        "train_years": train_years,
        "val_years": val_years,
        "test_years": test_years,
        "train_brier": train_brier,
        "validation": val_results,
        "test": test_results,
        "weights": optimized_weights,
    }

    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_path = CFG.backtest_dir / f"backtest_results_{date.today().isoformat()}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=_convert)
    print(f"[backtest] Results saved to {results_path}")
    print("[backtest] Done.")

    return results


# ---------------------------------------------------------------------------
# Game-Level Backtest (Statcast)
# ---------------------------------------------------------------------------

def _fetch_game_level_hr_data(year):
    """
    Pull Statcast pitch data for a season, aggregate to batter-game level.
    Returns DataFrame: game_date, batter_id, batter_name, home_team, away_team, homered
    """
    from pybaseball import statcast
    from data_fetcher import _load_cache, _save_cache

    cache_key = f"statcast_gamelevel_{year}"
    cached = _load_cache(cache_key)
    if cached is not None:
        print(f"[backtest_gl] Using cached game-level data for {year}")
        return cached

    months = [
        (f"{year}-03-20", f"{year}-03-31"),
        (f"{year}-04-01", f"{year}-04-30"),
        (f"{year}-05-01", f"{year}-05-31"),
        (f"{year}-06-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-07-31"),
        (f"{year}-08-01", f"{year}-08-31"),
        (f"{year}-09-01", f"{year}-09-30"),
        (f"{year}-10-01", f"{year}-10-06"),
    ]

    dfs = []
    for start, end in months:
        print(f"  Fetching {start} to {end}...")
        try:
            chunk = statcast(start_dt=start, end_dt=end)
            if chunk is not None and not chunk.empty:
                dfs.append(chunk)
        except Exception as e:
            print(f"  Warning: {e}")
            continue

    if not dfs:
        return pd.DataFrame()

    raw = pd.concat(dfs, ignore_index=True)
    print(f"[backtest_gl] Total pitches fetched: {len(raw)}")

    # Filter to rows that have an event (end of PA)
    pa_rows = raw[raw["events"].notna()].copy()
    pa_rows["homered"] = pa_rows["events"] == "home_run"

    # Aggregate to batter-game level
    game_level = pa_rows.groupby(
        ["game_date", "batter", "home_team", "away_team"]
    ).agg(
        homered=("homered", "any"),
        batter_name=("player_name", "first"),
        n_pa=("events", "count"),
    ).reset_index()

    game_level.rename(columns={"batter": "batter_id"}, inplace=True)
    game_level["homered"] = game_level["homered"].astype(bool)

    print(f"[backtest_gl] Aggregated to {len(game_level)} batter-game rows, "
          f"{game_level['homered'].sum()} HR events")

    _save_cache(cache_key, game_level)
    return game_level


def _find_batter_in_stats(batter_name_lastfirst, batter_id, batter_df):
    """
    Find a batter row in the stats DataFrame.
    Statcast uses 'Last, First' format; stats df uses 'First Last'.
    """
    if batter_df.empty:
        return None

    name_col = "Name" if "Name" in batter_df.columns else None
    if name_col is None:
        return None

    # Convert "Last, First" to "First Last"
    if isinstance(batter_name_lastfirst, str) and ", " in batter_name_lastfirst:
        parts = batter_name_lastfirst.split(", ", 1)
        search_name = f"{parts[1]} {parts[0]}"
    else:
        search_name = str(batter_name_lastfirst)

    # Try exact match
    rows = batter_df[batter_df[name_col] == search_name]
    if not rows.empty:
        return rows.iloc[0]

    # Try mlbID match
    if "mlbID" in batter_df.columns and batter_id:
        rows = batter_df[batter_df["mlbID"] == int(batter_id)]
        if not rows.empty:
            return rows.iloc[0]

    # Try last name match
    last_name = search_name.split()[-1] if search_name else ""
    if last_name:
        rows = batter_df[batter_df[name_col].str.contains(last_name, na=False)]
        if len(rows) == 1:
            return rows.iloc[0]

    return None


def run_game_level_backtest(years=None):
    """
    Game-level backtest using Statcast pitch data.
    For each batter-game in the year:
      1. Look up batter's prior-year season stats (no lookahead)
      2. Build features and score using the model
      3. Compare to actual HR outcome
      4. Track ROI by confidence tier
    """
    if years is None:
        years = [2024]

    from data_fetcher import get_batter_statcast, get_pitcher_statcast

    BET_SIZES = {"STRONG": 10.0, "STANDARD": 5.0, "SPECULATIVE": 1.0}
    PAYOUT = 2.0  # single-leg binary payout for ROI tracking

    results = {
        tier: {"bets": 0, "wins": 0, "wagered": 0.0, "returned": 0.0}
        for tier in BET_SIZES
    }
    total_scored = 0

    for year in years:
        print(f"\n[backtest_gl] === Processing {year} ===")

        # Fetch game-level HR data
        game_data = _fetch_game_level_hr_data(year)
        if game_data.empty:
            print(f"[backtest_gl] No data for {year}, skipping.")
            continue

        # Load prior year stats to avoid lookahead bias
        print(f"[backtest_gl] Loading {year-1} season stats for features...")
        batter_stats = get_batter_statcast(season=year - 1)
        pitcher_stats = get_pitcher_statcast(season=year - 1)

        if batter_stats.empty:
            print(f"[backtest_gl] No batter stats for {year-1}, skipping.")
            continue

        print(f"[backtest_gl] {len(game_data)} batter-game rows to score...")

        # Build feature dicts for all rows in bulk
        feature_dicts = []
        actuals = []
        skipped = 0

        for _, row in game_data.iterrows():
            b_row = _find_batter_in_stats(
                row.get("batter_name", ""), row.get("batter_id"), batter_stats
            )
            if b_row is None:
                skipped += 1
                continue

            feats = _extract_batter_features(b_row)

            # Use league-average pitcher features (no per-game matchup in bulk mode)
            feats["pitcher_hr_fb"] = 0.11
            feats["pitcher_fly_ball_rate"] = 0.34
            feats["pitcher_xfip"] = 4.20
            feats["pitcher_hard_hit_allowed"] = 0.35

            # Neutral context
            feats["park_hr_factor"] = 1.0
            feats["wind_bonus"] = 0.0
            feats["temp_bonus"] = 0.0
            feats["batting_order_position"] = 0.0
            feats["platoon_factor"] = 1.0

            # Add identity for the scorer
            feats["name"] = str(row.get("batter_name", ""))
            feats["team"] = str(row.get("home_team", ""))
            feats["opponent_pitcher"] = ""
            feats["game_id"] = str(row.get("game_date", ""))

            feature_dicts.append(feats)
            actuals.append(bool(row["homered"]))

        print(f"[backtest_gl] Built features for {len(feature_dicts)} rows "
              f"(skipped {skipped} unmatched batters)")

        if not feature_dicts:
            continue

        # Score all batters using the existing scorer
        from scorer import score_batters
        scored = score_batters(feature_dicts)

        # Map results back to actuals and track ROI
        for i, s in enumerate(scored):
            # scored is sorted by score desc, but we need original order
            # Actually score_batters sorts — we need to match by index
            pass

        # Since score_batters sorts, re-match by name+game_id
        scored_lookup = {}
        for s in scored:
            key = (s["name"], s["game_id"])
            scored_lookup[key] = s

        for i, feats in enumerate(feature_dicts):
            key = (feats["name"], feats["game_id"])
            s = scored_lookup.get(key)
            if s is None:
                continue

            tier = s["tier"]
            actual = actuals[i]
            bet = BET_SIZES[tier]

            results[tier]["bets"] += 1
            results[tier]["wagered"] += bet
            if actual:
                results[tier]["wins"] += 1
                results[tier]["returned"] += bet * PAYOUT

            total_scored += 1

        print(f"[backtest_gl] Scored {total_scored} batter-game rows for {year}")

    # Calculate final metrics
    for tier in results:
        r = results[tier]
        if r["wagered"] > 0:
            r["roi_pct"] = round((r["returned"] - r["wagered"]) / r["wagered"] * 100, 1)
            r["win_rate"] = round(r["wins"] / r["bets"] * 100, 2) if r["bets"] > 0 else 0
            r["hr_rate"] = round(r["wins"] / r["bets"], 4) if r["bets"] > 0 else 0
        else:
            r["roi_pct"] = 0
            r["win_rate"] = 0
            r["hr_rate"] = 0

    # Save results
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out = {
        "mode": "game_level",
        "date": date.today().isoformat(),
        "years": years,
        "total_batter_games_scored": total_scored,
        "results_by_tier": results,
    }

    CFG.backtest_dir.mkdir(parents=True, exist_ok=True)
    outpath = CFG.backtest_dir / f"gamelevel_results_{date.today().isoformat()}.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2, default=_convert)

    print(f"\n[backtest_gl] Results saved to {outpath}")
    print("\n=== GAME-LEVEL BACKTEST RESULTS ===")
    print(f"Total batter-games scored: {total_scored}")
    for tier, r in results.items():
        print(f"  {tier}: {r['bets']} bets, "
              f"{r['win_rate']}% hit rate ({r['wins']}/{r['bets']}), "
              f"ROI: {r['roi_pct']}%")

    # League avg HR rate for context
    total_hrs = sum(r["wins"] for r in results.values())
    total_bets = sum(r["bets"] for r in results.values())
    if total_bets > 0:
        print(f"\n  Overall HR rate: {total_hrs}/{total_bets} = "
              f"{total_hrs/total_bets*100:.2f}%")
        print(f"  (MLB avg ~3.2% of PA result in HR)")

    return out


# ---------------------------------------------------------------------------
# Totals Outcome Backtest
# ---------------------------------------------------------------------------

def _fetch_season_scores(year):
    """
    Fetch all regular-season game scores for a year via MLB Stats API.
    Returns list of dicts: game_id, home_team, away_team, stadium,
        home_pitcher_name, away_pitcher_name, home_score, away_score, total_runs
    Cached per year.
    """
    import requests
    from data_fetcher import _load_cache, _save_cache

    cache_key = f"season_scores_{year}"
    cached = _load_cache(cache_key)
    if cached is not None:
        print(f"[backtest_totals] Using cached scores for {year} ({len(cached)} games)")
        return cached

    # Pull month by month to avoid timeouts
    all_games = []
    for month in range(3, 11):  # March through October
        start = f"{year}-{month:02d}-01"
        if month == 10:
            end = f"{year}-10-10"
        elif month in (4, 6, 9):
            end = f"{year}-{month:02d}-30"
        else:
            end = f"{year}-{month:02d}-31"

        url = (
            f"https://statsapi.mlb.com/api/v1/schedule"
            f"?sportId=1&startDate={start}&endDate={end}"
            f"&gameType=R&hydrate=linescore,probablePitcher,venue"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  Warning: failed to fetch {start} to {end}: {e}")
            continue

        for d in data.get("dates", []):
            for g in d.get("games", []):
                ls = g.get("linescore", {})
                teams_ls = ls.get("teams", {})
                home_runs = teams_ls.get("home", {}).get("runs")
                away_runs = teams_ls.get("away", {}).get("runs")

                if home_runs is None or away_runs is None:
                    continue  # game not completed

                hp = g["teams"]["home"].get("probablePitcher", {})
                ap = g["teams"]["away"].get("probablePitcher", {})
                venue = g.get("venue", {}).get("name", "")

                all_games.append({
                    "game_id": g["gamePk"],
                    "home_team": g["teams"]["home"]["team"]["name"],
                    "away_team": g["teams"]["away"]["team"]["name"],
                    "stadium": venue,
                    "home_pitcher_name": hp.get("fullName", "TBD"),
                    "away_pitcher_name": ap.get("fullName", "TBD"),
                    "home_score": int(home_runs),
                    "away_score": int(away_runs),
                    "total_runs": int(home_runs) + int(away_runs),
                })

    print(f"[backtest_totals] Fetched {len(all_games)} completed games for {year}")
    _save_cache(cache_key, all_games)
    return all_games


def _precompute_game_features(games, batter_df, pitcher_df):
    """
    Pre-compute raw feature values for each game (expensive lookups done once).
    Returns list of dicts with: fip_dev, wrc_dev, park_adj, total_runs.
    The optimizer then just does: base + fip_dev*w1 + wrc_dev*w2 + park_adj.
    """
    from market_totals import (
        _find_pitcher_row, _get_stat, _find_team_wrc_plus,
        PARK_RUN_FACTORS, _BASE_RUNS,
    )

    features = []
    for g in games:
        hp_row = _find_pitcher_row(g.get("home_pitcher_name"), pitcher_df)
        ap_row = _find_pitcher_row(g.get("away_pitcher_name"), pitcher_df)

        home_fip = _get_stat(hp_row, ["FIP"], 4.20)
        away_fip = _get_stat(ap_row, ["FIP"], 4.20)
        fip_dev = (home_fip - 4.20) + (away_fip - 4.20)

        home_wrc = _find_team_wrc_plus(g.get("home_team", ""), batter_df)
        away_wrc = _find_team_wrc_plus(g.get("away_team", ""), batter_df)
        wrc_dev = (home_wrc - 100) + (away_wrc - 100)

        park_rf = PARK_RUN_FACTORS.get(g.get("stadium", ""), 1.0)
        park_adj = (park_rf - 1.0) * _BASE_RUNS

        features.append({
            "fip_dev": fip_dev,
            "wrc_dev": wrc_dev,
            "park_adj": park_adj,
            "total_runs": g["total_runs"],
        })

    return features


def _fast_accuracy(features, fip_w, wrc_w, base_runs):
    """Fast accuracy computation using pre-computed features (no lookups)."""
    better = 0
    total = len(features)
    for f in features:
        model = base_runs + f["fip_dev"] * fip_w + f["wrc_dev"] * wrc_w + f["park_adj"]
        actual = f["total_runs"]
        baseline_err = abs(base_runs - actual)
        model_err = abs(model - actual)
        if model_err < baseline_err:
            better += 1
    return better, total


def run_totals_backtest(years=None):
    """
    Totals outcome backtest: test model's prediction accuracy on historical game scores.
    Uses prior-year stats (no lookahead). Optimizes weights on train years, tests on holdout.
    """
    if years is None:
        years = [2021, 2022, 2023, 2024]

    from data_fetcher import get_batter_statcast, get_pitcher_statcast
    from market_totals import _BASE_RUNS

    print("[backtest_totals] === Totals Outcome Backtest ===")

    # --- Phase 1: Fetch scores and pre-compute features per year ---
    year_features = {}  # year -> list of feature dicts
    year_games_count = {}

    for year in years:
        print(f"\n[backtest_totals] Processing {year}...")
        games = _fetch_season_scores(year)
        if not games:
            print(f"[backtest_totals] No games for {year}, skipping.")
            continue

        print(f"  Loading {year-1} stats for model features...")
        batter_df = get_batter_statcast(season=year - 1)
        pitcher_df = get_pitcher_statcast(season=year - 1)

        print(f"  Pre-computing features for {len(games)} games...")
        feats = _precompute_game_features(games, batter_df, pitcher_df)
        year_features[year] = feats
        year_games_count[year] = len(feats)

        # Initial accuracy with default weights
        better, total = _fast_accuracy(feats, 0.5, 0.015, _BASE_RUNS)
        pct = better / total * 100 if total > 0 else 0
        print(f"  {year}: {total} games, model beats baseline in {better} ({pct:.1f}%)")

    if not year_features:
        print("[backtest_totals] No data available. Aborting.")
        return

    # --- Phase 2: Optimize weights on train set ---
    train_years = years[:-1]
    holdout_year = years[-1]

    # Combine train features into one list
    train_feats = []
    for y in train_years:
        if y in year_features:
            train_feats.extend(year_features[y])

    print(f"\n[backtest_totals] Optimizing on {len(train_feats)} train games "
          f"({train_years}), holdout: {holdout_year}")

    def _objective(params):
        better, total = _fast_accuracy(train_feats, params[0], params[1], _BASE_RUNS)
        return -(better / max(total, 1))

    initial = [0.5, 0.015]
    result = minimize(_objective, initial, method="Nelder-Mead",
                      options={"maxiter": 1000, "xatol": 1e-5})

    opt_fip_w = round(float(result.x[0]), 4)
    opt_wrc_w = round(float(result.x[1]), 6)
    train_acc = -result.fun * 100

    print(f"[backtest_totals] Optimized train accuracy: {train_acc:.1f}%")
    print(f"[backtest_totals] Calibrated: pitcher_fip_weight={opt_fip_w}, wrc_weight={opt_wrc_w}")

    optimized = {
        "pitcher_fip_weight": opt_fip_w,
        "wrc_weight": opt_wrc_w,
        "wind_out_bonus": 0.5,
        "wind_in_penalty": 0.3,
        "temp_hot_bonus": 0.2,
        "temp_cold_penalty": 0.4,
    }

    # --- Phase 3: Score each year with optimized weights ---
    year_results = {}
    for year in years:
        if year not in year_features:
            continue
        feats = year_features[year]
        better, total = _fast_accuracy(feats, opt_fip_w, opt_wrc_w, _BASE_RUNS)
        pct = better / total * 100 if total > 0 else 0
        year_results[year] = {
            "games": total,
            "model_better_than_baseline": better,
            "accuracy_pct": round(pct, 1),
        }

    # --- Save weights ---
    weights_path = CFG.base_dir / "weights_totals.json"
    with open(weights_path, "w") as f:
        json.dump(optimized, f, indent=2)
    print(f"\n[backtest_totals] Saved calibrated weights to {weights_path}")

    # --- Save full results ---
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    out = {
        "mode": "totals_backtest",
        "date": date.today().isoformat(),
        "years": years,
        "train_years": train_years,
        "holdout_year": holdout_year,
        "train_accuracy_pct": round(train_acc, 1),
        "optimized_weights": optimized,
        "year_results": {str(y): r for y, r in year_results.items()},
    }

    outpath = CFG.backtest_dir / f"totals_backtest_{date.today().isoformat()}.json"
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2, default=_convert)

    # --- Print summary table ---
    print(f"\n{'='*50}")
    print("=== TOTALS BACKTEST RESULTS ===")
    print(f"{'='*50}")
    print(f"{'Year':<6} {'Games':<7} {'Beats Base':<12} {'Accuracy':<10}")
    print("-" * 40)
    total_better_all = 0
    total_games_all = 0
    for y in years:
        if y in year_results:
            r = year_results[y]
            g = r["games"]
            b = r["model_better_than_baseline"]
            a = r["accuracy_pct"]
            marker = " <-- holdout" if y == holdout_year else ""
            total_better_all += b
            total_games_all += g
            print(f"{y:<6} {g:<7} {b:<12} {a:.1f}%{marker}")

    overall = total_better_all / total_games_all * 100 if total_games_all > 0 else 0
    print("-" * 40)
    print(f"{'Total':<6} {total_games_all:<7} {total_better_all:<12} {overall:.1f}%")
    print()
    print(f"Calibrated weights saved to {weights_path}")
    above54 = overall >= 54.0
    print(f"Key finding: model {'IS' if above54 else 'is NOT'} above 54% threshold")
    print(f"[backtest_totals] Done.")

    return out
