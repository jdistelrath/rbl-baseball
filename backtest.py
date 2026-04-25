"""
Module 6: Backtest.
Walk-forward validation harness. Run manually only -- not on cron.

Usage: python main.py backtest
"""

import json
import math
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
    """Load batting stats for a given year via pybaseball."""
    try:
        from pybaseball import batting_stats
        df = batting_stats(year, qual=50)
        return df
    except Exception as e:
        print(f"[backtest] Failed to load batting stats for {year}: {e}")
        return pd.DataFrame()


def _load_historical_pitching(year):
    """Load pitching stats for a given year via pybaseball."""
    try:
        from pybaseball import pitching_stats
        df = pitching_stats(year, qual=20)
        return df
    except Exception as e:
        print(f"[backtest] Failed to load pitching stats for {year}: {e}")
        return pd.DataFrame()


def _load_hr_actuals(year):
    """
    Load actual HR totals per player for a season using pybaseball.
    Returns dict {player_name: total_HR}.
    """
    try:
        from pybaseball import batting_stats
        df = batting_stats(year, qual=1)
        name_col = "Name" if "Name" in df.columns else df.columns[0]
        hr_col = "HR" if "HR" in df.columns else None
        if hr_col is None:
            return {}
        return dict(zip(df[name_col], df[hr_col]))
    except Exception as e:
        print(f"[backtest] Failed to load HR actuals for {year}: {e}")
        return {}


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
    """Extract feature vector from a batting_stats row."""
    barrel = _get_stat(batter_row, ["Barrel%", "barrel_rate"], 6.5)
    if barrel > 1:
        barrel /= 100.0

    hr_fb = _get_stat(batter_row, ["HR/FB", "HR_FB"], 12.0)
    if hr_fb > 1:
        hr_fb /= 100.0

    iso = _get_stat(batter_row, ["ISO"], 0.155)

    hard_hit = _get_stat(batter_row, ["Hard%", "HardHit%"], 35.0)
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
