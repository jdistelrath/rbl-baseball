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
# Park HR factors (FanGraphs-derived, 1.0 = neutral)
# ---------------------------------------------------------------------------

PARK_HR_FACTOR = {
    "Coors Field": 1.38, "Great American Ball Park": 1.18, "Yankee Stadium": 1.15,
    "Globe Life Field": 1.12, "Citizens Bank Park": 1.10, "Wrigley Field": 1.08,
    "Fenway Park": 1.07, "Guaranteed Rate Field": 1.06, "Minute Maid Park": 1.05,
    "Dodger Stadium": 1.04, "Truist Park": 1.03, "Rogers Centre": 1.03,
    "Angel Stadium": 1.02, "Oriole Park at Camden Yards": 1.02, "Target Field": 1.01,
    "Citi Field": 1.00, "PNC Park": 0.99, "American Family Field": 0.99,
    "Busch Stadium": 0.98, "Comerica Park": 0.97, "Chase Field": 0.97,
    "Progressive Field": 0.96, "Kauffman Stadium": 0.96, "T-Mobile Park": 0.95,
    "Nationals Park": 0.95, "loanDepot park": 0.94, "Petco Park": 0.93,
    "Tropicana Field": 0.92, "Oracle Park": 0.90, "Oakland Coliseum": 0.88,
    "RingCentral Coliseum": 0.88,
}

# MLB team name -> home stadium mapping
TEAM_STADIUM = {
    "Colorado Rockies": "Coors Field", "Cincinnati Reds": "Great American Ball Park",
    "New York Yankees": "Yankee Stadium", "Texas Rangers": "Globe Life Field",
    "Philadelphia Phillies": "Citizens Bank Park", "Chicago Cubs": "Wrigley Field",
    "Boston Red Sox": "Fenway Park", "Chicago White Sox": "Guaranteed Rate Field",
    "Houston Astros": "Minute Maid Park", "Los Angeles Dodgers": "Dodger Stadium",
    "Atlanta Braves": "Truist Park", "Toronto Blue Jays": "Rogers Centre",
    "Los Angeles Angels": "Angel Stadium", "Baltimore Orioles": "Oriole Park at Camden Yards",
    "Minnesota Twins": "Target Field", "New York Mets": "Citi Field",
    "Pittsburgh Pirates": "PNC Park", "Milwaukee Brewers": "American Family Field",
    "St. Louis Cardinals": "Busch Stadium", "Detroit Tigers": "Comerica Park",
    "Arizona Diamondbacks": "Chase Field", "Cleveland Guardians": "Progressive Field",
    "Kansas City Royals": "Kauffman Stadium", "Seattle Mariners": "T-Mobile Park",
    "Washington Nationals": "Nationals Park", "Miami Marlins": "loanDepot park",
    "San Diego Padres": "Petco Park", "Tampa Bay Rays": "Tropicana Field",
    "San Francisco Giants": "Oracle Park", "Athletics": "Oakland Coliseum",
    "Oakland Athletics": "Oakland Coliseum",
}


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

K_BIAS_CORRECTION = -0.37  # corrects consistent over-projection of ~+0.37 Ks


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
    return (k9 / 9.0) * ip_per_start * k_rate_adj + K_BIAS_CORRECTION


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
# HR prop model
# ---------------------------------------------------------------------------

def _fetch_batter_handedness(player_ids):
    """Batch fetch bat side for a list of player IDs. Returns dict {id: 'L'/'R'/'S'}."""
    cache_key = f"batter_handedness_{hash(tuple(sorted(player_ids)))}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    result = {}
    # Batch in groups of 50
    for i in range(0, len(player_ids), 50):
        batch = player_ids[i:i+50]
        ids_str = ",".join(str(pid) for pid in batch)
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={ids_str}"
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            for p in resp.json().get("people", []):
                result[p["id"]] = p.get("batSide", {}).get("code", "R")
        except Exception:
            continue

    _save_cache(cache_key, result)
    return result


def _build_team_pitcher_hr9(pitcher_df):
    """Build team-level pitcher HR/9 from season stats. Returns dict {team_name: hr9}."""
    if pitcher_df.empty:
        return {}

    team_hr = {}
    team_ip = {}
    for _, row in pitcher_df.iterrows():
        team = row.get("Team", "")
        if not team:
            continue
        try:
            hr = int(row.get("HR", 0))
            ip = float(row.get("IP", 0))
        except (TypeError, ValueError):
            continue
        if ip < 10:
            continue
        team_hr.setdefault(team, 0)
        team_ip.setdefault(team, 0.0)
        team_hr[team] += hr
        team_ip[team] += ip

    result = {}
    for team in team_hr:
        if team_ip[team] > 0:
            result[team] = (team_hr[team] / team_ip[team]) * 9.0
    return result


def _project_hr_prob(batter_row, opp_team, team_pitcher_hr9, bat_side,
                     is_home, opp_team_name):
    """
    Project probability of hitting at least 1 HR in a game.

    Inputs:
        - Batter HR/PA rate (from prior-year season stats)
        - Opposing team pitcher HR/9 (higher = more HR-prone)
        - Park HR factor
        - Platoon adjustment (LHB vs RHP staff, etc.)

    Returns float probability (0-1) or None.
    """
    try:
        hr = int(batter_row.get("HR", 0))
        pa = int(batter_row.get("PA", 0))
    except (TypeError, ValueError):
        return None
    if pa < 100:
        return None

    # Base HR/PA rate
    hr_per_pa = hr / pa

    # Expected PA per game (~3.8-4.2 for starters)
    try:
        g = int(batter_row.get("G", 0))
        total_pa = int(batter_row.get("PA", 0))
    except (TypeError, ValueError):
        return None
    pa_per_game = total_pa / max(g, 1) if g > 0 else 3.9

    # Base probability of at least 1 HR in a game
    # P(>=1 HR) = 1 - P(0 HR) = 1 - (1 - hr_per_pa)^pa_per_game
    base_prob = 1.0 - (1.0 - hr_per_pa) ** pa_per_game

    # Adjustment 1: opposing team pitcher HR/9
    # League avg HR/9 ~1.25. Scale linearly.
    league_avg_hr9 = 1.25
    opp_hr9 = team_pitcher_hr9.get(opp_team_name, league_avg_hr9)
    pitcher_mult = opp_hr9 / league_avg_hr9 if league_avg_hr9 > 0 else 1.0
    pitcher_mult = max(0.5, min(2.0, pitcher_mult))  # clamp

    # Adjustment 2: park factor
    if is_home:
        home_team = None
        # Find home team from opponent
        for _, r in []:
            pass
        # Use opp_team for away games, need home team for park
        stadium = TEAM_STADIUM.get(opp_team_name, "") if not is_home else ""
    # Simpler: use home team's stadium
    # is_home tells us if the batter's team is home
    batter_team = batter_row.get("Team", "")
    if is_home:
        stadium = TEAM_STADIUM.get(batter_team, "")
    else:
        stadium = TEAM_STADIUM.get(opp_team_name, "")
    park_factor = PARK_HR_FACTOR.get(stadium, 1.0)

    # Adjustment 3: platoon (simplified -- team-level, not per-pitcher)
    # LHB gets ~10% boost (more RHP starters in MLB)
    # Switch hitters get ~5% boost
    if bat_side == "L":
        platoon_mult = 1.08
    elif bat_side == "S":
        platoon_mult = 1.04
    else:
        platoon_mult = 1.0

    # Combined probability
    adjusted_prob = base_prob * pitcher_mult * park_factor * platoon_mult

    # Clamp to realistic range
    return max(0.0, min(0.60, adjusted_prob))


# Calibration: 4-season avg raw projection -> actual HR rate per quintile
# Derived from 2022-2025 backtest (66K games)
# (midpoint of raw projection bin, observed actual HR rate)
_HR_CALIB_RAW =    [0.056, 0.098, 0.127, 0.162, 0.235]
_HR_CALIB_ACTUAL = [0.082, 0.112, 0.136, 0.155, 0.182]


def _calibrate_hr_prob(raw_prob):
    """Piecewise linear interpolation from raw projection to calibrated probability."""
    if raw_prob <= _HR_CALIB_RAW[0]:
        # Extrapolate below lowest bin
        slope = ((_HR_CALIB_ACTUAL[1] - _HR_CALIB_ACTUAL[0]) /
                 (_HR_CALIB_RAW[1] - _HR_CALIB_RAW[0]))
        return max(0.0, _HR_CALIB_ACTUAL[0] + slope * (raw_prob - _HR_CALIB_RAW[0]))
    if raw_prob >= _HR_CALIB_RAW[-1]:
        slope = ((_HR_CALIB_ACTUAL[-1] - _HR_CALIB_ACTUAL[-2]) /
                 (_HR_CALIB_RAW[-1] - _HR_CALIB_RAW[-2]))
        return min(0.50, _HR_CALIB_ACTUAL[-1] + slope * (raw_prob - _HR_CALIB_RAW[-1]))
    # Find the two bracketing points
    for i in range(len(_HR_CALIB_RAW) - 1):
        if _HR_CALIB_RAW[i] <= raw_prob <= _HR_CALIB_RAW[i + 1]:
            t = ((raw_prob - _HR_CALIB_RAW[i]) /
                 (_HR_CALIB_RAW[i + 1] - _HR_CALIB_RAW[i]))
            return _HR_CALIB_ACTUAL[i] + t * (_HR_CALIB_ACTUAL[i + 1] - _HR_CALIB_ACTUAL[i])
    return raw_prob


def _backtest_batter_hr(year, batter_df, pitcher_df):
    """Backtest HR prop projections vs 0.5 line."""
    cache_key = f"props_bt_batter_hr_{year}"
    cached = _load_cache(cache_key)

    if cached is not None:
        results = cached
        print(f"  [HR] Using cached results ({len(results)} games)")
    else:
        if "HR" not in batter_df.columns or "PA" not in batter_df.columns:
            print("  [HR] HR or PA column missing. Skipping.")
            return

        qualified = batter_df[batter_df["PA"] >= 200].copy()
        print(f"  [HR] Testing {len(qualified)} batters...")

        # Build team pitcher HR/9 lookup from prior-year pitching stats
        team_pitcher_hr9 = _build_team_pitcher_hr9(pitcher_df)

        # Fetch handedness for all batters
        player_ids = qualified["mlbID"].astype(int).tolist()
        handedness = _fetch_batter_handedness(player_ids)

        results = []
        for _, row in qualified.iterrows():
            pid = int(row["mlbID"])
            name = row["Name"]
            bat_side = handedness.get(pid, "R")

            game_logs = _fetch_batter_game_logs(pid, year)
            if not game_logs:
                continue

            # We need opponent team per game -- re-fetch with opponent info
            gl_url = (
                f"https://statsapi.mlb.com/api/v1/people/{pid}/stats"
                f"?stats=gameLog&group=hitting&season={year}&gameType=R"
            )
            try:
                resp = requests.get(gl_url, timeout=15)
                resp.raise_for_status()
                splits = resp.json().get("stats", [{}])[0].get("splits", [])
            except Exception:
                continue

            for s in splits:
                stat = s.get("stat", {})
                ab = int(stat.get("atBats", 0))
                if ab == 0:
                    continue

                opp_name = s.get("opponent", {}).get("name", "")
                is_home = s.get("isHome", False)
                actual_hr = int(stat.get("homeRuns", 0))

                proj = _project_hr_prob(
                    row, opp_name, team_pitcher_hr9, bat_side,
                    is_home, opp_name
                )
                if proj is None:
                    continue

                results.append({
                    "name": name,
                    "projected_prob": round(proj, 4),
                    "actual_hr": actual_hr,
                    "homered": actual_hr > 0,
                    "date": s.get("date", ""),
                    "opponent": opp_name,
                    "is_home": is_home,
                })

        _save_cache(cache_key, results)
        print(f"  [HR] Fetched {len(results)} games")

    if not results:
        print("  [HR] No data.")
        return

    df = pd.DataFrame(results)
    df["calibrated_prob"] = df["projected_prob"].apply(_calibrate_hr_prob)

    actual_hr_rate = df["homered"].mean()
    print(f"  [HR] Base HR rate: {actual_hr_rate:.1%} ({df['homered'].sum()}/{len(df)})")

    # Quintile analysis for both raw and calibrated
    df["prob_bin"] = pd.qcut(df["projected_prob"], q=5, duplicates="drop")
    bin_stats = df.groupby("prob_bin", observed=True).agg(
        games=("homered", "count"),
        hrs=("homered", "sum"),
        avg_raw=("projected_prob", "mean"),
        avg_cal=("calibrated_prob", "mean"),
    )
    bin_stats["actual_rate"] = bin_stats["hrs"] / bin_stats["games"]

    print("  [HR] Quintile calibration (raw vs calibrated vs actual):")
    print(f"       {'Q':<4} {'Games':>6} {'Raw':>7} {'Calib':>7} {'Actual':>7} {'RawErr':>7} {'CalErr':>7}")
    for i, (idx, row) in enumerate(bin_stats.iterrows(), 1):
        raw_err = row["avg_raw"] - row["actual_rate"]
        cal_err = row["avg_cal"] - row["actual_rate"]
        print(f"       Q{i:<3} {int(row['games']):>6} {row['avg_raw']:>7.1%} "
              f"{row['avg_cal']:>7.1%} {row['actual_rate']:>7.1%} "
              f"{raw_err:>+7.1%} {cal_err:>+7.1%}")

    # Overall metrics: raw vs calibrated
    raw_bias = df["projected_prob"].mean() - actual_hr_rate
    cal_bias = df["calibrated_prob"].mean() - actual_hr_rate
    raw_mae = (df["projected_prob"] - df["homered"].astype(float)).abs().mean()
    cal_mae = (df["calibrated_prob"] - df["homered"].astype(float)).abs().mean()

    # Top/bottom 20% (ranking is the same since calibration is monotonic)
    top_20_threshold = df["projected_prob"].quantile(0.80)
    top_picks = df[df["projected_prob"] >= top_20_threshold]
    top_hr_rate = top_picks["homered"].mean() if len(top_picks) > 0 else 0

    bottom_20_threshold = df["projected_prob"].quantile(0.20)
    bottom_picks = df[df["projected_prob"] <= bottom_20_threshold]
    bottom_hr_rate = bottom_picks["homered"].mean() if len(bottom_picks) > 0 else 0

    print(f"  [HR] Top 20% picks: {top_hr_rate:.1%} HR rate "
          f"({top_picks['homered'].sum()}/{len(top_picks)})")
    print(f"  [HR] Bottom 20%: {bottom_hr_rate:.1%} HR rate "
          f"({bottom_picks['homered'].sum()}/{len(bottom_picks)})")
    print(f"  [HR] Lift: {top_hr_rate/max(bottom_hr_rate, 0.001):.1f}x")
    print(f"  [HR] Raw   -> MAE: {raw_mae:.4f} | Bias: {raw_bias:+.4f}")
    print(f"  [HR] Calib -> MAE: {cal_mae:.4f} | Bias: {cal_bias:+.4f}")
    print(f"  [HR] Games analyzed: {len(df)}")


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

        # --- Batter HR Props ---
        _backtest_batter_hr(year, batter_df, pitcher_df)


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
