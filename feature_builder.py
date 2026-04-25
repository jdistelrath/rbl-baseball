"""
Module 3: Feature Builder.
Transforms raw DataFrames into clean feature dicts per batter-game matchup.
"""

import math
import pandas as pd

from config import CFG
import data_fetcher as df


def _fuzzy_match_name(name, df_col):
    """Try to match player name in a DataFrame column. Returns best match or None."""
    if name in df_col.values:
        return name
    # Try last name match
    last = name.split()[-1] if name else ""
    matches = [n for n in df_col.values if isinstance(n, str) and last in n]
    if len(matches) == 1:
        return matches[0]
    return None


def _get_batter_row(batter_name, batter_df):
    """Find a batter's row in the batting stats DataFrame."""
    if batter_df.empty:
        return None
    name_col = None
    for col in ("Name", "name", "PlayerName"):
        if col in batter_df.columns:
            name_col = col
            break
    if name_col is None:
        return None

    matched = _fuzzy_match_name(batter_name, batter_df[name_col])
    if matched is None:
        return None
    rows = batter_df[batter_df[name_col] == matched]
    if rows.empty:
        return None
    return rows.iloc[0]


def _get_pitcher_row(pitcher_name, pitcher_df):
    """Find a pitcher's row in the pitching stats DataFrame."""
    if pitcher_df.empty:
        return None
    name_col = None
    for col in ("Name", "name", "PlayerName"):
        if col in pitcher_df.columns:
            name_col = col
            break
    if name_col is None:
        return None

    matched = _fuzzy_match_name(pitcher_name, pitcher_df[name_col])
    if matched is None:
        return None
    rows = pitcher_df[pitcher_df[name_col] == matched]
    if rows.empty:
        return None
    return rows.iloc[0]


def _safe_float(val, default=None):
    """Convert a value to float, returning default if impossible."""
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def _compute_platoon_factor(bat_side, pitcher_throws):
    """
    Platoon advantage factor.
    Favorable: LHB vs RHP or RHP vs LHP -> 1.15
    Unfavorable: same side -> 0.85
    Switch hitter or unknown: 1.0
    """
    if not bat_side or not pitcher_throws:
        return 1.0
    if bat_side == "S":  # switch hitter
        return 1.05  # slight advantage, they always have the platoon edge
    if bat_side != pitcher_throws:
        return 1.15  # favorable
    return 0.85  # unfavorable


def _compute_wind_bonus(weather, stadium_name):
    """
    Wind bonus based on speed and direction relative to typical CF orientation.
    Blowing out (roughly from home plate toward outfield) at >10mph: +0.05 to +0.15
    Blowing in: -0.05 to -0.10
    Crosswind or <5mph: 0
    """
    speed = weather.get("wind_speed_mph", 0)
    deg = weather.get("wind_dir_degrees", 0)

    if speed < 5:
        return 0.0

    # Most MLB parks face roughly NE-E from home plate.
    # Wind blowing OUT = wind coming FROM S/SW (180-240 deg range broadly)
    # Wind blowing IN = wind FROM N/NE (0-60 deg range)
    # This is a simplification; ideally per-park orientation would be used.

    # Normalize: "out" directions (blowing toward outfield, i.e. from behind home plate)
    # Treat 180-270 as "out" direction, 0-90 as "in"
    if 150 <= deg <= 270:
        # Blowing out
        factor = min(0.15, max(0.05, (speed - 10) * 0.02 + 0.05)) if speed >= 10 else 0.02
        return factor
    elif deg <= 60 or deg >= 330:
        # Blowing in
        factor = max(-0.10, min(-0.05, -(speed - 5) * 0.01 - 0.05)) if speed >= 5 else -0.02
        return factor
    else:
        # Crosswind
        return 0.0


def _compute_temp_bonus(weather):
    """Temp >80F: +0.03, <45F: -0.03, else 0."""
    temp = weather.get("temp_f", 70)
    if temp > 80:
        return 0.03
    elif temp < 45:
        return -0.03
    return 0.0


def _batting_order_feature(order_pos):
    """
    Batting order feature: 1-4 = 0.05 bonus, 5-6 = 0, 7-9 = -0.05 penalty.
    """
    if order_pos is None:
        return 0.0
    if order_pos <= 4:
        return 0.05
    elif order_pos <= 6:
        return 0.0
    else:
        return -0.05


# ---------------------------------------------------------------------------
# Position-average defaults (used when player data is missing)
# ---------------------------------------------------------------------------

_POSITION_AVG = {
    "barrel_rate": 0.065,
    "hr_fb_ratio": 0.12,
    "iso": 0.155,
    "hard_hit_pct": 0.35,
    "pitcher_hr_fb": 0.11,
    "pitcher_fly_ball_rate": 0.34,
    "pitcher_xfip": 4.20,
    "pitcher_hard_hit_allowed": 0.35,
}


def build_features_for_game(game, lineup, batter_df, pitcher_df, weather):
    """
    Build feature dicts for all batters in a confirmed lineup for a single game.

    Args:
        game: dict from get_today_schedule()
        lineup: dict from get_confirmed_lineups() {home: [...], away: [...]}
        batter_df: DataFrame from get_batter_statcast()
        pitcher_df: DataFrame from get_pitcher_statcast()
        weather: dict from get_weather()

    Returns:
        list of feature dicts, one per batter
    """
    features_list = []
    park_factor = df.get_park_factor_for_stadium(game["stadium"])

    wind_bonus = _compute_wind_bonus(weather, game["stadium"])
    temp_bonus = _compute_temp_bonus(weather)

    # Process each side: away batters face home pitcher, home batters face away pitcher
    for side, pitcher_key, pitcher_name_key in [
        ("away", "home_pitcher_id", "home_pitcher_name"),
        ("home", "away_pitcher_id", "away_pitcher_name"),
    ]:
        batters = lineup.get(side, [])
        opp_pitcher_name = game.get(pitcher_name_key, "TBD")
        opp_pitcher_id = game.get(pitcher_key)
        pitcher_throws = _get_pitcher_hand_from_df(opp_pitcher_name, pitcher_df)

        # Get pitcher stats
        p_row = _get_pitcher_row(opp_pitcher_name, pitcher_df)
        pitcher_hr_fb = _safe_float(
            _get_stat(p_row, ["HR/FB", "HR_FB", "HR/FB%"]),
            _POSITION_AVG["pitcher_hr_fb"]
        )
        pitcher_fb_rate = _safe_float(
            _get_stat(p_row, ["FB%", "FB_pct", "FBpct"]),
            _POSITION_AVG["pitcher_fly_ball_rate"]
        )
        # FB% often comes as percentage (e.g., 35.0 for 35%), normalize to 0-1
        if pitcher_fb_rate > 1:
            pitcher_fb_rate /= 100.0
        pitcher_xfip = _safe_float(
            _get_stat(p_row, ["xFIP"]),
            _POSITION_AVG["pitcher_xfip"]
        )
        pitcher_hh = _safe_float(
            _get_stat(p_row, ["Hard%", "HardHit%", "Hard_pct"]),
            _POSITION_AVG["pitcher_hard_hit_allowed"]
        )
        if pitcher_hh > 1:
            pitcher_hh /= 100.0

        # HR/FB% normalization
        if pitcher_hr_fb > 1:
            pitcher_hr_fb /= 100.0

        for batter in batters:
            b_row = _get_batter_row(batter["name"], batter_df)

            barrel_rate = _safe_float(
                _get_stat(b_row, ["Barrel%", "barrel_rate", "Barrel_pct"]),
                _POSITION_AVG["barrel_rate"]
            )
            if barrel_rate > 1:
                barrel_rate /= 100.0

            hr_fb = _safe_float(
                _get_stat(b_row, ["HR/FB", "HR_FB"]),
                _POSITION_AVG["hr_fb_ratio"]
            )
            if hr_fb > 1:
                hr_fb /= 100.0

            iso = _safe_float(
                _get_stat(b_row, ["ISO"]),
                _POSITION_AVG["iso"]
            )

            hard_hit = _safe_float(
                _get_stat(b_row, ["Hard%", "HardHit%", "Hard_pct"]),
                _POSITION_AVG["hard_hit_pct"]
            )
            if hard_hit > 1:
                hard_hit /= 100.0

            bat_side = batter.get("bat_side", "R")
            if not pitcher_throws:
                pitcher_throws = "R"
            platoon = _compute_platoon_factor(bat_side, pitcher_throws)

            order_bonus = _batting_order_feature(batter.get("batting_order"))

            features = {
                # Identity
                "name": batter["name"],
                "team": game["home_team"] if side == "home" else game["away_team"],
                "opponent_pitcher": opp_pitcher_name,
                "game_id": game["game_id"],
                "side": side,
                "batting_order": batter.get("batting_order"),
                "stadium": game["stadium"],

                # Batter features
                "barrel_rate": barrel_rate,
                "hr_fb_ratio": hr_fb,
                "iso": iso,
                "hard_hit_pct": hard_hit,
                "platoon_factor": platoon,

                # Pitcher features
                "pitcher_hr_fb": pitcher_hr_fb,
                "pitcher_fly_ball_rate": pitcher_fb_rate,
                "pitcher_xfip": pitcher_xfip,
                "pitcher_hard_hit_allowed": pitcher_hh,

                # Context features
                "park_hr_factor": park_factor,
                "wind_bonus": wind_bonus,
                "temp_bonus": temp_bonus,
                "batting_order_position": order_bonus,

                # Metadata for key_edge generation
                "_bat_side": bat_side,
                "_pitcher_throws": pitcher_throws,
                "_wind_speed": weather.get("wind_speed_mph", 0),
                "_wind_dir": weather.get("wind_dir_label", "calm"),
                "_temp_f": weather.get("temp_f", 70),
            }
            features_list.append(features)

    return features_list


def _get_stat(row, col_names):
    """Try multiple column names to extract a stat from a DataFrame row."""
    if row is None:
        return None
    for col in col_names:
        if col in row.index:
            return row[col]
    return None


def _get_pitcher_hand_from_df(pitcher_name, pitcher_df):
    """Try to get pitcher throwing hand from the stats DataFrame."""
    if pitcher_df.empty:
        return "R"
    row = _get_pitcher_row(pitcher_name, pitcher_df)
    if row is not None:
        for col in ("Throws", "throws", "ThrowHand"):
            if col in row.index:
                return str(row[col])
    # Fallback: fetch from MLB API
    return "R"
