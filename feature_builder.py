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
    if batter_df is None or batter_df.empty:
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

    # New v11 data pulls (cached + fail-safe)
    try:
        rolling_stats = df.get_batter_rolling_stats()  # {7: df, 14: df, 30: df}
    except Exception as e:
        print(f"[feature_builder] rolling stats unavailable: {e}")
        rolling_stats = {7: pd.DataFrame(), 14: pd.DataFrame(), 30: pd.DataFrame()}
    pool_medians = _compute_pool_medians(rolling_stats.get(14, pd.DataFrame()))

    # Process each side: away batters face home pitcher, home batters face away pitcher
    for side, pitcher_key, pitcher_name_key in [
        ("away", "home_pitcher_id", "home_pitcher_name"),
        ("home", "away_pitcher_id", "away_pitcher_name"),
    ]:
        batters = lineup.get(side, [])
        opp_pitcher_name = game.get(pitcher_name_key, "TBD")
        opp_pitcher_id = game.get(pitcher_key)
        pitcher_throws = _get_pitcher_hand_from_df(opp_pitcher_name, pitcher_df)

        # Pitcher-level recent form / pitch mix (fail-safe)
        try:
            pitcher_recent = df.get_pitcher_recent_starts(opp_pitcher_name)
        except Exception as e:
            print(f"[feature_builder] pitcher_recent failed for {opp_pitcher_name}: {e}")
            pitcher_recent = []
        try:
            pitch_mix = df.get_pitcher_pitch_mix(opp_pitcher_name)
        except Exception as e:
            print(f"[feature_builder] pitch_mix failed for {opp_pitcher_name}: {e}")
            pitch_mix = None

        recent_form = _summarize_pitcher_recent(pitcher_recent)
        pitch_mix_summary = _summarize_pitch_mix(pitch_mix)

        # Get pitcher stats
        p_row = _get_pitcher_row(opp_pitcher_name, pitcher_df)
        pitcher_hr_fb = _safe_float(
            _get_stat(p_row, ["pitcher_hr_fb", "HR/FB", "HR_FB", "HR/FB%"]),
            _POSITION_AVG["pitcher_hr_fb"]
        )
        pitcher_fb_rate = _safe_float(
            _get_stat(p_row, ["fly_ball_rate", "FB%", "FB_pct", "FBpct"]),
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
                _get_stat(b_row, ["barrel_rate", "Barrel%", "Barrel_pct"]),
                _POSITION_AVG["barrel_rate"]
            )
            if barrel_rate > 1:
                barrel_rate /= 100.0

            hr_fb = _safe_float(
                _get_stat(b_row, ["hr_fb_ratio", "HR/FB", "HR_FB"]),
                _POSITION_AVG["hr_fb_ratio"]
            )
            if hr_fb > 1:
                hr_fb /= 100.0

            iso = _safe_float(
                _get_stat(b_row, ["ISO"]),
                _POSITION_AVG["iso"]
            )

            hard_hit = _safe_float(
                _get_stat(b_row, ["hard_hit_pct", "Hard%", "HardHit%", "Hard_pct"]),
                _POSITION_AVG["hard_hit_pct"]
            )
            if hard_hit > 1:
                hard_hit /= 100.0

            bat_side = batter.get("bat_side", "R")
            if not pitcher_throws:
                pitcher_throws = "R"
            platoon = _compute_platoon_factor(bat_side, pitcher_throws)

            order_bonus = _batting_order_feature(batter.get("batting_order"))

            # --- v11: rolling form ---
            rolling = _lookup_rolling(batter["name"], rolling_stats)
            barrel_14d = rolling.get(14, {}).get("barrel_rate")
            hard_14d = rolling.get(14, {}).get("hard_hit_pct")
            hr_fb_14d = rolling.get(14, {}).get("hr_fb_ratio")
            barrel_7d = rolling.get(7, {}).get("barrel_rate")
            barrel_30d = rolling.get(30, {}).get("barrel_rate")

            barrel_rate_14d = barrel_14d if barrel_14d is not None else barrel_rate
            hard_hit_pct_14d = hard_14d if hard_14d is not None else hard_hit
            hr_fb_14d_v = hr_fb_14d if hr_fb_14d is not None else hr_fb

            if barrel_7d is not None and barrel_30d is not None and barrel_30d > 0:
                form_trend = (barrel_7d - barrel_30d) / barrel_30d
                form_trend = max(-1.0, min(1.0, form_trend))
            else:
                form_trend = 0.0

            # --- v11: H2H ---
            try:
                h2h = df.get_batter_pitcher_matchup(batter["name"], opp_pitcher_name)
            except Exception as e:
                print(f"[feature_builder] h2h failed for {batter['name']} vs {opp_pitcher_name}: {e}")
                h2h = None
            if h2h and h2h.get("pa", 0) >= 5:
                h2h_hr_rate = h2h["hr"] / max(h2h["pa"], 1)
                h2h_pa = h2h["pa"]
                h2h_hr = h2h["hr"]
            else:
                h2h_hr_rate = None
                h2h_pa = 0
                h2h_hr = 0

            # --- v11: pitcher recent form ---
            pitcher_recent_hr_per_ip = recent_form["recent_hr_per_ip"]
            pitcher_recent_hard_hit = recent_form["recent_hard_hit"]
            pitcher_high_workload = -0.05 if recent_form["high_workload"] else 0.0

            # --- v11: pitch type edge ---
            pitch_type_edge = _compute_pitch_type_edge(
                pitch_mix_summary,
                hard_hit_pct_14d,
                barrel_rate_14d,
                pool_medians,
            )

            # Track missing data fields per batter
            data_gaps = []
            if barrel_14d is None:
                data_gaps.append("barrel_rate_14d")
            if hard_14d is None:
                data_gaps.append("hard_hit_pct_14d")
            if not pitcher_recent:
                data_gaps.append("pitcher_recent")
            if pitch_mix is None:
                data_gaps.append("pitch_mix")
            if h2h_hr_rate is None:
                data_gaps.append("h2h")

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

                # v11 rolling form
                "barrel_rate_14d": barrel_rate_14d,
                "hard_hit_pct_14d": hard_hit_pct_14d,
                "hr_fb_14d": hr_fb_14d_v,
                "form_trend": form_trend,

                # v11 pitcher recent form
                "pitcher_recent_hr_per_ip": (
                    pitcher_recent_hr_per_ip
                    if pitcher_recent_hr_per_ip is not None
                    else pitcher_hr_fb
                ),
                "pitcher_recent_hard_hit": (
                    pitcher_recent_hard_hit
                    if pitcher_recent_hard_hit is not None
                    else pitcher_hh
                ),
                "pitcher_high_workload": pitcher_high_workload,

                # v11 pitch type edge
                "pitch_type_edge": pitch_type_edge,

                # v11 H2H scalar (handled separately by scorer; kept here for diagnostics)
                "h2h_hr_rate": h2h_hr_rate,

                # Metadata for key_edge generation
                "_bat_side": bat_side,
                "_pitcher_throws": pitcher_throws,
                "_wind_speed": weather.get("wind_speed_mph", 0),
                "_wind_dir": weather.get("wind_dir_label", "calm"),
                "_temp_f": weather.get("temp_f", 70),
                "_form_trend_raw": form_trend,
                "_h2h_pa": h2h_pa,
                "_h2h_hr": h2h_hr,
                "_recent_hr_per_ip_raw": pitcher_recent_hr_per_ip,
                "_primary_pitch": pitch_mix_summary.get("primary_pitch"),
                "data_gaps": data_gaps,
            }
            features_list.append(features)

    return features_list


# ---------------------------------------------------------------------------
# v11 helpers
# ---------------------------------------------------------------------------

def _compute_pool_medians(df_window):
    """Compute median hard_hit_pct and barrel_rate from a rolling window DataFrame."""
    medians = {"hard_hit_pct": 0.35, "barrel_rate": 0.065}
    if df_window is None or df_window.empty:
        return medians
    try:
        if "hard_hit_pct" in df_window.columns:
            v = pd.to_numeric(df_window["hard_hit_pct"], errors="coerce").dropna()
            if not v.empty:
                medians["hard_hit_pct"] = float(v.median())
        if "barrel_rate" in df_window.columns:
            v = pd.to_numeric(df_window["barrel_rate"], errors="coerce").dropna()
            if not v.empty:
                medians["barrel_rate"] = float(v.median())
    except Exception:
        pass
    return medians


def _lookup_rolling(batter_name, rolling_stats):
    """Pull rolling barrel_rate/hard_hit_pct/hr_fb_ratio per window for a batter name."""
    out = {7: {}, 14: {}, 30: {}}
    for w in (7, 14, 30):
        df_w = rolling_stats.get(w)
        if df_w is None or df_w.empty or "Name" not in df_w.columns:
            continue
        matched = _fuzzy_match_name(batter_name, df_w["Name"])
        if matched is None:
            continue
        row = df_w[df_w["Name"] == matched]
        if row.empty:
            continue
        r = row.iloc[0]
        out[w] = {
            "barrel_rate": _safe_float(r.get("barrel_rate"), None),
            "hard_hit_pct": _safe_float(r.get("hard_hit_pct"), None),
            "hr_fb_ratio": _safe_float(r.get("hr_fb_ratio"), None),
            "iso": _safe_float(r.get("ISO"), None),
        }
    return out


def _summarize_pitcher_recent(starts):
    """Aggregate last-N starts into recent_hr_per_ip / recent_hard_hit / high_workload."""
    if not starts:
        return {
            "recent_hr_per_ip": None,
            "recent_hard_hit": None,
            "high_workload": False,
        }
    total_ip = sum(s.get("ip", 0) or 0 for s in starts)
    total_hr = sum(s.get("hr_allowed", 0) or 0 for s in starts)
    recent_hr_per_ip = total_hr / total_ip if total_ip > 0 else None

    hh = [s.get("hard_hit_rate") for s in starts if s.get("hard_hit_rate") is not None]
    recent_hard_hit = (sum(hh) / len(hh)) if hh else None

    high_workload = bool(starts and (starts[0].get("pitches", 0) or 0) > 95)
    return {
        "recent_hr_per_ip": recent_hr_per_ip,
        "recent_hard_hit": recent_hard_hit,
        "high_workload": high_workload,
    }


_FASTBALL_TYPES = {"FF", "FT", "SI", "FA", "FC"}
_BREAKING_TYPES = {"SL", "CU", "KC", "SV", "ST", "SC"}
_OFFSPEED_TYPES = {"CH", "FS", "FO", "EP", "KN"}


def _summarize_pitch_mix(pitch_mix):
    """Bucket a pitcher's pitch_mix dict into fastball / breaking / offspeed pcts."""
    out = {
        "primary_pitch": None,
        "primary_pct": 0.0,
        "fastball_pct": 0.0,
        "breaking_pct": 0.0,
        "offspeed_pct": 0.0,
    }
    if not pitch_mix:
        return out
    out["primary_pitch"] = pitch_mix.get("primary_pitch")
    out["primary_pct"] = pitch_mix.get("primary_pct", 0.0) or 0.0
    mix = pitch_mix.get("pitch_mix", {}) or {}
    for pt, pct in mix.items():
        pt_u = (pt or "").upper()
        if pt_u in _FASTBALL_TYPES:
            out["fastball_pct"] += pct
        elif pt_u in _BREAKING_TYPES:
            out["breaking_pct"] += pct
        elif pt_u in _OFFSPEED_TYPES:
            out["offspeed_pct"] += pct
    return out


def _compute_pitch_type_edge(pitch_mix_summary, hard_hit_14d, barrel_14d, medians):
    """Pitch-type advantage: capped to [-0.10, +0.15], 0 if no data."""
    if not pitch_mix_summary or not pitch_mix_summary.get("primary_pitch"):
        return 0.0
    fb = pitch_mix_summary.get("fastball_pct", 0.0)
    br = pitch_mix_summary.get("breaking_pct", 0.0)
    off = pitch_mix_summary.get("offspeed_pct", 0.0)
    edge = 0.0
    hh_med = medians.get("hard_hit_pct", 0.35)
    barrel_med = medians.get("barrel_rate", 0.065)

    if fb > 0.60 and (hard_hit_14d or 0) > hh_med:
        edge += 0.08
    if br > 0.50 and (barrel_14d or 0) > barrel_med:
        edge += 0.05
    if off > 0.50:
        edge += 0.05
    return max(-0.10, min(0.15, edge))


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
