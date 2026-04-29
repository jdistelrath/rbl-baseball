"""
Module 4: Scorer.
Z-score normalize features, weighted sum, sigmoid transform, confidence tiers.
"""

import math

import numpy as np

from config import CFG

# Feature columns that feed into the composite score
FEATURE_COLS = [
    "barrel_rate",
    "hr_fb_ratio",
    "iso",
    "hard_hit_pct",
    "platoon_factor",
    "pitcher_hr_fb",
    "pitcher_fly_ball_rate",
    "pitcher_xfip",
    "pitcher_hard_hit_allowed",
    "park_hr_factor",
    "wind_bonus",
    "temp_bonus",
    "batting_order_position",
    # v11 additions
    "barrel_rate_14d",
    "hard_hit_pct_14d",
    "form_trend",
    "pitcher_recent_hr_per_ip",
    "pitcher_recent_hard_hit",
    "pitcher_high_workload",
    "pitch_type_edge",
]

# Features where higher = worse for HR probability (invert sign before scoring)
_INVERT_FEATURES = set()
# pitcher_xfip: lower xFIP = better pitcher = harder to homer off
# So higher xFIP = easier to homer off = good for batter. No inversion needed.


def _sigmoid(x):
    """Sigmoid squash to 0-1."""
    return 1.0 / (1.0 + math.exp(-x))


def _is_nan(v):
    try:
        return isinstance(v, float) and math.isnan(v)
    except (TypeError, ValueError):
        return False


def score_batters(feature_dicts):
    """
    Takes a list of feature dicts (from feature_builder), returns scored list.

    Steps:
    1. Z-score normalize each feature across all candidates
    2. Weighted sum using CFG.weights
    3. Sigmoid transform to probability
    4. Assign confidence tiers

    Returns list of dicts with added fields:
        score, prob, tier, bet_amount, key_edge
    """
    if not feature_dicts:
        return []

    n = len(feature_dicts)

    # Extract feature matrix; replace None with column mean (keeps row neutral)
    raw = {}
    col_means = {}
    for col in FEATURE_COLS:
        vals_raw = [f.get(col) for f in feature_dicts]
        valid = [float(v) for v in vals_raw if v is not None and not _is_nan(v)]
        mean_val = float(np.mean(valid)) if valid else 0.0
        col_means[col] = mean_val
        raw[col] = [
            float(v) if (v is not None and not _is_nan(v)) else mean_val
            for v in vals_raw
        ]

    # Z-score normalize
    z_scores = {}
    for col in FEATURE_COLS:
        vals = np.array(raw[col], dtype=float)
        mean = np.mean(vals)
        std = np.std(vals)
        if std < 1e-9:
            z_scores[col] = np.zeros(n)
        else:
            z_scores[col] = (vals - mean) / std

    # Weighted sum
    weights = CFG.weights
    composite = np.zeros(n)
    for col in FEATURE_COLS:
        w = weights.get(col, 1.0)
        composite += w * z_scores[col]

    # H2H scalar bonus (added directly post-z-score; missing = neutral)
    for i, fd in enumerate(feature_dicts):
        h2h_rate = fd.get("h2h_hr_rate")
        if h2h_rate is None:
            continue
        h2h_bonus = (h2h_rate - 0.035) * 5.0
        composite[i] += max(-0.5, min(0.5, h2h_bonus))

    # Sigmoid transform
    probs = np.array([_sigmoid(c) for c in composite])

    # Determine tier thresholds
    sorted_probs = np.sort(probs)[::-1]
    if n >= 3:
        strong_threshold = sorted_probs[max(0, int(n * 0.15) - 1)]
        standard_threshold = sorted_probs[max(0, int(n * 0.50) - 1)]
    else:
        strong_threshold = sorted_probs[0]
        standard_threshold = sorted_probs[-1]

    # Build scored results
    results = []
    for i, fd in enumerate(feature_dicts):
        prob = float(probs[i])
        score = float(composite[i])

        if prob >= strong_threshold:
            tier = "STRONG"
            bet = CFG.bet_high
        elif prob >= standard_threshold:
            tier = "STANDARD"
            bet = CFG.bet_standard
        else:
            tier = "SPECULATIVE"
            bet = CFG.bet_lottery

        key_edge = _build_key_edge(fd)

        results.append({
            "name": fd["name"],
            "team": fd["team"],
            "opponent_pitcher": fd["opponent_pitcher"],
            "game_id": fd["game_id"],
            "score": round(score, 4),
            "prob": round(prob, 4),
            "tier": tier,
            "bet_amount": bet,
            "key_edge": key_edge,
            "batting_order": fd.get("batting_order"),
            "stadium": fd.get("stadium", ""),
            "wind_label": f"{fd.get('_wind_speed', 0):.0f}mph {fd.get('_wind_dir', 'calm')}",
            "park_label": f"PF {fd.get('park_hr_factor', 1.0):.2f}",
            # Carry through raw features for stack_builder
            "barrel_rate": fd.get("barrel_rate", 0),
            "iso": fd.get("iso", 0),
            "hr_fb_ratio": fd.get("hr_fb_ratio", 0),
            "pitcher_hr_fb": fd.get("pitcher_hr_fb", 0),
            "pitcher_xfip": fd.get("pitcher_xfip", 0),
            "pitcher_fly_ball_rate": fd.get("pitcher_fly_ball_rate", 0),
            "_bat_side": fd.get("_bat_side", ""),
            "_pitcher_throws": fd.get("_pitcher_throws", ""),
            # v11 diagnostics
            "barrel_rate_14d": fd.get("barrel_rate_14d"),
            "form_trend": fd.get("form_trend", 0.0),
            "pitcher_recent_hr_per_ip": fd.get("_recent_hr_per_ip_raw"),
            "pitch_type_edge": fd.get("pitch_type_edge", 0.0),
            "h2h_hr_rate": fd.get("h2h_hr_rate"),
            "h2h_pa": fd.get("_h2h_pa", 0),
            "primary_pitch": fd.get("_primary_pitch"),
            "data_gaps": fd.get("data_gaps", []),
        })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def _build_key_edge(fd):
    """Generate a one-line explanation of the primary edge."""
    parts = []

    # Platoon
    bs = fd.get("_bat_side", "")
    pt = fd.get("_pitcher_throws", "")
    if bs and pt and bs != "S" and bs != pt:
        parts.append(f"{'L' if bs == 'L' else 'R'}HB vs {'R' if pt == 'R' else 'L'}HP")

    # Pitcher HR/FB
    phrfb = fd.get("pitcher_hr_fb", 0)
    if phrfb > 0.13:
        parts.append(f"{phrfb*100:.0f}% HR/FB allowed")

    # Barrel rate
    br = fd.get("barrel_rate", 0)
    if br > 0.08:
        parts.append(f"{br*100:.0f}% barrel rate")

    # Wind
    ws = fd.get("_wind_speed", 0)
    wd = fd.get("_wind_dir", "calm")
    if ws >= 10 and fd.get("wind_bonus", 0) > 0:
        parts.append(f"wind {ws:.0f}mph out to {wd}")

    # Park
    pf = fd.get("park_hr_factor", 1.0)
    if pf >= 1.10:
        parts.append(f"park factor {pf:.2f}")

    # v11: rolling form
    ft = fd.get("_form_trend_raw", fd.get("form_trend", 0.0)) or 0.0
    if ft > 0.15:
        parts.append(f"heating up (14d barrel +{ft*100:.0f}%)")
    elif ft < -0.15:
        parts.append(f"cooling off (14d barrel {ft*100:.0f}%)")

    # v11: H2H
    h2h_rate = fd.get("h2h_hr_rate")
    h2h_pa = fd.get("_h2h_pa", 0)
    h2h_hr = fd.get("_h2h_hr", 0)
    if h2h_rate is not None and h2h_rate >= 0.08 and h2h_pa >= 5:
        parts.append(f"{h2h_hr} HR in {h2h_pa} PA vs this pitcher")

    # v11: pitcher recent HR allowed
    rec_hr = fd.get("_recent_hr_per_ip_raw")
    if rec_hr is not None and rec_hr > 0.15:
        # Approximate HR count over last 3 starts (~18 IP typical)
        approx_hr = round(rec_hr * 18)
        parts.append(f"pitcher allowed {approx_hr} HR last 3 starts")

    if not parts:
        iso = fd.get("iso", 0)
        parts.append(f"ISO {iso:.3f}")

    return ", ".join(parts)
