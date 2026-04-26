"""
Market: HR Props -- straight bet EV analysis.
Compares batter's actual HR/game rate to book implied probability.
Only considers 0.5 HR lines (at least 1 HR). Skips multi-HR longshots.
Flags bets where model_prob > implied_prob + MIN_EDGE.
"""

import math

from ev_calculator import calculate_ev, kelly_fraction, suggested_bet_size

MIN_EDGE = 0.03  # minimum 3% edge to surface a bet
MAX_REASONABLE_ODDS = 2000  # skip lines beyond +2000 (multi-HR longshots)


def score_hr_props(scored_batters, hr_prop_lines, batter_df=None):
    """
    Args:
        scored_batters: list of dicts from scorer.py (with name, tier, key_edge fields)
        hr_prop_lines: list of dicts from data_fetcher.get_player_props("batter_home_runs")
        batter_df: DataFrame from data_fetcher.get_batter_statcast() for HR/game rates

    Returns list of +EV HR prop bets, sorted by edge descending.
    """
    if not scored_batters or not hr_prop_lines:
        return []

    # Filter to 0.5 lines only (binary HR prop) and reasonable odds
    valid_props = [
        p for p in hr_prop_lines
        if p.get("over_line") == 0.5 and p.get("over_odds", 99999) <= MAX_REASONABLE_ODDS
    ]

    if not valid_props:
        return []

    # Build lookup: player_name (lowercase) -> prop line
    prop_lookup = {}
    for prop in valid_props:
        key = prop["player_name"].lower().strip()
        prop_lookup[key] = prop

    # Build HR/game rate lookup from batter stats
    hr_rates = _build_hr_rate_lookup(batter_df)

    results = []

    for batter in scored_batters:
        name = batter.get("name", "")

        # Get actual HR/game rate from season stats
        model_prob = hr_rates.get(name)
        if model_prob is None:
            continue
        if model_prob < 0.01:
            continue  # skip batters with negligible HR rate

        prop = _find_prop(name, prop_lookup)
        if prop is None:
            continue

        implied_prob = prop["implied_over_prob"]
        edge = model_prob - implied_prob

        if edge < MIN_EDGE:
            continue

        over_odds = prop["over_odds"]
        ev = calculate_ev(model_prob, over_odds)
        k = kelly_fraction(model_prob, over_odds)
        bet = suggested_bet_size(ev, k)

        # Sanity: EV > 200% is a data error
        if ev > 2.0:
            continue

        results.append({
            "player_name": name,
            "team": batter.get("team", ""),
            "model_prob": round(model_prob, 4),
            "implied_prob": round(implied_prob, 4),
            "edge": round(edge, 4),
            "over_odds": over_odds,
            "ev_per_dollar": round(ev, 4),
            "kelly": round(k, 4),
            "suggested_bet": bet,
            "tier": batter.get("tier", "SPECULATIVE"),
            "key_edge": batter.get("key_edge", ""),
            "market": "HR",
            "description": f"{name} HR over 0.5",
        })

    results.sort(key=lambda x: x["edge"], reverse=True)
    return results


def _build_hr_rate_lookup(batter_df):
    """Build dict of player_name -> HR/game rate from season stats."""
    rates = {}
    if batter_df is None or batter_df.empty:
        return rates

    name_col = "Name" if "Name" in batter_df.columns else None
    if name_col is None:
        return rates

    for _, row in batter_df.iterrows():
        name = row.get(name_col, "")
        hr = 0
        g = 0
        try:
            hr = int(row.get("HR", 0))
            g = int(row.get("G", 0))
        except (TypeError, ValueError):
            continue
        if g >= 10:  # minimum sample size
            rates[name] = hr / g
    return rates


def _find_prop(player_name, prop_lookup):
    """Match player name to prop lookup. Requires both first and last name to match."""
    key = player_name.lower().strip()
    if key in prop_lookup:
        return prop_lookup[key]

    # Require both first and last name parts to appear in the prop key
    parts = key.split()
    if len(parts) < 2:
        return None

    first = parts[0]
    last = parts[-1]

    for k, v in prop_lookup.items():
        if first in k and last in k:
            return v
    return None
