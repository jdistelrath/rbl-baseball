"""
Market: HR Props -- straight bet EV analysis on DraftKings.
Compares model HR probability to DK implied probability.
Flags bets where model_prob > implied_prob + MIN_EDGE.
"""

from ev_calculator import calculate_ev, kelly_fraction, suggested_bet_size

MIN_EDGE = 0.03  # minimum 3% edge to surface a bet


def score_hr_props(scored_batters, hr_prop_lines):
    """
    Args:
        scored_batters: list of dicts from scorer.py (with prob, name, tier fields)
        hr_prop_lines: list of dicts from data_fetcher.get_player_props("batter_home_runs")

    Returns list of +EV HR prop bets, sorted by edge descending.
    """
    if not scored_batters or not hr_prop_lines:
        return []

    # Build lookup: player_name (lowercase, stripped) -> prop line
    prop_lookup = {}
    for prop in hr_prop_lines:
        key = prop["player_name"].lower().strip()
        prop_lookup[key] = prop

    results = []

    for batter in scored_batters:
        name = batter.get("name", "")
        model_prob = batter.get("prob", 0.0)

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


def _find_prop(player_name, prop_lookup):
    """Fuzzy match player name to prop lookup."""
    key = player_name.lower().strip()
    if key in prop_lookup:
        return prop_lookup[key]
    # Try last name match
    last = player_name.split()[-1].lower()
    matches = [v for k, v in prop_lookup.items() if last in k]
    if len(matches) == 1:
        return matches[0]
    return None
