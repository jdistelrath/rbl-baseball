"""
Market: Strikeout Props.
Over/under on strikeouts for a starting pitcher.
Only surfaces starters with K/9 > 7.5 and >= 0.5K model-vs-book difference.
"""

import math

from ev_calculator import calculate_ev, kelly_fraction, suggested_bet_size


def _safe_float(val, default):
    try:
        v = float(val)
        return v if not (math.isnan(v) or math.isinf(v)) else default
    except (TypeError, ValueError):
        return default


def _get_stat(row, col_names, default):
    if row is None:
        return default
    for col in col_names:
        if col in row.index:
            return _safe_float(row[col], default)
    return default


def _find_pitcher_row(pitcher_name, pitcher_df):
    if pitcher_df.empty or not pitcher_name or pitcher_name == "TBD":
        return None
    name_col = None
    for col in ("Name", "name", "PlayerName"):
        if col in pitcher_df.columns:
            name_col = col
            break
    if name_col is None:
        return None
    rows = pitcher_df[pitcher_df[name_col] == pitcher_name]
    if rows.empty:
        last = pitcher_name.split()[-1]
        rows = pitcher_df[pitcher_df[name_col].str.contains(last, na=False)]
        if len(rows) != 1:
            return None
    return rows.iloc[0]


def _find_team_k_rate(team_name, batter_df):
    """Estimate opposing team K% from batting stats."""
    if batter_df.empty:
        return 0.22  # league average
    team_col = None
    for col in ("Team", "team", "Tm"):
        if col in batter_df.columns:
            team_col = col
            break
    if team_col is None:
        return 0.22
    team_rows = batter_df[batter_df[team_col].astype(str).str.contains(
        team_name.split()[-1], case=False, na=False
    )]
    if team_rows.empty:
        return 0.22
    k_col = None
    for col in ("K%", "SO%", "Kpct"):
        if col in team_rows.columns:
            k_col = col
            break
    if k_col is None:
        return 0.22
    val = _safe_float(team_rows[k_col].mean(), 22.0)
    return val / 100.0 if val > 1 else val


def _over_probability(model_ks, book_line):
    """
    Logistic probability that actual Ks exceed book line.
    Each 0.5K difference ~ 10% probability shift.
    """
    diff = model_ks - book_line
    prob = 1.0 / (1.0 + math.exp(-0.40 * diff / 0.5))
    return prob


def score_strikeout_prop(game, side, pitcher_df, batter_df, weather, odds_lookup):
    """
    Score a strikeout prop for one pitcher in a game.

    Args:
        game: dict from schedule
        side: 'home' or 'away' (which pitcher to score)
        pitcher_df: pitching stats DataFrame
        batter_df: batting stats DataFrame
        weather: dict
        odds_lookup: odds data

    Returns dict or None if pitcher K/9 <= 7.5 or insufficient edge.
    """
    if side == "home":
        pitcher_name = game.get("home_pitcher_name", "TBD")
        team = game.get("home_team", "")
        opp_team = game.get("away_team", "")
    else:
        pitcher_name = game.get("away_pitcher_name", "TBD")
        team = game.get("away_team", "")
        opp_team = game.get("home_team", "")

    if pitcher_name == "TBD":
        return None

    p_row = _find_pitcher_row(pitcher_name, pitcher_df)

    # K/9 gate: only surface pitchers with K/9 > 7.5
    k9 = _get_stat(p_row, ["K/9", "K9", "SO9"], 0.0)
    if k9 <= 7.5:
        return None

    # K% for SwStr% proxy
    k_pct = _get_stat(p_row, ["K%", "Kpct", "SO%"], 22.0)
    if k_pct > 1:
        k_pct /= 100.0

    # Pitcher's average IP/start (cap at 6)
    ip = _get_stat(p_row, ["IP"], 150.0)
    gs = _get_stat(p_row, ["GS", "G"], 30.0)
    if gs > 0:
        ip_per_start = min(6.0, ip / gs)
    else:
        ip_per_start = 5.5

    # Base projection: (K/9 / 9) * expected innings
    base_ks = (k9 / 9.0) * ip_per_start

    # Opposing team K rate adjustment
    opp_k_rate = _find_team_k_rate(opp_team, batter_df)
    if opp_k_rate > 0.24:
        k_adj = 1.05  # +5% for high-K team
    elif opp_k_rate < 0.18:
        k_adj = 0.95  # -5% for low-K team
    else:
        k_adj = 1.0

    # Temperature adjustment
    temp = weather.get("temp_f", 70)
    temp_adj = -0.3 if temp < 45 else 0.0

    K_BIAS = -0.37
    model_ks = base_ks * k_adj + temp_adj + K_BIAS

    # Get strikeout prop odds
    book_line, over_odds, under_odds = _extract_k_odds(pitcher_name, game, odds_lookup)

    if book_line is None:
        # No odds; still provide projection but no EV
        return {
            "market": "STRIKEOUT",
            "game_id": game["game_id"],
            "pitcher_name": pitcher_name,
            "team": team,
            "opponent": opp_team,
            "model_k_projection": round(model_ks, 1),
            "book_line": None,
            "over_odds": None,
            "under_odds": None,
            "best_side": None,
            "ev_per_dollar": 0.0,
            "kelly": 0.0,
            "suggested_bet": 0.0,
            "key_factors": _build_factors(pitcher_name, k9, k_pct, opp_k_rate, opp_team, temp),
            "description": f"{pitcher_name} Ks (proj {model_ks:.1f})",
        }

    # Only surface if model projects >= 0.5K difference from book line
    if abs(model_ks - book_line) < 0.5:
        return None

    over_prob = _over_probability(model_ks, book_line)
    under_prob = 1.0 - over_prob

    ev_over = calculate_ev(over_prob, over_odds) if over_odds else -1.0
    ev_under = calculate_ev(under_prob, under_odds) if under_odds else -1.0

    if ev_over >= ev_under:
        best_side = "over"
        best_ev = ev_over
        best_prob = over_prob
        best_odds = over_odds
    else:
        best_side = "under"
        best_ev = ev_under
        best_prob = under_prob
        best_odds = under_odds

    if best_ev <= 0:
        return None

    k = kelly_fraction(best_prob, best_odds) if best_odds else 0.0
    bet = suggested_bet_size(best_ev, k)

    factors = _build_factors(pitcher_name, k9, k_pct, opp_k_rate, opp_team, temp)

    return {
        "market": "STRIKEOUT",
        "game_id": game["game_id"],
        "pitcher_name": pitcher_name,
        "team": team,
        "opponent": opp_team,
        "model_k_projection": round(model_ks, 1),
        "book_line": book_line,
        "over_odds": over_odds,
        "under_odds": under_odds,
        "best_side": best_side,
        "ev_per_dollar": round(best_ev, 4),
        "kelly": round(k, 4),
        "suggested_bet": bet,
        "key_factors": factors,
        "description": f"{pitcher_name} {best_side.upper()} {book_line} Ks",
    }


def _extract_k_odds(pitcher_name, game, odds_lookup):
    """Try to find pitcher strikeout prop odds."""
    if not odds_lookup:
        return None, None, None

    home = game.get("home_team", "")
    away = game.get("away_team", "")

    for event in odds_lookup if isinstance(odds_lookup, list) else []:
        if _teams_match(event, home, away):
            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    key = market.get("key", "")
                    if "strikeout" in key.lower() or "pitcher_k" in key.lower():
                        outcomes = market.get("outcomes", [])
                        # Find outcomes matching this pitcher
                        over_odds = None
                        under_odds = None
                        line = None
                        for o in outcomes:
                            desc = o.get("description", "")
                            name = o.get("name", "")
                            if pitcher_name.split()[-1] in desc or pitcher_name in desc:
                                if "Over" in name:
                                    over_odds = o.get("price")
                                    line = o.get("point")
                                elif "Under" in name:
                                    under_odds = o.get("price")
                                    if line is None:
                                        line = o.get("point")
                        if line is not None:
                            return line, over_odds, under_odds
    return None, None, None


def _teams_match(event, home, away):
    eh = event.get("home_team", "")
    ea = event.get("away_team", "")
    return (home in eh or eh in home) and (away in ea or ea in away)


def _build_factors(pitcher_name, k9, k_pct, opp_k_rate, opp_team, temp):
    factors = []
    if k9 >= 10:
        factors.append(f"{pitcher_name} elite K/9 ({k9:.1f})")
    elif k9 >= 8.5:
        factors.append(f"{pitcher_name} strong K/9 ({k9:.1f})")
    else:
        factors.append(f"{pitcher_name} K/9 {k9:.1f}")

    if opp_k_rate > 0.24:
        factors.append(f"{opp_team} high K% ({opp_k_rate*100:.0f}%)")
    elif opp_k_rate < 0.18:
        factors.append(f"{opp_team} low K% ({opp_k_rate*100:.0f}%)")

    if temp < 45:
        factors.append(f"cold ({temp:.0f}F) suppresses Ks")

    return factors
