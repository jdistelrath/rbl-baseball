"""
Market: First 5 Innings (F5) Lines.
Bet on which team wins the first 5 innings (isolates starting pitchers).
Only surfaces bets with >= 3% EV edge.
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


def _find_team_wrc_plus(team_name, batter_df):
    if batter_df.empty:
        return 100.0
    team_col = None
    for col in ("Team", "team", "Tm"):
        if col in batter_df.columns:
            team_col = col
            break
    if team_col is None:
        return 100.0
    team_rows = batter_df[batter_df[team_col].astype(str).str.contains(
        team_name.split()[-1], case=False, na=False
    )]
    if team_rows.empty:
        return 100.0
    wrc_col = None
    for col in ("wRC+", "wRC", "wRCplus"):
        if col in team_rows.columns:
            wrc_col = col
            break
    if wrc_col is None:
        return 100.0
    return _safe_float(team_rows[wrc_col].mean(), 100.0)


def score_f5(game, pitcher_df, batter_df, weather, odds_lookup):
    """
    Score a game's First 5 innings moneyline for EV.

    Returns dict or None if insufficient edge (< 3% EV).
    """
    home = game.get("home_team", "")
    away = game.get("away_team", "")
    home_pitcher = game.get("home_pitcher_name", "TBD")
    away_pitcher = game.get("away_pitcher_name", "TBD")

    # Start at 50/50
    home_win_prob = 0.50

    # Starter FIP differential: each 0.5 FIP diff = ~3% win prob shift
    hp_row = _find_pitcher_row(home_pitcher, pitcher_df)
    ap_row = _find_pitcher_row(away_pitcher, pitcher_df)

    home_fip = _get_stat(hp_row, ["FIP"], 4.20)
    away_fip = _get_stat(ap_row, ["FIP"], 4.20)
    fip_diff = away_fip - home_fip  # positive = home pitcher better
    home_win_prob += (fip_diff / 0.5) * 0.03

    # Opposing lineup wRC+ vs pitcher handedness
    # Higher wRC+ for away lineup = harder for home pitcher = lower home win prob
    home_wrc = _find_team_wrc_plus(home, batter_df)
    away_wrc = _find_team_wrc_plus(away, batter_df)
    wrc_diff = (home_wrc - away_wrc)  # positive = home lineup stronger
    home_win_prob += wrc_diff * 0.0008  # ~0.08% per wRC+ point

    # Home field advantage (~3% in F5)
    home_win_prob += 0.03

    # K/BB differential for starter dominance
    home_kbb = _get_stat(hp_row, ["K/BB", "KBB"], 2.5)
    away_kbb = _get_stat(ap_row, ["K/BB", "KBB"], 2.5)
    kbb_diff = home_kbb - away_kbb  # positive = home pitcher better K/BB
    home_win_prob += kbb_diff * 0.01

    # Clamp to reasonable range
    home_win_prob = max(0.25, min(0.75, home_win_prob))
    away_win_prob = 1.0 - home_win_prob

    # Get F5 odds from lookup
    home_f5_odds, away_f5_odds = _extract_f5_odds(game, odds_lookup)

    if home_f5_odds is None and away_f5_odds is None:
        # No odds; use h2h moneyline as proxy
        home_f5_odds, away_f5_odds = _extract_h2h_odds(game, odds_lookup)

    if home_f5_odds is None and away_f5_odds is None:
        return None  # Can't calculate EV without odds

    # Calculate EV for both sides
    ev_home = calculate_ev(home_win_prob, home_f5_odds) if home_f5_odds else -1.0
    ev_away = calculate_ev(away_win_prob, away_f5_odds) if away_f5_odds else -1.0

    if ev_home >= ev_away:
        best_side = "home"
        best_ev = ev_home
        best_prob = home_win_prob
        best_odds = home_f5_odds
        best_team = home
    else:
        best_side = "away"
        best_ev = ev_away
        best_prob = away_win_prob
        best_odds = away_f5_odds
        best_team = away

    # Sanity check: >200% EV is a data error, not a real edge
    if best_ev > 2.0:
        print(f"[market_f5] Discarding {home} vs {away}: EV {best_ev:.1%} is unrealistic "
              f"(prob={best_prob:.3f}, odds={best_odds})")
        return None

    # Only surface if >= 3% EV edge
    if best_ev < 0.03:
        return None

    k = kelly_fraction(best_prob, best_odds) if best_odds else 0.0
    bet = suggested_bet_size(best_ev, k)

    factors = _build_factors(home_fip, away_fip, home_pitcher, away_pitcher,
                             home_wrc, away_wrc, best_side)

    return {
        "market": "F5",
        "game_id": game["game_id"],
        "home_team": home,
        "away_team": away,
        "home_starter": home_pitcher,
        "away_starter": away_pitcher,
        "model_home_win_prob": round(home_win_prob, 4),
        "home_f5_odds": home_f5_odds,
        "away_f5_odds": away_f5_odds,
        "best_side": best_side,
        "best_team": best_team,
        "ev_per_dollar": round(best_ev, 4),
        "kelly": round(k, 4),
        "suggested_bet": bet,
        "key_factors": factors,
        "description": f"{best_team} F5 ML ({away} @ {home})",
    }


def _extract_f5_odds(game, odds_lookup):
    """Try to find F5 / first-half moneyline odds."""
    if not odds_lookup:
        return None, None

    home = game.get("home_team", "")
    away = game.get("away_team", "")

    for event in odds_lookup if isinstance(odds_lookup, list) else []:
        if _teams_match(event, home, away):
            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    key = market.get("key", "")
                    if "h2h_1st_half" in key or "first_half" in key or "h2h_h1" in key:
                        outcomes = market.get("outcomes", [])
                        home_odds = None
                        away_odds = None
                        for o in outcomes:
                            name = o.get("name", "")
                            if home in name or name in home:
                                home_odds = o.get("price")
                            elif away in name or name in away:
                                away_odds = o.get("price")
                        if home_odds or away_odds:
                            return home_odds, away_odds
    return None, None


def _extract_h2h_odds(game, odds_lookup):
    """Fallback: use full-game moneyline as F5 proxy."""
    if not odds_lookup:
        return None, None

    home = game.get("home_team", "")
    away = game.get("away_team", "")

    for event in odds_lookup if isinstance(odds_lookup, list) else []:
        if _teams_match(event, home, away):
            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") == "h2h":
                        outcomes = market.get("outcomes", [])
                        home_odds = None
                        away_odds = None
                        for o in outcomes:
                            name = o.get("name", "")
                            if home in name or name in home:
                                home_odds = o.get("price")
                            elif away in name or name in away:
                                away_odds = o.get("price")
                        if home_odds or away_odds:
                            return home_odds, away_odds
    return None, None


def _teams_match(event, home, away):
    eh = event.get("home_team", "")
    ea = event.get("away_team", "")
    return (home in eh or eh in home) and (away in ea or ea in away)


def _build_factors(home_fip, away_fip, home_pitcher, away_pitcher,
                   home_wrc, away_wrc, best_side):
    factors = []
    fip_diff = abs(home_fip - away_fip)
    if fip_diff > 0.5:
        better = home_pitcher if home_fip < away_fip else away_pitcher
        factors.append(f"strong SP edge ({better} {min(home_fip, away_fip):.2f} FIP)")

    wrc_diff = abs(home_wrc - away_wrc)
    if wrc_diff > 10:
        better_side = "home" if home_wrc > away_wrc else "away"
        factors.append(f"lineup edge ({better_side} wRC+ {max(home_wrc, away_wrc):.0f})")

    if not factors:
        if best_side == "home":
            factors.append(f"{home_pitcher} {home_fip:.2f} FIP + home advantage")
        else:
            factors.append(f"{away_pitcher} {away_fip:.2f} FIP vs weaker lineup")

    return factors
