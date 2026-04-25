"""
Market: Team Totals (over/under on total runs scored).
Scores every game for EV on the total.
"""

import json
import math
import os

from ev_calculator import calculate_ev, kelly_fraction, suggested_bet_size


# League average runs per game (both teams combined), ~2024 baseline
_BASE_RUNS = 8.8


def _load_totals_weights():
    path = os.path.join(os.path.dirname(__file__), "weights_totals.json")
    defaults = {
        "pitcher_fip_weight": 0.5,
        "wrc_weight": 0.015,
        "wind_out_bonus": 0.5,
        "wind_in_penalty": 0.3,
        "temp_hot_bonus": 0.2,
        "temp_cold_penalty": 0.4,
    }
    if os.path.exists(path):
        try:
            with open(path) as f:
                loaded = json.load(f)
                defaults.update(loaded)
        except Exception:
            pass
    return defaults


TOTALS_WEIGHTS = _load_totals_weights()

# Run factors by stadium (FanGraphs-derived, 1.0 = neutral)
PARK_RUN_FACTORS = {
    "Coors Field": 1.30,
    "Great American Ball Park": 1.12,
    "Yankee Stadium": 1.08,
    "Chase Field": 1.07,
    "Citizens Bank Park": 1.06,
    "Wrigley Field": 1.05,
    "Globe Life Field": 1.04,
    "Fenway Park": 1.03,
    "Guaranteed Rate Field": 1.02,
    "Minute Maid Park": 1.02,
    "Dodger Stadium": 1.01,
    "Truist Park": 1.01,
    "Rogers Centre": 1.01,
    "Angel Stadium": 1.00,
    "Oriole Park at Camden Yards": 1.00,
    "Target Field": 1.00,
    "Citi Field": 0.99,
    "PNC Park": 0.98,
    "American Family Field": 0.98,
    "Busch Stadium": 0.97,
    "Comerica Park": 0.97,
    "Progressive Field": 0.96,
    "Kauffman Stadium": 0.96,
    "T-Mobile Park": 0.95,
    "Nationals Park": 0.95,
    "loanDepot park": 0.94,
    "Petco Park": 0.93,
    "Tropicana Field": 0.93,
    "Oracle Park": 0.91,
    "Oakland Coliseum": 0.89,
    "RingCentral Coliseum": 0.89,
}


def _safe_float(val, default):
    try:
        v = float(val)
        return v if not (math.isnan(v) or math.isinf(v)) else default
    except (TypeError, ValueError):
        return default


def _get_stat(row, col_names, default):
    """Try multiple column names to find a stat."""
    if row is None:
        return default
    for col in col_names:
        if col in row.index:
            return _safe_float(row[col], default)
    return default


def _find_pitcher_row(pitcher_name, pitcher_df):
    """Find a pitcher in the stats DataFrame."""
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
        # Try last-name match
        last = pitcher_name.split()[-1]
        rows = pitcher_df[pitcher_df[name_col].str.contains(last, na=False)]
        if len(rows) != 1:
            return None
    return rows.iloc[0]


def _find_team_wrc_plus(team_name, batter_df):
    """Estimate team wRC+ as average wRC+ of batters on that team."""
    if batter_df.empty:
        return 100.0  # league average
    team_col = None
    for col in ("Team", "team", "Tm"):
        if col in batter_df.columns:
            team_col = col
            break
    if team_col is None:
        return 100.0

    # Team names in FanGraphs may be abbreviated; try substring match
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
    if wrc_col is not None:
        return _safe_float(team_rows[wrc_col].mean(), 100.0)

    # Fallback: approximate wRC+ from OPS (league avg OPS ~ 0.728 = 100 wRC+)
    if "OPS" in team_rows.columns:
        avg_ops = _safe_float(team_rows["OPS"].mean(), 0.728)
        return (avg_ops / 0.728) * 100.0

    return 100.0


def _weather_run_adjustment(weather, weights=None):
    """
    Wind blowing out >10mph: +wind_out_bonus runs
    Wind blowing in: -wind_in_penalty runs
    Temp <45F: -temp_cold_penalty runs
    Temp >85F: +temp_hot_bonus runs
    """
    w = weights or TOTALS_WEIGHTS
    adj = 0.0
    speed = weather.get("wind_speed_mph", 0)
    deg = weather.get("wind_dir_degrees", 0)

    if speed >= 10 and 150 <= deg <= 270:
        adj += w["wind_out_bonus"]
    elif speed >= 10 and (deg <= 60 or deg >= 330):
        adj -= w["wind_in_penalty"]

    temp = weather.get("temp_f", 70)
    if temp < 45:
        adj -= w["temp_cold_penalty"]
    elif temp > 85:
        adj += w["temp_hot_bonus"]

    return adj


def _over_probability(model_total, book_line):
    """
    Convert model total vs book line to an over probability.
    Uses a logistic function centered at the book line.
    Each 0.5 run difference ~ 8% probability shift.
    """
    diff = model_total - book_line
    # Steepness: 0.5 runs = ~8% shift -> k ~ 0.33
    prob = 1.0 / (1.0 + math.exp(-0.33 * diff / 0.5))
    return prob


def score_game_total(game, batter_df, pitcher_df, weather, odds_lookup,
                     weights_override=None):
    """
    Score a game for total runs over/under EV.

    Args:
        game: dict from schedule
        batter_df: batting stats DataFrame
        pitcher_df: pitching stats DataFrame
        weather: dict from get_weather()
        odds_lookup: dict keyed by game matchup string -> odds data
        weights_override: optional dict to override TOTALS_WEIGHTS

    Returns dict with model_total, book_line, best_side, ev_per_dollar, etc.
    Returns None if insufficient data.
    """
    w = weights_override or TOTALS_WEIGHTS
    stadium = game.get("stadium", "")
    home = game.get("home_team", "")
    away = game.get("away_team", "")

    # Pitcher adjustments (FIP-based: league avg FIP ~ 4.20)
    hp_row = _find_pitcher_row(game.get("home_pitcher_name"), pitcher_df)
    ap_row = _find_pitcher_row(game.get("away_pitcher_name"), pitcher_df)

    home_fip = _get_stat(hp_row, ["FIP"], 4.20)
    away_fip = _get_stat(ap_row, ["FIP"], 4.20)

    pitcher_adj = ((home_fip - 4.20) + (away_fip - 4.20)) * w["pitcher_fip_weight"]

    # Team offense adjustment (wRC+ based: 100 = league avg)
    home_wrc = _find_team_wrc_plus(home, batter_df)
    away_wrc = _find_team_wrc_plus(away, batter_df)
    offense_adj = ((home_wrc - 100) + (away_wrc - 100)) * w["wrc_weight"]

    # Park adjustment
    park_rf = PARK_RUN_FACTORS.get(stadium, 1.0)
    park_adj = (park_rf - 1.0) * _BASE_RUNS

    # Weather adjustment
    weather_adj = _weather_run_adjustment(weather, weights=w)

    model_total = _BASE_RUNS + pitcher_adj + offense_adj + park_adj + weather_adj

    # Get odds from lookup
    book_line, over_odds, under_odds = _extract_totals_odds(game, odds_lookup)

    if book_line is None:
        # No odds available; still return model prediction but no EV
        return {
            "market": "TOTAL",
            "game_id": game["game_id"],
            "home_team": home,
            "away_team": away,
            "model_total": round(model_total, 1),
            "book_line": None,
            "over_odds": None,
            "under_odds": None,
            "best_side": None,
            "ev_per_dollar": 0.0,
            "kelly": 0.0,
            "suggested_bet": 0.0,
            "key_factors": _build_factors(home_fip, away_fip, park_rf, weather,
                                          game.get("home_pitcher_name", "TBD"),
                                          game.get("away_pitcher_name", "TBD")),
            "description": f"{away} @ {home} total",
        }

    over_prob = _over_probability(model_total, book_line)
    under_prob = 1.0 - over_prob

    # Calculate EV for both sides
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

    k = kelly_fraction(best_prob, best_odds) if best_odds else 0.0
    bet = suggested_bet_size(best_ev, k)

    factors = _build_factors(home_fip, away_fip, park_rf, weather,
                             game.get("home_pitcher_name", "TBD"),
                             game.get("away_pitcher_name", "TBD"))

    return {
        "market": "TOTAL",
        "game_id": game["game_id"],
        "home_team": home,
        "away_team": away,
        "model_total": round(model_total, 1),
        "book_line": book_line,
        "over_odds": over_odds,
        "under_odds": under_odds,
        "best_side": best_side,
        "ev_per_dollar": round(best_ev, 4),
        "kelly": round(k, 4),
        "suggested_bet": bet,
        "key_factors": factors,
        "description": f"{away} @ {home} {best_side.upper()} {book_line} runs",
    }


_PREFERRED_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars"]


def _extract_totals_odds(game, odds_lookup):
    """Pull totals line + odds from the odds lookup dict."""
    if not odds_lookup:
        return None, None, None

    home = game.get("home_team", "")
    away = game.get("away_team", "")

    # The Odds API returns a list; find matching game
    for event in odds_lookup if isinstance(odds_lookup, list) else []:
        if _teams_match(event, home, away):
            # Prioritize sharp books
            bookmakers = sorted(
                event.get("bookmakers", []),
                key=lambda b: (0 if b.get("key", "") in _PREFERRED_BOOKS else 1)
            )
            for bookmaker in bookmakers:
                for market in bookmaker.get("markets", []):
                    if market.get("key") == "totals":
                        outcomes = market.get("outcomes", [])
                        over_odds = None
                        under_odds = None
                        line = None
                        for o in outcomes:
                            if o.get("name") == "Over":
                                over_odds = o.get("price")
                                line = o.get("point")
                            elif o.get("name") == "Under":
                                under_odds = o.get("price")
                                if line is None:
                                    line = o.get("point")
                        # Sanity check: main game totals are 5-12 runs
                        if line is not None and 5.0 <= line <= 12.0:
                            return line, over_odds, under_odds
                        elif line is not None:
                            print(f"[market_totals] Skipping {bookmaker.get('key', '?')} "
                                  f"totals line {line} (outside 5-12 range)")
    return None, None, None


def _teams_match(event, home, away):
    """Check if an odds event matches the game's teams."""
    eh = event.get("home_team", "")
    ea = event.get("away_team", "")
    # Exact or substring match
    return (home in eh or eh in home) and (away in ea or ea in away)


def _build_factors(home_fip, away_fip, park_rf, weather, home_pitcher, away_pitcher):
    """Build 2-3 key factor strings."""
    factors = []
    avg_fip = (home_fip + away_fip) / 2.0
    if avg_fip > 4.50:
        factors.append(f"weak pitching ({home_pitcher} {home_fip:.2f} / {away_pitcher} {away_fip:.2f} FIP)")
    elif avg_fip < 3.80:
        factors.append(f"strong pitching ({home_pitcher} {home_fip:.2f} / {away_pitcher} {away_fip:.2f} FIP)")

    if park_rf >= 1.05:
        factors.append(f"hitter-friendly park (RF {park_rf:.2f})")
    elif park_rf <= 0.95:
        factors.append(f"pitcher-friendly park (RF {park_rf:.2f})")

    temp = weather.get("temp_f", 70)
    wind = weather.get("wind_speed_mph", 0)
    if temp > 85 or (wind >= 10 and 150 <= weather.get("wind_dir_degrees", 0) <= 270):
        factors.append(f"favorable weather ({temp:.0f}F, {wind:.0f}mph wind)")
    elif temp < 45:
        factors.append(f"cold weather ({temp:.0f}F)")

    if not factors:
        factors.append(f"avg FIP {avg_fip:.2f}, park RF {park_rf:.2f}")

    return factors
