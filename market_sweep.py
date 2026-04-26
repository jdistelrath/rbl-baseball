"""
Market Sweep: hit all available MLB prop markets from The Odds API,
score each against a simple model projection, rank by EV,
and output a daily top-10 list.
"""

import math

from ev_calculator import (
    calculate_ev, kelly_fraction, suggested_bet_size, american_to_implied_prob,
)

# Markets to sweep in priority order
SWEEP_MARKETS = [
    "batter_total_bases",
    "batter_hits",
    "batter_home_runs",
    "pitcher_strikeouts",
    "batter_rbis",
    "batter_runs_scored",
]

MIN_EDGE = 0.03
MAX_ODDS = 2000       # skip longshots beyond +2000
MIN_GAMES = 10        # minimum games played for rate calc
OVER_UNDER_SD = {     # standard deviations for normal CDF per market
    "batter_total_bases": 1.2,
    "batter_hits": 0.8,
    "batter_home_runs": 0.4,
    "pitcher_strikeouts": 1.5,
    "batter_rbis": 0.8,
    "batter_runs_scored": 0.7,
}


def sweep_all_props(games, pitcher_df, batter_df):
    """
    Fetch all 6 markets from The Odds API, score each prop line
    against model projections, return +EV plays sorted by EV descending.

    Args:
        games: today's schedule (list of game dicts)
        pitcher_df: pitcher stats DataFrame
        batter_df: batter stats DataFrame

    Returns list of dicts, each representing a +EV play.
    """
    from data_fetcher import _get_mlb_event_ids, get_weather
    from config import CFG
    import requests

    if not CFG.odds_api_key or CFG.odds_api_key == "your_odds_api_key_here":
        print("[sweep] No Odds API key configured.")
        return []

    # Step 1: get event IDs (one call)
    events = _get_mlb_event_ids()
    if not events:
        print("[sweep] No MLB events found.")
        return []

    print(f"[sweep] {len(events)} events, sweeping {len(SWEEP_MARKETS)} markets...")

    # Build per-game rate lookups from batter/pitcher stats
    batter_rates = _build_batter_rates(batter_df)
    pitcher_k_rates = _build_pitcher_k_rates(pitcher_df, games)

    # Build weather map for K adjustments
    weather_map = {}
    for g in games:
        stadium = g.get("stadium", "")
        if stadium and stadium not in weather_map:
            weather_map[stadium] = get_weather(stadium, None)

    all_plays = []

    # Step 2: for each market, fetch props across all events
    for market in SWEEP_MARKETS:
        props = _fetch_market_props(events, market, CFG.odds_api_key)
        if not props:
            continue

        print(f"  {market}: {len(props)} lines", end="")

        plays = _score_market(market, props, batter_rates, pitcher_k_rates,
                              pitcher_df, batter_df, games, weather_map)
        print(f" -> {len(plays)} +EV")
        all_plays.extend(plays)

    all_plays.sort(key=lambda x: x["ev_per_dollar"], reverse=True)
    return all_plays


def _fetch_market_props(events, market, api_key):
    """Fetch a single market's props across all events."""
    import requests

    PREFERRED_BOOKS = ["draftkings", "fanduel", "betmgm"]
    results = []

    for event_info in events:
        event_id = event_info.get("id", "")
        home_team = event_info.get("home_team", "")
        away_team = event_info.get("away_team", "")
        if not event_id:
            continue

        url = (
            f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds"
            f"?apiKey={api_key}&regions=us&markets={market}&oddsFormat=american"
        )
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code != 200:
                continue
            event_data = resp.json()
        except Exception:
            continue

        bookmakers = sorted(
            event_data.get("bookmakers", []),
            key=lambda b: (0 if b.get("key", "") in PREFERRED_BOOKS else 1)
        )

        seen_players = set()
        for bookmaker in bookmakers:
            bk_key = bookmaker.get("key", "")
            for mkt in bookmaker.get("markets", []):
                if mkt.get("key") != market:
                    continue
                outcomes = mkt.get("outcomes", [])
                for outcome in outcomes:
                    player = outcome.get("description", outcome.get("name", ""))
                    side = outcome.get("name", "")
                    price = outcome.get("price")
                    line = outcome.get("point")

                    if not player or player in seen_players:
                        continue
                    if side != "Over" or price is None or line is None:
                        continue
                    if abs(int(price)) > MAX_ODDS:
                        continue

                    # Find matching under
                    under_odds = None
                    for o2 in outcomes:
                        if (o2.get("description", o2.get("name", "")) == player
                                and o2.get("name") == "Under"):
                            under_odds = o2.get("price")

                    results.append({
                        "player_name": player,
                        "home_team": home_team,
                        "away_team": away_team,
                        "market": market,
                        "over_line": float(line),
                        "over_odds": int(price),
                        "under_odds": int(under_odds) if under_odds else None,
                        "bookmaker": bk_key,
                    })
                    seen_players.add(player)

    return results


def _build_batter_rates(batter_df):
    """Build per-game rate lookup for all batter markets."""
    rates = {}
    if batter_df is None or batter_df.empty:
        return rates

    name_col = "Name" if "Name" in batter_df.columns else None
    if name_col is None:
        return rates

    for _, row in batter_df.iterrows():
        name = row.get(name_col, "")
        try:
            g = int(row.get("G", 0))
        except (TypeError, ValueError):
            continue
        if g < MIN_GAMES:
            continue

        def _safe(col):
            try:
                return int(row.get(col, 0))
            except (TypeError, ValueError):
                return 0

        rates[name] = {
            "tb_per_game": _safe("TB") / g,
            "h_per_game": _safe("H") / g,
            "hr_per_game": _safe("HR") / g,
            "rbi_per_game": _safe("RBI") / g,
            "r_per_game": _safe("R") / g,
            "g": g,
        }
    return rates


def _build_pitcher_k_rates(pitcher_df, games):
    """Build K projection lookup for pitchers in today's games."""
    rates = {}
    if pitcher_df is None or pitcher_df.empty:
        return rates

    name_col = "Name" if "Name" in pitcher_df.columns else None
    if name_col is None:
        return rates

    for _, row in pitcher_df.iterrows():
        name = row.get(name_col, "")
        try:
            k9 = float(row.get("K9", row.get("K/9", 0)))
            ip = float(row.get("IP", 0))
            gs = float(row.get("GS", 1))
        except (TypeError, ValueError):
            continue
        if gs < 3 or k9 < 1:
            continue
        ip_per_start = min(ip / max(gs, 1), 6.5)
        rates[name] = {
            "k9": k9,
            "ip_per_start": ip_per_start,
            "projected_k": (k9 / 9.0) * ip_per_start,
        }
    return rates


def _score_market(market, props, batter_rates, pitcher_k_rates,
                  pitcher_df, batter_df, games, weather_map):
    """Score all props in a single market, return +EV plays."""
    results = []

    for prop in props:
        player = prop["player_name"]
        line = prop["over_line"]
        over_odds = prop["over_odds"]
        under_odds = prop.get("under_odds")

        # Get model projection
        projection = _get_projection(market, player, line, batter_rates,
                                     pitcher_k_rates, games, weather_map,
                                     batter_df)
        if projection is None:
            continue

        model_rate = projection["rate"]
        sd = OVER_UNDER_SD.get(market, 1.0)

        # Over probability via normal CDF
        try:
            from scipy.stats import norm
            over_prob = 1.0 - norm.cdf(line, loc=model_rate, scale=sd)
        except ImportError:
            diff = model_rate - line
            over_prob = 1.0 / (1.0 + math.exp(-1.5 * diff / sd))
        under_prob = 1.0 - over_prob

        # Determine best side
        diff = model_rate - line
        if diff > 0:
            best_side = "over"
            best_prob = over_prob
            best_odds = over_odds
        elif under_odds is not None:
            best_side = "under"
            best_prob = under_prob
            best_odds = under_odds
        else:
            continue

        try:
            implied = american_to_implied_prob(best_odds)
        except (ValueError, TypeError):
            continue

        edge = best_prob - implied
        if edge < MIN_EDGE:
            continue

        ev = calculate_ev(best_prob, best_odds)
        if ev > 2.0:
            continue  # sanity cap

        kf = kelly_fraction(best_prob, best_odds)
        bet = suggested_bet_size(ev, kf)

        market_label = _market_label(market)
        results.append({
            "player_name": player,
            "market": market_label,
            "best_side": best_side,
            "line": line,
            "model_projection": round(model_rate, 2),
            "over_odds": over_odds,
            "under_odds": under_odds,
            "best_odds": best_odds,
            "model_prob": round(best_prob, 4),
            "implied_prob": round(implied, 4),
            "edge": round(edge, 4),
            "ev_per_dollar": round(ev, 4),
            "kelly": round(kf, 4),
            "suggested_bet": bet,
            "bookmaker": prop.get("bookmaker", ""),
            "description": f"{player} {best_side.upper()} {line} {market_label}",
        })

    return results


def _get_projection(market, player_name, line, batter_rates, pitcher_k_rates,
                    games, weather_map, batter_df):
    """Get model projection for a player in a specific market."""
    if market == "pitcher_strikeouts":
        rate_info = _fuzzy_lookup(player_name, pitcher_k_rates)
        if rate_info is None:
            return None
        proj = rate_info["projected_k"]
        # Temperature adjustment
        for g in games:
            for role in ("home_pitcher_name", "away_pitcher_name"):
                if player_name in g.get(role, ""):
                    weather = weather_map.get(g.get("stadium", ""), {})
                    if weather.get("temp_f", 70) < 45:
                        proj -= 0.3
                    break
        return {"rate": proj}

    # Batter markets
    rate_info = _fuzzy_lookup(player_name, batter_rates)
    if rate_info is None:
        return None

    rate_key = {
        "batter_total_bases": "tb_per_game",
        "batter_hits": "h_per_game",
        "batter_home_runs": "hr_per_game",
        "batter_rbis": "rbi_per_game",
        "batter_runs_scored": "r_per_game",
    }.get(market)

    if rate_key is None or rate_key not in rate_info:
        return None

    return {"rate": rate_info[rate_key]}


def _fuzzy_lookup(name, lookup):
    """Match player name to lookup dict. Requires first+last name match."""
    key = name.strip()
    if key in lookup:
        return lookup[key]

    parts = key.lower().split()
    if len(parts) < 2:
        return None
    first = parts[0]
    last = parts[-1]

    for k, v in lookup.items():
        kl = k.lower()
        if first in kl and last in kl:
            return v
    return None


def _market_label(market):
    """Short human label for a market key."""
    return {
        "batter_total_bases": "TB",
        "batter_hits": "Hits",
        "batter_home_runs": "HR",
        "pitcher_strikeouts": "Ks",
        "batter_rbis": "RBI",
        "batter_runs_scored": "Runs",
    }.get(market, market)
