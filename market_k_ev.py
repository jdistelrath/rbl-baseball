"""
Market: Strikeout Props -- straight bet EV analysis on DraftKings.
Only targets starters with K/9 > 7.5.
"""

import math

from ev_calculator import (
    calculate_ev, kelly_fraction, suggested_bet_size, american_to_implied_prob,
)

MIN_K9 = 7.5
MIN_EDGE = 0.03


def score_k_props(games, pitcher_df, batter_df, weather_map, k_prop_lines):
    """
    Args:
        games: today's schedule (list of game dicts)
        pitcher_df: pitcher stats DataFrame
        batter_df: batter stats DataFrame
        weather_map: dict of stadium -> weather dict
        k_prop_lines: list from data_fetcher.get_player_props("pitcher_strikeouts")

    Returns list of +EV K prop bets sorted by edge descending.
    """
    if not games or not k_prop_lines:
        return []

    # Build prop lookup
    prop_lookup = {}
    for prop in k_prop_lines:
        key = prop["player_name"].lower().strip()
        prop_lookup[key] = prop

    results = []

    for game in games:
        for role in ("home_pitcher_name", "away_pitcher_name"):
            pitcher_name = game.get(role, "TBD")
            if not pitcher_name or pitcher_name == "TBD":
                continue

            pitcher_row = _find_pitcher(pitcher_name, pitcher_df)
            if pitcher_row is None:
                continue

            k9 = _get_stat(pitcher_row, ["K9", "SO9", "K/9"], 0.0)
            if k9 < MIN_K9:
                continue

            # Project Ks
            ip = _get_stat(pitcher_row, ["IP"], 150.0)
            gs = _get_stat(pitcher_row, ["GS"], 25.0)
            ip_per_start = min(ip / max(gs, 1), 6.5)

            # Opposing team K rate
            if role == "home_pitcher_name":
                opp_team = game.get("away_team", "")
            else:
                opp_team = game.get("home_team", "")
            opp_k_rate = _get_team_k_rate(opp_team, batter_df)
            k_rate_adj = 1.0 + (opp_k_rate - 0.225) * 0.5

            # Weather
            stadium = game.get("stadium", "")
            weather = weather_map.get(stadium, {"temp_f": 70})
            temp_adj = -0.3 if weather.get("temp_f", 70) < 45 else 0.0

            K_BIAS = -0.37
            model_k = max(0.0, round((k9 / 9.0) * ip_per_start * k_rate_adj + temp_adj + K_BIAS, 1))

            # Find prop line
            prop = _find_prop(pitcher_name, prop_lookup)
            if prop is None:
                continue

            book_line = prop["over_line"]
            over_odds = prop["over_odds"]
            under_odds = prop.get("under_odds")

            diff = model_k - book_line
            if abs(diff) < 0.5:
                continue

            # Normal distribution probability (SD ~1.5 Ks)
            try:
                from scipy.stats import norm
                over_prob = 1.0 - norm.cdf(book_line, loc=model_k, scale=1.5)
            except ImportError:
                # Fallback: logistic approximation
                over_prob = 1.0 / (1.0 + math.exp(-0.6 * diff))
            under_prob = 1.0 - over_prob

            if diff > 0:
                best_side = "over"
                best_prob = over_prob
                best_odds = over_odds
            else:
                best_side = "under"
                best_prob = under_prob
                best_odds = under_odds

            if best_odds is None:
                continue

            try:
                implied = american_to_implied_prob(int(best_odds))
            except (ValueError, TypeError):
                continue

            edge = best_prob - implied
            if edge < MIN_EDGE:
                continue

            ev = calculate_ev(best_prob, int(best_odds))
            k_frac = kelly_fraction(best_prob, int(best_odds))
            bet = suggested_bet_size(ev, k_frac)

            # Sanity cap
            if ev > 2.0:
                continue

            factors = [
                f"{pitcher_name} K/9 {k9:.1f}",
                f"model {model_k} Ks vs line {book_line}",
            ]
            if opp_k_rate > 0.26:
                factors.append(f"high-K opponent ({opp_k_rate:.0%} K rate)")

            results.append({
                "pitcher_name": pitcher_name,
                "team": game.get("home_team") if role == "home_pitcher_name" else game.get("away_team"),
                "opponent": opp_team,
                "model_k_projection": model_k,
                "book_line": book_line,
                "over_odds": over_odds,
                "under_odds": under_odds,
                "best_side": best_side,
                "edge": round(edge, 4),
                "ev_per_dollar": round(ev, 4),
                "kelly": round(k_frac, 4),
                "suggested_bet": bet,
                "key_factors": factors,
                "market": "K",
                "description": f"{pitcher_name} {best_side.upper()} {book_line} Ks",
            })

    results.sort(key=lambda x: x["edge"], reverse=True)
    return results


def _find_pitcher(name, df):
    if df.empty:
        return None
    for col in ("Name", "name"):
        if col in df.columns:
            rows = df[df[col] == name]
            if not rows.empty:
                return rows.iloc[0]
            last = name.split()[-1]
            rows = df[df[col].str.contains(last, na=False)]
            if len(rows) == 1:
                return rows.iloc[0]
    return None


def _get_stat(row, cols, default):
    if row is None:
        return default
    for c in cols:
        if c in row.index:
            try:
                v = float(row[c])
                if not (math.isnan(v) or math.isinf(v)):
                    return v
            except (TypeError, ValueError):
                pass
    return default


def _get_team_k_rate(team_name, batter_df):
    if batter_df.empty:
        return 0.225
    for col in ("Team", "Tm", "team"):
        if col in batter_df.columns:
            last_word = team_name.split()[-1] if team_name else ""
            rows = batter_df[batter_df[col].astype(str).str.contains(
                last_word, case=False, na=False
            )]
            if not rows.empty:
                for kcol in ("k_rate", "SO%", "K%"):
                    if kcol in rows.columns:
                        val = rows[kcol].mean()
                        if not math.isnan(val):
                            return val / 100.0 if val > 1 else val
    return 0.225


def _find_prop(name, lookup):
    key = name.lower().strip()
    if key in lookup:
        return lookup[key]
    last = name.split()[-1].lower()
    matches = [v for k, v in lookup.items() if last in k]
    if len(matches) == 1:
        return matches[0]
    return None
