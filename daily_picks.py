"""
Daily Picks Pipeline.
Pulls today's confirmed lineups and starting pitchers, runs K/Hits/TB/HR
models, outputs sharp_brief.txt and hr_list.txt.

Usage: python main.py daily_picks
"""

import math
from datetime import date

import requests
from scipy.stats import poisson

from config import CFG
from backtest_props import (
    PARK_HR_FACTOR, TEAM_STADIUM,
    _calibrate_hr_prob, _project_hr_prob, _build_team_pitcher_hr9,
    K_BIAS_CORRECTION,
)

# ---------------------------------------------------------------------------
# Odds API helpers
# ---------------------------------------------------------------------------

ODDS_MARKETS = [
    "pitcher_strikeouts",
    "batter_hits",
    "batter_total_bases",
    "batter_home_runs",
]
ODDS_BOOKS = ["draftkings", "fanduel"]
BOOK_DISPLAY = {"draftkings": "DK", "fanduel": "FD", "underdog": "UD"}


def _american_to_implied(price):
    """Convert American odds to implied probability (no-vig)."""
    if price < 0:
        return abs(price) / (abs(price) + 100)
    return 100 / (price + 100)


def _poisson_over_prob(lam, line):
    """P(X > line) using Poisson CDF.  line is typically x.5."""
    return 1 - poisson.cdf(math.floor(line), lam)


def fetch_odds_lines():
    """Fetch today's MLB player prop lines from The Odds API.

    Returns dict keyed by (normalized_name, market) -> list of
    {book, line, price, implied_prob, name_raw}.
    """
    api_key = CFG.odds_api_key
    if not api_key or api_key == "your_odds_api_key_here":
        print("[odds] THE_ODDS_API_KEY not set — skipping line lookup")
        return {}

    # Step 1: get today's event IDs
    events_url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
    try:
        resp = requests.get(events_url, params={"apiKey": api_key}, timeout=15)
        if resp.status_code == 401:
            print("[odds] API key invalid or expired (401). Skipping line lookup.")
            return {}
        resp.raise_for_status()
        events = resp.json()
    except requests.RequestException as exc:
        print(f"[odds] Events fetch failed: {exc}")
        return {}

    if not isinstance(events, list) or not events:
        print("[odds] No MLB events found (games may have already started)")
        return {}

    print(f"[odds] Found {len(events)} MLB events, fetching player props...")

    lines = {}  # (norm_name, market) -> [line_entries]

    for event in events:
        eid = event["id"]
        props_url = (
            f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{eid}/odds"
        )
        try:
            r = requests.get(props_url, params={
                "apiKey": api_key,
                "regions": "us",
                "markets": ",".join(ODDS_MARKETS),
                "oddsFormat": "american",
                "bookmakers": ",".join(ODDS_BOOKS),
            }, timeout=15)
            r.raise_for_status()
        except requests.RequestException as exc:
            print(f"[odds] Failed to fetch props for event {eid}: {exc}")
            continue

        data = r.json()
        for bk in data.get("bookmakers", []):
            book_key = bk["key"]
            if book_key not in ODDS_BOOKS:
                continue
            for mkt in bk.get("markets", []):
                market_key = mkt["key"]
                for outcome in mkt.get("outcomes", []):
                    if outcome["name"] != "Over":
                        continue
                    player_raw = outcome.get("description", "")
                    if not player_raw:
                        continue
                    norm = _normalize_name(player_raw)
                    point = outcome.get("point")
                    price = outcome.get("price")
                    if point is None or price is None:
                        continue
                    key = (norm, market_key)
                    entry = {
                        "book": book_key,
                        "line": point,
                        "price": price,
                        "implied_prob": _american_to_implied(price),
                        "name_raw": player_raw,
                    }
                    lines.setdefault(key, []).append(entry)

    print(f"[odds] Loaded {len(lines)} player/market lines")
    return lines


def fetch_underdog_lines():
    """Fetch MLB player prop lines from Underdog Fantasy public API.

    Returns dict keyed by (normalized_name, market) -> list of
    {book, line, price, implied_prob, name_raw} — same format as fetch_odds_lines.
    Underdog uses pick'em (no American odds), so we use -110 as a proxy price.
    """
    STAT_TO_MARKET = {
        "Strikeouts": "pitcher_strikeouts",
        "Hits": "batter_hits",
        "Total Bases": "batter_total_bases",
        "Home Runs": "batter_home_runs",
    }

    try:
        resp = requests.get(
            "https://api.underdogfantasy.com/beta/v5/over_under_lines",
            timeout=10, headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[underdog] Fetch failed: {e}")
        return {}

    players = {p["id"]: p for p in data.get("players", [])}
    appearances = {a["id"]: a for a in data.get("appearances", [])}
    games = {g["id"]: g for g in data.get("games", [])}

    lines = {}
    count = 0

    for l in data.get("over_under_lines", []):
        ou = l.get("over_under", {})
        app_stat = ou.get("appearance_stat", {})
        app_id = app_stat.get("appearance_id", "")
        app = appearances.get(app_id, {})
        game = games.get(app.get("match_id", ""), {})

        if game.get("sport_id") != "MLB":
            continue

        stat_name = app_stat.get("display_stat", "")
        market = STAT_TO_MARKET.get(stat_name)
        if not market:
            continue

        player_id = app.get("player_id", "")
        player = players.get(player_id, {})
        name_raw = f"{player.get('first_name', '')} {player.get('last_name', '')}".strip()
        if not name_raw:
            continue

        line_val = l.get("stat_value")
        if line_val is None:
            continue

        norm = _normalize_name(name_raw)
        # Underdog is pick'em (no odds), treat as -110 proxy
        entry = {
            "book": "underdog",
            "line": float(line_val),
            "price": -110,
            "implied_prob": 0.524,
            "name_raw": name_raw,
        }
        key = (norm, market)
        lines.setdefault(key, []).append(entry)
        count += 1

    print(f"[underdog] Loaded {count} MLB prop lines")
    return lines


def _normalize_name(name):
    """Lowercase, strip suffixes like Jr./III, collapse whitespace."""
    n = name.lower().strip()
    for suffix in (" jr.", " jr", " sr.", " sr", " ii", " iii", " iv"):
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    return " ".join(n.split())


def _lookup_line(lines, player_name, market):
    """Find best book line for a player+market.  Prefer DK, then FD, then UD."""
    norm = _normalize_name(player_name)
    entries = lines.get((norm, market), [])
    if not entries:
        return None
    for pref in ["draftkings", "fanduel", "underdog"]:
        for e in entries:
            if e["book"] == pref:
                return e
    return entries[0]


def _lookup_all_books(lines, player_name, market):
    """Return dict of {book_display: entry} for all books with a line."""
    norm = _normalize_name(player_name)
    entries = lines.get((norm, market), [])
    result = {}
    for e in entries:
        bk = BOOK_DISPLAY.get(e["book"], e["book"])
        if bk not in result:
            result[bk] = e
    return result


def _format_american(price):
    """Format American odds with sign."""
    return f"+{price}" if price > 0 else str(price)


# ---------------------------------------------------------------------------
# Model projections
# ---------------------------------------------------------------------------

def _project_ks(pitcher_row, opp_k_rate=0.225):
    """Project strikeouts for a start. Same model as backtest with bias correction."""
    try:
        k9 = float(pitcher_row.get("K9", pitcher_row.get("K/9", 0)))
        ip = float(pitcher_row.get("IP", 0))
        gs = float(pitcher_row.get("GS", 1))
    except (TypeError, ValueError):
        return None
    if gs < 3 or k9 < 1:
        return None
    ip_per_start = min(ip / max(gs, 1), 6.5)
    k_rate_adj = 1.0 + (opp_k_rate - 0.225) * 0.5
    return (k9 / 9.0) * ip_per_start * k_rate_adj + K_BIAS_CORRECTION


def _project_batter(batter_row):
    """Project hits/game, TB/game, and HR prob for a batter."""
    try:
        g = int(batter_row.get("G", 0))
        h = int(batter_row.get("H", 0))
        tb = int(batter_row.get("TB", 0))
        hr = int(batter_row.get("HR", 0))
        pa = int(batter_row.get("PA", 0))
    except (TypeError, ValueError):
        return None
    if g < 10 or pa < 50:
        return None
    return {
        "h_per_game": h / g,
        "tb_per_game": tb / g,
        "hr_per_game": hr / g,
        "hr_pa_rate": hr / pa if pa > 0 else 0,
        "pa_per_game": pa / g,
    }


def _get_team_k_rate(team_name, batter_df):
    """Get team strikeout rate from batter stats."""
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


def _find_pitcher_row(name, pitcher_df):
    """Find a pitcher in the stats DataFrame."""
    if pitcher_df.empty or not name or name == "TBD":
        return None
    name_col = "Name" if "Name" in pitcher_df.columns else None
    if name_col is None:
        return None
    rows = pitcher_df[pitcher_df[name_col] == name]
    if not rows.empty:
        return rows.iloc[0]
    last = name.split()[-1]
    rows = pitcher_df[pitcher_df[name_col].str.contains(last, na=False)]
    if len(rows) == 1:
        return rows.iloc[0]
    return None


def _find_batter_row(name, batter_df):
    """Find a batter in the stats DataFrame."""
    if batter_df.empty or not name:
        return None
    name_col = "Name" if "Name" in batter_df.columns else None
    if name_col is None:
        return None
    rows = batter_df[batter_df[name_col] == name]
    if not rows.empty:
        return rows.iloc[0]
    last = name.split()[-1]
    rows = batter_df[batter_df[name_col].str.contains(last, na=False)]
    if len(rows) == 1:
        return rows.iloc[0]
    return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_daily_picks():
    """Run the full daily picks pipeline."""
    from data_fetcher import (
        get_today_schedule, get_confirmed_lineups,
        get_batter_statcast, get_pitcher_statcast,
    )

    today = date.today().strftime("%B %d, %Y")
    print(f"[daily_picks] === Daily Picks for {today} ===")

    games = get_today_schedule()
    if not games:
        print("[daily_picks] No games today.")
        return

    batter_df = get_batter_statcast()
    pitcher_df = get_pitcher_statcast()
    team_pitcher_hr9 = _build_team_pitcher_hr9(pitcher_df)

    k_picks = []       # pitcher K projections
    batter_picks = []   # batter H/TB/HR projections

    confirmed_games = 0

    # Pre-fetch weather per stadium for wind context
    from data_fetcher import get_weather
    weather_map = {}
    for game in games:
        stadium = game.get("stadium", "")
        if stadium and stadium not in weather_map:
            weather_map[stadium] = get_weather(stadium, None)

    for game in games:
        game_id = game["game_id"]
        home = game["home_team"]
        away = game["away_team"]
        stadium = game.get("stadium", "")

        # --- Score both pitchers for K props ---
        for role, opp_team in [
            ("home_pitcher_name", away),
            ("away_pitcher_name", home),
        ]:
            pitcher_name = game.get(role, "TBD")
            if not pitcher_name or pitcher_name == "TBD":
                continue

            p_row = _find_pitcher_row(pitcher_name, pitcher_df)
            if p_row is None:
                continue

            opp_k_rate = _get_team_k_rate(opp_team, batter_df)
            proj_k = _project_ks(p_row, opp_k_rate)
            if proj_k is None or proj_k < 2:
                continue

            try:
                k9 = float(p_row.get("K9", 0))
                era = float(p_row.get("ERA", 0))
            except (TypeError, ValueError):
                k9, era = 0, 0

            team = home if role == "home_pitcher_name" else away
            k_picks.append({
                "name": pitcher_name,
                "team": team,
                "opponent": opp_team,
                "proj_k": round(proj_k, 1),
                "k9": round(k9, 1),
                "era": round(era, 2),
                "opp_k_rate": round(opp_k_rate, 3),
                "note": f"vs {opp_team} ({opp_k_rate:.0%} K rate)",
            })

        # --- Score batters (need confirmed lineups) ---
        lineup = get_confirmed_lineups(game_id)
        if lineup is None:
            continue
        confirmed_games += 1

        for side, opp_team in [("home", away), ("away", home)]:
            batters = lineup.get(side, [])
            batter_team = home if side == "home" else away
            is_home = side == "home"

            for batter in batters:
                name = batter["name"]
                bat_side = batter.get("bat_side", "R")

                b_row = _find_batter_row(name, batter_df)
                if b_row is None:
                    continue

                proj = _project_batter(b_row)
                if proj is None:
                    continue

                # HR probability (raw then calibrated)
                raw_hr = _project_hr_prob(
                    b_row, opp_team, team_pitcher_hr9, bat_side,
                    is_home, opp_team
                )
                cal_hr = _calibrate_hr_prob(raw_hr) if raw_hr is not None else None

                # Opposing pitcher name for matchup note
                if side == "home":
                    opp_pitcher = game.get("away_pitcher_name", "TBD")
                else:
                    opp_pitcher = game.get("home_pitcher_name", "TBD")

                # Park factor
                if is_home:
                    park_stadium = TEAM_STADIUM.get(batter_team, "")
                else:
                    park_stadium = TEAM_STADIUM.get(opp_team, "")
                pf = PARK_HR_FACTOR.get(park_stadium, 1.0)

                # Wind context
                wx = weather_map.get(stadium, {})
                wind_speed = wx.get("wind_speed_mph", 0)
                wind_deg = wx.get("wind_dir_degrees", 0)
                wind_label = wx.get("wind_dir_label", "calm")
                if wind_speed >= 10 and 150 <= wind_deg <= 270:
                    wind_tag = f"WIND OUT {wind_speed:.0f}mph {wind_label}"
                elif wind_speed >= 10 and (wind_deg <= 60 or wind_deg >= 330):
                    wind_tag = f"wind in {wind_speed:.0f}mph {wind_label}"
                else:
                    wind_tag = ""

                note_parts = [f"vs {opp_pitcher}", f"PF {pf:.2f}"]
                if wind_tag:
                    note_parts.append(wind_tag)

                batter_picks.append({
                    "name": name,
                    "team": batter_team,
                    "opponent": opp_team,
                    "opp_pitcher": opp_pitcher,
                    "bat_order": batter.get("batting_order"),
                    "h_proj": round(proj["h_per_game"], 2),
                    "tb_proj": round(proj["tb_per_game"], 2),
                    "hr_prob": round(cal_hr, 3) if cal_hr else 0,
                    "hr_prob_raw": round(raw_hr, 3) if raw_hr else 0,
                    "park_factor": pf,
                    "bat_side": bat_side,
                    "wind_speed": round(wind_speed, 0),
                    "wind_out": wind_speed >= 10 and 150 <= wind_deg <= 270,
                    "wind_in": wind_speed >= 10 and (wind_deg <= 60 or wind_deg >= 330),
                    "wind_tag": wind_tag,
                    "note": ", ".join(note_parts),
                })

    print(f"[daily_picks] {len(games)} games, {confirmed_games} with lineups")
    print(f"[daily_picks] {len(k_picks)} K projections, {len(batter_picks)} batter projections")

    # Debug: top 10 batters by H/G projection
    if batter_picks:
        by_h = sorted(batter_picks, key=lambda x: x["h_proj"], reverse=True)
        print(f"[daily_picks] Top 10 by Hits projection:")
        for b in by_h[:10]:
            print(f"  {b['name']:<25} {b['h_proj']:.2f} H/G  "
                  f"({b['team']} vs {b['opp_pitcher']})")

    # --- Fetch odds lines (DK/FD + Underdog) ---
    odds_lines = fetch_odds_lines()
    ud_lines = fetch_underdog_lines()
    # Merge: add Underdog lines to existing dict
    for key, entries in ud_lines.items():
        odds_lines.setdefault(key, []).extend(entries)

    # --- Attach line info to picks ---
    _attach_k_lines(k_picks, odds_lines)
    _attach_batter_lines(batter_picks, odds_lines)

    # --- Telegram high-EV alerts ---
    _send_high_ev_alerts(k_picks, batter_picks)

    # --- Generate sharp brief ---
    _write_sharp_brief(k_picks, batter_picks, today)

    # --- Generate HR list ---
    _write_hr_list(batter_picks, today)

    print(f"[daily_picks] Output written to outputs/sharp_brief.txt and outputs/hr_list.txt")


def _attach_k_lines(k_picks, odds_lines):
    """Attach book line, implied prob, and model edge to each K pick."""
    for p in k_picks:
        entry = _lookup_line(odds_lines, p["name"], "pitcher_strikeouts")
        if entry:
            p["book"] = BOOK_DISPLAY.get(entry["book"], entry["book"])
            p["book_line"] = entry["line"]
            p["book_price"] = entry["price"]
            p["book_impl"] = entry["implied_prob"]
            p["model_prob"] = _poisson_over_prob(p["proj_k"], entry["line"])
            p["edge"] = p["model_prob"] - p["book_impl"]
            p["has_line"] = True
        else:
            p["has_line"] = False


def _attach_batter_lines(batter_picks, odds_lines):
    """Attach book lines for hits, TB, and HR markets to each batter pick."""
    market_map = [
        ("batter_hits", "h_proj"),
        ("batter_total_bases", "tb_proj"),
        ("batter_home_runs", None),  # HR uses hr_prob directly
    ]
    for b in batter_picks:
        b["has_any_line"] = False
        for market, proj_key in market_map:
            short = market.replace("batter_", "")  # hits, total_bases, home_runs
            entry = _lookup_line(odds_lines, b["name"], market)
            if entry:
                b[f"{short}_book"] = BOOK_DISPLAY.get(entry["book"], entry["book"])
                b[f"{short}_line"] = entry["line"]
                b[f"{short}_price"] = entry["price"]
                b[f"{short}_impl"] = entry["implied_prob"]
                if market == "batter_home_runs":
                    # hr_prob is already P(>=1 HR) which equals P(over 0.5)
                    b[f"{short}_model"] = b["hr_prob"]
                else:
                    lam = b[proj_key]
                    b[f"{short}_model"] = _poisson_over_prob(lam, entry["line"])
                b[f"{short}_edge"] = b[f"{short}_model"] - b[f"{short}_impl"]
                b["has_any_line"] = True
            else:
                b[f"{short}_book"] = None


def _send_high_ev_alerts(k_picks, batter_picks):
    """Send Telegram alerts for any pick with edge > 10% and a confirmed book line."""
    from messenger import _send_telegram

    alerts = []

    for p in k_picks:
        if p.get("has_line") and p.get("edge", 0) > 0.10:
            alerts.append(
                f"\U0001f6a8 High EV Alert: {p['name']} Ks OVER {p['book_line']} "
                f"-- Model: {p['model_prob']:.0%}, Book: {p['book_impl']:.0%}, "
                f"Edge: {p['edge']:.0%}"
            )

    for b in batter_picks:
        for mkt, label in [("hits", "Hits"), ("total_bases", "TB"), ("home_runs", "HR")]:
            if b.get(f"{mkt}_book") and b.get(f"{mkt}_edge", 0) > 0.10:
                alerts.append(
                    f"\U0001f6a8 High EV Alert: {b['name']} {label} OVER {b[f'{mkt}_line']} "
                    f"-- Model: {b[f'{mkt}_model']:.0%}, Book: {b[f'{mkt}_impl']:.0%}, "
                    f"Edge: {b[f'{mkt}_edge']:.0%}"
                )

    if not alerts:
        print("[daily_picks] No high-EV alerts (>10% edge) to send.")
        return

    text = "\n\n".join(alerts)
    print(f"[daily_picks] Sending {len(alerts)} high-EV Telegram alerts...")
    _send_telegram(text)


def _format_k_line(p):
    """Format line availability string for a K pick."""
    if not p.get("has_line"):
        return "NO LINE -- model only"
    return (f"{p['book']} O{p['book_line']} ({_format_american(p['book_price'])})  "
            f"model {p['model_prob']:.0%} vs impl {p['book_impl']:.0%}  "
            f"edge {p['edge']:+.1%}")


def _format_batter_line(b, market):
    """Format line availability string for a batter market."""
    if b.get(f"{market}_book") is None:
        return "NO LINE -- model only"
    return (f"{b[f'{market}_book']} O{b[f'{market}_line']} "
            f"({_format_american(b[f'{market}_price'])})  "
            f"model {b[f'{market}_model']:.0%} vs impl {b[f'{market}_impl']:.0%}  "
            f"edge {b[f'{market}_edge']:+.1%}")


def _write_sharp_brief(k_picks, batter_picks, today):
    """Write the sharp brief with top picks per market."""
    out_dir = CFG.outputs_dir
    out_dir.mkdir(exist_ok=True)

    lines = [f"SHARP BRIEF -- {today}", "=" * 50, ""]

    # --- K Props: 4.5 and 6.5 lines ---
    lines.append("PITCHER K PROPS")
    lines.append("-" * 40)

    k_sorted = sorted(k_picks, key=lambda x: x["proj_k"], reverse=True)

    # Over 6.5 candidates
    over_65 = [p for p in k_sorted if p["proj_k"] > 6.5]
    if over_65:
        lines.append("")
        lines.append("  OVER 6.5 Ks:")
        for p in over_65[:5]:
            conf = "HIGH" if p["proj_k"] >= 7.5 else "MED"
            line_str = _format_k_line(p)
            lines.append(f"    {p['name']:<22} proj {p['proj_k']:>4} Ks  "
                         f"K/9 {p['k9']:>4}  [{conf}]  {line_str}  {p['note']}")

    # Over 4.5 candidates (proj > 4.5, exclude those already in 6.5)
    over_45 = [p for p in k_sorted if 4.5 < p["proj_k"] <= 6.5]
    if over_45:
        lines.append("")
        lines.append("  OVER 4.5 Ks:")
        for p in over_45[:5]:
            conf = "HIGH" if p["proj_k"] >= 5.5 else "MED"
            line_str = _format_k_line(p)
            lines.append(f"    {p['name']:<22} proj {p['proj_k']:>4} Ks  "
                         f"K/9 {p['k9']:>4}  [{conf}]  {line_str}  {p['note']}")

    lines.append("")

    # --- Hits: 1.5 line ---
    lines.append("BATTER HITS PROPS (vs 1.5 line)")
    lines.append("-" * 40)

    hits_sorted = sorted(batter_picks, key=lambda x: x["h_proj"], reverse=True)
    over_15h = [b for b in hits_sorted if b["h_proj"] > 1.5]
    if over_15h:
        for b in over_15h[:8]:
            conf = "HIGH" if b["h_proj"] >= 1.8 else "MED"
            line_str = _format_batter_line(b, "hits")
            lines.append(f"    {b['name']:<22} proj {b['h_proj']:>4} H/G  "
                         f"[{conf}]  {line_str}  {b['note']}")
    else:
        lines.append("    No batters projecting over 1.5 H/G today")
    lines.append("")

    # --- TB: 2.5 line ---
    lines.append("BATTER TOTAL BASES (vs 2.5 line)")
    lines.append("-" * 40)

    tb_sorted = sorted(batter_picks, key=lambda x: x["tb_proj"], reverse=True)
    over_25tb = [b for b in tb_sorted if b["tb_proj"] > 2.5]
    if over_25tb:
        for b in over_25tb[:8]:
            conf = "HIGH" if b["tb_proj"] >= 3.0 else "MED"
            line_str = _format_batter_line(b, "total_bases")
            lines.append(f"    {b['name']:<22} proj {b['tb_proj']:>4} TB/G  "
                         f"[{conf}]  {line_str}  {b['note']}")
    else:
        lines.append("    No batters projecting over 2.5 TB/G today")
    lines.append("")

    # --- HR: top quintile ---
    lines.append("HR PROPS (top quintile, calibrated)")
    lines.append("-" * 40)

    hr_sorted = sorted(batter_picks, key=lambda x: x["hr_prob"], reverse=True)
    # Top quintile = top 20%
    n_top = max(1, len(hr_sorted) // 5)
    top_hr = hr_sorted[:n_top]
    for b in top_hr[:8]:
        if b["hr_prob"] < 0.05:
            break
        conf = "HIGH" if b["hr_prob"] >= 0.18 else "MED" if b["hr_prob"] >= 0.14 else "LOW"
        line_str = _format_batter_line(b, "home_runs")
        lines.append(f"    {b['name']:<22} {b['hr_prob']:>5.1%} HR prob  "
                     f"PF {b['park_factor']:.2f}  [{conf}]  {line_str}  {b['note']}")

    lines.append("")
    lines.append(f"Generated: {today}")
    lines.append(f"K picks: {len(k_picks)} | Batter picks: {len(batter_picks)}")

    text = "\n".join(lines)
    with open(out_dir / "sharp_brief.txt", "w") as f:
        f.write(text)
    print(f"[daily_picks] sharp_brief.txt written ({len(lines)} lines)")


def _write_hr_list(batter_picks, today):
    """Write the HR list: top 10 batters most likely to homer, by calibrated HR probability."""
    out_dir = CFG.outputs_dir
    out_dir.mkdir(exist_ok=True)

    ranked = sorted(batter_picks, key=lambda b: b.get("hr_prob", 0), reverse=True)

    lines = []
    for i, b in enumerate(ranked[:10], 1):
        prob_pct = f"{b['hr_prob']*100:.1f}%" if b.get("hr_prob") else "—"
        lines.append(f"{i}. {b['name']} ({b['team']}) — {prob_pct}")

    if not lines:
        lines.append("No batter projections available today.")

    text = "\n".join(lines)
    with open(out_dir / "hr_list.txt", "w") as f:
        f.write(text)
    print(f"[daily_picks] hr_list.txt written ({len(lines)} names)")
