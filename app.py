"""
Flask web app for MLB prop model.
Run: python app.py   (serves on localhost:5050)
"""

import json
import math
import os
import traceback
from datetime import date, datetime, timedelta

from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# Bootstrap — ensure .env is loaded before any project imports
# ---------------------------------------------------------------------------
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent / ".env")

from config import CFG
from data_fetcher import (
    get_today_schedule, get_confirmed_lineups,
    get_batter_statcast, get_pitcher_statcast,
)
from daily_picks import (
    _project_ks, _project_batter, _find_pitcher_row, _find_batter_row,
    _get_team_k_rate, fetch_odds_lines, fetch_underdog_lines,
    _lookup_line, _lookup_all_books, _normalize_name,
    _poisson_over_prob, _american_to_implied, BOOK_DISPLAY, ODDS_BOOKS,
)
from backtest_props import (
    PARK_HR_FACTOR, TEAM_STADIUM,
    _calibrate_hr_prob, _project_hr_prob, _build_team_pitcher_hr9,
)

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/picks")
def api_picks():
    """Run daily picks pipeline and return JSON."""
    try:
        data = _run_picks_pipeline()
        return jsonify(data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/log_picks", methods=["POST"])
def api_log_picks():
    """Log today's picks snapshot for CLV tracking."""
    try:
        from clv_tracker import log_snapshot
        data = _run_picks_pipeline()
        count = log_snapshot(data["k_picks"], data["batter_picks"])
        return jsonify({"logged": count})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/close_lines", methods=["POST"])
def api_close_lines():
    """Pull closing lines for today's logged picks."""
    try:
        from clv_tracker import close_lines
        closed = close_lines()
        return jsonify({"closed": closed})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/update_outcomes", methods=["POST"])
def api_update_outcomes():
    """Update actual outcomes for logged picks."""
    try:
        from clv_tracker import update_outcomes
        updated = update_outcomes()
        return jsonify({"updated": updated})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/clv_data")
def api_clv_data():
    """Return CLV tracking data for the CLV Tracker tab."""
    try:
        from clv_tracker import get_clv_summary
        return jsonify(get_clv_summary())
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/parlay_builder")
def api_parlay_builder():
    """Return today's eligible legs with model probs and book odds for the parlay builder."""
    try:
        data = _run_picks_pipeline()

        legs = []
        # K legs
        for p in data["k_picks"]:
            if not p.get("book"):
                continue
            legs.append({
                "id": f"k_{p['name']}",
                "player": p["name"], "team": p["team"],
                "type": "Ks", "line": p["book_line"], "side": "Over",
                "book": p["book"], "price": p["book_price"],
                "model_prob": p["model_prob"], "impl_prob": p["book_impl"],
                "edge": p["edge"],
                "kelly_frac": p.get("kelly_frac", 0),
                "note": p["note"],
            })

        # Batter legs (each market is a separate leg)
        for b in data["batter_picks"]:
            for mkt, label in [("hits", "Hits"), ("total_bases", "TB"), ("home_runs", "HR")]:
                if not b.get(f"{mkt}_book"):
                    continue
                edge = b.get(f"{mkt}_edge", 0) or 0
                legs.append({
                    "id": f"{mkt}_{b['name']}",
                    "player": b["name"], "team": b["team"],
                    "type": label, "line": b[f"{mkt}_line"], "side": "Over",
                    "book": b[f"{mkt}_book"], "price": b[f"{mkt}_price"],
                    "model_prob": b.get(f"{mkt}_model", 0),
                    "impl_prob": b.get(f"{mkt}_impl", 0),
                    "edge": edge,
                    "kelly_frac": b.get(f"{mkt}_kelly_frac", 0),
                    "note": f"vs {b['opp_pitcher']}",
                    "is_hr": mkt == "home_runs",
                })

        # Sort by edge descending
        legs.sort(key=lambda x: x["edge"], reverse=True)

        # Build recommended parlays: top 3-5 leg combos from green legs, no same player
        green = [l for l in legs if l["edge"] > 0.05]
        recommended = _build_recommended_parlays(green)

        return jsonify({"legs": legs, "recommended": recommended})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _build_recommended_parlays(green_legs):
    """Build top 5 parlay combos from green legs, no duplicate players or same team."""
    from itertools import combinations
    if len(green_legs) < 3:
        return []

    results = []
    for size in (3, 4):
        if len(green_legs) < size:
            continue
        for combo in combinations(green_legs, size):
            players = set(l["player"] for l in combo)
            if len(players) < size:
                continue  # same player in multiple legs
            teams = [l["team"] for l in combo]
            if len(set(teams)) < len(teams):
                continue  # same team stacking

            combined_prob = 1.0
            combined_dec = 1.0
            for l in combo:
                combined_prob *= l["model_prob"]
                p = l["price"]
                dec = (p / 100.0 + 1.0) if p > 0 else (100.0 / abs(p) + 1.0)
                combined_dec *= dec

            ev = combined_prob * combined_dec - 1.0
            results.append({
                "legs": [{"player": l["player"], "type": l["type"], "line": l["line"],
                          "price": l["price"], "edge": l["edge"]} for l in combo],
                "size": size,
                "combined_prob": round(combined_prob, 4),
                "combined_dec_odds": round(combined_dec, 2),
                "combined_american": round((combined_dec - 1) * 100, 0) if combined_dec >= 2 else round(-100 / (combined_dec - 1), 0),
                "ev_per_dollar": round(ev, 4),
            })

    results.sort(key=lambda x: x["ev_per_dollar"], reverse=True)
    return results[:5]


@app.route("/api/backtest_hr", methods=["POST"])
def api_backtest_hr():
    """Run HR list daily backtest and return JSON results."""
    try:
        params = request.get_json(force=True)
        print(f"[api/backtest_hr] Running with params: {params}")
        data = _run_hr_backtest(params)
        print(f"[api/backtest_hr] Complete: {data.get('total_bets', 0)} bets, "
              f"PnL=${data.get('pnl', 0):.2f}")
        return jsonify(data)
    except Exception as e:
        print(f"[api/backtest_hr] EXCEPTION: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    """Run strategy backtester and return JSON results."""
    try:
        params = request.get_json(force=True)
        print(f"[api/backtest] Running with params: {params}")
        data = _run_backtest(params)
        print(f"[api/backtest] Complete: {data.get('total_bets', 0)} bets, "
              f"PnL=${data.get('pnl', 0):.2f}")
        return jsonify(data)
    except Exception as e:
        print(f"[api/backtest] EXCEPTION: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Picks pipeline (reuses daily_picks logic, returns structured data)
# ---------------------------------------------------------------------------

def _run_picks_pipeline():
    games = get_today_schedule()
    if not games:
        return {"k_picks": [], "batter_picks": [], "hr_list": [], "meta": {"games": 0}}

    batter_df = get_batter_statcast()
    pitcher_df = get_pitcher_statcast()
    team_pitcher_hr9 = _build_team_pitcher_hr9(pitcher_df)

    k_picks = []
    batter_picks = []
    confirmed = 0

    for game in games:
        gid = game["game_id"]
        home = game["home_team"]
        away = game["away_team"]

        # K projections for both starters
        for role, opp_team in [("home_pitcher_name", away), ("away_pitcher_name", home)]:
            pname = game.get(role, "TBD")
            if not pname or pname == "TBD":
                continue
            p_row = _find_pitcher_row(pname, pitcher_df)
            if p_row is None:
                continue
            opp_k = _get_team_k_rate(opp_team, batter_df)
            proj = _project_ks(p_row, opp_k)
            if proj is None or proj < 2:
                continue
            try:
                k9 = float(p_row.get("K9", 0))
                era = float(p_row.get("ERA", 0))
            except (TypeError, ValueError):
                k9, era = 0, 0
            team = home if role == "home_pitcher_name" else away
            k_picks.append({
                "name": pname, "team": team, "opponent": opp_team,
                "proj_k": round(proj, 1), "k9": round(k9, 1), "era": round(era, 2),
                "opp_k_rate": round(opp_k, 3),
                "note": f"vs {opp_team} ({opp_k:.0%} K rate)",
            })

        # Batter projections
        lineup = get_confirmed_lineups(gid)
        if lineup is None:
            continue
        confirmed += 1

        for side, opp_team in [("home", away), ("away", home)]:
            batter_team = home if side == "home" else away
            is_home = side == "home"
            opp_pitcher = game.get("away_pitcher_name" if side == "home" else "home_pitcher_name", "TBD")

            if is_home:
                park_stadium = TEAM_STADIUM.get(batter_team, "")
            else:
                park_stadium = TEAM_STADIUM.get(opp_team, "")
            pf = PARK_HR_FACTOR.get(park_stadium, 1.0)

            for batter in lineup.get(side, []):
                name = batter["name"]
                bat_side = batter.get("bat_side", "R")
                b_row = _find_batter_row(name, batter_df)
                if b_row is None:
                    continue
                proj = _project_batter(b_row)
                if proj is None:
                    continue

                raw_hr = _project_hr_prob(b_row, opp_team, team_pitcher_hr9, bat_side, is_home, opp_team)
                cal_hr = _calibrate_hr_prob(raw_hr) if raw_hr else 0

                batter_picks.append({
                    "name": name, "team": batter_team, "opponent": opp_team,
                    "opp_pitcher": opp_pitcher, "bat_order": batter.get("batting_order"),
                    "h_proj": round(proj["h_per_game"], 2),
                    "tb_proj": round(proj["tb_per_game"], 2),
                    "hr_prob": round(cal_hr, 3) if cal_hr else 0,
                    "park_factor": pf, "bat_side": bat_side,
                })

    # Attach odds lines (DK/FD + Underdog)
    odds_lines = {}
    try:
        odds_lines = fetch_odds_lines()
    except Exception as e:
        print(f"[app] Odds fetch failed: {e}")
    try:
        ud = fetch_underdog_lines()
        for key, entries in ud.items():
            odds_lines.setdefault(key, []).extend(entries)
    except Exception as e:
        print(f"[app] Underdog fetch failed: {e}")

    for p in k_picks:
        entry = _lookup_line(odds_lines, p["name"], "pitcher_strikeouts")
        all_bks = _lookup_all_books(odds_lines, p["name"], "pitcher_strikeouts")
        if entry:
            p["book"] = BOOK_DISPLAY.get(entry["book"], entry["book"])
            p["book_line"] = entry["line"]
            p["book_price"] = entry["price"]
            p["book_impl"] = round(entry["implied_prob"], 3)
            p["model_prob"] = round(_poisson_over_prob(p["proj_k"], entry["line"]), 3)
            p["edge"] = round(p["model_prob"] - p["book_impl"], 3)
            p["all_books"] = list(all_bks.keys())
        else:
            p["book"] = None
            p["edge"] = 0
            p["all_books"] = []

    for b in batter_picks:
        for mkt, proj_key in [("batter_hits", "h_proj"), ("batter_total_bases", "tb_proj"),
                               ("batter_home_runs", None)]:
            short = mkt.replace("batter_", "")
            entry = _lookup_line(odds_lines, b["name"], mkt)
            all_bks = _lookup_all_books(odds_lines, b["name"], mkt)
            if entry:
                b[f"{short}_book"] = BOOK_DISPLAY.get(entry["book"], entry["book"])
                b[f"{short}_line"] = entry["line"]
                b[f"{short}_price"] = entry["price"]
                b[f"{short}_impl"] = round(entry["implied_prob"], 3)
                if mkt == "batter_home_runs":
                    b[f"{short}_model"] = b["hr_prob"]
                else:
                    b[f"{short}_model"] = round(_poisson_over_prob(b[proj_key], entry["line"]), 3)
                b[f"{short}_edge"] = round(b[f"{short}_model"] - b[f"{short}_impl"], 3)
                b[f"{short}_all_books"] = list(all_bks.keys())
            else:
                b[f"{short}_book"] = None
                b[f"{short}_edge"] = 0
                b[f"{short}_all_books"] = []

    # Kelly sizing — return raw fraction, client applies mode/cap
    def _kelly_raw(model_prob, american_odds):
        """Raw Kelly fraction (uncapped)."""
        if american_odds > 0:
            dec = (american_odds / 100.0) + 1.0
        else:
            dec = (100.0 / abs(american_odds)) + 1.0
        b = dec - 1.0
        if b <= 0:
            return 0.0
        q = 1.0 - model_prob
        f = (model_prob * b - q) / b
        return round(max(0.0, f), 4)

    for p in k_picks:
        if p.get("book") and p.get("model_prob"):
            p["kelly_frac"] = _kelly_raw(p["model_prob"], p["book_price"])
        else:
            p["kelly_frac"] = 0

    for b in batter_picks:
        for mkt in ["hits", "total_bases", "home_runs"]:
            if b.get(f"{mkt}_book") and b.get(f"{mkt}_model"):
                b[f"{mkt}_kelly_frac"] = _kelly_raw(b[f"{mkt}_model"], b[f"{mkt}_price"])
            else:
                b[f"{mkt}_kelly_frac"] = 0

    # Build HR list: top 10 batters by calibrated HR probability
    hr_list = sorted(batter_picks, key=lambda x: x.get("hr_prob", 0), reverse=True)[:10]
    hr_list = [{"name": b["name"], "team": b["team"], "hr_prob": b["hr_prob"]} for b in hr_list]

    lines_note = ""
    if not odds_lines:
        lines_note = "Lines available pre-game only. Check back tomorrow morning."

    return {
        "k_picks": sorted(k_picks, key=lambda x: x["proj_k"], reverse=True),
        "batter_picks": sorted(batter_picks, key=lambda x: x["hr_prob"], reverse=True),
        "hr_list": hr_list,
        "meta": {
            "date": date.today().isoformat(),
            "games": len(games),
            "confirmed": confirmed,
            "odds_lines": len(odds_lines),
            "lines_note": lines_note,
        },
    }


# ---------------------------------------------------------------------------
# Strategy backtester
# ---------------------------------------------------------------------------

def _run_hr_backtest(params):
    """
    HR List daily backtest: for each game-day, rank batters by calibrated HR prob,
    pick top N, check if any homered. Track individual bet P&L at -110.
    """
    start_year = int(params.get("start_year", 2024))
    end_year = int(params.get("end_year", start_year))
    top_n = int(params.get("top_n", 5))

    from backtest_props import (
        _fetch_batter_game_logs, _fetch_batter_handedness,
        _build_team_pitcher_hr9, _project_hr_prob, _calibrate_hr_prob,
    )
    from collections import defaultdict

    STAKE = 1.0
    PAYOUT = 0.909  # profit on a win at -110

    results_by_week = {}
    total_bets = 0
    total_wins = 0
    total_pnl = 0.0
    total_days = 0
    days_with_hr = 0

    years = list(range(start_year, end_year + 1))

    for year in years:
        print(f"[api/backtest_hr] Processing {year}...")
        batter_df = get_batter_statcast(season=year - 1)
        pitcher_df = get_pitcher_statcast(season=year - 1)

        if batter_df.empty or "PA" not in batter_df.columns:
            continue

        team_pitcher_hr9 = _build_team_pitcher_hr9(pitcher_df)

        # Get qualified batters and their HR projections
        qualified = batter_df[batter_df["PA"] >= 200].copy()
        if qualified.empty:
            continue

        player_ids = qualified["mlbID"].astype(int).tolist()
        handedness = _fetch_batter_handedness(player_ids)

        # Score each batter's HR prob (using neutral context — no per-game matchup)
        batter_scores = []
        for _, row in qualified.iterrows():
            pid = int(row["mlbID"])
            name = row["Name"]
            bat_side = handedness.get(pid, "R")
            # Use league-average opponent for ranking
            raw_prob = _project_hr_prob(row, "", team_pitcher_hr9, bat_side, True, "")
            if raw_prob is None or raw_prob < 0.01:
                continue
            cal_prob = _calibrate_hr_prob(raw_prob)
            batter_scores.append({
                "pid": pid, "name": name, "cal_prob": cal_prob,
            })

        batter_scores.sort(key=lambda x: x["cal_prob"], reverse=True)
        top_batters = batter_scores[:top_n]

        if not top_batters:
            continue

        print(f"  Top {top_n}: {', '.join(b['name'] for b in top_batters[:3])}...")

        # Fetch game logs for top N batters, bucket by date
        daily_results = defaultdict(dict)  # date -> {pid: homered}

        for b in top_batters:
            import requests as _req
            url = (
                f"https://statsapi.mlb.com/api/v1/people/{b['pid']}/stats"
                f"?stats=gameLog&group=hitting&season={year}&gameType=R"
            )
            try:
                resp = _req.get(url, timeout=15)
                resp.raise_for_status()
                splits = resp.json().get("stats", [{}])[0].get("splits", [])
            except Exception:
                continue

            for s in splits:
                stat = s.get("stat", {})
                ab = int(stat.get("atBats", 0))
                if ab == 0:
                    continue
                game_date = s.get("date", "")
                hr = int(stat.get("homeRuns", 0))
                daily_results[game_date][b["pid"]] = hr > 0

        # Score each day
        for game_date in sorted(daily_results.keys()):
            day_data = daily_results[game_date]
            if not day_data:
                continue

            total_days += 1
            any_hr = False

            for pid, homered in day_data.items():
                total_bets += 1
                if homered:
                    total_wins += 1
                    total_pnl += STAKE * PAYOUT
                    any_hr = True
                else:
                    total_pnl -= STAKE

            if any_hr:
                days_with_hr += 1

            # Weekly P&L
            try:
                dt = datetime.strptime(game_date, "%Y-%m-%d")
                week_key = dt.strftime("%Y-W%W")
                day_pnl = sum(
                    (STAKE * PAYOUT) if h else -STAKE for h in day_data.values()
                )
                results_by_week.setdefault(week_key, 0.0)
                results_by_week[week_key] += day_pnl
            except (ValueError, TypeError):
                pass

    win_rate = total_wins / total_bets * 100 if total_bets > 0 else 0
    roi = total_pnl / (total_bets * STAKE) * 100 if total_bets > 0 else 0
    day_hit_rate = days_with_hr / total_days * 100 if total_days > 0 else 0

    # Chart data
    weeks_sorted = sorted(results_by_week.keys())
    cum = 0.0
    chart_labels = []
    chart_data = []
    for w in weeks_sorted:
        cum += results_by_week[w]
        chart_labels.append(w)
        chart_data.append(round(cum, 2))

    # Best/worst month
    monthly = {}
    for w, pnl in results_by_week.items():
        try:
            dt = datetime.strptime(w + "-1", "%Y-W%W-%w")
            month = dt.strftime("%Y-%m")
        except (ValueError, TypeError):
            month = w[:7]
        monthly.setdefault(month, 0.0)
        monthly[month] += pnl

    best_month = max(monthly, key=monthly.get) if monthly else "N/A"
    worst_month = min(monthly, key=monthly.get) if monthly else "N/A"

    return {
        "total_days": total_days,
        "days_with_hr": days_with_hr,
        "day_hit_rate": round(day_hit_rate, 1),
        "total_bets": total_bets,
        "total_wins": total_wins,
        "win_rate": round(win_rate, 1),
        "pnl": round(total_pnl, 2),
        "roi": round(roi, 1),
        "best_month": f"{best_month} (${monthly.get(best_month, 0):.2f})" if monthly else "N/A",
        "worst_month": f"{worst_month} (${monthly.get(worst_month, 0):.2f})" if monthly else "N/A",
        "chart_labels": chart_labels,
        "chart_data": chart_data,
        "top_n": top_n,
    }


def _run_backtest(params):
    prop_type = params.get("prop_type", "pitcher_strikeouts")
    min_edge = float(params.get("min_edge", 0.03))
    start_year = int(params.get("start_year", params.get("year", 2024)))
    end_year = int(params.get("end_year", start_year))
    team_filter = params.get("team", "all")
    line = float(params.get("line", 5.5))

    from backtest_props import (
        _fetch_pitcher_game_logs, _fetch_batter_game_logs,
        _load_cache, _save_cache, _project_pitcher_ks, _project_batter_rates,
        K_BIAS_CORRECTION,
    )

    STAKE = 1.0
    JUICE_BRK = 0.5238  # breakeven at -110

    results_by_week = {}
    total_bets = 0
    total_wins = 0
    total_pnl = 0.0

    # Map abbreviations to full team names for filtering
    ABBR_TO_TEAM = {
        "AZ": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
        "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CWS": "Chicago White Sox",
        "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
        "DET": "Detroit Tigers", "HOU": "Houston Astros", "KC": "Kansas City Royals",
        "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
        "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "New York Mets",
        "NYY": "New York Yankees", "OAK": "Athletics", "PHI": "Philadelphia Phillies",
        "PIT": "Pittsburgh Pirates", "SD": "San Diego Padres", "SF": "San Francisco Giants",
        "SEA": "Seattle Mariners", "STL": "St. Louis Cardinals", "TB": "Tampa Bay Rays",
        "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays", "WSH": "Washington Nationals",
    }
    team_full = ABBR_TO_TEAM.get(team_filter, "") if team_filter != "all" else ""

    years = list(range(start_year, end_year + 1))

    for year in years:
        print(f"[api/backtest] Processing {year}...")
        batter_df = get_batter_statcast(season=year - 1)
        pitcher_df = get_pitcher_statcast(season=year - 1)

        if prop_type == "pitcher_strikeouts":
            starters = pitcher_df[(pitcher_df["GS"] >= 10) & (pitcher_df["K9"] > 0)]
            if team_full:
                starters = starters[starters["Team"] == team_full]

            for _, row in starters.iterrows():
                pid = int(row["mlbID"])
                proj = _project_pitcher_ks(row)
                if proj is None:
                    continue
                model_prob = _poisson_over_prob(proj, line)
                if model_prob - JUICE_BRK < min_edge:
                    continue

                logs = _fetch_pitcher_game_logs(pid, year)
                for g in logs:
                    actual = g["actual_k"]
                    won = actual > line
                    pnl = (STAKE * 0.909) if won else -STAKE
                    total_bets += 1
                    total_wins += int(won)
                    total_pnl += pnl
                    try:
                        dt = datetime.strptime(g["date"], "%Y-%m-%d")
                        week_key = dt.strftime("%Y-W%W")
                        results_by_week.setdefault(week_key, 0.0)
                        results_by_week[week_key] += pnl
                    except (ValueError, TypeError):
                        pass

        elif prop_type in ("batter_hits", "batter_total_bases"):
            proj_key = "h" if prop_type == "batter_hits" else "tb"

            if "H" not in batter_df.columns:
                continue

            qualified = batter_df[batter_df["G"] >= 50]
            if team_full:
                qualified = qualified[qualified["Team"] == team_full]

            for _, row in qualified.iterrows():
                pid = int(row["mlbID"])
                rates = _project_batter_rates(row)
                if rates[0] is None:
                    continue
                proj_val = rates[0] if proj_key == "h" else rates[1]
                model_prob = _poisson_over_prob(proj_val, line)
                if model_prob - JUICE_BRK < min_edge:
                    continue

                logs = _fetch_batter_game_logs(pid, year)
                for g in logs:
                    actual = g[f"actual_{proj_key}"]
                    won = actual > line
                    pnl = (STAKE * 0.909) if won else -STAKE
                    total_bets += 1
                    total_wins += int(won)
                    total_pnl += pnl
                    try:
                        dt = datetime.strptime(g["date"], "%Y-%m-%d")
                        week_key = dt.strftime("%Y-W%W")
                        results_by_week.setdefault(week_key, 0.0)
                        results_by_week[week_key] += pnl
                    except (ValueError, TypeError):
                        pass

    win_rate = total_wins / total_bets if total_bets > 0 else 0
    roi = total_pnl / (total_bets * STAKE) * 100 if total_bets > 0 else 0

    # Build cumulative P&L chart data
    weeks_sorted = sorted(results_by_week.keys())
    cum = 0.0
    chart_labels = []
    chart_data = []
    for w in weeks_sorted:
        cum += results_by_week[w]
        chart_labels.append(w)
        chart_data.append(round(cum, 2))

    # Best/worst month
    monthly = {}
    for w, pnl in results_by_week.items():
        month = w[:7]  # "2024-W" -> use year-month from week
        try:
            dt = datetime.strptime(w + "-1", "%Y-W%W-%w")
            month = dt.strftime("%Y-%m")
        except (ValueError, TypeError):
            pass
        monthly.setdefault(month, 0.0)
        monthly[month] += pnl

    best_month = max(monthly, key=monthly.get) if monthly else "N/A"
    worst_month = min(monthly, key=monthly.get) if monthly else "N/A"

    return {
        "total_bets": total_bets,
        "total_wins": total_wins,
        "win_rate": round(win_rate * 100, 1),
        "pnl": round(total_pnl, 2),
        "roi": round(roi, 1),
        "best_month": f"{best_month} (${monthly.get(best_month, 0):.2f})" if monthly else "N/A",
        "worst_month": f"{worst_month} (${monthly.get(worst_month, 0):.2f})" if monthly else "N/A",
        "chart_labels": chart_labels,
        "chart_data": chart_data,
    }


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=True)
