"""
Flask web app for MLB prop model.
Run: python app.py   (serves on localhost:5050)
"""

import json
import math
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
    _get_team_k_rate, fetch_odds_lines, _lookup_line, _normalize_name,
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


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    """Run strategy backtester and return JSON results."""
    try:
        params = request.get_json(force=True)
        data = _run_backtest(params)
        return jsonify(data)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Picks pipeline (reuses daily_picks logic, returns structured data)
# ---------------------------------------------------------------------------

def _run_picks_pipeline():
    games = get_today_schedule()
    if not games:
        return {"k_picks": [], "batter_picks": [], "floor_list": [], "meta": {"games": 0}}

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

    # Attach odds lines
    odds_lines = {}
    try:
        odds_lines = fetch_odds_lines()
    except Exception as e:
        print(f"[app] Odds fetch failed: {e}")

    for p in k_picks:
        entry = _lookup_line(odds_lines, p["name"], "pitcher_strikeouts")
        if entry:
            p["book"] = BOOK_DISPLAY.get(entry["book"], entry["book"])
            p["book_line"] = entry["line"]
            p["book_price"] = entry["price"]
            p["book_impl"] = round(entry["implied_prob"], 3)
            p["model_prob"] = round(_poisson_over_prob(p["proj_k"], entry["line"]), 3)
            p["edge"] = round(p["model_prob"] - p["book_impl"], 3)
        else:
            p["book"] = None
            p["edge"] = 0

    for b in batter_picks:
        for mkt, proj_key in [("batter_hits", "h_proj"), ("batter_total_bases", "tb_proj"),
                               ("batter_home_runs", None)]:
            short = mkt.replace("batter_", "")
            entry = _lookup_line(odds_lines, b["name"], mkt)
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
            else:
                b[f"{short}_book"] = None
                b[f"{short}_edge"] = 0

    # Build floor list (composite score)
    floor = []
    if batter_picks:
        max_h = max((b["h_proj"] for b in batter_picks), default=1) or 1
        max_tb = max((b["tb_proj"] for b in batter_picks), default=1) or 1
        max_hr = max((b["hr_prob"] for b in batter_picks), default=1) or 0.01
        for b in batter_picks:
            score = (b["h_proj"]/max_h)*0.30 + (b["tb_proj"]/max_tb)*0.35 + (b["hr_prob"]/max_hr)*0.35
            floor.append({"name": b["name"], "team": b["team"], "type": "B", "score": round(score, 3)})
    if k_picks:
        max_k = max((p["proj_k"] for p in k_picks), default=1) or 1
        for p in k_picks:
            floor.append({"name": p["name"], "team": p["team"], "type": "P", "score": round((p["proj_k"]/max_k)*0.8, 3)})
    floor.sort(key=lambda x: x["score"], reverse=True)

    lines_note = ""
    if not odds_lines:
        lines_note = "Lines available pre-game only. Check back tomorrow morning."

    return {
        "k_picks": sorted(k_picks, key=lambda x: x["proj_k"], reverse=True),
        "batter_picks": sorted(batter_picks, key=lambda x: x["hr_prob"], reverse=True),
        "floor_list": floor[:10],
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

def _run_backtest(params):
    prop_type = params.get("prop_type", "pitcher_strikeouts")
    min_edge = float(params.get("min_edge", 0.03))
    year = int(params.get("year", 2024))

    from backtest_props import (
        _fetch_pitcher_game_logs, _fetch_batter_game_logs,
        _load_cache, _save_cache, _project_pitcher_ks, _project_batter_rates,
        K_BIAS_CORRECTION,
    )

    batter_df = get_batter_statcast(season=year - 1)
    pitcher_df = get_pitcher_statcast(season=year - 1)

    STAKE = 1.0
    JUICE_BRK = 0.5238  # breakeven at -110

    results_by_week = {}
    total_bets = 0
    total_wins = 0
    total_pnl = 0.0

    if prop_type == "pitcher_strikeouts":
        starters = pitcher_df[(pitcher_df["GS"] >= 10) & (pitcher_df["K9"] > 0)]
        line = float(params.get("line", 5.5))

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

                # Weekly bucketing
                try:
                    dt = datetime.strptime(g["date"], "%Y-%m-%d")
                    week_key = dt.strftime("%Y-W%W")
                    results_by_week.setdefault(week_key, 0.0)
                    results_by_week[week_key] += pnl
                except (ValueError, TypeError):
                    pass

    elif prop_type in ("batter_hits", "batter_total_bases"):
        line = float(params.get("line", 1.5 if prop_type == "batter_hits" else 2.5))
        proj_key = "h" if prop_type == "batter_hits" else "tb"

        if "H" not in batter_df.columns:
            return {"error": "Batter stats missing H/TB columns. Clear cache and re-run."}

        qualified = batter_df[batter_df["G"] >= 50]
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
