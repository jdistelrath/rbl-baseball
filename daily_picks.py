"""
Daily Picks Pipeline.
Pulls today's confirmed lineups and starting pitchers, runs K/Hits/TB/HR
models, outputs sharp_brief.txt and floor_list.txt.

Usage: python main.py daily_picks
"""

import math
from datetime import date

import requests

from config import CFG
from backtest_props import (
    PARK_HR_FACTOR, TEAM_STADIUM,
    _calibrate_hr_prob, _project_hr_prob, _build_team_pitcher_hr9,
    K_BIAS_CORRECTION,
)


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
                    "note": f"vs {opp_pitcher}, PF {pf:.2f}",
                })

    print(f"[daily_picks] {len(games)} games, {confirmed_games} with lineups")
    print(f"[daily_picks] {len(k_picks)} K projections, {len(batter_picks)} batter projections")

    # --- Generate sharp brief ---
    _write_sharp_brief(k_picks, batter_picks, today)

    # --- Generate floor list ---
    _write_floor_list(k_picks, batter_picks, today)

    print(f"[daily_picks] Output written to outputs/sharp_brief.txt and outputs/floor_list.txt")


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
            lines.append(f"    {p['name']:<22} proj {p['proj_k']:>4} Ks  "
                         f"K/9 {p['k9']:>4}  [{conf}]  {p['note']}")

    # Over 4.5 candidates (proj > 4.5, exclude those already in 6.5)
    over_45 = [p for p in k_sorted if 4.5 < p["proj_k"] <= 6.5]
    if over_45:
        lines.append("")
        lines.append("  OVER 4.5 Ks:")
        for p in over_45[:5]:
            conf = "HIGH" if p["proj_k"] >= 5.5 else "MED"
            lines.append(f"    {p['name']:<22} proj {p['proj_k']:>4} Ks  "
                         f"K/9 {p['k9']:>4}  [{conf}]  {p['note']}")

    lines.append("")

    # --- Hits: 1.5 line ---
    lines.append("BATTER HITS PROPS (vs 1.5 line)")
    lines.append("-" * 40)

    hits_sorted = sorted(batter_picks, key=lambda x: x["h_proj"], reverse=True)
    over_15h = [b for b in hits_sorted if b["h_proj"] > 1.5]
    if over_15h:
        for b in over_15h[:8]:
            conf = "HIGH" if b["h_proj"] >= 1.8 else "MED"
            lines.append(f"    {b['name']:<22} proj {b['h_proj']:>4} H/G  "
                         f"[{conf}]  {b['note']}")
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
            lines.append(f"    {b['name']:<22} proj {b['tb_proj']:>4} TB/G  "
                         f"[{conf}]  {b['note']}")
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
        lines.append(f"    {b['name']:<22} {b['hr_prob']:>5.1%} HR prob  "
                     f"PF {b['park_factor']:.2f}  [{conf}]  {b['note']}")

    lines.append("")
    lines.append(f"Generated: {today}")
    lines.append(f"K picks: {len(k_picks)} | Batter picks: {len(batter_picks)}")

    text = "\n".join(lines)
    with open(out_dir / "sharp_brief.txt", "w") as f:
        f.write(text)
    print(f"[daily_picks] sharp_brief.txt written ({len(lines)} lines)")


def _write_floor_list(k_picks, batter_picks, today):
    """Write the floor list: top 10 names ranked by composite score."""
    out_dir = CFG.outputs_dir
    out_dir.mkdir(exist_ok=True)

    # Composite score for batters: normalize each metric to 0-1 and average
    scored = []
    if batter_picks:
        max_h = max(b["h_proj"] for b in batter_picks) or 1
        max_tb = max(b["tb_proj"] for b in batter_picks) or 1
        max_hr = max(b["hr_prob"] for b in batter_picks) or 1

        for b in batter_picks:
            composite = (
                (b["h_proj"] / max_h) * 0.30 +
                (b["tb_proj"] / max_tb) * 0.35 +
                (b["hr_prob"] / max_hr) * 0.35
            )
            scored.append((b["name"], b["team"], composite))

    # Add K pitchers with a normalized score
    if k_picks:
        max_k = max(p["proj_k"] for p in k_picks) or 1
        for p in k_picks:
            composite = (p["proj_k"] / max_k) * 0.80  # K pitchers score slightly lower weight
            scored.append((p["name"], p["team"], composite))

    scored.sort(key=lambda x: x[2], reverse=True)

    lines = []
    for i, (name, team, _) in enumerate(scored[:10], 1):
        lines.append(f"{i}. {name}")

    text = "\n".join(lines)
    with open(out_dir / "floor_list.txt", "w") as f:
        f.write(text)
    print(f"[daily_picks] floor_list.txt written ({len(lines)} names)")
