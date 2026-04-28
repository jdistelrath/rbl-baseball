"""
Underdog Draft module.

Projects fantasy points for today's confirmed starters under Underdog's
6-man (1 P / 2 IF / 2 OF / 1 FLEX) snake-draft scoring, then builds a
draft cheat sheet (rounds, position boards, optimal lineup).
"""

from daily_picks import _find_pitcher_row, _find_batter_row


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

HITTER_SCORING = {
    "single": 3.0,
    "double": 6.0,
    "triple": 8.0,
    "home_run": 10.0,
    "walk": 3.0,
    "hbp": 3.0,
    "rbi": 2.0,
    "run": 2.0,
    "stolen_base": 4.0,
}

PITCHER_SCORING = {
    "win": 2.0,
    "quality_start": 3.0,
    "strikeout": 1.0,
    "inning_pitched": 1.0,
    "earned_run": -1.0,
}


# ---------------------------------------------------------------------------
# Position classification — IF / OF / P. DH counts as OF per Underdog rules.
# Catcher (C) is folded into IF so backstops still get drafted.
# ---------------------------------------------------------------------------

IF_POSITIONS = {"1B", "2B", "3B", "SS", "C", "IF"}
OF_POSITIONS = {"LF", "CF", "RF", "DH", "OF"}
P_POSITIONS = {"P", "SP", "RP"}


def classify_position(pos):
    if not pos:
        return None
    p = pos.upper().strip()
    if p in P_POSITIONS:
        return "P"
    if p in IF_POSITIONS:
        return "IF"
    if p in OF_POSITIONS:
        return "OF"
    return None


# Lineup-spot multipliers for league-baseline R/RBI rates.
# Index 0 unused; 1..9 = batting order spots.
RUN_MULT = [0, 1.20, 1.15, 1.05, 1.00, 0.95, 0.90, 0.85, 0.80, 0.85]
RBI_MULT = [0, 0.65, 0.85, 1.20, 1.30, 1.20, 1.05, 0.95, 0.85, 0.75]

LEAGUE_RBI_PER_PA = 0.12
LEAGUE_R_PER_PA = 0.11


# ---------------------------------------------------------------------------
# Hitter projection
# ---------------------------------------------------------------------------

def _safe_int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _safe_float(v, default=0.0):
    try:
        f = float(v)
        if f != f:  # NaN
            return default
        return f
    except (TypeError, ValueError):
        return default


def project_hitter(name, team, opponent, position, lineup_spot, batter_row):
    """Return projection dict for one hitter, or None if insufficient data."""
    if batter_row is None:
        return None

    pa = _safe_int(batter_row.get("PA", 0))
    g = _safe_int(batter_row.get("G", 0))
    if pa < 30 or g < 5:
        return None

    h = _safe_int(batter_row.get("H", 0))
    hr = _safe_int(batter_row.get("HR", 0))
    doubles = _safe_int(batter_row.get("2B", 0))
    triples = _safe_int(batter_row.get("3B", 0))
    obp = _safe_float(batter_row.get("OBP", 0))
    ba = _safe_float(batter_row.get("BA", 0))

    singles = max(0, h - doubles - triples - hr)

    single_per_pa = singles / pa
    double_per_pa = doubles / pa
    triple_per_pa = triples / pa
    hr_per_pa = hr / pa
    h_per_pa = h / pa

    # Walk+HBP rate ≈ OBP - H/PA (ignoring SF, close enough for fantasy projection)
    bb_hbp_per_pa = max(0.0, obp - h_per_pa)

    # Per-PA expected fantasy points from hits + walks
    fp_per_pa = (
        single_per_pa * HITTER_SCORING["single"]
        + double_per_pa * HITTER_SCORING["double"]
        + triple_per_pa * HITTER_SCORING["triple"]
        + hr_per_pa * HITTER_SCORING["home_run"]
        + bb_hbp_per_pa * HITTER_SCORING["walk"]
    )

    # Projected PAs based on lineup slot
    spot = lineup_spot if 1 <= lineup_spot <= 9 else 5
    proj_pa = 4.3 if spot <= 5 else 3.8

    # RBI / R from league baselines, lineup-adjusted
    rbi_proj = LEAGUE_RBI_PER_PA * RBI_MULT[spot] * proj_pa
    run_proj = LEAGUE_R_PER_PA * RUN_MULT[spot] * proj_pa

    # Component totals (used for breakdown + total)
    hits_fp = (single_per_pa * HITTER_SCORING["single"]
               + double_per_pa * HITTER_SCORING["double"]
               + triple_per_pa * HITTER_SCORING["triple"]) * proj_pa
    hr_fp = hr_per_pa * HITTER_SCORING["home_run"] * proj_pa
    walk_fp = bb_hbp_per_pa * HITTER_SCORING["walk"] * proj_pa
    rbi_fp = rbi_proj * HITTER_SCORING["rbi"]
    run_fp = run_proj * HITTER_SCORING["run"]
    sb_fp = 0.0  # No SB data available in current batter_df

    total_fp = hits_fp + hr_fp + walk_fp + rbi_fp + run_fp + sb_fp

    # Edge note
    edge_bits = []
    if hr_per_pa >= 0.05:
        edge_bits.append(f"{hr_per_pa*100:.1f}% HR/PA")
    if obp >= 0.360:
        edge_bits.append(f"{obp:.3f} OBP")
    if ba >= 0.290:
        edge_bits.append(f"{ba:.3f} BA")
    if spot <= 3:
        edge_bits.append(f"hits {spot}")
    elif spot in (4, 5):
        edge_bits.append("RBI spot")
    edge = " · ".join(edge_bits) if edge_bits else "—"

    return {
        "name": name,
        "team": team,
        "opponent": opponent,
        "position": classify_position(position) or "OF",
        "raw_position": position,
        "lineup_spot": spot,
        "projected_fp": round(total_fp, 1),
        "fp_breakdown": {
            "hits": round(hits_fp, 1),
            "hr": round(hr_fp, 1),
            "walks": round(walk_fp, 1),
            "rbi": round(rbi_fp, 1),
            "runs": round(run_fp, 1),
            "sb": round(sb_fp, 1),
        },
        "tier": _tier(total_fp),
        "key_edge": edge,
    }


# ---------------------------------------------------------------------------
# Pitcher projection
# ---------------------------------------------------------------------------

def project_pitcher(name, team, opponent, pitcher_row):
    if pitcher_row is None:
        return None

    gs = _safe_int(pitcher_row.get("GS", 0))
    ip_total = _safe_float(pitcher_row.get("IP", 0))
    k9 = _safe_float(pitcher_row.get("K9", pitcher_row.get("K/9", 0)))
    era = _safe_float(pitcher_row.get("ERA", 0))

    # IP per start, capped at 6.0
    if gs >= 1 and ip_total > 0:
        ip_per_start = min(ip_total / gs, 6.0)
    else:
        ip_per_start = 5.0

    if ip_per_start < 3.0 or k9 <= 0:
        return None

    proj_ip = ip_per_start
    proj_k = k9 * proj_ip / 9.0
    proj_er = era * proj_ip / 9.0 if era > 0 else 2.5  # league-ish default

    win_prob = 0.5

    if proj_ip >= 6.0 and era > 0 and era <= 3.50:
        qs_prob = 0.45
    elif proj_ip >= 6.0 and era > 0 and era <= 4.20:
        qs_prob = 0.30
    elif proj_ip >= 6.0:
        qs_prob = 0.15
    else:
        qs_prob = 0.10

    ip_fp = proj_ip * PITCHER_SCORING["inning_pitched"]
    k_fp = proj_k * PITCHER_SCORING["strikeout"]
    er_fp = proj_er * PITCHER_SCORING["earned_run"]
    win_fp = win_prob * PITCHER_SCORING["win"]
    qs_fp = qs_prob * PITCHER_SCORING["quality_start"]

    total_fp = ip_fp + k_fp + er_fp + win_fp + qs_fp

    edge_bits = []
    if k9 >= 9.5:
        edge_bits.append(f"{k9:.1f} K/9")
    elif k9 >= 8.0:
        edge_bits.append(f"solid {k9:.1f} K/9")
    if era > 0 and era <= 3.50:
        edge_bits.append(f"{era:.2f} ERA")
    elif era > 0 and era >= 5.0:
        edge_bits.append(f"shaky {era:.2f} ERA")
    if proj_ip >= 6.0:
        edge_bits.append("QS upside")
    edge = " · ".join(edge_bits) if edge_bits else "—"

    return {
        "name": name,
        "team": team,
        "opponent": opponent,
        "position": "P",
        "raw_position": "SP",
        "lineup_spot": 0,
        "projected_fp": round(total_fp, 1),
        "fp_breakdown": {
            "ip": round(ip_fp, 1),
            "k": round(k_fp, 1),
            "er": round(er_fp, 1),
            "win": round(win_fp, 1),
            "qs": round(qs_fp, 1),
        },
        "tier": _tier(total_fp),
        "key_edge": edge,
    }


# ---------------------------------------------------------------------------
# Tiering
# ---------------------------------------------------------------------------

def _tier(fp):
    if fp > 30:
        return "ELITE"
    if fp >= 20:
        return "SOLID"
    if fp >= 12:
        return "VALUE"
    return "AVOID"


# ---------------------------------------------------------------------------
# Top-level: project every player on the slate
# ---------------------------------------------------------------------------

def project_all_players(games_with_lineups, batter_df, pitcher_df):
    """
    Args:
      games_with_lineups: list of game dicts. Each dict should contain:
        home_team, away_team, home_pitcher_name, away_pitcher_name,
        and 'lineups' = {home: [...], away: [...]} from get_confirmed_lineups
        (may be None if lineups not yet posted).
      batter_df, pitcher_df: season stat DataFrames.

    Returns: list of player projection dicts.
    """
    players = []

    for game in games_with_lineups:
        home = game.get("home_team", "")
        away = game.get("away_team", "")

        # Pitchers — use both starters even if lineups aren't out yet.
        for role, opp in [("home_pitcher_name", away), ("away_pitcher_name", home)]:
            pname = game.get(role, "TBD")
            if not pname or pname == "TBD":
                continue
            p_row = _find_pitcher_row(pname, pitcher_df)
            if p_row is None:
                continue
            p_team = home if role == "home_pitcher_name" else away
            proj = project_pitcher(pname, p_team, opp, p_row)
            if proj:
                players.append(proj)

        # Hitters — need confirmed lineups
        lineup = game.get("lineups")
        if not lineup:
            continue

        for side, opp in [("home", away), ("away", home)]:
            bteam = home if side == "home" else away
            for batter in lineup.get(side, []):
                name = batter.get("name", "")
                if not name:
                    continue
                pos_class = classify_position(batter.get("position", ""))
                if pos_class is None or pos_class == "P":
                    # Skip pitchers in the lineup (NL DH avoids this anyway)
                    continue
                b_row = _find_batter_row(name, batter_df)
                if b_row is None:
                    continue
                proj = project_hitter(
                    name=name,
                    team=bteam,
                    opponent=opp,
                    position=batter.get("position", ""),
                    lineup_spot=batter.get("batting_order", 0) or 0,
                    batter_row=b_row,
                )
                if proj:
                    players.append(proj)

    return players


# ---------------------------------------------------------------------------
# Cheat sheet
# ---------------------------------------------------------------------------

def _build_optimal_lineup(players):
    """
    Greedy: best P, best 2 IF, best 2 OF, then best remaining IF/OF for FLEX.
    If the resulting 6 are all on one team, swap the FLEX for the next-best
    different-team IF/OF eligible player.
    """
    pitchers = sorted([p for p in players if p["position"] == "P"],
                      key=lambda x: x["projected_fp"], reverse=True)
    infielders = sorted([p for p in players if p["position"] == "IF"],
                        key=lambda x: x["projected_fp"], reverse=True)
    outfielders = sorted([p for p in players if p["position"] == "OF"],
                         key=lambda x: x["projected_fp"], reverse=True)

    if not pitchers or len(infielders) < 2 or len(outfielders) < 2:
        return None

    p = pitchers[0]
    if1, if2 = infielders[0], infielders[1]
    of1, of2 = outfielders[0], outfielders[1]

    used_names = {p["name"], if1["name"], if2["name"], of1["name"], of2["name"]}
    flex_pool = [x for x in (infielders + outfielders)
                 if x["name"] not in used_names]
    flex_pool.sort(key=lambda x: x["projected_fp"], reverse=True)
    flex = flex_pool[0] if flex_pool else None
    if flex is None:
        return None

    lineup = {"P": p, "IF1": if1, "IF2": if2, "OF1": of1, "OF2": of2, "FLEX": flex}

    # Two-team rule: roster must include players from at least 2 teams.
    teams = {x["team"] for x in lineup.values()}
    if len(teams) < 2 and flex_pool:
        sole_team = next(iter(teams))
        for candidate in flex_pool:
            if candidate["team"] != sole_team:
                lineup["FLEX"] = candidate
                teams = {x["team"] for x in lineup.values()}
                break

    total_fp = round(sum(x["projected_fp"] for x in lineup.values()), 1)
    return {
        **lineup,
        "total_fp": total_fp,
        "teams": sorted(teams),
    }


def build_draft_cheat_sheet(players):
    if not players:
        return {
            "round_1": [], "round_2": [], "round_3": [],
            "pitchers": [], "if_targets": [], "of_targets": [],
            "optimal_lineup": None,
        }

    overall = sorted(players, key=lambda x: x["projected_fp"], reverse=True)

    pitchers = [p for p in overall if p["position"] == "P"]
    infielders = [p for p in overall if p["position"] == "IF"]
    outfielders = [p for p in overall if p["position"] == "OF"]

    return {
        "round_1": overall[:4],
        "round_2": overall[4:8],
        "round_3": overall[8:12],
        "pitchers": pitchers[:5],
        "if_targets": infielders[:5],
        "of_targets": outfielders[:5],
        "optimal_lineup": _build_optimal_lineup(players),
    }
