# Solomon Prompt v9 — Underdog Draft Tab

You are **Solomon**, the Builder. Execute this spec literally. Read existing files before modifying them. Do not break anything working.

## Working Directory
`/Users/jimsmini/Projects/baseball/`

## Context
The Flask app (`app.py`) has multiple tabs. `python3 app.py` serves on port 5050, bound to 0.0.0.0. `python3 main.py dry_run` must still work after this build.

## What to Build
A new **Underdog Draft** tab in the Flask app that projects fantasy points for today's players and produces a draft cheat sheet.

---

## Underdog Scoring System

### Hitters
| Stat | Points |
|---|---|
| Single | 3.0 |
| Double | 6.0 |
| Triple | 8.0 |
| Home Run | 10.0 |
| Walk | 3.0 |
| Hit By Pitch | 3.0 |
| RBI | 2.0 |
| Run | 2.0 |
| Stolen Base | 4.0 |

### Pitchers
| Stat | Points |
|---|---|
| Win | 2.0 |
| Quality Start (6+ IP, ≤3 ER) | 3.0 |
| Strikeout | 1.0 |
| Inning Pitched | 1.0 |
| Earned Run | -1.0 |

---

## Roster Requirements
- **6 players total:** 1 P, 2 IF, 2 OF, 1 FLEX (IF or OF)
- **Snake draft** — no salary cap, rank-based
- **Two-team rule:** roster must include players from at least 2 different teams
- FLEX = IF or OF only (DH counts as OF)

---

## New File: `market_underdog_draft.py`

Build this module. It projects fantasy points for all confirmed starters today.

### Hitter Projection

Project each of these per plate appearance, then multiply by projected PAs (4.3 for lineup spots 1-5, 3.8 for spots 6-9):

```python
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
```

Use existing batter Statcast data (barrel rate, ISO, HR/FB, etc.) already pulled by `data_fetcher.py`. Build per-PA rates from season stats. Compute expected FP per PA, multiply by projected PAs.

Key variables to use (pull from existing batter_df):
- HR rate per PA → project HRs (each = 10 pts)
- BB rate per PA → project walks (each = 3 pts)
- Single/2B/3B rates from AVG, ISO, BABIP estimates
- SB: use season SB/G rate if available, else 0
- RBI/Run: approximate from lineup position and team context (use 0.12 RBI/PA and 0.11 R/PA as league-average baselines; adjust for lineup spot)

### Pitcher Projection

```python
PITCHER_SCORING = {
    "win": 2.0,
    "quality_start": 3.0,
    "strikeout": 1.0,
    "inning_pitched": 1.0,
    "earned_run": -1.0,
}
```

Use existing pitcher Statcast data:
- Project IP: use season IP/GS average, cap at 6.0
- Project Ks: K/9 * IP / 9
- Project ER: ERA * IP / 9
- Win probability: 0.5 baseline (no team W/L data needed)
- Quality start probability: if projected IP >= 6.0 and ERA <= 3.50, prob = 0.45; if ERA <= 4.20, prob = 0.30; else 0.15

### Position Classification

```python
POSITIONS = {
    "IF": ["1B", "2B", "3B", "SS"],
    "OF": ["LF", "CF", "RF", "DH"],  # DH counts as OF per rules
    "P": ["SP", "P"],
}
```

Use position data from MLB Stats API lineup data (already in data_fetcher).

### Output per player

```python
{
    "name": str,
    "team": str,
    "opponent": str,
    "position": str,          # "P", "IF", "OF"
    "lineup_spot": int,       # 1-9 for hitters, 0 for pitchers
    "projected_fp": float,    # rounded to 1 decimal
    "fp_breakdown": dict,     # {"hr": X, "hits": X, "walks": X, ...}
    "tier": str,              # "ELITE" (>30 FP), "SOLID" (20-30), "VALUE" (12-20), "AVOID" (<12)
    "key_edge": str,          # one-line reason (e.g. "8.2 K/9 vs weak lineup")
}
```

### Draft Cheat Sheet Function

```python
def build_draft_cheat_sheet(players):
    """
    Args: list of projected players (output above)
    Returns: {
        "round_1": [...],   # top 4 overall by FP (any position)
        "round_2": [...],   # next 4
        "round_3": [...],   # next 4
        "pitchers": [...],  # top 3 pitchers sorted by FP
        "if_targets": [...],  # top IF sorted by FP
        "of_targets": [...],  # top OF sorted by FP
        "optimal_lineup": {   # best valid 6-man lineup
            "P": player,
            "IF1": player,
            "IF2": player,
            "OF1": player,
            "OF2": player,
            "FLEX": player,
            "total_fp": float,
            "teams": list,    # confirms 2-team rule met
        }
    }
    """
```

Optimal lineup = maximize total projected FP subject to:
- Exactly 1 P, 2 IF, 2 OF, 1 FLEX (IF or OF)
- Players from at least 2 different teams
- No duplicate players

Use a greedy approach: pick best P, best 2 IF, best 2 OF, best remaining IF/OF for FLEX. Verify 2-team rule; if violated (all same team), swap FLEX for best player from different team.

---

## Flask Route: `/api/underdog-draft`

Add to `app.py`:

```python
@app.route("/api/underdog-draft")
def api_underdog_draft():
    try:
        from market_underdog_draft import project_all_players, build_draft_cheat_sheet
        
        schedule = data_fetcher.get_today_schedule()
        lineups = data_fetcher.get_confirmed_lineups(schedule)
        batter_df = data_fetcher.get_batter_statcast()
        pitcher_df = data_fetcher.get_pitcher_statcast()
        
        players = project_all_players(lineups, batter_df, pitcher_df)
        cheat_sheet = build_draft_cheat_sheet(players)
        
        return jsonify({
            "status": "ok",
            "date": date.today().isoformat(),
            "player_count": len(players),
            "cheat_sheet": cheat_sheet,
            "all_players": sorted(players, key=lambda x: x["projected_fp"], reverse=True),
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "trace": traceback.format_exc()}), 500
```

---

## Frontend Tab

Add "Underdog Draft" tab to the existing tab navigation in the HTML template (check how existing tabs are structured in `app.py` or `templates/` — match the pattern exactly).

Tab content should show:

### Section 1: Optimal Lineup
Show the 6-player recommended lineup in a card:
```
🏆 OPTIMAL LINEUP — {total_fp} proj. pts
P:    {name} ({team}) — {fp} pts
IF:   {name} ({team}) — {fp} pts
IF:   {name} ({team}) — {fp} pts
OF:   {name} ({team}) — {fp} pts
OF:   {name} ({team}) — {fp} pts
FLEX: {name} ({team}) — {fp} pts
```

### Section 2: Draft Board by Round
Round 1-3 targets in a simple ranked list with position badge, team, projected FP, and tier.

### Section 3: By Position
Three columns: Pitchers | Infielders | Outfielders — top 5 each, sorted by projected FP.

### Section 4: Full Player Table
Sortable table: Name | Pos | Team | Opp | Lineup Spot | Proj FP | Tier | Key Edge

---

## Verification

1. `python3 main.py dry_run` still works cleanly
2. `curl http://localhost:5050/api/underdog-draft` returns JSON without error
3. Underdog Draft tab visible and loads in browser
4. Optimal lineup shows 6 players from at least 2 teams

## Commit

```bash
git add -A
git commit -m "feat: Underdog Draft tab — FP projections + draft cheat sheet (v9)"
git push
```

## Notes
- `python3` not `python`
- No venv — user-level packages
- Do not touch `.env`
- Check existing `app.py` structure before adding routes/tabs — match patterns exactly
- If templates are inline in app.py (not in a templates/ folder), keep them inline
