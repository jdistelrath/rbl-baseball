# Solomon Prompt v10 — Draft Projection Logging + Actuals Comparison

You are **Solomon**, the Builder. Read existing files before modifying. Do not break anything working.

## Working Directory
`/Users/jimsmini/Projects/baseball/`

## What to Build
1. Save today's draft projections to disk at brief time
2. Pull yesterday's actual stats from MLB Stats API and compute actual FP
3. Show projection vs. actual accuracy on the Draft tab

---

## Part 1: Save Projections — `market_underdog_draft.py`

Add a function `save_projections(players, date_str=None)`:

```python
def save_projections(players, date_str=None):
    """
    Save today's player projections to outputs/draft_projections_YYYY-MM-DD.json
    Call this once per day at brief time (10:50am).
    Skips if file already exists (idempotent).
    """
    from pathlib import Path
    import json
    from datetime import date
    
    date_str = date_str or date.today().isoformat()
    out_dir = Path(__file__).parent / "outputs" / "draft_projections"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date_str}.json"
    
    if out_path.exists():
        return  # already saved today
    
    with open(out_path, "w") as f:
        json.dump({
            "date": date_str,
            "players": players,
        }, f, indent=2)
    print(f"[draft] Projections saved: {out_path}")
```

---

## Part 2: Pull Actuals — new function `get_draft_actuals(date_str)` in `data_fetcher.py`

```python
def get_draft_actuals(date_str):
    """
    Pull actual game stats for a given date from MLB Stats API.
    Returns list of dicts:
    {
        name: str,
        team: str,
        position: str,   # "P", "IF", "OF"
        actual_fp: float,
        fp_breakdown: dict,  # {"hr": X, "hits": X, ...}
        game_id: int,
    }
    
    Endpoint: https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=boxscore
    
    For each completed game, pull:
    - Batters: H (singles=H-2B-3B-HR, 2B, 3B, HR), BB, HBP, RBI, R, SB
    - Pitchers: W (win), QS (IP>=6 and ER<=3), SO, IP, ER
    
    Apply Underdog scoring:
    HITTER: single*3 + double*6 + triple*8 + hr*10 + bb*3 + hbp*3 + rbi*2 + r*2 + sb*4
    PITCHER: win*2 + qs*3 + so*1 + ip*1 + er*(-1)
    
    Return [] gracefully on failure.
    Cache result with _save_cache / _load_cache (cache_key = f"actuals_{date_str}").
    """
```

Position classification: use existing `market_underdog_draft.classify_position()` on the player's position from the boxscore. Import it at top of the function.

---

## Part 3: Accuracy Engine — new function `compare_projections_to_actuals(date_str)` in `market_underdog_draft.py`

```python
def compare_projections_to_actuals(date_str):
    """
    Load saved projections for date_str, fetch actuals, compare.
    Returns dict:
    {
        "date": date_str,
        "status": "ok" | "no_projections" | "no_actuals",
        "players": [
            {
                name, team, position,
                projected_fp, actual_fp,
                error,        # actual - projected
                pct_error,    # error / projected * 100
                rank_proj,    # rank by projected FP (1 = highest)
                rank_actual,  # rank by actual FP (1 = highest)
                rank_delta,   # rank_actual - rank_proj (negative = outperformed projection)
            }
        ],
        "summary": {
            "mae",           # mean absolute error
            "rmse",          # root mean squared error
            "rank_corr",     # Spearman rank correlation (projected rank vs actual rank)
            "top5_hit_rate", # % of projected top-5 that appeared in actual top-5
            "player_count",  # number of matched players
        }
    }
    
    Player matching: exact name first, then last-name substring fallback.
    Only include players who appear in both projections and actuals.
    Require at least 5 matched players to compute summary stats.
    """
```

Use `scipy.stats.spearmanr` for rank correlation.

---

## Part 4: Flask Route `/api/draft-accuracy`

Add to `app.py`:

```python
@app.route("/api/draft-accuracy")
def api_draft_accuracy():
    try:
        from market_underdog_draft import compare_projections_to_actuals
        from datetime import date, timedelta
        
        # Default: yesterday. Allow ?date=YYYY-MM-DD override.
        date_str = request.args.get("date", (date.today() - timedelta(days=1)).isoformat())
        result = compare_projections_to_actuals(date_str)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "error": str(e), "trace": traceback.format_exc()}), 500
```

---

## Part 5: Wire projection saving into morning brief

In `main.py`, find the morning_brief mode. After the underdog draft projections are computed (look for where `project_all_players` is called or where `/api/underdog-draft` data originates), add:

```python
# Save projections for actuals comparison tomorrow
try:
    from market_underdog_draft import project_all_players, save_projections
    # Re-use already-fetched data if available, otherwise fetch fresh
    draft_players = project_all_players(lineups, batter_df, pitcher_df)
    save_projections(draft_players)
except Exception as e:
    print(f"[main] Draft projection save failed: {e}")
```

If `project_all_players` is already called in morning_brief, don't call it twice — just pass the result to `save_projections`.

Also add projection saving to `mode_dry_run()` for testing, but use `date_str="TEST"` so it doesn't overwrite real projections:
```python
save_projections(draft_players, date_str="TEST")
```

---

## Part 6: Draft Tab — Add Accuracy Section

In the Underdog Draft tab HTML, add a new section **"Yesterday's Accuracy"** that:
- Fetches `/api/draft-accuracy` on page load
- If `status == "no_projections"`: shows "No projections saved yet — accuracy tracking starts today"
- If `status == "ok"`: shows a summary card + sortable table

Summary card:
```
📊 ACCURACY — {date}
MAE: {mae} FP  |  RMSE: {rmse} FP  |  Rank Corr: {rank_corr}
Top-5 Hit Rate: {top5_hit_rate}%  |  {player_count} players matched
```

Table columns: Player | Pos | Proj FP | Actual FP | Error | Rank Proj | Rank Actual | Δ Rank
- Color code Error: green if error < 0 (actual > projected = underrated), red if error > 0 (actual < projected = overrated)
- Sort default: by |error| descending (biggest misses first)

---

## Verification

1. `python3 main.py dry_run` — confirm projections saved to `outputs/draft_projections/TEST.json`
2. `curl http://localhost:5050/api/draft-accuracy` — should return `{"status": "no_projections"}` (yesterday has none yet)
3. `curl http://localhost:5050/api/draft-accuracy?date=TEST` — should return comparison data
4. Draft tab loads with "No projections saved yet" message in accuracy section
5. App restarts cleanly

## Commit

```bash
git add -A
git commit -m "feat: draft projection logging + actuals comparison (v10)"
git push
```

## Notes
- `python3` not `python`
- No venv — user-level packages
- Do not touch `.env`
- `dry_run` only, never `morning_brief`
- Check existing morning_brief code carefully before adding — don't duplicate data fetches
