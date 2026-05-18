# Solomon Prompt v6 — Totals Outcome Backtest + Calibration (2021-2024)

You are **Solomon**, the Builder. Add a totals outcome backtest to the existing codebase. Surgical addition only — do not break anything existing.

## Working Directory
`/Users/jimsmini/.openclaw/workspace/rbl/baseball/`

## What to Build

Add `run_totals_backtest(years=[2021,2022,2023,2024,2025])` to `backtest.py` and wire it into `main.py` as `python main.py backtest_totals`.

### Data Source

Use the MLB Stats API to pull historical game scores — no pybaseball needed, no FanGraphs.

```
https://statsapi.mlb.com/api/v1/schedule?sportId=1&season={year}&gameType=R&hydrate=linescore
```

This returns every regular season game with final scores. Parse home_score + away_score = actual total runs.

For pitcher data: use `data_fetcher.get_pitcher_statcast(year-1)` (prior season stats, no lookahead bias).
For batter data: use `data_fetcher.get_batter_statcast(year-1)`.
For weather: skip (not available historically) — use neutral defaults (temp=70, wind=0).
For park factors: use existing `config.py` PARK_RUN_FACTORS dict.

### Logic

```python
def run_totals_backtest(years=[2021, 2022, 2023, 2024]):
    """
    For each completed game in the given years:
    1. Pull final score (actual total runs)
    2. Run market_totals.score_game_total() with prior-year stats
    3. Record: model_total, actual_total, model_side (over/under), correct (bool)
    4. Track accuracy by year and overall
    5. Run scipy optimizer to find weights that maximize accuracy
    6. Save calibrated totals weights to weights_totals.json
    """
```

Pull games month by month per year to avoid timeout. Cache each month's game results to `cache/scores_{year}_{month}.pkl`.

For each game:
- Build a minimal game dict: game_id, home_team, away_team, stadium, home_pitcher_name, away_pitcher_name
- Get pitcher names from the schedule hydrate (probables)
- Run `score_game_total()` → get model_total
- Compare model_total to actual_total
- If model_total > actual (book line proxy = actual total): model said OVER
- Record correct if actual > model_total when model said OVER, or actual < model_total when model said UNDER

**Important:** We don't have historical book lines, so use the actual game total as a proxy for the line. This measures directional accuracy only — whether the model's predicted direction was correct — not true EV against a book line.

### Calibration

After collecting all game-level results, run scipy.optimize.minimize to find totals model weights that maximize directional accuracy:

The totals model has these internal parameters (currently hardcoded in `market_totals.py`):
- `pitcher_fip_weight`: how much each FIP point above/below 4.20 shifts the total (currently 0.5)
- `wrc_weight`: how much each wRC+ point above/below 100 shifts total (currently 0.015)
- `wind_out_bonus`: runs added for wind blowing out >10mph (currently 0.5)
- `wind_in_penalty`: runs subtracted for wind blowing in (currently 0.3)
- `temp_hot_bonus`: runs added for temp >85F (currently 0.2)
- `temp_cold_penalty`: runs subtracted for temp <45F (currently 0.4)

Parameterize these in `market_totals.py` so they can be overridden via a `totals_weights.json` file (similar to how `weights.json` works for HR).

Optimize to maximize correct predictions on 2021-2023 train set. Validate on 2024. Report accuracy on 2025 holdout (games played so far this season — use 2024 stats as prior year).

Save results to:
- `outputs/backtest/totals_backtest_{date}.json` — full results by year
- `weights_totals.json` — calibrated parameters

### Output Format

```
=== TOTALS BACKTEST RESULTS ===
Year  Games  Correct  Accuracy
2021   2430     1287    53.0%  ← train
2022   2430     1312    54.0%  ← train
2023   2430     1298    53.4%  ← train
2024   2430     1318    54.2%  ← validation
2025    ~500     ~270    54.0%  ← holdout (live season so far)
Overall: 53.7% directional accuracy

Calibrated weights saved to weights_totals.json
Key finding: model is / is not above 54% threshold for DK betting
```

### Wire into main.py

```python
"backtest_totals": mode_backtest_totals,

def mode_backtest_totals():
    print("[main] === Totals Outcome Backtest ===")
    from backtest import run_totals_backtest
    run_totals_backtest(years=[2021, 2022, 2023, 2024, 2025])
```

### Parameterize market_totals.py

At the top of `market_totals.py`, add:

```python
import json, os

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
        with open(path) as f:
            loaded = json.load(f)
            defaults.update(loaded)
    return defaults

TOTALS_WEIGHTS = _load_totals_weights()
```

Use `TOTALS_WEIGHTS["pitcher_fip_weight"]` etc. throughout `score_game_total()` instead of hardcoded values.

## After Build

1. Run `python main.py backtest_totals` — let it run (will take 20-40 min, pulling 4 years of scores)
2. Report the accuracy table when done
3. Report whether overall accuracy is above or below 54%
4. Confirm `weights_totals.json` was written
5. `git add` and commit
