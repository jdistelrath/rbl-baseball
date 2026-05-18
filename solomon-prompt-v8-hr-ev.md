# Solomon Prompt v8 — HR Prop EV Engine + Odds API Integration

You are **Solomon**, the Builder. You receive specs and execute them literally. Every number traces to a source. Do not invent features not in the spec. Do not break existing functionality.

## Working Directory
`/Users/jimsmini/.openclaw/workspace/rbl/baseball/`

## Current State (as of Apr 28 2026)
The following modules are **live and working**:
- `data_fetcher.py` — MLB Stats API for schedules, lineups, batter/pitcher stats + OWM weather
- `feature_builder.py` — rolling metrics, platoon splits, park factors
- `scorer.py` — composite HR probability, z-score normalization, tiers
- `stack_builder.py` — target pitcher identification + correlated stacks
- `market_totals.py` — game totals EV (F5 + full game)
- `market_f5.py` — F5 moneyline EV
- `market_strikeouts.py` — K prop model (projects Ks from pitcher stats, no Odds API yet)
- `messenger.py` — Telegram output
- `main.py` — orchestrator, dry_run works
- `ev_calculator.py` — EV, Kelly, bet sizing helpers
- `config.py` + `config.yaml` + `.env`

**`python3 main.py dry_run` runs cleanly** — do not break this.

The following are **NOT yet built**:
- `market_hr_ev.py` — HR prop EV vs book lines
- Odds API integration in `data_fetcher.py` (`get_player_props()`)
- Bullpen ERA in `market_totals.py`

## What to Build (in order)

---

### Step 1: Add `get_player_props(market)` to `data_fetcher.py`

Add this function. Do not change any existing functions.

```python
def get_player_props(market="batter_home_runs"):
    """
    Pull player prop lines from The Odds API.
    market: "batter_home_runs" or "pitcher_strikeouts"
    
    Returns list of dicts:
    {
        player_name: str,
        home_team: str,
        away_team: str,
        market: str,
        over_line: float,
        over_odds: int,
        under_odds: int or None,
        implied_over_prob: float,
        bookmaker: str
    }
    
    Returns [] gracefully if THE_ODDS_API_KEY not set or request fails.
    Prefer DraftKings, fall back to FanDuel, then others.
    One entry per player (best available book).
    """
```

Use THE_ODDS_API_KEY from os.getenv. Endpoint:
`https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/?apiKey={key}&regions=us&markets={market}&oddsFormat=american`

Use `ev_calculator.american_to_implied_prob()` for implied prob conversion.

---

### Step 2: Build `market_hr_ev.py`

New file. HR prop EV engine comparing model probability to DK book lines.

```python
"""
Market: HR Props — straight bet EV vs DraftKings lines.
MIN_EDGE = 0.03 (3%)
Only surfaces bets where model_prob > implied_prob + MIN_EDGE.
"""
MIN_EDGE = 0.03

def score_hr_props(scored_batters, hr_prop_lines):
    """
    Args:
        scored_batters: list of dicts from scorer.py (fields: name, prob, tier, team, key_edge)
        hr_prop_lines: list from data_fetcher.get_player_props("batter_home_runs")
    
    Returns list sorted by edge desc:
    {
        player_name, team,
        model_prob, implied_prob, edge,
        over_odds, ev_per_dollar, kelly, suggested_bet,
        tier, key_edge
    }
    """
```

Player name matching: exact lowercase first, then last-name substring fallback.

Use `ev_calculator.calculate_ev`, `kelly_fraction`, `suggested_bet_size`.

---

### Step 3: Update `market_strikeouts.py` to use Odds API lines

Currently `market_strikeouts.py` scores pitchers but has no book line comparison (EV shows as 0%).

Add an optional `k_prop_lines` parameter to the main scoring function. When lines are available:
- Compare model K projection to book line
- Calculate edge = (model_prob_of_outcome) - (implied_prob_from_book_odds)
- Only surface bets with edge >= 0.03
- If no lines available (empty list), keep existing behavior (show top projections, 0% EV)

Check the existing function signatures in `market_strikeouts.py` first and extend them — do not rewrite from scratch.

---

### Step 4: Add bullpen ERA to `market_totals.py`

Add `get_bullpen_stats(season=None)` to `data_fetcher.py`:
- Pulls relief ERA by team from MLB Stats API (relievers = pitchers with GS < 5)
- Returns `{team_name: weighted_avg_era}` — IP-weighted
- Cache with existing `_load_cache`/`_save_cache` pattern
- Return `{}` gracefully on failure

In `market_totals.py`, after existing pitcher adjustments:
```python
# Bullpen ERA adj: each 1.0 ERA above/below 4.20 league avg = 0.3 run shift
bullpen_stats = data_fetcher.get_bullpen_stats()
home_bullpen = bullpen_stats.get(home_team, 4.20)
away_bullpen = bullpen_stats.get(away_team, 4.20)
bullpen_adj = ((home_bullpen - 4.20) + (away_bullpen - 4.20)) * 0.3
```

Integrate `bullpen_adj` into the model total. Check `market_totals.py` to find the exact variable names for home_team, away_team, and model_total before writing.

---

### Step 5: Add `send_ev_props(hr_props, k_props)` to `messenger.py`

```
⚾ DK +EV PROPS — {date}

💰 HR PROPS
• {Player} ({Team}): model {X}% vs book {Y}% → {odds} | ${bet}
  {key_edge}
[or "No +EV HR props today" if empty]

⚡ K PROPS  
• {Pitcher} {OVER/UNDER} {line} Ks → {odds} | ${bet}
  {key_factors joined by ", "}
[or "No +EV K props today" if empty]

Min edge: 3% | Straight bets only
```

Only sends if `hr_props` or `k_props` is non-empty. Prints to console in dry_run mode (check how existing messenger functions handle this pattern).

---

### Step 6: Wire into `main.py`

In the morning brief and dry_run modes, after the existing EV pipeline:

```python
# HR + K prop EV (requires Odds API key)
hr_prop_lines = data_fetcher.get_player_props("batter_home_runs")
k_prop_lines = data_fetcher.get_player_props("pitcher_strikeouts")

from market_hr_ev import score_hr_props
hr_ev_plays = score_hr_props(all_scored_batters, hr_prop_lines) if all_scored_batters else []

# Pass k_prop_lines to existing strikeouts scorer (extend its call signature)
# Check current call in main.py and update accordingly

k_ev_plays = [p for p in k_scored if p.get("edge", 0) >= 0.03]

if hr_ev_plays or k_ev_plays:
    messenger.send_ev_props(hr_ev_plays[:5], k_ev_plays[:3])
```

Find the existing variable names for `all_scored_batters` and `k_scored` in `main.py` before writing — match exact names.

---

## Verification Steps (run after build)

1. `python3 main.py dry_run` — must complete cleanly with no import errors
2. Confirm EV pipeline section still outputs top 3 plays
3. If THE_ODDS_API_KEY is not set in .env, confirm no crash (graceful [] return)
4. Print a count of how many HR prop lines and K prop lines were fetched (even if 0)

## Commit

After successful dry_run:
```bash
git add -A
git commit -m "feat: HR prop EV engine, Odds API integration, bullpen ERA in totals (v8)"
```

## Notes
- `python3` is the binary (not `python`)
- No venv exists — packages installed at user level
- Do not create or modify `.env` — API keys are Jim's to add
- Do not run `morning_brief` mode (sends to Telegram) — dry_run only
