# Solomon Build Prompt v2 — MLB Model Expansion
# Expand existing HR prop model to include full EV analysis across 4 markets

You are **Solomon**, the Builder. You are extending an existing working codebase — do not rewrite what works. Add to it surgically. Every number traces to a source. No stubs, no placeholders.

## Context

A working MLB HR prop model already exists at:
`/Users/jimsmini/.openclaw/workspace/rbl/baseball/`

It has 8 modules (config.py, data_fetcher.py, feature_builder.py, scorer.py, stack_builder.py, backtest.py, messenger.py, main.py) and runs end-to-end. The Odds API key is already in `.env` as `THE_ODDS_API_KEY`.

## Your Mission

Extend the model to cover 4 markets and produce a **daily top 3 highest-EV plays** output across all markets, in addition to the existing HR outputs.

---

## The 4 Markets

### 1. HR Props (existing — keep working, do not break)
- Platform: PrizePicks Power Play
- Already built and tested

### 2. Team Totals (new)
- Over/under on total runs scored in a game
- Key variables: starting pitcher ERA/FIP/K%, bullpen ERA, team wRC+, park run factor, weather (wind + temp strongly affects run scoring), implied total from odds
- EV calculation: compare model's predicted run total to book's line

### 3. First 5 Innings (F5) Lines (new)
- Bet on which team wins the first 5 innings (isolates starting pitchers, removes bullpen)
- Key variables: starting pitcher quality (FIP, K/BB, recent form), opposing lineup wRC+ vs. handedness, park, weather
- EV calculation: compare model's win probability to implied probability from F5 moneyline odds

### 4. Strikeout Props (new)
- Over/under on strikeouts for a starting pitcher
- Key variables: pitcher K/9, K%, SwStr%, opposing team K rate, park (some parks favor strikeouts), temperature
- Only run this for starters with confirmed starts and K/9 > 7.5 — don't surface weak strikeout pitchers
- EV calculation: compare model's projected K total to book's line

---

## What to Build

### New file: `ev_calculator.py`

Single responsibility: given a model probability and a book's American odds line, calculate EV.

```python
def american_to_implied_prob(american_odds: int) -> float:
    """Convert American odds to implied probability."""
    if american_odds > 0:
        return 100 / (american_odds + 100)
    else:
        return abs(american_odds) / (abs(american_odds) + 100)

def calculate_ev(model_prob: float, american_odds: int, stake: float = 1.0) -> float:
    """
    Calculate expected value of a bet.
    EV = (model_prob * payout) - ((1 - model_prob) * stake)
    Returns EV per dollar staked.
    """

def kelly_fraction(model_prob: float, american_odds: int) -> float:
    """
    Kelly criterion fraction (full Kelly — cap at 0.25 for safety).
    Returns fraction of bankroll to bet.
    """

def suggested_bet_size(ev_per_dollar: float, kelly: float, bankroll: float = 100.0) -> float:
    """
    Given EV and Kelly fraction, return suggested bet size in dollars.
    Minimum $1, maximum $50.
    Scale: kelly * bankroll, floored/capped.
    """
```

### New file: `market_totals.py`

Score every game for team total over/under EV.

```python
def score_game_total(game: dict, batter_stats: df, pitcher_stats: df, weather: dict, park_factors: df, odds: dict) -> dict:
    """
    Returns {
        game_id, home_team, away_team,
        model_total,        # predicted total runs
        book_line,          # book's over/under line
        over_odds,          # American odds for over
        under_odds,         # American odds for under
        best_side,          # 'over' or 'under'
        ev_per_dollar,
        kelly,
        suggested_bet,
        key_factors         # list of 2-3 strings explaining the model's call
    }
    """
```

Key inputs for model_total:
- Home + away starter FIP (lower FIP = fewer runs)
- Home + away team wRC+ (offensive strength)
- Park run factor (from FanGraphs — add run_factor column to park_factors dict in config.py)
- Wind bonus: wind blowing out >10mph adds ~0.5 runs; blowing in subtracts ~0.3
- Temp: <45°F subtracts ~0.4 runs; >85°F adds ~0.2

Simple linear model: `predicted_total = base_runs + pitcher_adjustment + park_adjustment + weather_adjustment`
Use league average ~8.8 runs/game as base. Adjust up/down from there.

### New file: `market_f5.py`

Score every game's First 5 innings moneyline for EV.

```python
def score_f5(game: dict, pitcher_stats: df, batter_stats: df, weather: dict, odds: dict) -> dict:
    """
    Returns {
        game_id, home_team, away_team,
        home_starter, away_starter,
        model_home_win_prob,   # probability home team wins F5
        home_f5_odds,          # American odds
        away_f5_odds,
        best_side,             # 'home' or 'away'
        ev_per_dollar,
        kelly,
        suggested_bet,
        key_factors
    }
    """
```

Win probability model: start at 50/50, adjust for:
- Starter FIP differential (each 0.5 FIP = ~3% win prob shift)
- Opposing lineup wRC+ vs. pitcher handedness
- Home field advantage (~3%)
- Weather (extreme conditions slightly favor better pitcher)

Only surface F5 bets where |ev_per_dollar| > 0.03 (3% edge minimum).

### New file: `market_strikeouts.py`

Score strikeout props for confirmed starters with K/9 > 7.5.

```python
def score_strikeout_prop(game: dict, pitcher_stats: df, batter_stats: df, weather: dict, odds: dict) -> dict:
    """
    Returns {
        game_id, pitcher_name, team,
        model_k_projection,    # projected strikeouts
        book_line,             # over/under line (e.g. 6.5)
        over_odds,
        under_odds,
        best_side,
        ev_per_dollar,
        kelly,
        suggested_bet,
        key_factors
    }
    """
```

K projection: `projected_Ks = (pitcher_K9 / 9) * expected_innings * opp_k_rate_adjustment * temp_adjustment`
- Expected innings: use pitcher's season average IP/start, cap at 6
- Opp K rate adjustment: if opposing team K% > 24%, add 5%; if < 18%, subtract 5%
- Temp <45°F: subtract 0.3 Ks (cold = less swing-and-miss)
- Only surface if model projects >0.5K difference from the book line

### Extend `data_fetcher.py`

Add one new function:

```python
def get_odds_for_markets(markets="h2h,totals,spreads") -> dict:
    """
    Pulls odds for MLB games from The Odds API.
    Returns dict keyed by away_team+home_team for easy lookup.
    Gracefully returns {} if API key not set or request fails.
    Markets: h2h (moneyline), totals (over/under)
    Also pulls alternate markets for F5 if available.
    """
```

### Extend `messenger.py`

Add one new function:

```python
def send_top_ev_plays(plays: list):
    """
    Sends the top 3 EV plays of the day, ranked by ev_per_dollar.
    plays is a list of dicts with keys: market, description, ev_per_dollar, suggested_bet, key_factors

    Format:
    ⚾ TOP EV PLAYS — {date}

    #1 {market}: {description}
    Edge: {ev_per_dollar:.1%} EV | Bet: ${suggested_bet}
    Why: {key_factors}

    #2 ...
    #3 ...
    """
```

### Extend `main.py`

In `mode_morning_brief()`, after existing HR pipeline:
1. Call `get_odds_for_markets()` 
2. Run `market_totals.score_game_total()` for each confirmed game
3. Run `market_f5.score_f5()` for each confirmed game
4. Run `market_strikeouts.score_strikeout_prop()` for confirmed starters with K/9 > 7.5
5. Combine all results, sort by `ev_per_dollar` descending
6. Take top 3, call `messenger.send_top_ev_plays(top3)`

In `mode_dry_run()`: print top EV plays to console instead of sending.

---

## Park Factors Extension

In `config.py`, extend the stadium dict to include run factors (for totals model):

Add `run_factor` alongside `hr_factor` for each stadium. Approximate values:
- Coors Field: 1.30
- Great American Ball Park: 1.12
- Yankee Stadium: 1.08
- Chase Field: 1.07 (roof open)
- Citizens Bank Park: 1.06
- Wrigley Field: 1.05 (wind-dependent)
- Globe Life Field: 1.04
- Fenway Park: 1.03
- All others: scale proportionally from hr_factor

---

## Build Rules

1. **Do not touch existing modules unless extending them.** HR pipeline must continue to work exactly as before.
2. Build in this order: ev_calculator.py → market_totals.py → market_f5.py → market_strikeouts.py → extend data_fetcher.py → extend messenger.py → extend main.py
3. After each new file: import it in a test, verify no errors.
4. After all files done: run `python main.py dry_run` — confirm both HR output AND top EV plays output appear.
5. Missing odds data never crashes the pipeline — degrade gracefully, skip that market.
6. Final step: git add + commit everything.
