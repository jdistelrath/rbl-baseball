# Solomon Bug Fix Prompt — MLB Model v2 Patches

You are **Solomon**, the Builder. Fix specific bugs in an existing codebase. Do not rewrite anything that isn't broken. Surgical fixes only.

## Working Directory
`/Users/jimsmini/.openclaw/workspace/rbl/baseball/`

---

## Bug 1: F5 EV is 1278% (wildly wrong)

**Root cause:** `_extract_h2h_odds()` is falling back to full-game moneyline odds (e.g. -150, +130) but the EV calculator is receiving these as raw integers. The implied probability calculation for negative American odds is correct, but the model win probability (50% ± adjustments) is being compared against the wrong side's odds — creating a massive apparent edge.

**Fix in `market_f5.py`:**

1. In `score_f5()`, add a debug print before returning to verify `home_f5_odds`, `away_f5_odds`, `home_win_prob`, and the resulting EV values. This will confirm the exact source of the inflated number.

2. Add a sanity check: if `ev_per_dollar > 2.0` (200% EV), treat it as a data error and return None. No real betting edge is ever 1278%.

3. In `_extract_h2h_odds()`, verify that the price values coming back are American odds integers (e.g. -150, +130), not decimal odds (e.g. 1.67, 2.30). The Odds API returns American odds when `oddsFormat=american` is specified. Check `get_odds_for_markets()` in `data_fetcher.py` to confirm `oddsFormat=american` is in the request params. If it's returning decimal odds, the EV formula breaks completely.

4. In `ev_calculator.py`, add input validation to `american_to_implied_prob()`:
   - If odds is between -3 and +3, it's likely decimal odds — raise ValueError with a clear message
   - If odds is 0, return 0.5 (neutral)
   - Log a warning if odds seems unreasonable (e.g. > +5000 or < -5000)

---

## Bug 2: Totals line showing 16.0 runs (should be ~8-9)

**Root cause:** The totals market is pulling from the wrong odds field. The Odds API `totals` market returns `point` as the line value. The code looks correct but may be hitting a different bookmaker's alternate totals market (which has inflated lines like 16.0).

**Fix in `market_totals.py` → `_extract_totals_odds()`:**

1. Add bookmaker priority: prefer DraftKings (`draftkings`) or FanDuel (`fanduel`) over other bookmakers. Only fall back to others if neither is available. Add this before the inner loop:
```python
# Prioritize sharp books
PREFERRED_BOOKS = ["draftkings", "fanduel", "betmgm", "caesars"]
bookmakers = sorted(
    event.get("bookmakers", []),
    key=lambda b: (0 if b.get("key", "") in PREFERRED_BOOKS else 1)
)
```

2. Add a sanity check on the line: if `line > 12` or `line < 5`, skip this bookmaker's totals market — it's an alternate line, not the main game total.

3. After extracting the line, print it to console in dry_run mode so we can verify it's sane.

---

## Bug 3: Output still says "PrizePicks" — should say "Underdog"

**Fix in `messenger.py`:**
Replace all instances of "PrizePicks" with "Underdog Fantasy". 
Replace "PrizePicks 3-leg Power Play = 5x" with "Underdog 3-leg Pick'em = 3x payout".
Replace "PrizePicks Power Play" with "Underdog Pick'em".

Also update the multiplier table (Underdog differs slightly from PrizePicks):
- 2-leg = 3x
- 3-leg = 5x  
- 4-leg = 10x
- 5-leg = 20x
- 6-leg = 40x

(These are the same — good, no logic change needed, just text.)

---

## Bug 4: Bet sizing too aggressive ($25 on unvalidated model)

The model is suggesting $25 bets before any backtesting has run. Cap suggested_bet_size at $5 until `weights.json` exists (indicating backtest has been run).

**Fix in `ev_calculator.py` → `suggested_bet_size()`:**
```python
import os

def suggested_bet_size(ev_per_dollar, kelly, bankroll=100.0):
    weights_exist = os.path.exists(
        os.path.join(os.path.dirname(__file__), "weights.json")
    )
    max_bet = 50.0 if weights_exist else 5.0  # cap at $5 until backtest calibrates weights
    ...
```

---

## After All Fixes

Run `python main.py dry_run` and confirm:
1. F5 EV is a reasonable number (between -50% and +50% for most plays)
2. Totals line is between 5.0 and 12.0 runs
3. No "PrizePicks" anywhere in output
4. Suggested bets are $1-5 (no weights.json yet)
5. All other outputs still working normally

Then `git add` and commit.
