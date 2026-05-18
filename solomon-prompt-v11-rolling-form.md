# Solomon Prompt v11 — Rolling Form + Pitcher Recent Starts + H2H + Pitch Type

You are **Solomon**, the Builder. Execute this spec literally. Read every file you touch before modifying it. Do not break anything working.

## Working Directory
`/Users/jimsmini/Projects/baseball/`

## Verify First
Run `python3 main.py dry_run` before starting. Confirm it exits cleanly. If it fails, stop and report the error.

## What to Build

Add four new signal layers to the HR prop model:

1. **Batter rolling form** — last 7/14/30 day rolling windows from Statcast
2. **Head-to-head matchup history** — batter vs. specific pitcher career stats
3. **Pitcher recent starts** — last 3 starts: HR/9, hard hit rate, pitch count context
4. **Batter vs. pitch type** — batter performance vs. pitcher's primary pitch

---

## Module Changes

### `data_fetcher.py` — Add 4 new fetch functions

#### 1. `get_batter_rolling_stats(days=30)`

```python
def get_batter_rolling_stats(days=30):
    """
    Pull batter Statcast data for the last N days using pybaseball.
    Returns a DataFrame with barrel_rate, hard_hit_pct, hr_fb_ratio, iso per batter.
    Cache key: f"batter_rolling_{days}d"
    """
```

Use `pybaseball.statcast_batter_exitvelo_barrels(year, minPA=10)` or
`pybaseball.batting_stats_range(start_dt, end_dt)` for the rolling window.
- Pull for 7, 14, and 30 days (three separate cached calls)
- Return a dict: `{7: df_7, 14: df_14, 30: df_30}`
- Cache each window separately: `batter_rolling_7d`, `batter_rolling_14d`, `batter_rolling_30d`
- If pybaseball call fails (network, no data), return empty DataFrame — do not crash

#### 2. `get_batter_pitcher_matchup(batter_name, pitcher_name)`

```python
def get_batter_pitcher_matchup(batter_name, pitcher_name):
    """
    Pull career H2H stats between batter and pitcher using MLB Stats API.
    Endpoint: https://statsapi.mlb.com/api/v1/stats?stats=vsPlayer&group=hitting
              &playerId={batter_id}&opposingPlayerId={pitcher_id}&season={year}
    Returns dict: {pa, hr, hard_contact_rate, avg} or None if <5 PA or not found.
    Minimum 5 PA to be usable — return None below threshold.
    Cache key: f"h2h_{batter_name}_{pitcher_name}" (sanitize to alphanum+underscore)
    """
```

Resolve player IDs from the MLB Stats API people search endpoint:
`https://statsapi.mlb.com/api/v1/people?sportId=1&search={name}`

#### 3. `get_pitcher_recent_starts(pitcher_name, n=3)`

```python
def get_pitcher_recent_starts(pitcher_name, n=3):
    """
    Pull game log for pitcher — last N starts.
    MLB Stats API: /api/v1/people/{pitcher_id}/stats?stats=gameLog&group=pitching&season={year}
    Returns list of dicts (most recent first):
      [{date, ip, hr_allowed, hard_hit_rate, pitches, era_in_start}, ...]
    Returns [] if not found.
    Cache key: f"pitcher_starts_{pitcher_name}"
    """
```

#### 4. `get_pitcher_pitch_mix(pitcher_name)`

```python
def get_pitcher_pitch_mix(pitcher_name):
    """
    Pull pitcher's primary pitch type and usage % from Statcast.
    Use pybaseball.statcast_pitcher_pitch_arsenal(year, minP=100).
    Returns dict: {primary_pitch: str, primary_pct: float, pitch_mix: {pitch_type: pct}}
    Returns None if not found.
    Cache key: f"pitch_mix_{pitcher_name}"
    """
```

---

### `feature_builder.py` — Add new features

At the top of `build_features_for_game()`, after existing pitcher/batter fetches, add:

```python
# New data pulls (fail gracefully — never crash build_features_for_game)
rolling_stats = df.get_batter_rolling_stats()   # {7: df, 14: df, 30: df}
pitcher_recent = df.get_pitcher_recent_starts(opp_pitcher_name)
pitch_mix = df.get_pitcher_pitch_mix(opp_pitcher_name)
```

For each batter, add these features to the features dict:

#### Rolling Form Features

```python
# 14-day rolling (primary signal)
"barrel_rate_14d": float or None -> fall back to season barrel_rate if None
"hard_hit_pct_14d": float or None -> fall back to season hard_hit_pct if None
"hr_fb_14d": float or None -> fall back to season hr_fb_ratio if None

# 7-day trend vs 30-day baseline (momentum signal)
# positive = trending up, negative = trending down
"form_trend": (barrel_rate_7d - barrel_rate_30d) / barrel_rate_30d if both available else 0.0
```

Fallback rules:
- If rolling data unavailable, use season value (no penalty, no bonus)
- `form_trend` = 0.0 if insufficient data

#### H2H Feature

```python
h2h = df.get_batter_pitcher_matchup(batter["name"], opp_pitcher_name)
"h2h_hr_rate": h2h["hr"] / h2h["pa"] if h2h and h2h["pa"] >= 5 else None
# If None, exclude from scoring (not 0 — just absent)
```

#### Pitcher Recent Form Features

```python
# From pitcher_recent (last 3 starts)
if pitcher_recent:
    recent_hr_per_ip = sum(s["hr_allowed"] for s in pitcher_recent) / max(sum(s["ip"] for s in pitcher_recent), 1)
    recent_hard_hit = mean([s["hard_hit_rate"] for s in pitcher_recent if s.get("hard_hit_rate")])
    high_workload = pitcher_recent[0]["pitches"] > 95  # short rest risk flag
else:
    recent_hr_per_ip = None
    recent_hard_hit = None
    high_workload = False

"pitcher_recent_hr_per_ip": recent_hr_per_ip   # None = fall back to season pitcher_hr_fb
"pitcher_recent_hard_hit": recent_hard_hit      # None = fall back to season pitcher_hard_hit_allowed
"pitcher_high_workload": -0.05 if high_workload else 0.0  # small penalty if tired
```

#### Pitch Type vs Batter Feature

```python
# Simple version: does batter have a platoon-style advantage vs primary pitch?
# Pull batter's wOBA vs pitch type from Statcast if available
# Fallback: use platoon_factor as proxy if pitch type data unavailable
"pitch_type_edge": float  # range -0.10 to +0.15, 0.0 if unknown
```

For pitch type edge:
- If pitcher is >60% fastball and batter has above-median hard_hit_pct → +0.08
- If pitcher is >50% breaking ball and batter barrel_rate is above median → +0.05
- If pitcher is >50% offspeed (changeup/splitter) and batter pull% is high → +0.05
- Otherwise → 0.0
- Use median values from the current day's batter pool

---

### `scorer.py` — Add new features to FEATURE_COLS and weights

Append to `FEATURE_COLS`:
```python
"barrel_rate_14d",       # replaces/supplements season barrel_rate
"hard_hit_pct_14d",      # replaces/supplements season hard_hit_pct
"form_trend",            # momentum signal
"pitcher_recent_hr_per_ip",  # pitcher recent form
"pitcher_recent_hard_hit",   # pitcher recent form
"pitcher_high_workload", # direct add (already a scalar bonus/penalty)
"pitch_type_edge",       # direct add (already a scalar bonus/penalty)
```

For H2H: handle separately — add as a direct scalar bonus to composite score after z-score weighting:
```python
# After computing composite[i]:
h2h_rate = fd.get("h2h_hr_rate")
if h2h_rate is not None:
    # Normalize: league avg HR rate per PA ~0.035; above that = bonus
    h2h_bonus = (h2h_rate - 0.035) * 5.0  # scale: 0.10 HR/PA = +0.325 bonus
    h2h_bonus = max(-0.5, min(0.5, h2h_bonus))  # cap
    composite[i] += h2h_bonus
```

Missing value handling for z-score normalization:
- Replace `None` with the column mean before z-scoring (not 0 — mean keeps them neutral)
- Track which batters had missing data; include `"data_gaps": [list of missing fields]` in output

Add to `weights.json` defaults (equal weight to start, backtest will calibrate):
```json
"barrel_rate_14d": 1.2,
"hard_hit_pct_14d": 1.0,
"form_trend": 0.8,
"pitcher_recent_hr_per_ip": 1.2,
"pitcher_recent_hard_hit": 0.8,
"pitcher_high_workload": 1.0,
"pitch_type_edge": 1.0
```

---

### `feature_builder.py` — Update `_build_key_edge()`

Add to key edge generation:
- If `form_trend > 0.15`: add "heating up (14d barrel rate +X%)"
- If `form_trend < -0.15`: add "cooling off (14d barrel rate -X%)"
- If h2h_hr_rate >= 0.08 and PA >= 5: add "X HR in Y PA vs this pitcher"
- If recent_hr_per_ip > 0.15: add "pitcher allowed X HR last 3 starts"

---

## Backtest

After building, run:
```bash
python3 main.py backtest
```

If the backtest completes without error, report the top-5 feature weights by calibrated value and the overall accuracy vs. prior run (backtest_results_2026-04-25.json for comparison).

---

## Verification

1. `python3 main.py dry_run` exits cleanly with no errors
2. New features appear in dry_run output for at least 3 batters
3. `form_trend`, `pitcher_recent_hr_per_ip`, `pitch_type_edge` all show non-zero values for at least some batters
4. H2H data loads (or gracefully returns None) — no crashes
5. `python3 main.py backtest` completes
6. All existing tabs in Flask app still work: `curl http://localhost:5050/api/hr-brief` returns 200

---

## Commit

```bash
git add -A
git commit -m "feat: v11 — rolling form, H2H, pitcher recent starts, pitch type edge"
git push
```

---

## Notes
- `python3` not `python`
- No venv — user-level packages
- Do not touch `.env`
- All new data fetches must fail gracefully (try/except, return None or empty) — a failed API call should never crash the morning brief
- Prefer cached results over re-fetching within the same day
- The morning brief must still fire even if every new feature returns None
