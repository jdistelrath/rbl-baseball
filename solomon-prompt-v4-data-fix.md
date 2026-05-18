# Solomon Bug Fix Prompt v4 — Data Fetcher: FanGraphs → Baseball Reference

You are **Solomon**, the Builder. Fix a specific data source problem. Surgical changes only — do not touch anything that isn't broken.

## Working Directory
`/Users/jimsmini/.openclaw/workspace/rbl/baseball/`

## The Problem

`data_fetcher.py` uses `pybaseball.batting_stats()` and `pybaseball.pitching_stats()` which hit FanGraphs' legacy scraper endpoint. FanGraphs is returning 403 on all calls. The backtest cannot run and live stat features degrade to position averages.

**Confirmed working alternative:** `pybaseball.batting_stats_bref()` and `pybaseball.pitching_stats_bref()` pull from Baseball Reference and return successfully.

## Available Columns (confirmed live)

**batting_stats_bref(year) columns:**
Name, Age, Tm, G, PA, AB, R, H, 2B, 3B, HR, RBI, BB, SO, BA, OBP, SLG, OPS, SB, CS, mlbID

**pitching_stats_bref(year) columns:**
Name, Age, Tm, G, GS, W, L, IP, H, R, ER, BB, SO, HR, ERA, GB/FB, WHIP, BAbip, SO9, SO/W, mlbID

## What to Change in `data_fetcher.py`

### Fix `get_batter_statcast()`

Replace the pybaseball call with `batting_stats_bref`. Derive features from available columns:

```python
from pybaseball import batting_stats_bref

def get_batter_statcast(season=None):
    season = season or date.today().year
    cache_key = f"batter_stats_bref_{season}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached
    try:
        df = batting_stats_bref(season)
        # Derive ISO = SLG - BA
        df["ISO"] = df["SLG"] - df["BA"]
        # Derive HR/FB proxy: HR / (AB * 0.35) — approximate fly ball count
        df["hr_fb_ratio"] = df["HR"] / (df["AB"] * 0.35).replace(0, 1)
        # K rate as proxy for hard contact (inverse — low K = more contact)
        df["k_rate"] = df["SO"] / df["PA"].replace(0, 1)
        # barrel_rate and hard_hit_pct not available from BRef — set to NaN (will use position average)
        df["barrel_rate"] = float("nan")
        df["hard_hit_pct"] = float("nan")
        _save_cache(cache_key, df)
        return df
    except Exception as e:
        print(f"[data_fetcher] Batter stats fetch failed: {e}")
        return pd.DataFrame()
```

### Fix `get_pitcher_statcast()`

Replace with `pitching_stats_bref`. Derive features:

```python
from pybaseball import pitching_stats_bref

def get_pitcher_statcast(season=None):
    season = season or date.today().year
    cache_key = f"pitcher_stats_bref_{season}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached
    try:
        df = pitching_stats_bref(season)
        # FIP proxy: use ERA as fallback (BRef doesn't have FIP directly)
        df["FIP"] = df["ERA"]
        # HR/FB: HR / (IP * 1.2) — approximate fly balls allowed
        df["pitcher_hr_fb"] = df["HR"] / (df["IP"] * 1.2).replace(0, 1)
        # K/9 from SO9 column
        df["K9"] = df["SO9"]
        # K/BB
        df["K/BB"] = df["SO/W"]
        # xFIP not available — use ERA
        df["xFIP"] = df["ERA"]
        # Fly ball rate proxy from GB/FB ratio: FB% = 1 / (1 + GB/FB)
        def safe_fb_rate(gbfb):
            try:
                v = float(gbfb)
                return 1.0 / (1.0 + v) if v > 0 else 0.45
            except:
                return 0.45
        df["fly_ball_rate"] = df["GB/FB"].apply(safe_fb_rate)
        df["pitcher_hard_hit_allowed"] = float("nan")  # not available from BRef
        _save_cache(cache_key, df)
        return df
    except Exception as e:
        print(f"[data_fetcher] Pitcher stats fetch failed: {e}")
        return pd.DataFrame()
```

### Fix `backtest.py`

The backtest loads historical stats and HR actuals. Update it to use the same bref functions:

1. Replace any calls to `batting_stats()` / `pitching_stats()` with `batting_stats_bref()` / `pitching_stats_bref()`
2. For HR actuals per game: use `pybaseball.statcast(start_dt, end_dt)` — this pulls from Baseball Savant which is confirmed working. Filter for `events == 'home_run'` to get actual HR events per batter per game.
3. Column name mapping — update any references to FanGraphs-specific column names (wRC+, barrel%, hard_hit%) to the BRef equivalents (ISO, hr_fb_ratio, k_rate) or NaN fallbacks.

### Fix `feature_builder.py`

Update column lookups to match BRef column names:

- `wRC+` / `wRC` / `wRCplus` → also try `OPS` as fallback (not perfect but directionally correct)
- `barrel_rate` → use NaN fallback (position average) since BRef doesn't have it
- `hard_hit_pct` → use NaN fallback
- `hr_fb_ratio` → column is now available directly
- `ISO` → column now available directly
- `FIP` → now derived from ERA, column available
- `K/BB` → now available as `SO/W`
- `fly_ball_rate` → now available as derived column

In `_find_team_wrc_plus()` (in both `market_totals.py` and `market_f5.py`): add `OPS` as a fallback column when `wRC+` is not found. Scale OPS to approximate wRC+: `wrc_approx = (OPS / 0.728) * 100` where 0.728 is approx league average OPS.

## After Fixes

1. Run `python main.py dry_run` — confirm batter/pitcher stats load successfully (no "fetch failed" messages for stats)
2. Run `python main.py backtest` — confirm it runs through all 4 years and produces `outputs/backtest/backtest_results_YYYY-MM-DD.json` and `weights.json`
3. Verify `weights.json` exists after backtest
4. Run `python main.py dry_run` again — confirm bet sizes are now above $5 (weights.json exists)
5. `git add` and commit all changes
