# Solomon Prompt v5 — Game-Level Backtest (2024 Single Season)

You are **Solomon**, the Builder. Extend the existing backtest to use game-level Statcast data instead of season aggregates. Surgical addition — do not break anything existing.

## Working Directory
`/Users/jimsmini/.openclaw/workspace/rbl/baseball/`

## The Problem

The current `backtest.py` uses season-level batting/pitching stats to simulate picks. This produces tiny samples (20 STRONG bets per season) because it can't distinguish game-by-game conditions. We need game-level simulation.

## What to Build

Add a new mode to `backtest.py`: `run_game_level_backtest(years=[2024])`

Also wire it into `main.py` as a new CLI mode: `python main.py backtest_gamelevel`

### Data Pull Strategy

Use `pybaseball.statcast(start_dt, end_dt)` — pulls from Baseball Savant, confirmed working. Returns one row per pitch. We need HR outcomes per batter per game.

```python
from pybaseball import statcast

def fetch_game_level_hr_data(year):
    """
    Pull all Statcast data for a season, aggregate to batter-game level.
    Returns DataFrame with columns:
        game_date, batter_name, batter_id, pitcher_name, pitcher_id,
        home_team, away_team, homered (bool)
    """
    # Pull month by month to avoid timeouts
    months = [
        (f"{year}-04-01", f"{year}-04-30"),
        (f"{year}-05-01", f"{year}-05-31"),
        (f"{year}-06-01", f"{year}-06-30"),
        (f"{year}-07-01", f"{year}-07-31"),
        (f"{year}-08-01", f"{year}-08-31"),
        (f"{year}-09-01", f"{year}-09-30"),
        (f"{year}-10-01", f"{year}-10-01"),  # playoffs start
    ]
    
    dfs = []
    for start, end in months:
        print(f"  Fetching {start} to {end}...")
        try:
            df = statcast(start_dt=start, end_dt=end)
            if df is not None and not df.empty:
                dfs.append(df)
        except Exception as e:
            print(f"  Warning: {e}")
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    raw = pd.concat(dfs, ignore_index=True)
    
    # Aggregate to batter-game level
    # A batter homered in a game if any of their PAs had events == 'home_run'
    raw["homered"] = raw["events"] == "home_run"
    
    game_level = raw.groupby(
        ["game_date", "batter", "pitcher", "home_team", "away_team"]
    ).agg(
        homered=("homered", "any"),
        batter_name=("player_name", "first"),
    ).reset_index()
    
    game_level.rename(columns={"batter": "batter_id", "pitcher": "pitcher_id"}, inplace=True)
    
    return game_level
```

### Game-Level Simulation

```python
def run_game_level_backtest(years=[2024]):
    """
    For each game-day in the year:
    1. Get batter season stats (rolling — use stats from prior season to avoid lookahead)
    2. Score each batter using the model
    3. Compare predicted score to actual HR outcome
    4. Track ROI by confidence tier
    """
    results = {
        "STRONG": {"bets": 0, "wins": 0, "wagered": 0.0, "returned": 0.0},
        "STANDARD": {"bets": 0, "wins": 0, "wagered": 0.0, "returned": 0.0},
        "SPECULATIVE": {"bets": 0, "wins": 0, "wagered": 0.0, "returned": 0.0},
    }
    
    BET_SIZES = {"STRONG": 10.0, "STANDARD": 5.0, "SPECULATIVE": 1.0}
    # Underdog 1-leg payout: 1x (binary, not a parlay leg alone — skip singles)
    # For backtest purposes treat each pick as a straight bet at 2x payout (over/not)
    PAYOUT = 2.0  # simplification for single-leg ROI tracking
    
    for year in years:
        print(f"\n[backtest_gl] Fetching game-level data for {year}...")
        game_data = fetch_game_level_hr_data(year)
        if game_data.empty:
            print(f"[backtest_gl] No data for {year}, skipping.")
            continue
        
        # Use prior year stats to avoid lookahead bias
        batter_stats = get_batter_statcast(year - 1)
        pitcher_stats = get_pitcher_statcast(year - 1)
        
        print(f"[backtest_gl] {len(game_data)} batter-game rows for {year}")
        
        # Score each batter-game row
        correct = 0
        total = 0
        
        for _, row in game_data.iterrows():
            # Build a minimal game dict for feature_builder
            game = {
                "game_id": f"{row['game_date']}_{row['home_team']}",
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "stadium": "",  # park factor will default to neutral
                "home_pitcher_name": "",
                "away_pitcher_name": "",
            }
            weather = {"temp_f": 70, "wind_speed_mph": 5, "wind_dir_degrees": 180}
            
            # Score this batter
            try:
                from feature_builder import build_features_for_batter
                from scorer import score_batter
                
                features = build_features_for_batter(
                    str(row.get("batter_name", "")),
                    game,
                    batter_stats,
                    pitcher_stats,
                    weather,
                    1.0,  # neutral park factor
                    batting_order_pos=3,  # assume mid-order
                )
                scored = score_batter(features, [features])  # single-item list for normalization
                tier = scored.get("tier", "SPECULATIVE")
                prob = scored.get("prob", 0.0)
                actual = bool(row["homered"])
            except Exception:
                continue
            
            bet = BET_SIZES[tier]
            results[tier]["bets"] += 1
            results[tier]["wagered"] += bet
            
            if actual:
                results[tier]["wins"] += 1
                results[tier]["returned"] += bet * PAYOUT
            
            total += 1
        
        print(f"[backtest_gl] Scored {total} batter-game rows")
    
    # Calculate ROI
    for tier in results:
        r = results[tier]
        if r["wagered"] > 0:
            r["roi_pct"] = round((r["returned"] - r["wagered"]) / r["wagered"] * 100, 1)
            r["win_rate"] = round(r["wins"] / r["bets"] * 100, 1) if r["bets"] > 0 else 0
        else:
            r["roi_pct"] = 0
            r["win_rate"] = 0
    
    # Save results
    import json
    from datetime import date
    out = {
        "mode": "game_level",
        "years": years,
        "total_batter_games": total,
        "results": results,
    }
    outpath = f"outputs/backtest/gamelevel_results_{date.today()}.json"
    os.makedirs("outputs/backtest", exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    
    print(f"\n[backtest_gl] Results saved to {outpath}")
    print("\n=== GAME-LEVEL BACKTEST RESULTS ===")
    for tier, r in results.items():
        print(f"{tier}: {r['bets']} bets, {r['win_rate']}% hit rate, {r['roi_pct']}% ROI")
    
    return out
```

### Wire into `main.py`

Add to the MODES dict:
```python
"backtest_gamelevel": mode_backtest_gamelevel,
```

Add the function:
```python
def mode_backtest_gamelevel():
    print("[main] === Game-Level Backtest ===")
    from backtest import run_game_level_backtest
    run_game_level_backtest(years=[2024])
```

## After Build

1. Run `python main.py backtest_gamelevel` — confirm it starts fetching data month by month
2. Let it run to completion (will take 15-30 minutes for 2024)
3. Report the final hit rates and ROI by tier
4. `git add` and commit
