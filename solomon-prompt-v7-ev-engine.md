# Solomon Prompt v7 — Full EV Engine: HR Props + K Props + Totals Improvements

You are **Solomon**, the Builder. This is a significant extension of the existing codebase. Build in order, verify each piece works before moving to the next. Do not break anything existing.

## Working Directory
`/Users/jimsmini/.openclaw/workspace/rbl/baseball/`

## Overview

Build a true expected value engine across three baseball markets:
1. **HR Props (DK straight bets)** — compare model HR probability to DK implied probability, flag +EV single bets
2. **Strikeout Props** — build a K model, compare to DK lines, flag +EV bets  
3. **Totals improvements** — add bullpen ERA + recent form, rerun calibration

---

## Part 1: Pull Player Prop Lines from The Odds API

### Extend `data_fetcher.py`

Add new function `get_player_props(market)`:

```python
def get_player_props(market="batter_home_runs"):
    """
    Pull player prop lines from The Odds API.
    market options:
        "batter_home_runs" — HR props (over 0.5 HRs)
        "pitcher_strikeouts" — K props (over/under X.5 Ks)
    
    Returns list of dicts:
    {
        player_name: str,
        team: str,
        market: str,
        over_line: float,    # the prop line (e.g. 0.5 for HR, 6.5 for Ks)
        over_odds: int,      # American odds for over
        under_odds: int,     # American odds for under
        implied_over_prob: float,
        bookmaker: str
    }
    
    Gracefully returns [] if API key not set or request fails.
    Prefer DraftKings, fall back to FanDuel, then others.
    """
    api_key = os.getenv("THE_ODDS_API_KEY", "")
    if not api_key:
        return []
    
    url = (
        f"https://api.the-odds-api.com/v4/sports/baseball_mlb/odds/"
        f"?apiKey={api_key}&regions=us&markets={market}&oddsFormat=american"
    )
    
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        events = resp.json()
    except Exception as e:
        print(f"[data_fetcher] Player props fetch failed ({market}): {e}")
        return []
    
    results = []
    PREFERRED_BOOKS = ["draftkings", "fanduel", "betmgm"]
    
    for event in events:
        home_team = event.get("home_team", "")
        away_team = event.get("away_team", "")
        
        # Sort bookmakers by preference
        bookmakers = sorted(
            event.get("bookmakers", []),
            key=lambda b: (0 if b.get("key", "") in PREFERRED_BOOKS else 1)
        )
        
        seen_players = set()
        
        for bookmaker in bookmakers:
            bk_key = bookmaker.get("key", "")
            for mkt in bookmaker.get("markets", []):
                if mkt.get("key") != market:
                    continue
                for outcome in mkt.get("outcomes", []):
                    player = outcome.get("description", outcome.get("name", ""))
                    side = outcome.get("name", "")  # "Over" or "Under"
                    price = outcome.get("price")
                    line = outcome.get("point")
                    
                    if not player or player in seen_players:
                        continue
                    
                    if side == "Over" and price is not None and line is not None:
                        from ev_calculator import american_to_implied_prob
                        impl_prob = american_to_implied_prob(int(price))
                        
                        # Find matching under
                        under_odds = None
                        for o2 in mkt.get("outcomes", []):
                            if o2.get("description", o2.get("name","")) == player and o2.get("name") == "Under":
                                under_odds = o2.get("price")
                        
                        results.append({
                            "player_name": player,
                            "home_team": home_team,
                            "away_team": away_team,
                            "market": market,
                            "over_line": line,
                            "over_odds": int(price),
                            "under_odds": int(under_odds) if under_odds else None,
                            "implied_over_prob": round(impl_prob, 4),
                            "bookmaker": bk_key,
                        })
                        seen_players.add(player)
    
    return results
```

---

## Part 2: HR Prop EV Engine

### New file: `market_hr_ev.py`

```python
"""
Market: HR Props — straight bet EV analysis on DraftKings.
Compares model HR probability to DK implied probability.
Flags bets where model_prob > implied_prob + MIN_EDGE.
"""

MIN_EDGE = 0.03  # minimum 3% edge to surface a bet

def score_hr_props(scored_batters, hr_prop_lines):
    """
    Args:
        scored_batters: list of dicts from scorer.py (with prob, name, tier fields)
        hr_prop_lines: list of dicts from data_fetcher.get_player_props("batter_home_runs")
    
    Returns list of +EV HR prop bets, sorted by edge descending:
    {
        player_name,
        team,
        model_prob,       # our model's HR probability
        implied_prob,     # book's implied probability
        edge,             # model_prob - implied_prob
        over_odds,        # DK American odds
        ev_per_dollar,
        kelly,
        suggested_bet,
        tier,
        key_edge,
    }
    """
    from ev_calculator import calculate_ev, kelly_fraction, suggested_bet_size
    
    # Build lookup: player_name (lowercase, stripped) -> prop line
    prop_lookup = {}
    for prop in hr_prop_lines:
        key = prop["player_name"].lower().strip()
        prop_lookup[key] = prop
    
    results = []
    
    for batter in scored_batters:
        name = batter.get("name", "")
        model_prob = batter.get("prob", 0.0)
        
        # Try to match player name to prop line
        prop = _find_prop(name, prop_lookup)
        if prop is None:
            continue
        
        implied_prob = prop["implied_over_prob"]
        edge = model_prob - implied_prob
        
        if edge < MIN_EDGE:
            continue  # not enough edge
        
        over_odds = prop["over_odds"]
        ev = calculate_ev(model_prob, over_odds)
        k = kelly_fraction(model_prob, over_odds)
        bet = suggested_bet_size(ev, k)
        
        results.append({
            "player_name": name,
            "team": batter.get("team", ""),
            "model_prob": round(model_prob, 4),
            "implied_prob": round(implied_prob, 4),
            "edge": round(edge, 4),
            "over_odds": over_odds,
            "ev_per_dollar": round(ev, 4),
            "kelly": round(k, 4),
            "suggested_bet": bet,
            "tier": batter.get("tier", "SPECULATIVE"),
            "key_edge": batter.get("key_edge", ""),
        })
    
    results.sort(key=lambda x: x["edge"], reverse=True)
    return results


def _find_prop(player_name, prop_lookup):
    """Fuzzy match player name to prop lookup."""
    key = player_name.lower().strip()
    if key in prop_lookup:
        return prop_lookup[key]
    # Try last name match
    last = player_name.split()[-1].lower()
    for k, v in prop_lookup.items():
        if last in k:
            return v
    return None
```

---

## Part 3: Strikeout Prop EV Engine

### New file: `market_k_ev.py`

```python
"""
Market: Strikeout Props — straight bet EV analysis on DraftKings.
Only targets starters with K/9 > 7.5.
"""

MIN_K9 = 7.5
MIN_EDGE = 0.03

def score_k_props(games, pitcher_df, batter_df, weather_map, k_prop_lines):
    """
    Args:
        games: today's schedule (list of game dicts with pitcher names)
        pitcher_df: pitcher stats DataFrame
        batter_df: batter stats DataFrame  
        weather_map: dict of stadium -> weather dict
        k_prop_lines: list from data_fetcher.get_player_props("pitcher_strikeouts")
    
    Returns list of +EV K prop bets:
    {
        pitcher_name, team, opponent,
        model_k_projection,
        book_line,
        over_odds, under_odds,
        best_side,          # "over" or "under"
        edge,
        ev_per_dollar,
        suggested_bet,
        key_factors,
    }
    """
    from ev_calculator import calculate_ev, kelly_fraction, suggested_bet_size, american_to_implied_prob
    
    results = []
    
    # Build prop lookup
    prop_lookup = {}
    for prop in k_prop_lines:
        key = prop["player_name"].lower().strip()
        prop_lookup[key] = prop
    
    for game in games:
        for role in ("home_pitcher_name", "away_pitcher_name"):
            pitcher_name = game.get(role, "TBD")
            if not pitcher_name or pitcher_name == "TBD":
                continue
            
            pitcher_row = _find_pitcher(pitcher_name, pitcher_df)
            if pitcher_row is None:
                continue
            
            k9 = _get_stat(pitcher_row, ["K9", "SO9", "K/9"], 0.0)
            if k9 < MIN_K9:
                continue  # not a strikeout pitcher
            
            # Project Ks
            ip_per_start = _get_stat(pitcher_row, ["IP"], 150.0) / max(_get_stat(pitcher_row, ["GS"], 25.0), 1)
            ip_per_start = min(ip_per_start, 6.5)  # cap at 6.5 innings
            
            # Opposing team K rate
            opp_team = game.get("away_team") if role == "home_pitcher_name" else game.get("home_team")
            opp_k_rate = _get_team_k_rate(opp_team, batter_df)
            k_rate_adj = 1.0 + (opp_k_rate - 0.225) * 0.5  # 22.5% league avg K rate
            
            # Weather
            stadium = game.get("stadium", "")
            weather = weather_map.get(stadium, {"temp_f": 70})
            temp_adj = -0.3 if weather.get("temp_f", 70) < 45 else 0.0
            
            model_k = (k9 / 9.0) * ip_per_start * k_rate_adj + temp_adj
            model_k = max(0.0, round(model_k, 1))
            
            # Find prop line
            prop = _find_prop(pitcher_name, prop_lookup)
            if prop is None:
                continue
            
            book_line = prop["over_line"]
            over_odds = prop["over_odds"]
            under_odds = prop.get("under_odds")
            
            diff = model_k - book_line
            if abs(diff) < 0.5:
                continue  # not enough model/line gap
            
            # Convert to probability
            # Use normal distribution: SD of ~1.5 Ks around model projection
            import scipy.stats as st
            over_prob = 1.0 - st.norm.cdf(book_line, loc=model_k, scale=1.5)
            under_prob = 1.0 - over_prob
            
            if diff > 0:  # model says more Ks than book line → over
                best_side = "over"
                best_prob = over_prob
                best_odds = over_odds
            else:  # model says fewer Ks → under
                best_side = "under"
                best_prob = under_prob
                best_odds = under_odds
            
            if best_odds is None:
                continue
            
            implied = american_to_implied_prob(int(best_odds))
            edge = best_prob - implied
            
            if edge < MIN_EDGE:
                continue
            
            ev = calculate_ev(best_prob, int(best_odds))
            k_frac = kelly_fraction(best_prob, int(best_odds))
            bet = suggested_bet_size(ev, k_frac)
            
            factors = [
                f"{pitcher_name} K/9 {k9:.1f}",
                f"model projects {model_k} Ks vs line {book_line}",
            ]
            if opp_k_rate > 0.26:
                factors.append(f"high-K opponent ({opp_k_rate:.1%} K rate)")
            
            results.append({
                "pitcher_name": pitcher_name,
                "team": game.get("home_team") if role == "home_pitcher_name" else game.get("away_team"),
                "opponent": opp_team,
                "model_k_projection": model_k,
                "book_line": book_line,
                "over_odds": over_odds,
                "under_odds": under_odds,
                "best_side": best_side,
                "edge": round(edge, 4),
                "ev_per_dollar": round(ev, 4),
                "suggested_bet": bet,
                "key_factors": factors,
                "description": f"{pitcher_name} {best_side.upper()} {book_line} Ks",
            })
    
    results.sort(key=lambda x: x["edge"], reverse=True)
    return results


def _find_pitcher(name, df):
    if df.empty: return None
    for col in ("Name", "name"):
        if col in df.columns:
            rows = df[df[col] == name]
            if not rows.empty: return rows.iloc[0]
            last = name.split()[-1]
            rows = df[df[col].str.contains(last, na=False)]
            if len(rows) == 1: return rows.iloc[0]
    return None

def _get_stat(row, cols, default):
    import math
    if row is None: return default
    for c in cols:
        if c in row.index:
            try:
                v = float(row[c])
                return v if not (math.isnan(v) or math.isinf(v)) else default
            except: pass
    return default

def _get_team_k_rate(team_name, batter_df):
    if batter_df.empty: return 0.225
    for col in ("Team", "Tm", "team"):
        if col in batter_df.columns:
            rows = batter_df[batter_df[col].astype(str).str.contains(
                team_name.split()[-1] if team_name else "", case=False, na=False
            )]
            if not rows.empty:
                for kcol in ("SO%", "K%", "k_rate"):
                    if kcol in rows.columns:
                        import math
                        val = rows[kcol].mean()
                        if not math.isnan(val):
                            return val / 100.0 if val > 1 else val
    return 0.225

def _find_prop(name, lookup):
    key = name.lower().strip()
    if key in lookup: return lookup[key]
    last = name.split()[-1].lower()
    for k, v in lookup.items():
        if last in k: return v
    return None
```

---

## Part 4: Bullpen ERA + Recent Form in Totals Model

### Extend `data_fetcher.py`

Add `get_bullpen_stats(season=None)`:

```python
def get_bullpen_stats(season=None):
    """
    Pull relief pitcher ERA by team from MLB Stats API.
    Returns dict: {team_name: bullpen_era}
    """
    season = season or date.today().year
    cache_key = f"bullpen_stats_{season}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached
    
    url = (
        f"https://statsapi.mlb.com/api/v1/stats?stats=season&group=pitching"
        f"&gameType=R&season={season}&sportId=1&playerPool=All"
        f"&fields=stats,splits,stat,era,team,name&limit=1000"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        # Aggregate relief ERA by team (pitchers with GS < 5 = relievers)
        team_era = {}
        team_counts = {}
        
        for split in data.get("stats", [{}])[0].get("splits", []):
            stat = split.get("stat", {})
            team = split.get("team", {}).get("name", "")
            gs = stat.get("gamesStarted", 0)
            era = stat.get("era", None)
            ip = stat.get("inningsPitched", "0")
            
            if gs > 5 or not team or era is None:
                continue
            try:
                era_f = float(era)
                ip_f = float(ip)
                if ip_f < 5:
                    continue
                if team not in team_era:
                    team_era[team] = 0.0
                    team_counts[team] = 0.0
                team_era[team] += era_f * ip_f
                team_counts[team] += ip_f
            except:
                continue
        
        # Weighted average ERA by team
        result = {}
        for team in team_era:
            if team_counts[team] > 0:
                result[team] = round(team_era[team] / team_counts[team], 2)
        
        _save_cache(cache_key, result)
        return result
    except Exception as e:
        print(f"[data_fetcher] Bullpen stats fetch failed: {e}")
        return {}
```

### Update `market_totals.py`

Add bullpen ERA adjustment to `score_game_total()`:

```python
# After existing pitcher_adj calculation, add:
bullpen_stats = data_fetcher.get_bullpen_stats() if hasattr(data_fetcher, 'get_bullpen_stats') else {}

home_bullpen_era = bullpen_stats.get(home, 4.20)
away_bullpen_era = bullpen_stats.get(away, 4.20)

# Each 1.0 ERA above/below league avg (4.20) in bullpen = 0.3 run shift
# (Less impact than starter since bullpen pitches fewer innings)
bullpen_adj = ((home_bullpen_era - 4.20) + (away_bullpen_era - 4.20)) * 0.3

model_total = _BASE_RUNS + pitcher_adj + bullpen_adj + offense_adj + park_adj + weather_adj
```

Also add bullpen info to the `_build_factors()` output when bullpen ERA is significantly above/below average.

---

## Part 5: Unified EV Output in `messenger.py`

Add new function `send_ev_props(hr_props, k_props)`:

```python
def send_ev_props(hr_props, k_props):
    """
    Send +EV straight bet recommendations for DraftKings.
    Only sends if there are actual +EV plays (edge > MIN_EDGE).
    
    Format:
    ⚾ DK +EV PROPS — {date}

    💰 HR PROPS (straight bets)
    • {Player} +EV: model {X}% vs book {Y}% → {odds} | ${bet}
      Edge: {key_edge}

    ⚡ K PROPS
    • {Pitcher} OVER/UNDER {line} Ks → {odds} | ${bet}
      {key_factors}
    
    Only bet when model edge > 3%.
    """
```

---

## Part 6: Wire into `main.py`

In `mode_morning_brief()` and `mode_dry_run()`, after existing pipelines:

```python
# Pull player props
hr_prop_lines = fetcher.get_player_props("batter_home_runs")
k_prop_lines = fetcher.get_player_props("pitcher_strikeouts")

# Score HR props vs DK lines
from market_hr_ev import score_hr_props
hr_ev_plays = score_hr_props(all_scored_batters, hr_prop_lines)

# Score K props
weather_map = {g["stadium"]: fetcher.get_weather(g["stadium"], None) for g in games}
from market_k_ev import score_k_props
k_ev_plays = score_k_props(games, pitcher_df, batter_df, weather_map, k_prop_lines)

# Send/print
messenger.send_ev_props(hr_ev_plays[:5], k_ev_plays[:3])
```

---

## Build Order

1. Extend `data_fetcher.py` (player props + bullpen)
2. Build `market_hr_ev.py`
3. Build `market_k_ev.py`
4. Update `market_totals.py` (bullpen adjustment)
5. Update `messenger.py` (send_ev_props)
6. Update `main.py` (wire everything together)

## After Build

1. Run `python main.py dry_run` — confirm all sections output cleanly
2. Verify HR prop lines are being pulled from DK
3. Verify K prop lines are being pulled
4. Verify +EV plays section appears (may be empty if no edge today — that's fine)
5. Run `python main.py backtest_totals` — this runs 2021-2025 (already configured in v6). Confirm bullpen ERA improves accuracy above previous 51.3%
6. Run `python main.py backtest_gamelevel` with years=[2021,2022,2023,2024,2025] — update the function call in main.py to include 2025 for the HR game-level backtest as well
6. `git add` all new/changed files and commit
