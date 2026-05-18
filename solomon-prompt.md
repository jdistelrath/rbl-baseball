# Solomon Build Prompt — MLB HR Prop Model

You are **Solomon**, the Builder. You receive a complete architecture specification and execute it literally. Every number traces to a source. You write real, working code — not stubs, not pseudocode, not placeholders. You verify imports are installable, handle missing data gracefully, and do not proceed to the next module until the current one works.

---

## Your Mission

Build a daily MLB home run prop model that runs on a Mac Mini (Apple Silicon, macOS, Python 3.11+), sends a morning brief to Telegram, and powers two outputs:

1. **Sharp brief (Jim):** Correlated HR stacks with scoring, reasoning, and confidence tiers. Variable bet sizing based on EV.
2. **Floor guy's list:** Top 10 player names only, no context — for a coworker who plays 10-leg PrizePicks parlays.

**Platform:** PrizePicks Power Play (binary HR props — did they homer or not, line is always 0.5)
- 3-leg = 5x | 4-leg = 10x | 5-leg = 20x | 6-leg = 40x
- No correlation restrictions — stack freely

---

## Target Directory

`/Users/jimsmini/.openclaw/workspace/rbl/baseball/`

Create all files here. The directory already exists.

---

## Credentials & Config

Create a `.env` file template (do not populate values — Jim will fill them in):

```
OWM_API_KEY=your_openweathermap_key_here
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
THE_ODDS_API_KEY=your_odds_api_key_here  # optional, for line shopping
```

Create a `config.yaml`:

```yaml
bet_sizing:
  high_confidence: 10   # $ amount
  standard: 5
  lottery: 1

brief_times:
  morning: "10:50"      # ET
  followup: "13:00"     # ET

lineup_check_interval_minutes: 10
lineup_check_window:
  start: "09:00"
  end: "14:00"

parlay:
  sharp_legs: 3
  lottery_legs: 6
  target_pitchers_per_day: 3
  batters_per_stack: 4

backtesting:
  start_year: 2021
  end_year: 2024        # leave 2025 as holdout
  holdout_year: 2025
```

---

## 8 Modules — Build in This Order

### 1. `config.py`
- Load `.env` via `python-dotenv`
- Load `config.yaml` via PyYAML
- Auto-load `weights.json` if it exists (calibrated backtest weights); fall back to equal weights
- Expose a single `CFG` object used by all other modules
- Define stadium coordinates dict (lat/lon for every MLB ballpark) — used for weather lookups

### 2. `data_fetcher.py`
All network I/O lives here. Other modules never make HTTP calls.

**Functions to implement:**

`get_today_schedule()` → list of games (home team, away team, start time ET, stadium, game ID)
- Source: MLB Stats API `https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}`

`get_confirmed_lineups(game_id)` → dict of `{home: [player_ids...], away: [player_ids...]}` or None if unconfirmed
- Source: MLB Stats API `https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore`
- Return None if lineup not yet posted

`get_probable_pitchers(game_id)` → `{home_pitcher_id, away_pitcher_id}` or None
- Source: MLB Stats API schedule endpoint (probables field)

`get_batter_statcast(player_name, season_range=(2022, 2025))` → DataFrame
- Source: `pybaseball.statcast_batter_exitvelo_barrels()` and `pybaseball.batting_stats()`
- Cache to `cache/statcast_batters_{date}.pkl` — refresh once per day

`get_pitcher_statcast(player_name, season_range=(2022, 2025))` → DataFrame
- Source: `pybaseball.pitching_stats()` with FanGraphs fields (HR/FB, xFIP, fly ball rate, hard hit rate)
- Cache to `cache/statcast_pitchers_{date}.pkl`

`get_weather(stadium_name, game_time_utc)` → dict `{temp_f, wind_speed_mph, wind_dir_degrees, wind_dir_label}`
- Source: OpenWeatherMap One Call API (or Current Weather API if One Call unavailable on free tier)
- Use stadium coordinates from `config.py`
- Return neutral defaults if API key missing (temp=70, wind=0) — do not crash

`get_park_factors()` → DataFrame
- Source: FanGraphs park factors page (scrape or use pybaseball if available)
- Cache to `cache/park_factors_{year}.pkl`

`get_odds(sport="baseball_mlb", markets="h2h,totals")` → dict (optional)
- Source: The Odds API `https://api.the-odds-api.com/v4/sports/{sport}/odds/`
- Skip gracefully if `THE_ODDS_API_KEY` not set

### 3. `feature_builder.py`
Takes raw DataFrames from `data_fetcher.py`, returns a clean feature dict per batter-game matchup.

**Batter features (30-day rolling where available, season otherwise):**
- `barrel_rate` — barrels / batted ball events
- `hr_fb_ratio` — HR / fly balls hit
- `iso` — isolated power (SLG - AVG)
- `hard_hit_pct` — hard hit % (exit velo > 95mph)
- `platoon_factor` — 1.0 neutral, 1.15 favorable (batter/pitcher handedness advantage), 0.85 unfavorable

**Pitcher features:**
- `pitcher_hr_fb` — HR/FB rate allowed (season)
- `pitcher_fly_ball_rate` — fly ball % allowed
- `pitcher_xfip` — xFIP (lower = better pitcher, penalize for high values)
- `pitcher_hard_hit_allowed` — hard hit % allowed

**Context features:**
- `park_hr_factor` — from FanGraphs (1.0 = neutral, >1.0 = hitter friendly)
- `wind_bonus` — calculated from wind speed + direction relative to CF:
  - Blowing out to CF/RF/LF at >10mph: +0.05 to +0.15 linear
  - Blowing in: -0.05 to -0.10
  - Crosswind or <5mph: 0
- `temp_bonus` — temp >80°F: +0.03, temp <45°F: -0.03, else 0
- `batting_order_position` — 1-4 = prime, 5-6 = neutral, 7-9 = penalty

Handle missing data: if a feature is unavailable, use position-average for that feature (not 0, not drop the player).

### 4. `scorer.py`
Takes feature dicts, produces composite HR probability score and confidence tier.

**Scoring:**
1. Z-score normalize each feature across all batter-game candidates for the day
2. Weighted sum using weights from `config.py` (loaded from `weights.json` if calibrated, else equal)
3. Sigmoid transform to 0-1 probability range
4. Confidence tiers:
   - **STRONG** (top 15% of scores): $10 suggested bet
   - **STANDARD** (next 35%): $5 suggested bet
   - **SPECULATIVE** (bottom 50%): $1 lottery only

Output per batter: `{name, team, opponent_pitcher, score, prob, tier, wind_label, park_label, key_edge}`
- `key_edge`: one-line string explaining the primary reason for the pick (e.g., "LHB vs. RHP with 18% HR/FB allowed, wind 12mph to RF")

### 5. `stack_builder.py`
Takes scored batters, builds parlay recommendations.

**Logic:**
1. Rank all pitchers by HR-proneness score (composite of HR/FB, xFIP, fly ball rate)
2. Identify top 2-3 "target pitchers" for the day
3. For each target pitcher, rank their opposing batters by score
4. Build stacks: take top 3-4 batters per target pitcher

**Output parcels:**
- **Sharp parlay:** Best 3-leg correlated stack (same target pitcher), STRONG/STANDARD tier only
- **Lottery parlay:** Best 6-leg stack — can mix target pitchers if needed to reach 6
- **Floor guy list:** Top 10 individual batters by score, any pitcher

### 6. `backtest.py`
Walk-forward validation harness. Run manually only — not on cron.

**Logic:**
- Pull historical Statcast data 2021–2024 via pybaseball
- For each game-day in range: build features for all batters → score → check if they actually homered (binary)
- Walk-forward: train weights on 2021–2022, validate on 2023, test on 2024, holdout 2025
- Metric: Brier score + calibration curve + simulated PrizePicks ROI at each confidence tier
- Output: `outputs/backtest_results_{date}.json` + `weights.json` (calibrated weights)
- Print summary to console

### 7. `messenger.py`
Formats and sends Telegram messages. No other module sends Telegram messages directly.

**Sharp brief format:**
```
⚾ MLB HR Model — {date}

🎯 SHARP PLAY (3-leg, ~${bet})
Stack: {team} batters vs. {pitcher}
• {Batter1} — {key_edge}
• {Batter2} — {key_edge}
• {Batter3} — {key_edge}
PrizePicks 3-leg Power Play = 5x

📊 LOTTERY STACK (6-leg, $1)
{Batter1}, {Batter2}, {Batter3}, {Batter4}, {Batter5}, {Batter6}

🏟️ TODAY'S TOP 10 (floor picks)
1. {Name} ({Team})
2. {Name} ({Team})
...

⚠️ PENDING: {N} day games — follow-up at 1pm if lineups confirm
```

**Follow-up format (only if new lineups):**
```
⚾ HR Model Update — {time}

New lineups confirmed:
[same format for newly added picks only]
```

**Functions:**
- `send_brief(parcels, pending_games)` — morning brief
- `send_followup(new_parcels)` — 1pm update, only if new data
- `send_error(msg)` — alert Donna if something breaks

### 8. `main.py`
Orchestrator. All modes invokable via CLI arg.

```
python main.py morning_brief
python main.py followup
python main.py lineup_check
python main.py backtest
python main.py dry_run        # full pipeline, no Telegram send, print to console
```

**State file:** `state/daily_state_{date}.json`
- Tracks which game IDs have been sent, lineup status per game
- Prevents duplicate sends on follow-up

**morning_brief mode:**
1. Get today's schedule
2. For each game: get lineups (confirmed only) + probable pitchers
3. Build features → score → stack
4. Send brief (include pending count)
5. Write state file

**followup mode:**
1. Load state file
2. Check lineups for previously-pending games
3. If new lineups confirmed: build features → score → stack for new games only
4. If anything new: send follow-up message
5. Update state file

**lineup_check mode:**
- Silent — just checks lineups and updates state file
- Used by cron every 10 min; triggers followup internally if new lineups found between 9am–2pm

---

## Directory Structure to Create

```
rbl/baseball/
├── main.py
├── config.py
├── data_fetcher.py
├── feature_builder.py
├── scorer.py
├── stack_builder.py
├── backtest.py
├── messenger.py
├── config.yaml
├── .env               # template only, no real values
├── requirements.txt
├── README.md
├── cache/             # .gitignore this
├── state/             # .gitignore this
└── outputs/
    └── backtest/
```

---

## requirements.txt

```
pybaseball>=2.2.7
pandas>=2.0
numpy>=1.24
scipy>=1.10
requests>=2.28
python-dotenv>=1.0
PyYAML>=6.0
scikit-learn>=1.3
```

---

## README.md

Include:
- Setup instructions (pip install, .env population)
- How to run dry_run first
- How to run backtest
- Cron setup instructions for Mac (launchd or crontab)
- Note: OWM free tier (1000 calls/day) is sufficient; The Odds API optional

---

## Cron Setup (document in README, don't configure — Jim will enable)

```
# Lineup check every 10 min, 9am-2pm ET
*/10 9-13 * * * cd /Users/jimsmini/.openclaw/workspace/rbl/baseball && python main.py lineup_check

# Morning brief 10:50am ET
50 10 * * * cd /Users/jimsmini/.openclaw/workspace/rbl/baseball && python main.py morning_brief

# Follow-up 1pm ET
0 13 * * * cd /Users/jimsmini/.openclaw/workspace/rbl/baseball && python main.py followup
```

---

## Build Rules

1. Build modules in order (1→8). Do not skip ahead.
2. After each module: verify it imports cleanly, write a quick smoke test, confirm it works before moving on.
3. No stubs. If a feature is hard to implement, implement it simply — but implement it.
4. Missing data never crashes the pipeline. Degrade gracefully.
5. When done: run `python main.py dry_run` and confirm output looks correct before declaring done.
6. Final step: git add + commit everything in the baseball/ directory.
