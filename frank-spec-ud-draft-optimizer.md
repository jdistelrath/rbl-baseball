# Frank Spec — UD Draft Optimizer (Historical → Forward)
*Produced: April 30, 2026*
*Frank is the Architect. This doc is for Solomon (the Builder).*

---

## 0. Architecture Overview

```
[Data Pull]       scripts/build_game_logs.py
      ↓
[Score History]   scripts/score_historical.py
      ↓
[Solve Rosters]   scripts/solve_optimal_rosters.py
      ↓
[Feature Eng.]    scripts/build_features.py
      ↓
[Train Model]     scripts/train_model.py
      ↓
[Inference]       model_ud_draft.py  ← replaces heuristic in market_underdog_draft.py
```

All scripts run from `/Users/jimsmini/Projects/baseball/`. All data written to `/Users/jimsmini/Projects/baseball/data/`. All models written to `/Users/jimsmini/Projects/baseball/models/`.

---

## 1. Data Acquisition Plan

### 1.1 Primary Source Decision

The existing `batter_stats_mlb_YYYY_*.pkl` files are **season aggregates** (130 rows/year, 25 columns). They are NOT per-game logs. Per-game counting stats (H, 2B, 3B, HR, RBI, R, BB, HBP, SB) must be pulled fresh.

**Source:** MLB Stats API via the `statsapi` package (already installed; `pip install mlb-statsapi` if missing).

This is preferred over Statcast pitch-by-pitch because:
- Statcast does not include RBI or R (required for UD scoring)
- Statcast SB is unreliable at game aggregate level
- MLB Stats API returns clean per-game box scores directly

### 1.2 Player ID Map

The existing season aggregates include `mlbID` (integer). Use these as the authoritative ID source.

```python
# At start of build_game_logs.py, build master ID map:
import pandas as pd, glob

id_map = {}  # {mlb_id: name}
for path in glob.glob("data/batter_stats_mlb_*.pkl"):
    df = pd.read_pickle(path)
    for _, row in df.iterrows():
        id_map[int(row['mlbID'])] = row['Name']

pitcher_id_map = {}
for path in glob.glob("data/pitcher_stats_mlb_*.pkl"):
    df = pd.read_pickle(path)
    for _, row in df.iterrows():
        pitcher_id_map[int(row['mlbID'])] = row['Name']
```

### 1.3 Batter Game Logs Pull

```python
import statsapi, time

def pull_batter_game_logs(mlb_id: int, year: int) -> list[dict]:
    """Returns list of per-game stat dicts for one batter-season."""
    result = statsapi.player_stat_data(
        personId=mlb_id,
        group='hitting',
        type='gameLog',
        season=year
    )
    return result.get('stats', [])
```

**Rate limiting:** Sleep 0.3 seconds between API calls. If 429 returned, sleep 60 seconds and retry once.

**Output columns per row** (extract from the `stat` dict inside each game entry):

| Column | Source key | Type | Notes |
|---|---|---|---|
| `mlb_id` | (passed in) | int | |
| `game_date` | `'date'` | str→date | format: 'MM/DD/YYYY' → parse to datetime.date |
| `game_pk` | `'gamePk'` | int | MLB game ID |
| `team` | `'team'` | str | abbreviation |
| `opponent` | `'opponent'` | str | abbreviation |
| `home_away` | `'isHome'` | bool | True if home |
| `lineup_spot` | `'battingOrder'` | int | 0-based from API → add 1 |
| `AB` | `stat['atBats']` | int | |
| `R` | `stat['runs']` | int | |
| `H` | `stat['hits']` | int | |
| `doubles` | `stat['doubles']` | int | |
| `triples` | `stat['triples']` | int | |
| `HR` | `stat['homeRuns']` | int | |
| `RBI` | `stat['rbi']` | int | |
| `BB` | `stat['baseOnBalls']` | int | |
| `HBP` | `stat['hitByPitch']` | int | |
| `SB` | `stat['stolenBases']` | int | |
| `SO` | `stat['strikeOuts']` | int | |

**Edge case:** If `stat['atBats']` is None/missing, set all counting stats to 0. This handles games where player was listed but DNP.

**Filter:** Drop rows where `AB == 0 AND BB == 0 AND HBP == 0` — player did not bat.

### 1.4 Pitcher Game Logs Pull

```python
def pull_pitcher_game_logs(mlb_id: int, year: int) -> list[dict]:
    result = statsapi.player_stat_data(
        personId=mlb_id,
        group='pitching',
        type='gameLog',
        season=year
    )
    return result.get('stats', [])
```

**Output columns per row:**

| Column | Source key | Type | Notes |
|---|---|---|---|
| `mlb_id` | (passed in) | int | |
| `game_date` | `'date'` | str→date | |
| `game_pk` | `'gamePk'` | int | |
| `team` | `'team'` | str | |
| `opponent` | `'opponent'` | str | |
| `home_away` | `'isHome'` | bool | |
| `IP` | `stat['inningsPitched']` | float | "6.2" → convert: int_part + decimal_part/3 |
| `ER` | `stat['earnedRuns']` | int | |
| `SO` | `stat['strikeOuts']` | int | |
| `BB_allowed` | `stat['baseOnBalls']` | int | |
| `H_allowed` | `stat['hits']` | int | |
| `HR_allowed` | `stat['homeRuns']` | int | |
| `win` | `stat['wins']` | int | 1 if win, 0 otherwise |
| `GS` | `stat['gamesStarted']` | int | 1 if starter |

**IP conversion (critical):** MLB API returns IP as string like "6.2" where the decimal part is OUTS, not tenths. Convert:
```python
def convert_ip(ip_str: str) -> float:
    parts = str(ip_str).split('.')
    full_innings = int(parts[0])
    outs = int(parts[1]) if len(parts) > 1 else 0
    return full_innings + outs / 3.0
```

**Filter:** Keep only rows where `GS == 1` (starters only). Relief appearances excluded from v1.

### 1.5 Storage Format

```python
# One file per year per player type
# batters:
pd.DataFrame(rows).to_pickle(f"data/game_logs_batters_{year}.pkl")
# pitchers:
pd.DataFrame(rows).to_pickle(f"data/game_logs_pitchers_{year}.pkl")
```

**Expected volumes:**
- Batters: ~130 players × ~140 games × 4 years ≈ 72,800 rows total
- Pitchers: ~60 SP × ~30 starts × 4 years ≈ 7,200 rows total

### 1.6 Years to Pull

Pull: 2022, 2023, 2024, 2025 (partial — through last completed game).

Use players present in the corresponding `batter_stats_mlb_YYYY_*.pkl` files as the pull list. Do not attempt to pull all MLB players — scope to the ~130 batters and ~60 pitchers already in the cache.

### 1.7 Script: `scripts/build_game_logs.py`

```
CLI: python3 scripts/build_game_logs.py --years 2022 2023 2024 2025
Output: data/game_logs_batters_{year}.pkl, data/game_logs_pitchers_{year}.pkl
Idempotent: skip year if file exists and --force not set
Logging: print progress every 10 players, log errors to data/build_logs_errors.txt
```

---

## 2. Historical Scoring Engine

### 2.1 Hitter UD Fantasy Points Formula

```python
UD_HITTER_SCORING = {
    "single": 3.0,
    "double": 6.0,
    "triple": 8.0,
    "home_run": 10.0,
    "walk": 3.0,
    "hbp": 3.0,
    "rbi": 2.0,
    "run": 2.0,
    "stolen_base": 4.0,
}

def compute_hitter_fp(row: pd.Series) -> float:
    """
    row must have: H, doubles, triples, HR, BB, HBP, RBI, R, SB
    Returns total UD fantasy points for this game.
    """
    singles = row['H'] - row['doubles'] - row['triples'] - row['HR']
    singles = max(0, singles)  # guard against data anomalies

    fp = (
        singles      * 3.0 +
        row['doubles'] * 6.0 +
        row['triples'] * 8.0 +
        row['HR']      * 10.0 +
        row['BB']      * 3.0 +
        row['HBP']     * 3.0 +
        row['RBI']     * 2.0 +
        row['R']       * 2.0 +
        row['SB']      * 4.0
    )
    return round(fp, 1)
```

**Edge cases:**
- `singles < 0`: set to 0 (can happen if API data has errors; log a warning with player/date)
- Any field missing or NaN: treat as 0
- Intentional walks (IBB): included in `BB` from API; count normally (they still score 3 pts)

### 2.2 Pitcher UD Fantasy Points Formula

```python
def compute_pitcher_fp(row: pd.Series) -> float:
    """
    row must have: IP (float), ER, SO, win (0/1)
    Returns total UD fantasy points for this start.
    """
    ip = row['IP']  # already converted to float innings
    qs = 1 if (ip >= 6.0 and row['ER'] <= 3) else 0

    fp = (
        row['win']  * 2.0 +
        qs          * 3.0 +
        row['SO']   * 1.0 +
        ip          * 1.0 +
        row['ER']   * -1.0
    )
    return round(fp, 1)
```

**Edge cases:**
- QS definition: IP >= 6.0 AND ER <= 3 (strict — 5.2 IP does not qualify)
- A pitcher can earn both Win (2) and QS (3) — they are not mutually exclusive
- Minimum FP is bounded by ER penalty; no floor imposed

### 2.3 Script: `scripts/score_historical.py`

```
Input: data/game_logs_batters_{year}.pkl, data/game_logs_pitchers_{year}.pkl (all years)
Output: data/historical_fp_batters.pkl, data/historical_fp_pitchers.pkl
Columns added: actual_fp, singles (derived), qs_flag (pitchers)
```

Concatenate all years before saving. Add `year` column. Final schema for `historical_fp_batters.pkl`:
```
mlb_id, game_date, game_pk, team, opponent, home_away, lineup_spot,
AB, R, H, doubles, triples, HR, RBI, BB, HBP, SB, SO,
singles, actual_fp, year
```

Final schema for `historical_fp_pitchers.pkl`:
```
mlb_id, game_date, game_pk, team, opponent, home_away,
IP, ER, SO, BB_allowed, H_allowed, HR_allowed, win, GS, qs_flag,
actual_fp, year
```

---

## 3. Optimal Roster Solver

### 3.1 Solver Choice: ILP (not greedy)

**Decision: Integer Linear Program via PuLP.**

Justification: The historical optimal rosters are the training labels. Label quality is the single most important determinant of model quality. Greedy misses the true optimum by an estimated 3–8% on typical slates. Over 1,000+ training days that noise degrades the model. ILP is exact, runs <200ms per day, and trivially parallelizes across all historical days using Python's `multiprocessing.Pool`. The added `pulp` dependency is acceptable.

### 3.2 Problem Formulation

**Variables:**
```
x_i ∈ {0, 1}  — player i is selected
z_t ∈ {0, 1}  — at least one player from team t is selected
```

**Objective:**
```
maximize Σ(projected_fp_i × x_i)     [for historical: use actual_fp as projected_fp]
```

**Constraints:**
```
(1)  Σ(x_i for i in P_set)  == 1         # exactly 1 pitcher
(2)  Σ(x_i for i in IF_set) >= 2         # at least 2 infielders
(3)  Σ(x_i for i in OF_set) >= 2         # at least 2 outfielders
(4)  Σ(x_i for i in IF_set) + Σ(x_i for i in OF_set) == 5  # FLEX is IF or OF
(5)  Σ(x_i) == 6                          # exactly 6 total players
(6)  x_i <= z_{team(i)}  ∀i              # link player to team binary
(7)  Σ(z_t) >= 2                          # players from at least 2 teams
(8)  z_t ∈ {0, 1},  x_i ∈ {0, 1}
```

Note on constraint (4): Combined with (2) and (3), this forces exactly one FLEX slot taken by an IF or OF player (1P + 2IF + 2OF + 1 more IF-or-OF = 6).

**Position eligibility rules:**
```python
IF_POSITIONS = {"1B", "2B", "3B", "SS"}
OF_POSITIONS = {"LF", "CF", "RF", "DH"}  # DH counts as OF per UD rules
P_POSITIONS  = {"SP", "P"}
```

Players with no recognized position: exclude from that day's solve with a warning.

### 3.3 Historical Solve Loop

For each unique `game_date` in `historical_fp_batters.pkl`:

1. Collect all players with a game on that date (batters + pitchers)
2. Use `actual_fp` as the objective coefficient
3. Run ILP solve
4. Store result: `{game_date, P_player, IF1_player, IF2_player, OF1_player, OF2_player, FLEX_player, total_fp}`
5. Also store `optimal_roster_flag` back to the player-game rows (1 if in optimal, 0 otherwise)

**Parallelization:**
```python
from multiprocessing import Pool
with Pool(processes=8) as pool:
    results = pool.map(solve_one_day, list_of_game_dates)
```

### 3.4 Script: `scripts/solve_optimal_rosters.py`

```
Input: data/historical_fp_batters.pkl, data/historical_fp_pitchers.pkl
       data/player_positions.pkl  (position lookup — see §3.5)
Output: data/historical_optimal_rosters.pkl  (one row per game_date)
        data/historical_fp_batters_with_flags.pkl  (adds optimal_roster_flag column)
        data/historical_fp_pitchers_with_flags.pkl
```

### 3.5 Position Data

The season aggregate files do not include position. Position must be derived from the MLB Stats API.

Add to `build_game_logs.py`: for each player, call `statsapi.lookup_player(mlb_id)` and store primary position. Write `data/player_positions.pkl` as `{mlb_id: position_abbrev}` dict.

Fallback: if position unknown, classify as OF (conservative — they can still fill FLEX).

---

## 4. Feature Engineering

### 4.1 Feature Set — Batters

Computed per `(mlb_id, game_date)` observation. `game_date` is the date of the game being observed. Rolling windows look backward from `game_date` (exclusive — do not include the current game).

**Season aggregate features** (join from `batter_stats_mlb_YYYY_*.pkl` on `mlb_id + year`):
```
season_BA, season_OBP, season_SLG, season_OPS, season_ISO, season_wRC_plus,
season_hr_fb_ratio, season_k_rate, season_barrel_rate, season_hard_hit_pct,
season_PA, season_G
```

**Rolling window features** (computed from `historical_fp_batters.pkl`, looking back from game_date):

For windows of 7, 14, and 30 calendar days:
```
avg_fp_{N}d          — mean actual_fp over last N calendar days (games played only)
games_played_{N}d    — count of games in last N calendar days (availability signal)
hr_rate_{N}d         — HR / AB (last N days)
bb_rate_{N}d         — BB / (AB + BB + HBP) (last N days)
sb_per_game_{N}d     — SB / games_played (last N days)
singles_rate_{N}d    — singles / AB (last N days)
k_rate_{N}d          — SO / AB (last N days)
```

**Context features** (from current game row):
```
lineup_spot          — int 1-9
home_away            — int 1 if home, 0 if away
month                — int 1-12 (seasonality)
year                 — int 2022-2025 (trend)
days_since_last_game — int (calendar days since previous game log entry; cap at 7)
```

**Total batter features: ~43**

### 4.2 Feature Set — Pitchers

**Season aggregate features** (join from `pitcher_stats_mlb_YYYY_*.pkl`):
```
season_ERA, season_k_per_9, season_ip_per_gs, season_whip,
season_bb_pct, season_k_pct, season_hr_fb_ratio_allowed
```

**Rolling window features** (for 7, 14, 30 days):
```
avg_fp_{N}d          — mean actual_fp (starts only)
starts_{N}d          — number of starts in window
k_per_ip_{N}d        — SO / IP
er_per_ip_{N}d       — ER / IP
qs_rate_{N}d         — qs_flag mean (fraction of starts that were QS)
ip_per_start_{N}d    — IP / starts_{N}d
win_rate_{N}d        — win / starts_{N}d
```

**Context features:**
```
home_away            — int 1/0
month                — int 1-12
year                 — int 2022-2025
days_rest            — calendar days since last start (cap at 10)
```

**Total pitcher features: ~31**

### 4.3 Script: `scripts/build_features.py`

```
Input: data/historical_fp_batters_with_flags.pkl
       data/historical_fp_pitchers_with_flags.pkl
       data/batter_stats_mlb_*.pkl  (season aggregates)
       data/pitcher_stats_mlb_*.pkl
Output: data/training_features_batters.pkl
        data/training_features_pitchers.pkl
```

Both output files include the label columns defined in §5.1.

---

## 5. Model Architecture

### 5.1 Training Label Definition — Rolling Expected Value

**Rolling-EV label:** For each `(mlb_id, game_date)` observation, compute the player's average actual FP across all games they played in the **30 calendar days following** `game_date` (exclusive of `game_date` itself).

```python
def compute_rolling_ev_label(player_games: pd.DataFrame, game_date, window_days=30):
    """
    player_games: all game log rows for one mlb_id, sorted by game_date
    Returns mean actual_fp over next 30 days, or NaN if < 3 games exist.
    """
    end = game_date + timedelta(days=window_days)
    future = player_games[
        (player_games['game_date'] > game_date) &
        (player_games['game_date'] <= end)
    ]
    if len(future) < 3:
        return float('nan')
    return future['actual_fp'].mean()
```

**Drop NaN labels** before training. This removes late-season observations where the forward window doesn't have enough games (last ~30 days of 2025).

**Why this label:** Rolling-EV measures sustained expected FP per game — exactly what a drafter wants to optimize. It smooths single-game variance and rewards consistency. Raw FP for one game would reward flukes.

**Secondary label (also compute, used for analysis only):**
```
optimal_roster_flag  — 1 if player appeared in the ILP-optimal roster for that day
```
Do not train the primary model on this binary label. Use it to evaluate whether high-EV players also tend to appear in optimal rosters (sanity check).

### 5.2 Train/Validation/Test Split

```
Train:       game_date in [2022-04-01, 2023-08-31]
Validation:  game_date in [2023-09-01, 2023-12-31]
Test:        game_date in [2024-04-01, 2024-09-30]
Holdout:     game_date in [2025-04-01, present]  — do not touch until model finalized
```

**Important:** Split is temporal (not random). Never shuffle across time boundaries. Data leakage risk: rolling features for a 2023-09-01 validation row look back into the training period — this is correct and expected. The forward-looking label is what must not leak.

### 5.3 Model Type: LightGBM Regressor (Separate Models for Batters and Pitchers)

**Choice justification:** LightGBM handles tabular data with mixed feature types, missing values (NaN in rolling windows for players with few games), and non-linear interactions without preprocessing. Faster to train than XGBoost, equivalent accuracy on DFS tabular tasks. Two separate models (batter/pitcher) because feature sets are entirely different and joint training would require padding/masking complexity with no benefit.

**Model configuration:**
```python
import lightgbm as lgb

batter_model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    min_child_samples=30,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
)
```

**Hyperparameter tuning:** Use Optuna with 50 trials, optimizing validation MAE. Tune: `n_estimators`, `learning_rate`, `max_depth`, `num_leaves`, `min_child_samples`. Save best params to `models/batter_lgbm_best_params.json`.

**Training call:**
```python
batter_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='mae',
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)
```

### 5.4 Evaluation Metrics

Compute on test set (2024):

| Metric | Meaning | Target |
|---|---|---|
| MAE | Mean absolute error on rolling-EV label | < 2.5 FP |
| RMSE | Root mean squared error | < 4.0 FP |
| Spearman ρ | Rank correlation of predicted vs actual rolling-EV | > 0.35 |
| Top-6 FP delta | Mean FP of model's top-6 vs random 6 from same slate | > +15% |
| vs. heuristic | Mean FP of model's top-6 vs heuristic's top-6 on 2024 test days | positive delta |

The `Top-6 FP delta` and `vs. heuristic` metrics are the business-meaningful success criteria from the Clete scope.

### 5.5 Script: `scripts/train_model.py`

```
Input: data/training_features_batters.pkl, data/training_features_pitchers.pkl
Output:
  models/ud_batter_lgbm_{YYYYMMDD}.pkl
  models/ud_pitcher_lgbm_{YYYYMMDD}.pkl
  models/batter_lgbm_best_params.json
  models/pitcher_lgbm_best_params.json
  models/eval_report_{YYYYMMDD}.txt  (all metrics above)
  models/feature_importance_batters.csv
  models/feature_importance_pitchers.csv
```

Symlink after training:
```bash
ln -sf ud_batter_lgbm_{YYYYMMDD}.pkl models/ud_batter_lgbm_latest.pkl
ln -sf ud_pitcher_lgbm_{YYYYMMDD}.pkl models/ud_pitcher_lgbm_latest.pkl
```

The inference module always loads `*_latest.pkl` so no config change is needed after retraining.

---

## 6. Integration Plan

### 6.1 Decision: Replace Heuristic Projections

**Decision: Replace, not augment.**

The existing `project_all_players()` in `market_underdog_draft.py` is a heuristic PA-rate model. Running two competing projection systems in parallel creates confusion about which to trust and doubles maintenance burden. The model replaces the heuristic entirely.

The `build_draft_cheat_sheet()` function, Flask route, and frontend tab are **unchanged**. Only the projection input changes.

### 6.2 New File: `model_ud_draft.py`

This module has one public function that exactly mirrors the output schema of the heuristic `project_all_players()`:

```python
# model_ud_draft.py

import pandas as pd
import lightgbm as lgb
import pickle
from datetime import date, timedelta
from pathlib import Path

MODEL_DIR = Path(__file__).parent / "models"
DATA_DIR  = Path(__file__).parent / "data"

BATTER_MODEL_PATH  = MODEL_DIR / "ud_batter_lgbm_latest.pkl"
PITCHER_MODEL_PATH = MODEL_DIR / "ud_pitcher_lgbm_latest.pkl"

def project_all_players_ml(
    lineups: dict,         # same format as data_fetcher output
    batter_df: pd.DataFrame,   # season aggregate Statcast (existing cache)
    pitcher_df: pd.DataFrame,  # season aggregate Statcast (existing cache)
) -> list[dict]:
    """
    Returns list of player projection dicts matching the existing schema:
    {name, team, opponent, position, lineup_spot, projected_fp,
     fp_breakdown, tier, key_edge}

    Uses trained LightGBM models. Falls back to heuristic if model files missing.
    """
    ...
```

**Feature engineering at inference time:**

For each player in today's lineups:
1. Pull their rolling window stats from the most recent 30 days of game logs in `data/game_logs_batters_2025.pkl`
2. Join season aggregate features from the existing batter/pitcher pkl files
3. Construct feature vector matching the training schema exactly (same column names, same order)
4. Call `model.predict(X)` → this is the projected rolling-EV, used directly as `projected_fp`

**Inference feature engineering function:**
```python
def build_inference_features_batter(mlb_id, lineup_spot, home_away, season_df, game_logs_df):
    """
    Returns pd.DataFrame with one row, all batter feature columns.
    Missing rolling data (player not in 2025 logs yet): fill with 0 for rates, NaN for averages.
    LightGBM handles NaN natively — do not impute.
    """
    ...

def build_inference_features_pitcher(mlb_id, home_away, season_df, game_logs_df):
    """Same pattern for pitchers."""
    ...
```

**Fallback:** If `BATTER_MODEL_PATH` does not exist, log a warning and import + call the original `project_all_players` from `market_underdog_draft.py`. This ensures the app never breaks if the model hasn't been trained yet.

**Tier and key_edge derivation:**
- Tier thresholds unchanged: ELITE >30, SOLID 20–30, VALUE 12–20, AVOID <12
- `key_edge`: Use top-1 SHAP feature name + value for that player (requires `shap` package); or fall back to rule-based string if SHAP is slow (e.g., "30d avg FP: {X:.1f}")

### 6.3 Modifications to `market_underdog_draft.py`

Add at top of file:
```python
try:
    from model_ud_draft import project_all_players_ml as _ml_project
    _USE_ML = True
except ImportError:
    _USE_ML = False
```

Modify `project_all_players()`:
```python
def project_all_players(lineups, batter_df, pitcher_df):
    if _USE_ML:
        return _ml_project(lineups, batter_df, pitcher_df)
    # ... existing heuristic code unchanged below ...
```

**No other changes to `market_underdog_draft.py`.** Flask route, `build_draft_cheat_sheet()`, and all other functions remain untouched.

### 6.4 Daily Retraining Cadence (Optional v1 Cron)

Not required for v1 launch. Model retrained manually by running `scripts/train_model.py`. If Jim wants automated retraining, add a cron entry (separate task).

---

## 7. Output Format — What Solomon Delivers

### 7.1 Files Produced

```
data/
  game_logs_batters_2022.pkl
  game_logs_batters_2023.pkl
  game_logs_batters_2024.pkl
  game_logs_batters_2025.pkl
  game_logs_pitchers_2022.pkl
  game_logs_pitchers_2023.pkl
  game_logs_pitchers_2024.pkl
  game_logs_pitchers_2025.pkl
  player_positions.pkl
  historical_fp_batters.pkl
  historical_fp_pitchers.pkl
  historical_optimal_rosters.pkl
  historical_fp_batters_with_flags.pkl
  historical_fp_pitchers_with_flags.pkl
  training_features_batters.pkl
  training_features_pitchers.pkl

models/
  ud_batter_lgbm_YYYYMMDD.pkl
  ud_pitcher_lgbm_YYYYMMDD.pkl
  ud_batter_lgbm_latest.pkl     (symlink)
  ud_pitcher_lgbm_latest.pkl    (symlink)
  batter_lgbm_best_params.json
  pitcher_lgbm_best_params.json
  eval_report_YYYYMMDD.txt
  feature_importance_batters.csv
  feature_importance_pitchers.csv

scripts/
  build_game_logs.py
  score_historical.py
  solve_optimal_rosters.py
  build_features.py
  train_model.py

model_ud_draft.py               (new inference module, project root)
market_underdog_draft.py        (modified — 2 lines changed, nothing broken)
```

### 7.2 CLI Commands (Solomon must verify each runs clean)

```bash
# Step 1: Pull per-game historical data (slow — ~45 min due to rate limiting)
python3 scripts/build_game_logs.py --years 2022 2023 2024 2025

# Step 2: Score all historical games
python3 scripts/score_historical.py

# Step 3: Solve optimal rosters for all historical days (parallelized)
python3 scripts/solve_optimal_rosters.py

# Step 4: Build training feature matrices
python3 scripts/build_features.py

# Step 5: Train models + evaluate
python3 scripts/train_model.py

# Step 6: Verify integration
python3 main.py dry_run
curl http://localhost:5050/api/underdog-draft
```

### 7.3 Verification Checklist (Solomon must check all before committing)

- [ ] `python3 main.py dry_run` exits 0 with no errors
- [ ] `curl http://localhost:5050/api/underdog-draft` returns `{"status": "ok", ...}` with `projected_fp` values that look ML-derived (not all suspiciously round numbers)
- [ ] `models/eval_report_YYYYMMDD.txt` exists and shows test Spearman ρ > 0.25 (if not, flag for Jim before pushing)
- [ ] Manually score 3 batter game rows from `historical_fp_batters.pkl` by hand → within ±0.5 FP of `actual_fp`
- [ ] At least one 2024 test-day optimal roster has total_fp in range 70–140 (sanity check)
- [ ] `data/build_logs_errors.txt` reviewed — failed player pulls documented

### 7.4 Git Commit

```bash
git add -A
git commit -m "feat: ML draft optimizer — game log pipeline + LightGBM projections (v10)"
git push
```

---

## 8. Dependency List

New packages required (check before starting):
```
pulp          # ILP solver
lightgbm      # ML model
optuna        # hyperparameter tuning
mlb-statsapi  # MLB Stats API (package name: mlb-statsapi, import: statsapi)
shap          # feature importance at inference (optional — skip if slow)
```

Install check:
```bash
pip show pulp lightgbm optuna mlb-statsapi
```

Install if missing:
```bash
pip install pulp lightgbm optuna mlb-statsapi shap
```

---

## 9. Known Risks and Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| MLB Stats API rate limits during bulk pull | Medium | 0.3s sleep between calls; retry on 429; build_game_logs.py is idempotent |
| Player mlbID not in API (retired, minor league) | Low | Log and skip; don't crash |
| Rolling window empty for new 2025 players | Medium | LightGBM handles NaN; inference still runs |
| ILP infeasible (e.g., all players same team on low-game day) | Low | Catch infeasible status; log and skip that day from training |
| Model file missing at inference time | Low | Explicit fallback to heuristic; app never breaks |
| IP string format from API changes | Low | `convert_ip()` function; add assertion test |

---

*End of Frank Spec. Solomon, execute against this. Every ambiguous word above is intentional precision — do not interpret loosely.*
