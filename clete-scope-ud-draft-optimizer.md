# Clete Scope — UD Draft Optimizer (Historical → Forward)
*Produced: April 30, 2026*
*Clete is the Auditor. This doc is for Frank (the Architect).*

---

## 1. Problem Statement

The existing `market_underdog_draft.py` projects today's fantasy points forward using heuristics and season-average rates. It has no feedback loop — it cannot learn which features actually predicted high FP days.

**What we're building:** A training pipeline that:
1. Retroactively scores 3+ years of historical batter/pitcher game logs using UD FP rules
2. Solves the optimal 6-man roster for every historical game day (what would have won)
3. Uses those optimal rosters + player features as labeled training data to build a forward-looking draft ranking model calibrated to rolling expected value (not win rate)

The output is a **calibrated ranking model** that replaces the current heuristic scoring in `market_underdog_draft.py` with data-driven FP projections.

---

## 2. Data Inventory

### What We Have (Confirmed)

| Dataset | Rows | Fields (estimated) | Source |
|---|---|---|---|
| Batter game logs | ~66K rows | Statcast: barrel%, ISO, HR/FB, BB%, K%, BABIP, xBA, xSLG, sprint speed, lineup spot, game_date, player_id, team | pybaseball |
| Pitcher game logs | Unknown (est. 15–25K) | K/9, ERA, xFIP, IP/GS, GB%, FB%, WHIP, HR/FB allowed, game_date, player_id | pybaseball |

### Gaps — Data Frank's Pipeline Must Either Pull or Accept as Missing

| Gap | Criticality | Source |
|---|---|---|
| **Hit-type splits per game** (1B/2B/3B/HR) | **CRITICAL** — required to compute actual FP | Statcast game logs have these; confirm columns exist |
| **Stolen bases per game** | High | Statcast game logs (SB column) |
| **RBI/Runs per game** | High | Baseball Reference or MLB Stats API game logs |
| **HBP per game** | Medium | Statcast or game logs |
| **Pitcher Win/QS per game** | High | Baseball Reference game logs or retrosheet |
| **Pitcher ER per game** (for `earned_run` penalty) | **CRITICAL** | Statcast pitching logs (ER column) |
| **Position by game** (IF/OF/P classification) | High | MLB Stats API historical lineups |
| **Opposing pitcher per batter game** | Medium | Cross-join batter/pitcher by game_pk |
| **Park factor (run environment)** | Low | FanGraphs park factors (static lookup) |

**Assumption to validate:** The 66K batter-game rows contain enough per-game hit-type detail (1B, 2B, 3B, HR, BB, HBP, SB, RBI, R) to compute actual UD FP directly. If they don't, the historical scoring step requires a data pull before anything else.

---

## 3. Key Decisions for Frank

Frank must resolve these to write a complete buildable spec:

### Decision 1: Historical Scoring — Compute In-Place or Pull Supplemental Data?
The 66K rows may already contain all hit-type columns needed to compute FP directly. Or they may only contain aggregate Statcast metrics (barrel%, xSLG) that are predictive but not scoreable.
- **Path A:** Rows already have 1B/2B/3B/HR/BB/HBP/SB/RBI/R per game → score directly
- **Path B:** Rows are Statcast-only → must join with Baseball Reference or Retrosheet game logs for counting stats
Frank must inspect the actual column list and decide which path to build.

### Decision 2: Optimal Roster Solver — Greedy or Exact?
For each historical day we need the optimal 1P+2IF+2OF+1FLEX from all confirmed starters, maximizing total FP, subject to the 2-team rule. Running 3+ years of daily data means potentially 1,000+ solve instances.
- **Greedy (fast):** Same approach as current `build_draft_cheat_sheet` — pick best P, best 2 IF, best 2 OF, best remaining FLEX, fix 2-team violations. O(n), runs instantly per day but may miss optimal by 3–8%.
- **Exact (ILP):** PuLP/scipy integer linear program. True optimal, ~50–200ms per day, trivially parallelizable across years. Adds a dependency.
Frank must choose. Recommendation: ILP — the label quality directly determines model quality.

### Decision 3: Model Architecture for Forward Prediction
The labeled dataset (historical player features → actual FP earned, plus did-they-appear-in-optimal-roster flag) trains the forward model. Two reasonable options:
- **Gradient Boosted Trees (XGBoost/LightGBM):** Fast, interpretable, handles tabular data well, easy to retrain. Predicts FP per player per day. Standard choice for DFS ML.
- **Ensemble + Calibration:** GBM for raw FP prediction, separate calibrated probability that player appears in top-N that day. More complex but closer to rolling-EV objective.
Frank must decide the model type and whether the output is a raw FP regression or a probability-weighted EV estimate.

### Decision 4: Rolling-EV Training Objective — Window and Weighting
Jim specified rolling-EV as the optimization objective (not win rate, not pure hindsight FP). Frank must define:
- Rolling window size (e.g., 30-day, 60-day, or season-weighted)
- How EV is computed from historical optimal rosters (is EV = avg FP of drafted player over next N days? Is it regret minimization vs. random drafter?)
- Whether the model trains on EV labels directly or FP labels with a rolling weight scheme

### Decision 5: Integration — Replace or Augment Existing Module?
The trained model must power daily recommendations. Two paths:
- **Replace:** New `model_underdog_draft.py` replaces the heuristic FP projections in `market_underdog_draft.py`; Flask tab + API endpoint unchanged
- **Augment:** Add a `model_confidence` score alongside the current heuristic FP, surfaced as a secondary signal in the UI
Frank must decide which to build. Recommendation: Replace — augmenting two competing projection systems creates confusion.

---

## 4. Out of Scope (v1)

- **Pick'em / Power Play / PrizePicks** — draft only (Jim confirmed)
- **Multi-day draft optimization** — scope is single-day slate only
- **Live in-game updates** — projections set before first pitch, not updated mid-game
- **Player injury/availability modeling** — confirmed starters only, same as current system
- **Team strength / opponent quality context** — may be a v2 feature; exclude from v1 feature set
- **Neural network architectures** — GBM is sufficient; deep learning adds complexity without clear gain on tabular DFS data
- **Model serving infrastructure / API** — runs locally on Mac Mini via cron, same as existing system
- **Salary / ADP data** — UD Draft is snake, not salary-cap; irrelevant

---

## 5. Success Criteria

The system works when all three layers pass:

**Layer 1 — Historical Scoring Accuracy**
- Verify: manually score 20 random player-game rows by hand using UD rules → computed FP matches within ±0.5 pts
- Verify: daily optimal rosters produce FP totals that are plausible (est. 80–130 pts for a strong day)

**Layer 2 — Model Predictive Validity**
- Walk-forward validation: train on 2022–2023, test on 2024
- Success: model's projected top-6 players for a given day, when scored by actual FP, outperform a random 6-man roster drawn from the same slate by >15% on average FP
- Baseline to beat: current heuristic `market_underdog_draft.py` optimal lineup FP on 2024 test set

**Layer 3 — Production Integration**
- `curl http://localhost:5050/api/underdog-draft` returns model-powered projections (not heuristic) without error
- `python3 main.py dry_run` still passes
- Daily morning brief fires at 10:50am ET with updated projections

---

## 6. Open Questions (Jim Must Answer)

Only two — everything else is Frank's call:

1. **Column inventory:** Can you run `python3 -c "import pandas as pd; df = pd.read_csv('<your_batter_file>'); print(df.columns.tolist()); print(len(df))"` and paste the column list? This determines whether historical scoring is a join operation or a direct compute — it's the biggest architectural fork in the road.

2. **File locations:** Where do the 66K batter rows and pitcher data actually live on disk? (`/Users/jimsmini/Projects/baseball/data/`? Confirm path and filename pattern.) Frank needs this to spec the data loader.

That's it. Everything else — model type, solver choice, integration approach, feature set — Clete and Frank own.

---

## Appendix: UD Scoring Reference (for Frank's implementation)

```python
# Hitters (per event)
HITTER_FP = {"single": 3, "double": 6, "triple": 8, "home_run": 10,
             "walk": 3, "hbp": 3, "rbi": 2, "run": 2, "stolen_base": 4}

# Pitchers (per event)
PITCHER_FP = {"win": 2, "quality_start": 3, "strikeout": 1,
              "inning_pitched": 1, "earned_run": -1}
# QS = 6+ IP and ≤3 ER

# Roster slots
ROSTER = {"P": 1, "IF": 2, "OF": 2, "FLEX": 1}  # FLEX = IF or OF
# Constraint: players from ≥2 different teams
```
