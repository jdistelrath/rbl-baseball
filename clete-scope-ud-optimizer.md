# Clete Scope — UD Power Play Optimizer
*Produced: April 30, 2026*
*Auditor: Clete | Next: Frank (Architecture Spec)*

---

## 1. Problem Statement

We have 4 seasons of Statcast batter-game data (~66K rows, 2022–2025). We want to:

1. **Label** every historical batter-game with binary outcomes for each UD prop threshold (2+ hits, 3+ TB, 1+ HR, 5+ K pitcher, 7+ K pitcher).
2. **Reconstruct** what the optimal UD Power Play entry would have been on each historical game day — the combination of legs that maximized expected ROI given perfect hindsight.
3. **Train a forward-looking algorithm** on the patterns that distinguish optimal from suboptimal picks, so we can generate daily Power Play recommendations with positive expected value.

The output is a ranked daily slate: which prop, which player, which multiplier tier (2–5 pick), and a confidence score calibrated to UD's fixed breakeven thresholds.

---

## 2. Data Inventory

### What We Have
| Asset | Description | Status |
|---|---|---|
| Statcast batter-game rows | ~66K rows, 2022–2025, game-level stats per batter | ✅ Available |
| Existing feature pipeline | `feature_builder.py`, `scorer.py` — rolling Statcast metrics | ✅ Built |
| Existing data fetcher | `data_fetcher.py` — pybaseball + MLB Stats API | ✅ Built |
| HR model (PrizePicks) | Current production scorer for HR props | ✅ Validated |
| Park factors | FanGraphs HR park multipliers | ✅ In cache |

### What We Need to Derive
| Asset | How to Get It | Gap |
|---|---|---|
| Per-game hits (H) | Statcast batter-game already has H | ✅ Likely present |
| Per-game total bases (TB) | Statcast: 1B + 2×2B + 3×3B + 4×HR per game | ✅ Derivable |
| Per-game HR (binary) | Statcast: HR > 0 | ✅ Present |
| Per-game pitcher K (SP only) | Statcast pitcher-game rows, SP filter | ✅ Need to verify SP isolation |
| UD prop labels (binary) | Apply thresholds to each row | 🔧 Build |
| Historical "optimal lineup" per day | Retroactive: which pick set had highest ROI | 🔧 Build |

### Data Gaps to Confirm
- **Pitcher K data:** Do we have pitcher-game rows (not just batter-game) for 2022–2025 in the 66K dataset, or is it batter-only? Pitcher props (4.5K, 6.5K) require SP-level game logs, not batter rows.
- **Column completeness:** Confirm H, 2B, 3B, HR columns exist per row (not just aggregated season stats). If Statcast is game-level, this should be present.
- **Lineup position:** Do rows include batting order spot? Needed for plate appearance projection.

---

## 3. Key Decisions for Frank

### Decision 1: How Do We Define "Optimal" for Retroactive Labeling?
The retroactive optimizer needs a single objective. Three candidates:

- **Option A — Maximize ROI per day:** Find the 2–5 pick combo with highest realized payout given actual outcomes. Pure hindsight. Best for training signal but can't account for correlation on unplayed dates.
- **Option B — Maximize hit rate at breakeven:** Find pick combos where all legs hit, weighted by how far above breakeven each individual leg's historical rate sits. Better for calibration.
- **Option C — Maximize expected value using empirical hit rates:** For each player-prop, compute their rolling historical hit rate → calculate EV across all possible 2–5 pick combos using multiplier structure → select highest-EV combo as "optimal."

**Recommendation:** Option C. It treats retroactive labeling as a simulation of what a sharp bettor *would have* done with available-at-the-time data, which makes it the cleanest training signal for the forward model. Frank should confirm.

### Decision 2: Correlation Handling
UD Power Plays require all legs to hit (parlay structure). Correlation matters:

- **Positive correlation (stack same team):** Batters on the same team share game environment — weather, opposing pitcher, run environment. More hits → more TB → correlated upside.
- **Negative correlation (pitcher vs batter same game):** If you pick a pitcher's K prop AND a batter facing that pitcher's TB prop, they're inversely correlated (K = out, TB = hit).
- **Cross-prop same player:** Hits 1.5 and TB 2.5 for the same batter are *highly* correlated (a 2B clears both). Is this allowed on UD? Does it count as two legs?

Frank needs to decide: (a) whether to model leg correlations explicitly in the optimizer, or (b) apply simple rules (no pitcher-batter cross picks, allow same-team stacks). V1 recommendation: rules-based, no explicit correlation model.

### Decision 3: Pitcher Props vs. Batter Props — Unified or Separate Models?
Pitcher K props (4.5K / 6.5K) are fundamentally different from batter props:

- Different data source (pitcher-game logs vs batter-game logs)
- Different feature set (K/9, opponent K rate, IP, pitch mix)
- Different variance profile (starter K outcomes have higher game-to-game variance)
- Different breakeven sensitivity at 4.5K vs 6.5K

Options: (a) Single unified scorer that handles both prop types with position flag, or (b) separate models, each feeding a shared lineup optimizer. V1 recommendation: separate feature engineering, shared optimizer layer. Frank to decide if that means separate Python modules or a single parameterized model class.

### Decision 4: Bankroll and Bet Sizing Model
The optimizer needs to know how to size recommendations:

- **Flat betting:** Same dollar amount every entry regardless of confidence. Simple. Testable.
- **Kelly fraction:** Size proportional to edge over breakeven. Mathematically optimal but requires calibrated probabilities.
- **Tiered:** High confidence → 5-pick; medium → 3-pick; low → 2-pick. Implicit sizing via multiplier selection.

For v1 training purposes, we need to pick a bet-sizing assumption to calculate historical P&L. Flat betting is the right default for initial calibration — avoids the compounding complexity of Kelly until the probability model is validated.

### Decision 5: Optimization Scope Per Day — All Combos or Top-N Pre-Filter?
On a full slate (~15 games, ~180 batter starters, ~30 SP), exhaustive 5-pick combination search is computationally large (C(180,5) ≈ 1.5 billion). Need a pre-filter strategy:

- Score all player-props individually → take top N (e.g., top 30 by confidence) → optimize combos within that set
- Or: cluster by game/team → pick best-per-cluster → combine

Frank needs to define the pre-filter heuristic and the maximum combo depth the retroactive search will evaluate.

---

## 4. Out of Scope — V1

- **Live odds / market signals:** UD has no lines. We're not scraping odds or comparing to books.
- **Bullpen/relievers:** Pitcher props are starter-only. Reliever K props are not modeled.
- **Stolen base props:** Not a UD Power Play prop type in current scope.
- **RBI / runs props:** Not included in the 5 defined UD thresholds.
- **Bankroll management system:** V1 outputs picks and confidence. No auto-bet, no bankroll tracking, no P&L ledger.
- **Multi-entry optimization:** UD allows multiple entries per slate. V1 produces one optimal entry per day.
- **Real-time lineup substitutions:** If a player is scratched after picks are locked, that's a manual adjustment.
- **Calibration to UD's actual pick universe:** We don't know which players UD offers as options on a given day. V1 assumes any confirmed starter is a valid pick candidate.

---

## 5. Success Criteria

### Retroactive (Training Signal Quality)
- Retroactive labeler correctly classifies binary outcomes for all 5 prop types across all 66K batter-game rows with <1% error rate (spot-check validated).
- Retroactive "optimal" lineup for each historical game day is identifiable and produces a positive simulated ROI over the full 2022–2025 dataset at flat betting.
- Retroactive P&L distribution is plausible: not dominated by lucky 5-pick days; meaningful hit rate at 3-pick level.

### Forward Model (Algorithm Performance)
- **Calibration:** Model's predicted hit rate per leg is within 3pp of actual hit rate in holdout (2025 season as validation set).
- **ROI:** Simulated forward picks (trained on 2022–2024, validated on 2025) produce positive ROI at flat 3-pick entries over a full season (162-game slate window).
- **Breakeven test:** At least 58.5% of individual legs predicted as "high confidence" actually hit in holdout data (3-pick breakeven threshold).
- **Sharpe-equivalent:** Risk-adjusted return (mean daily ROI / std dev) is positive and stable — not driven by one hot streak.

### System Quality
- Daily picks delivered to Telegram before first pitch, with prop type, player, confidence tier, and pick count recommendation.
- Reruns cleanly on Mac Mini M4 in under 5 minutes.

---

## 6. Open Questions (Jim's Input Required Before Frank Specs)

**Q1: Pitcher K data availability**
Do the 66K batter-game rows include pitcher game logs, or is it batter-only? If pitcher K data isn't in the current dataset, we need to pull it separately via pybaseball `statcast_pitcher_game_logs()`. Confirm what's in the existing data before Frank specs the data layer.

**Q2: Cross-prop same player — is it allowed on UD?**
Can you pick Hits 1.5 AND TB 2.5 for the same player as two separate legs in one Power Play? If yes, it's the single highest-correlation pair in the model (they share almost all outcomes). If no, we rule it out in the combo generator. Need a definitive answer before Frank specs the combo logic.

**Q3: "Optimal" definition preference**
Frank will need to implement one retroactive optimization objective. Options are in Decision 1 above. Jim to confirm: do you want the optimizer to find what *actually would have won* (pure hindsight ROI), or what *should have been picked* given available-at-the-time data (rolling-EV forward simulation)?

**Q4: Validation year**
Should 2025 be the holdout set (train on 2022–2024, test on 2025), or do we use a rolling walk-forward approach across all 4 years? Walk-forward is more statistically honest but more complex to implement. Jim's call on acceptable build complexity.

**Q5: Existing model reuse**
The current HR scorer in `scorer.py` is validated for PrizePicks HR props. Should Frank reuse and extend it (adding Hits/TB/K prop types as additional feature sets), or build a separate module from scratch? Reuse is faster; scratch is cleaner. This depends on how different the feature sets turn out to be.

---

*End of Clete Scope. Frank, all five decisions above plus the five open questions define the full architecture surface. The open questions need Jim's answers before you can finalize the data layer and combo optimizer specs.*
