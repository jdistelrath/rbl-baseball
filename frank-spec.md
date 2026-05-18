# Frank Architecture Spec — MLB HR Prop Model
*Produced: April 25, 2026*

## Platform
**PrizePicks** — Power Play format
- 3-leg = 5x | 4-leg = 10x | 5-leg = 20x | 6-leg = 40x
- No correlation restrictions (stack same team freely)
- HR prop: 0.5 HRs over/under (binary: homered or not)
- $1/bet standard; $2-5 on sharp 3-leg

## Day Game Lineup Problem — Resolution
- Morning brief fires at **10:50am ET** with all confirmed games
- Day games with unconfirmed lineups flagged ⚠️ PENDING (not held)
- Second pass at **1:00pm ET** — follow-up only if new lineups came in
- State file prevents duplicates

## 8 Modules

| Module | Responsibility |
|---|---|
| `data_fetcher.py` | All network I/O — Statcast cache, MLB API, OWM weather |
| `feature_builder.py` | Rolling metrics, missing data handling |
| `scorer.py` | Composite formula, z-score normalization, confidence tiers |
| `stack_builder.py` | Target pitcher ID, batter stacking, Power Play selection |
| `backtest.py` | Walk-forward harness, weight calibration (manual, not cron) |
| `messenger.py` | Telegram formatting and delivery |
| `config.py` | Loads .env + config.yaml, auto-loads calibrated weights |
| `main.py` | Orchestrator — modes: morning_brief, followup, lineup_check, backtest, dry_run |

## Scoring Formula
- Z-score normalized across all features → weighted sum
- Starting weights equal across factors; calibrated via backtesting
- Missing weather → 0 bonus (not penalized)
- Platoon disadvantage = penalty only (neutral/favorable matchups unaffected)

## Cron Schedule
- Every 10 min, 9am–2pm ET: lineup check
- 10:50am ET: morning brief
- 1:00pm ET: follow-up check (silent if nothing new)

## Credentials Required
- **OpenWeatherMap API key** — Jim has this, needs storing on Mini
- No other paid APIs

## Target Directory
`workspace/rbl/baseball/`

## Build Method
Claude Code (ACP, thread-bound persistent session) — Solomon persona
Estimated build time: 30–45 min
