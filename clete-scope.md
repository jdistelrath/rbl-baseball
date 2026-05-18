# Clete Scope — MLB HR Prop Model
*Produced: April 25, 2026*

## Data Sources (all free)
- **pybaseball** (Statcast, FanGraphs) — batter/pitcher metrics
- **MLB Stats API** — official lineups, no key required
- **OpenWeatherMap** — wind, temp, humidity at game time
- **FanGraphs park factors** — HR park multipliers

## Key Predictive Variables

### Batter
- Barrel rate (30-day rolling)
- HR/FB ratio
- ISO
- Platoon split

### Pitcher
- HR/FB allowed
- Fly ball rate
- xFIP
- Hard hit rate allowed

### Context
- Park HR factor
- Wind speed/direction
- Temperature
- Batting order position

## Stacking Logic
Score every starting pitcher daily on HR-proneness → identify 2-3 "target pitchers" → rank their opposing batters → build 2-4 batter stacks per target pitcher → combine into 3-6 leg Power Plays (PrizePicks).

## Daily Outputs
1. **Sharp brief (Jim):** stack, reasoning, scores, suggested bet size
2. **Floor guy's list:** top 10 names, no explanation

## Backtesting
- 2021–2025 Statcast data, walk-forward validation
- Weather excluded from backtest initially, added live in Phase 2
- ~109k batter-game rows, runs in under 10 min

## Build Size
~1,200–1,500 lines of Python across 8 modules. All free dependencies. Runs on Mac Mini via cron, pushes to Telegram.

## Key Timing Issue
Day game lineups often unconfirmed by 11am ET → Frank to decide on brief timing strategy.
