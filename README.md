# MLB HR Prop Model

Daily MLB home run prop model for Underdog Pick'em. Produces a morning brief with correlated HR stacks and a top-10 floor list.

## Setup

```bash
cd /Users/jimsmini/.openclaw/workspace/rbl/baseball
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configure

Edit `.env` with your API keys:

- **OWM_API_KEY** - OpenWeatherMap (free tier, 1000 calls/day is plenty)
- **TELEGRAM_BOT_TOKEN** - Create via @BotFather on Telegram
- **TELEGRAM_CHAT_ID** - Your chat/group ID
- **THE_ODDS_API_KEY** - Optional, for line shopping

## Usage

### Dry run (no Telegram, prints to console)
```bash
python main.py dry_run
```

### Morning brief (sends to Telegram)
```bash
python main.py morning_brief
```

### Follow-up (checks for new lineups, sends update if any)
```bash
python main.py followup
```

### Lineup check (silent, updates state file)
```bash
python main.py lineup_check
```

### Backtest (manual, trains weights)
```bash
python main.py backtest
```

## Cron Setup (Mac)

Add to crontab (`crontab -e`):

```cron
# Lineup check every 10 min, 9am-2pm ET
*/10 9-13 * * * cd /Users/jimsmini/.openclaw/workspace/rbl/baseball && /Users/jimsmini/.openclaw/workspace/rbl/baseball/venv/bin/python main.py lineup_check

# Morning brief 10:50am ET
50 10 * * * cd /Users/jimsmini/.openclaw/workspace/rbl/baseball && /Users/jimsmini/.openclaw/workspace/rbl/baseball/venv/bin/python main.py morning_brief

# Follow-up 1pm ET
0 13 * * * cd /Users/jimsmini/.openclaw/workspace/rbl/baseball && /Users/jimsmini/.openclaw/workspace/rbl/baseball/venv/bin/python main.py followup
```

Or use launchd for more robust scheduling on macOS.

## How It Works

1. Pulls today's schedule, confirmed lineups, and probable pitchers from MLB Stats API
2. Fetches Statcast batting/pitching data via pybaseball
3. Builds features: barrel rate, ISO, HR/FB, platoon splits, park factors, weather
4. Scores each batter-game matchup with weighted composite + sigmoid transform
5. Identifies HR-prone pitchers and builds correlated stacks
6. Sends sharp 3-leg parlay, lottery 6-leg parlay, and top-10 floor list to Telegram

## Backtest

Run `python main.py backtest` to:
- Walk-forward train on 2021-2022, validate on 2023, test on 2024
- Outputs calibrated `weights.json` (auto-loaded on next run)
- Results saved to `outputs/backtest/`
- 2025 reserved as holdout
