# Solomon Prompt v12 — Filter Started Games from Draft Board & Picks

You are **Solomon**, the Builder. Read every file you touch before modifying it. Do not break anything working.

## Working Directory
`/Users/jimsmini/Projects/baseball/`

## Verify First
Run `python3 main.py dry_run` before starting. Confirm it exits cleanly.

## Problem
The morning brief and draft board include players from games that have already started. These players are no longer available for prop bets or Underdog drafts. They must be filtered out.

## What to Build

### 1. `data_fetcher.py` — Add `is_game_started(game)` helper

```python
def is_game_started(game):
    """
    Returns True if the game's start time has already passed (game is live or complete).
    Uses game['start_time_utc'] (ISO 8601 UTC string from MLB Stats API).
    Adds a 5-minute grace period buffer (game is considered started 5 min after scheduled start).
    Returns False if start_time_utc is missing or unparseable.
    """
    from datetime import datetime, timezone, timedelta
    start_str = game.get("start_time_utc", "")
    if not start_str:
        return False
    try:
        # MLB API returns ISO format like "2026-04-29T17:10:00Z"
        start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return now > (start_dt + timedelta(minutes=5))
    except Exception:
        return False


def filter_active_games(games):
    """
    Returns only games that haven't started yet.
    Logs how many games were filtered out.
    """
    active = [g for g in games if not is_game_started(g)]
    filtered = len(games) - len(active)
    if filtered > 0:
        print(f"[data_fetcher] Filtered {filtered} started game(s) — {len(active)} active remaining")
    return active
```

### 2. `main.py` — Apply filter after schedule fetch

Find where `get_today_schedule()` is called in the morning brief / picks flow. After that call, add:

```python
schedule = data_fetcher.filter_active_games(schedule)
```

Apply this filter in ALL modes that generate picks or draft projections:
- `morning_brief`
- `followup`
- `dry_run`
- Any Flask route that calls `get_today_schedule()`

Do NOT apply it in `backtest` mode (backtest needs full schedule).

### 3. `app.py` — Apply filter in all API routes

In every Flask route that calls `data_fetcher.get_today_schedule()`, add the filter immediately after:

```python
schedule = data_fetcher.get_today_schedule()
schedule = data_fetcher.filter_active_games(schedule)  # ADD THIS
```

Routes to update: `/api/picks`, `/api/hr-brief` (if it exists), `/api/underdog-draft`, and any other route that calls get_today_schedule.

### 4. `messenger.py` — Add note to brief when games are filtered

If any games were filtered, include a line in the Telegram message:

```
⏱️ {N} game(s) already started — excluded from picks
```

Place this near the top of the brief, after the date/header line.

To implement: have `main.py` pass the count of filtered games to the messenger, or compute it by comparing pre/post filter lengths.

---

## Verification

1. `python3 main.py dry_run` exits cleanly
2. If run after some games have started today, the dry_run output should show fewer players and log the filtered count
3. `curl http://localhost:5050/api/picks` returns only players from games that haven't started
4. All existing tabs still load

## Commit

```bash
git add -A
git commit -m "feat: v12 — filter started games from picks and draft board"
git push
```

## Notes
- `python3` not `python`
- No venv — user-level packages
- Do not touch `.env`
- The filter must be time-aware in real time — not cached. Do not cache the filtered result; always recompute at call time
- If ALL games have started (e.g. late night run), the brief should say "No active games today" rather than returning an empty list with no explanation
