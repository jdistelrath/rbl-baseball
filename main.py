"""
Module 8: Main orchestrator.
All modes invokable via CLI arg.

Usage:
    python main.py morning_brief
    python main.py followup
    python main.py lineup_check
    python main.py backtest
    python main.py dry_run
"""

import json
import sys
from datetime import date, datetime

from config import CFG
import data_fetcher
import feature_builder
import scorer
import stack_builder
import messenger
import market_totals
import market_f5
import market_strikeouts
import market_hr_ev
import market_k_ev


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _state_path():
    return CFG.state_dir / f"daily_state_{date.today().isoformat()}.json"


def _load_state():
    p = _state_path()
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return {"sent_game_ids": [], "lineup_status": {}, "date": date.today().isoformat()}


def _save_state(state):
    with open(_state_path(), "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Pipeline core
# ---------------------------------------------------------------------------

def _run_pipeline(games, dry_run=False):
    """
    Run the full pipeline for a set of games.
    Returns (parcels, confirmed_count, pending_count).
    """
    if not games:
        print("[main] No games found for today.")
        return None, 0, 0

    print(f"[main] Processing {len(games)} games...")

    # Load statcast data (cached daily)
    batter_df = data_fetcher.get_batter_statcast()
    pitcher_df = data_fetcher.get_pitcher_statcast()

    all_features = []
    confirmed_count = 0
    pending_count = 0
    confirmed_game_ids = []

    for game in games:
        game_id = game["game_id"]
        print(f"  Game {game_id}: {game['away_team']} @ {game['home_team']}")

        # Check lineups
        lineup = data_fetcher.get_confirmed_lineups(game_id)
        if lineup is None:
            print(f"    Lineups not yet posted. Skipping.")
            pending_count += 1
            continue

        confirmed_count += 1
        confirmed_game_ids.append(game_id)

        # Get weather
        weather = data_fetcher.get_weather(game["stadium"], game.get("start_time_utc"))

        # Build features
        game_features = feature_builder.build_features_for_game(
            game, lineup, batter_df, pitcher_df, weather
        )
        all_features.extend(game_features)
        print(f"    {len(game_features)} batter matchups built.")

    if not all_features:
        print("[main] No confirmed lineups with features available.")
        return None, confirmed_count, pending_count

    # Score all batters
    print(f"[main] Scoring {len(all_features)} batter matchups...")
    scored = scorer.score_batters(all_features)

    # Build stacks
    parcels = stack_builder.build_stacks(scored)
    parcels["_confirmed_game_ids"] = confirmed_game_ids
    parcels["_scored_batters"] = scored

    print(f"[main] Sharp parlay: {len(parcels['sharp_parlay'])} legs")
    print(f"[main] Lottery parlay: {len(parcels['lottery_parlay'])} legs")
    print(f"[main] Floor list: {len(parcels['floor_list'])} players")

    return parcels, confirmed_count, pending_count


# ---------------------------------------------------------------------------
# Multi-market EV pipeline
# ---------------------------------------------------------------------------

def _run_ev_pipeline(games, dry_run=False):
    """
    Run EV analysis across totals, F5, and strikeout markets.
    Returns top 3 plays sorted by ev_per_dollar descending.
    Never crashes -- degrades gracefully if odds or data are missing.
    """
    print("[main] === EV Pipeline ===")

    # Fetch odds
    odds_data = data_fetcher.get_odds_for_markets()
    if not odds_data:
        print("[main] No odds data available. EV pipeline will run without book lines.")

    # Load stats (cached, same as HR pipeline)
    batter_df = data_fetcher.get_batter_statcast()
    pitcher_df = data_fetcher.get_pitcher_statcast()

    all_ev_plays = []

    for game in games:
        game_id = game["game_id"]
        stadium = game.get("stadium", "")
        weather = data_fetcher.get_weather(stadium, game.get("start_time_utc"))

        # --- Market 1: Team Totals ---
        try:
            total_result = market_totals.score_game_total(
                game, batter_df, pitcher_df, weather, odds_data
            )
            if total_result and total_result.get("ev_per_dollar", 0) > 0:
                all_ev_plays.append(total_result)
        except Exception as e:
            print(f"  [totals] Error scoring {game_id}: {e}")

        # --- Market 2: First 5 Innings ---
        try:
            f5_result = market_f5.score_f5(
                game, pitcher_df, batter_df, weather, odds_data
            )
            if f5_result:
                all_ev_plays.append(f5_result)
        except Exception as e:
            print(f"  [f5] Error scoring {game_id}: {e}")

        # --- Market 3: Strikeout Props (both pitchers) ---
        for side in ("home", "away"):
            try:
                k_result = market_strikeouts.score_strikeout_prop(
                    game, side, pitcher_df, batter_df, weather, odds_data
                )
                if k_result:
                    all_ev_plays.append(k_result)
            except Exception as e:
                pitcher_key = f"{side}_pitcher_name"
                print(f"  [strikeouts] Error scoring {game.get(pitcher_key, '?')}: {e}")

    # Sort by EV descending, take top 3
    all_ev_plays.sort(key=lambda x: x.get("ev_per_dollar", 0), reverse=True)
    top3 = all_ev_plays[:3]

    print(f"[main] EV plays found: {len(all_ev_plays)} total, top 3 selected")
    for i, play in enumerate(top3, 1):
        print(f"  #{i} {play.get('market')}: {play.get('description')} "
              f"(EV: {play.get('ev_per_dollar', 0):.1%})")

    return top3


# ---------------------------------------------------------------------------
# DK Prop EV pipeline
# ---------------------------------------------------------------------------

def _run_prop_ev_pipeline(games, parcels, dry_run=False):
    """
    Pull DK player prop lines, compare to model, surface +EV bets.
    """
    print("[main] === DK Prop EV Pipeline ===")

    scored_batters = parcels.get("_scored_batters", []) if parcels else []

    # Pull prop lines
    hr_prop_lines = data_fetcher.get_player_props("batter_home_runs")
    k_prop_lines = data_fetcher.get_player_props("pitcher_strikeouts")

    # HR prop EV
    hr_ev_plays = []
    if scored_batters and hr_prop_lines:
        hr_ev_plays = market_hr_ev.score_hr_props(scored_batters, hr_prop_lines)
        print(f"[main] HR +EV plays: {len(hr_ev_plays)}")
    else:
        print("[main] No HR prop lines or scored batters available.")

    # K prop EV
    k_ev_plays = []
    if games and k_prop_lines:
        batter_df = data_fetcher.get_batter_statcast()
        pitcher_df = data_fetcher.get_pitcher_statcast()
        weather_map = {}
        for g in games:
            stadium = g.get("stadium", "")
            if stadium and stadium not in weather_map:
                weather_map[stadium] = data_fetcher.get_weather(stadium, None)
        k_ev_plays = market_k_ev.score_k_props(
            games, pitcher_df, batter_df, weather_map, k_prop_lines
        )
        print(f"[main] K +EV plays: {len(k_ev_plays)}")
    else:
        print("[main] No K prop lines available.")

    # Send/print
    if hr_ev_plays or k_ev_plays:
        messenger.send_ev_props(hr_ev_plays[:5], k_ev_plays[:3], dry_run=dry_run)

    return hr_ev_plays, k_ev_plays


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_morning_brief(dry_run=False):
    """Morning brief: full pipeline + send."""
    print("[main] === Morning Brief ===")
    games = data_fetcher.get_today_schedule()
    parcels, confirmed, pending = _run_pipeline(games, dry_run)

    if parcels is None:
        msg = "No confirmed lineups available yet. Will retry at follow-up."
        if dry_run:
            print(msg)
        else:
            messenger.send_error(msg)
        # Still run EV pipeline (doesn't require confirmed lineups)
        try:
            top_ev = _run_ev_pipeline(games, dry_run=dry_run)
            if top_ev:
                messenger.send_top_ev_plays(top_ev, dry_run=dry_run)
        except Exception as e:
            print(f"[main] EV pipeline error (non-fatal): {e}")
        # Save state with pending games
        state = _load_state()
        state["lineup_status"] = {
            str(g["game_id"]): "pending" for g in games
        }
        _save_state(state)
        return

    # Send HR brief
    messenger.send_brief(parcels, pending_count=pending, dry_run=dry_run)

    # Run multi-market EV pipeline (all games, not just confirmed lineups)
    try:
        top_ev = _run_ev_pipeline(games, dry_run=dry_run)
        if top_ev:
            messenger.send_top_ev_plays(top_ev, dry_run=dry_run)
    except Exception as e:
        print(f"[main] EV pipeline error (non-fatal): {e}")

    # Run DK prop EV pipeline (HR + K props vs book lines)
    try:
        _run_prop_ev_pipeline(games, parcels, dry_run=dry_run)
    except Exception as e:
        print(f"[main] Prop EV pipeline error (non-fatal): {e}")

    # Update state
    state = _load_state()
    for gid in parcels.get("_confirmed_game_ids", []):
        state["sent_game_ids"].append(gid)
        state["lineup_status"][str(gid)] = "sent"
    # Mark pending games
    for g in games:
        gid_str = str(g["game_id"])
        if gid_str not in state["lineup_status"]:
            state["lineup_status"][gid_str] = "pending"
    _save_state(state)
    print("[main] State saved.")


def mode_followup(dry_run=False):
    """Follow-up: check for new lineups on previously-pending games."""
    print("[main] === Follow-up ===")
    state = _load_state()

    # Find pending games
    pending_ids = [
        int(gid) for gid, status in state.get("lineup_status", {}).items()
        if status == "pending"
    ]

    if not pending_ids:
        print("[main] No pending games to check.")
        return

    print(f"[main] Checking {len(pending_ids)} pending games...")

    # Re-fetch schedule to get game info
    games = data_fetcher.get_today_schedule()
    pending_games = [g for g in games if g["game_id"] in pending_ids]

    if not pending_games:
        print("[main] Could not find pending games in schedule.")
        return

    parcels, confirmed, pending = _run_pipeline(pending_games, dry_run)

    if parcels is None:
        print("[main] Still no new confirmed lineups.")
        return

    # Send follow-up
    messenger.send_followup(parcels, dry_run=dry_run)

    # Update state
    for gid in parcels.get("_confirmed_game_ids", []):
        state["sent_game_ids"].append(gid)
        state["lineup_status"][str(gid)] = "sent"
    _save_state(state)
    print("[main] State updated.")


def mode_lineup_check(dry_run=False):
    """
    Silent lineup check. Updates state file.
    If new lineups found between 9am-2pm, triggers follow-up internally.
    """
    print("[main] === Lineup Check ===")
    state = _load_state()

    pending_ids = [
        int(gid) for gid, status in state.get("lineup_status", {}).items()
        if status == "pending"
    ]

    if not pending_ids:
        print("[main] No pending games.")
        return

    # Check each pending game for lineups
    newly_confirmed = []
    for gid in pending_ids:
        lineup = data_fetcher.get_confirmed_lineups(gid)
        if lineup is not None:
            newly_confirmed.append(gid)
            state["lineup_status"][str(gid)] = "confirmed"

    if not newly_confirmed:
        print("[main] No new lineups confirmed.")
        _save_state(state)
        return

    print(f"[main] {len(newly_confirmed)} new lineups confirmed!")
    _save_state(state)

    # Check if we're in the follow-up window
    now = datetime.now()
    hour = now.hour
    start_h = int(CFG.lineup_window_start.split(":")[0])
    end_h = int(CFG.lineup_window_end.split(":")[0])

    if start_h <= hour <= end_h:
        print("[main] Within follow-up window. Triggering follow-up...")
        mode_followup(dry_run=dry_run)
    else:
        print("[main] Outside follow-up window. Lineups noted for next run.")


def mode_backtest():
    """Run backtesting harness."""
    print("[main] === Backtest ===")
    import backtest
    backtest.run_backtest()


def mode_backtest_gamelevel():
    """Run game-level backtesting via Statcast data."""
    print("[main] === Game-Level Backtest ===")
    from backtest import run_game_level_backtest
    run_game_level_backtest(years=[2024, 2025])


def mode_backtest_totals():
    """Run totals model backtesting and calibration."""
    print("[main] === Totals Outcome Backtest ===")
    from backtest import run_totals_backtest
    run_totals_backtest(years=[2021, 2022, 2023, 2024, 2025])


def mode_dry_run():
    """Full pipeline, no Telegram send, print to console."""
    print("[main] === Dry Run ===")
    mode_morning_brief(dry_run=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

MODES = {
    "morning_brief": mode_morning_brief,
    "followup": mode_followup,
    "lineup_check": mode_lineup_check,
    "backtest": mode_backtest,
    "backtest_gamelevel": mode_backtest_gamelevel,
    "backtest_totals": mode_backtest_totals,
    "dry_run": mode_dry_run,
}


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python main.py <{'|'.join(MODES.keys())}>")
        sys.exit(1)

    mode = sys.argv[1]
    if mode not in MODES:
        print(f"Unknown mode: {mode}")
        print(f"Available: {', '.join(MODES.keys())}")
        sys.exit(1)

    try:
        MODES[mode]()
    except Exception as e:
        print(f"[main] Fatal error: {e}")
        import traceback
        traceback.print_exc()
        try:
            messenger.send_error(f"Pipeline failed in {mode}: {e}")
        except Exception:
            pass
        sys.exit(1)


if __name__ == "__main__":
    main()
