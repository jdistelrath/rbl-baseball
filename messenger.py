"""
Module 7: Messenger.
Formats and sends Telegram messages. No other module sends Telegram messages directly.
"""

from datetime import date, datetime

import sys

import requests

from config import CFG


def _safe_print(text):
    """Print text, replacing unencodable characters on Windows consoles."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
            sys.stdout.encoding or "utf-8", errors="replace"))


def _send_telegram(text):
    """Send a message via Telegram Bot API. Returns True on success."""
    if not CFG.telegram_bot_token or CFG.telegram_bot_token == "your_bot_token_here":
        print("[messenger] Telegram not configured, skipping send.")
        return False
    if not CFG.telegram_chat_id or CFG.telegram_chat_id == "your_chat_id_here":
        print("[messenger] Telegram chat ID not configured, skipping send.")
        return False

    url = f"https://api.telegram.org/bot{CFG.telegram_bot_token}/sendMessage"
    payload = {
        "chat_id": CFG.telegram_chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[messenger] Telegram send failed: {e}")
        return False


def _format_batter_line(b):
    """Format a single batter line for the brief."""
    return f"  \u2022 {b['name']} \u2014 {b['key_edge']}"


def send_brief(parcels, pending_count=0, started_count=0, dry_run=False):
    """
    Send the morning brief to Telegram.

    Args:
        parcels: dict from stack_builder.build_stacks()
        pending_count: number of games with unconfirmed lineups
        started_count: number of games already in progress (excluded from picks)
        dry_run: if True, print to console instead of sending
    """
    today = date.today().strftime("%B %d, %Y")

    sharp = parcels.get("sharp_parlay", [])
    lottery = parcels.get("lottery_parlay", [])
    floor = parcels.get("floor_list", [])

    lines = [f"\u26be MLB HR Model \u2014 {today}"]
    if started_count > 0:
        lines.append(f"\u23f1\ufe0f {started_count} game(s) already started \u2014 excluded from picks")
    lines.append("")

    # Sharp play
    if sharp:
        opp_pitcher = sharp[0].get("opponent_pitcher", "TBD")
        team = sharp[0].get("team", "")
        total_bet = sum(b.get("bet_amount", 5) for b in sharp[:1])  # bet on the parlay, not per leg
        bet_amount = sharp[0].get("bet_amount", 10)

        lines.append(f"\ud83c\udfaf SHARP PLAY (3-leg, ${bet_amount})")
        lines.append(f"Stack: {team} batters vs. {opp_pitcher}")
        for b in sharp:
            lines.append(_format_batter_line(b))
        lines.append("Underdog 3-leg Pick'em = 3x payout")
        lines.append("")

    # Lottery stack
    if lottery:
        lines.append(f"\ud83d\udcca LOTTERY STACK (6-leg, ${CFG.bet_lottery})")
        names = ", ".join(b["name"] for b in lottery)
        lines.append(names)
        lines.append("")

    # Floor guy list
    if floor:
        lines.append("\ud83c\udfe2 TODAY'S TOP 10 (floor picks)")
        for i, b in enumerate(floor, 1):
            lines.append(f"{i}. {b['name']} ({b['team']})")
        lines.append("")

    # Pending games
    if pending_count > 0:
        lines.append(f"\u26a0\ufe0f PENDING: {pending_count} day games \u2014 follow-up at 1pm if lineups confirm")

    text = "\n".join(lines)

    if dry_run:
        _safe_print("=" * 60)
        _safe_print("DRY RUN - Morning Brief")
        _safe_print("=" * 60)
        _safe_print(text)
        _safe_print("=" * 60)
        return True

    return _send_telegram(text)


def send_followup(new_parcels, dry_run=False):
    """
    Send follow-up message for newly confirmed lineups.
    Only sends if there are new picks.

    Args:
        new_parcels: dict from stack_builder.build_stacks() (new games only)
        dry_run: if True, print to console
    """
    sharp = new_parcels.get("sharp_parlay", [])
    floor = new_parcels.get("floor_list", [])

    if not sharp and not floor:
        return False

    now = datetime.now().strftime("%I:%M %p ET")
    lines = [f"\u26be HR Model Update \u2014 {now}", ""]
    lines.append("New lineups confirmed:")
    lines.append("")

    if sharp:
        opp_pitcher = sharp[0].get("opponent_pitcher", "TBD")
        team = sharp[0].get("team", "")
        lines.append(f"\ud83c\udfaf NEW STACK: {team} vs. {opp_pitcher}")
        for b in sharp:
            lines.append(_format_batter_line(b))
        lines.append("")

    if floor:
        lines.append("Updated floor picks:")
        for i, b in enumerate(floor, 1):
            lines.append(f"{i}. {b['name']} ({b['team']})")

    text = "\n".join(lines)

    if dry_run:
        _safe_print("=" * 60)
        _safe_print("DRY RUN - Follow-up")
        _safe_print("=" * 60)
        _safe_print(text)
        _safe_print("=" * 60)
        return True

    return _send_telegram(text)


def send_top_ev_plays(plays, dry_run=False):
    """
    Send the top 3 EV plays of the day across all markets.

    Args:
        plays: list of dicts with keys: market, description, ev_per_dollar,
               suggested_bet, key_factors
        dry_run: if True, print to console
    """
    if not plays:
        return False

    today = date.today().strftime("%B %d, %Y")
    lines = [f"\u26be TOP EV PLAYS \u2014 {today}", ""]

    for i, play in enumerate(plays[:3], 1):
        market = play.get("market", "")
        desc = play.get("description", "")
        ev = play.get("ev_per_dollar", 0)
        bet = play.get("suggested_bet", 1)
        factors = play.get("key_factors", [])

        lines.append(f"#{i} {market}: {desc}")
        lines.append(f"Edge: {ev:.1%} EV | Bet: ${bet:.0f}")
        if factors:
            lines.append(f"Why: {'; '.join(factors)}")
        lines.append("")

    text = "\n".join(lines)

    if dry_run:
        _safe_print("=" * 60)
        _safe_print("DRY RUN - Top EV Plays")
        _safe_print("=" * 60)
        _safe_print(text)
        _safe_print("=" * 60)
        return True

    return _send_telegram(text)


def send_ev_props(hr_props, k_props, dry_run=False):
    """
    Send +EV straight bet recommendations for DraftKings.
    Only sends if there are actual +EV plays.
    """
    if not hr_props and not k_props:
        return False

    today = date.today().strftime("%B %d, %Y")
    lines = [f"\u26be DK +EV PROPS \u2014 {today}", ""]

    if hr_props:
        lines.append("\U0001f4b0 HR PROPS (straight bets)")
        for p in hr_props[:5]:
            name = p.get("player_name", "")
            mp = p.get("model_prob", 0)
            ip = p.get("implied_prob", 0)
            odds = p.get("over_odds", 0)
            bet = p.get("suggested_bet", 1)
            edge_str = p.get("key_edge", "")
            lines.append(
                f"  \u2022 {name} +EV: model {mp:.0%} vs book {ip:.0%} "
                f"\u2192 {odds:+d} | ${bet:.0f}"
            )
            if edge_str:
                lines.append(f"    {edge_str}")
        lines.append("")

    if k_props:
        lines.append("\u26a1 K PROPS")
        for p in k_props[:3]:
            pitcher = p.get("pitcher_name", "")
            side = p.get("best_side", "over").upper()
            bline = p.get("book_line", 0)
            odds = p.get("over_odds", 0) if p.get("best_side") == "over" else p.get("under_odds", 0)
            bet = p.get("suggested_bet", 1)
            factors = p.get("key_factors", [])
            lines.append(
                f"  \u2022 {pitcher} {side} {bline} Ks \u2192 {odds:+d} | ${bet:.0f}"
            )
            if factors:
                lines.append(f"    {'; '.join(factors)}")
        lines.append("")

    lines.append("Only bet when model edge > 3%.")

    text = "\n".join(lines)

    if dry_run:
        _safe_print("=" * 60)
        _safe_print("DRY RUN - DK +EV Props")
        _safe_print("=" * 60)
        _safe_print(text)
        _safe_print("=" * 60)
        return True

    return _send_telegram(text)


def send_error(msg, dry_run=False):
    """Send an error alert via Telegram."""
    text = f"\u26a0\ufe0f HR Model Error\n\n{msg}"

    if dry_run:
        _safe_print(f"[ERROR] {msg}")
        return True

    return _send_telegram(text)
