"""
Closing Line Value (CLV) Tracker.
Logs pick snapshots, pulls closing lines, records outcomes, calculates CLV.

NOTE: clv_log.json is the most critical persistent data file. On Railway's
ephemeral filesystem it will be lost on redeploy. Needs a persistent storage
solution (Railway Volume, S3, or database) for production use.
"""

import json
import os
from datetime import date, datetime

from config import CFG

CLV_LOG_PATH = CFG.outputs_dir / "clv_log.json"


def _load_log():
    if CLV_LOG_PATH.exists():
        with open(CLV_LOG_PATH, "r") as f:
            return json.load(f)
    return []


def _save_log(entries):
    CLV_LOG_PATH.parent.mkdir(exist_ok=True)
    with open(CLV_LOG_PATH, "w") as f:
        json.dump(entries, f, indent=2)


def _american_to_implied(price):
    price = int(price)
    if price < 0:
        return abs(price) / (abs(price) + 100)
    return 100 / (price + 100)


# ---------------------------------------------------------------------------
# 1. Snapshot: log today's picks
# ---------------------------------------------------------------------------

def log_snapshot(k_picks, batter_picks):
    """Save a snapshot of today's picks with opening odds to clv_log.json."""
    entries = _load_log()
    today = date.today().isoformat()

    # Remove any existing entries for today (re-snapshot)
    entries = [e for e in entries if e.get("date") != today]

    new_entries = []

    for p in k_picks:
        if not p.get("book"):
            continue
        new_entries.append({
            "date": today,
            "player": p["name"],
            "team": p.get("team", ""),
            "prop_type": "Ks",
            "line": p.get("book_line"),
            "side": "Over",
            "book": p.get("book", ""),
            "opening_odds": p.get("book_price"),
            "opening_impl": round(_american_to_implied(p["book_price"]), 4),
            "model_prob": p.get("model_prob", 0),
            "edge": p.get("edge", 0),
            "closing_odds": None,
            "closing_impl": None,
            "clv": None,
            "outcome": None,
            "pnl": None,
        })

    for b in batter_picks:
        for mkt, label in [("hits", "Hits"), ("total_bases", "TB"), ("home_runs", "HR")]:
            if not b.get(f"{mkt}_book"):
                continue
            price = b.get(f"{mkt}_price")
            if price is None:
                continue
            new_entries.append({
                "date": today,
                "player": b["name"],
                "team": b.get("team", ""),
                "prop_type": label,
                "line": b.get(f"{mkt}_line"),
                "side": "Over",
                "book": b.get(f"{mkt}_book", ""),
                "opening_odds": price,
                "opening_impl": round(_american_to_implied(price), 4),
                "model_prob": b.get(f"{mkt}_model", 0),
                "edge": b.get(f"{mkt}_edge", 0),
                "closing_odds": None,
                "closing_impl": None,
                "clv": None,
                "outcome": None,
                "pnl": None,
            })

    entries.extend(new_entries)
    _save_log(entries)
    print(f"[clv] Logged {len(new_entries)} picks for {today}")
    return len(new_entries)


# ---------------------------------------------------------------------------
# 2. Close lines: pull final odds after games start
# ---------------------------------------------------------------------------

def close_lines():
    """Pull current odds for today's logged picks and record as closing line."""
    from daily_picks import fetch_odds_lines, fetch_underdog_lines, _normalize_name

    entries = _load_log()
    today = date.today().isoformat()
    today_entries = [e for e in entries if e.get("date") == today and e.get("closing_odds") is None]

    if not today_entries:
        print("[clv] No open picks to close today.")
        return 0

    # Fetch current lines
    odds = fetch_odds_lines()
    try:
        ud = fetch_underdog_lines()
        for key, vals in ud.items():
            odds.setdefault(key, []).extend(vals)
    except Exception:
        pass

    MARKET_MAP = {"Ks": "pitcher_strikeouts", "Hits": "batter_hits",
                  "TB": "batter_total_bases", "HR": "batter_home_runs"}

    closed = 0
    for e in today_entries:
        market = MARKET_MAP.get(e["prop_type"])
        if not market:
            continue
        norm = _normalize_name(e["player"])
        line_entries = odds.get((norm, market), [])
        # Find same book or any book
        closing_entry = None
        book_key = {"DK": "draftkings", "FD": "fanduel", "UD": "underdog"}.get(e["book"], "")
        for le in line_entries:
            if le["book"] == book_key:
                closing_entry = le
                break
        if not closing_entry and line_entries:
            closing_entry = line_entries[0]

        if closing_entry:
            closing_price = closing_entry["price"]
            e["closing_odds"] = closing_price
            e["closing_impl"] = round(_american_to_implied(closing_price), 4)
            e["clv"] = round(e["closing_impl"] - e["opening_impl"], 4)
            closed += 1

    _save_log(entries)
    print(f"[clv] Closed {closed}/{len(today_entries)} lines")
    return closed


# ---------------------------------------------------------------------------
# 3. Update outcomes: check actual results after games complete
# ---------------------------------------------------------------------------

def update_outcomes():
    """Check actual game outcomes for logged picks using MLB Stats API game logs."""
    import requests as _req

    entries = _load_log()
    pending = [e for e in entries if e.get("outcome") is None and e.get("opening_odds") is not None]

    if not pending:
        print("[clv] No pending outcomes to update.")
        return 0

    # Group by date to batch
    by_date = {}
    for e in pending:
        by_date.setdefault(e["date"], []).append(e)

    updated = 0

    for dt, picks in by_date.items():
        # Pull schedule for this date
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={dt}&gameType=R&hydrate=linescore"
        try:
            resp = _req.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue

        # Check if games are complete
        games_complete = True
        for d in data.get("dates", []):
            for g in d.get("games", []):
                status = g.get("status", {}).get("abstractGameState", "")
                if status != "Final":
                    games_complete = False
                    break

        if not games_complete:
            continue

        # For each pick, look up the player's game log for that date
        for e in picks:
            player_name = e["player"]
            prop_type = e["prop_type"]
            line = e.get("line", 0)

            # Find player ID from our stats
            from data_fetcher import get_batter_statcast, get_pitcher_statcast
            if prop_type == "Ks":
                df = get_pitcher_statcast()
            else:
                df = get_batter_statcast()

            if df.empty:
                continue

            name_col = "Name" if "Name" in df.columns else None
            if not name_col:
                continue

            rows = df[df[name_col] == player_name]
            if rows.empty:
                last = player_name.split()[-1]
                rows = df[df[name_col].str.contains(last, na=False)]
                if len(rows) != 1:
                    continue

            pid = int(rows.iloc[0]["mlbID"])
            year = int(dt[:4])

            # Fetch game log
            if prop_type == "Ks":
                gl_url = f"https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=gameLog&group=pitching&season={year}&gameType=R"
            else:
                gl_url = f"https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=gameLog&group=hitting&season={year}&gameType=R"

            try:
                resp = _req.get(gl_url, timeout=15)
                resp.raise_for_status()
                splits = resp.json().get("stats", [{}])[0].get("splits", [])
            except Exception:
                continue

            for s in splits:
                if s.get("date") != dt:
                    continue
                stat = s.get("stat", {})

                if prop_type == "Ks":
                    actual = int(stat.get("strikeOuts", 0))
                elif prop_type == "Hits":
                    actual = int(stat.get("hits", 0))
                elif prop_type == "TB":
                    actual = int(stat.get("totalBases", 0))
                elif prop_type == "HR":
                    actual = int(stat.get("homeRuns", 0))
                else:
                    continue

                won = actual > line
                e["outcome"] = "W" if won else "L"
                e["actual"] = actual
                # P&L at -110
                e["pnl"] = round(0.909, 2) if won else -1.0
                updated += 1
                break

    _save_log(entries)
    print(f"[clv] Updated {updated} outcomes")
    return updated


# ---------------------------------------------------------------------------
# 4. Summary stats
# ---------------------------------------------------------------------------

def get_clv_summary():
    """Return CLV stats for the CLV Tracker tab."""
    entries = _load_log()

    if not entries:
        return {"entries": [], "stats": {}}

    # Filter to entries with data
    with_clv = [e for e in entries if e.get("clv") is not None]
    with_outcome = [e for e in entries if e.get("outcome") is not None]

    stats = {
        "total_logged": len(entries),
        "with_closing": len(with_clv),
        "with_outcome": len(with_outcome),
    }

    if with_clv:
        clvs = [e["clv"] for e in with_clv]
        stats["avg_clv"] = round(sum(clvs) / len(clvs), 4)
        stats["positive_clv_pct"] = round(sum(1 for c in clvs if c > 0) / len(clvs) * 100, 1)

        # CLV by prop type
        by_type = {}
        for e in with_clv:
            pt = e["prop_type"]
            by_type.setdefault(pt, []).append(e["clv"])
        stats["clv_by_type"] = {k: round(sum(v)/len(v), 4) for k, v in by_type.items()}

        # CLV by book
        by_book = {}
        for e in with_clv:
            bk = e.get("book", "?")
            by_book.setdefault(bk, []).append(e["clv"])
        stats["clv_by_book"] = {k: round(sum(v)/len(v), 4) for k, v in by_book.items()}

        # Histogram buckets
        buckets = {}
        for c in clvs:
            bucket = round(c * 20) / 20  # 5% increments
            buckets[bucket] = buckets.get(bucket, 0) + 1
        stats["histogram"] = dict(sorted(buckets.items()))

    if with_outcome:
        wins = sum(1 for e in with_outcome if e["outcome"] == "W")
        stats["win_rate"] = round(wins / len(with_outcome) * 100, 1)
        stats["total_pnl"] = round(sum(e.get("pnl", 0) for e in with_outcome), 2)

        # Win rate by CLV quartile
        oc_with_clv = [e for e in with_outcome if e.get("clv") is not None]
        if len(oc_with_clv) >= 4:
            sorted_oc = sorted(oc_with_clv, key=lambda e: e["clv"])
            q = len(sorted_oc) // 4
            quartiles = [sorted_oc[:q], sorted_oc[q:2*q], sorted_oc[2*q:3*q], sorted_oc[3*q:]]
            stats["win_by_clv_quartile"] = []
            for i, qr in enumerate(quartiles):
                w = sum(1 for e in qr if e["outcome"] == "W")
                avg_c = sum(e["clv"] for e in qr) / len(qr) if qr else 0
                stats["win_by_clv_quartile"].append({
                    "label": f"Q{i+1}",
                    "count": len(qr),
                    "wins": w,
                    "win_rate": round(w / len(qr) * 100, 1) if qr else 0,
                    "avg_clv": round(avg_c, 4),
                })

    # Return recent entries (last 100) for display
    display_entries = sorted(entries, key=lambda e: e.get("date", ""), reverse=True)[:100]

    return {"entries": display_entries, "stats": stats}
