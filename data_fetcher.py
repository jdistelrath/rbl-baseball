"""
Module 2: Data Fetcher.
All network I/O lives here. Other modules never make HTTP calls.
"""

import pickle
import math
from datetime import datetime, date
from pathlib import Path

import pandas as pd
import requests

from config import CFG

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _today_str():
    return date.today().isoformat()


def _cache_path(name):
    return CFG.cache_dir / f"{name}_{_today_str()}.pkl"


def _load_cache(name):
    p = _cache_path(name)
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(name, obj):
    p = _cache_path(name)
    with open(p, "wb") as f:
        pickle.dump(obj, f)


# ---------------------------------------------------------------------------
# MLB Stats API
# ---------------------------------------------------------------------------

def get_today_schedule(target_date=None):
    """
    Returns list of dicts:
      {game_id, home_team, away_team, start_time_et, stadium, home_pitcher_id,
       away_pitcher_id, home_pitcher_name, away_pitcher_name}
    """
    dt = target_date or _today_str()
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={dt}&hydrate=probablePitcher,venue,linescore"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[data_fetcher] Schedule fetch failed: {e}")
        return []

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            game_id = g["gamePk"]
            home = g["teams"]["home"]["team"]["name"]
            away = g["teams"]["away"]["team"]["name"]
            venue = g.get("venue", {}).get("name", "Unknown")
            game_date = g.get("gameDate", "")  # UTC ISO

            # Probable pitchers
            hp = g["teams"]["home"].get("probablePitcher", {})
            ap = g["teams"]["away"].get("probablePitcher", {})

            games.append({
                "game_id": game_id,
                "home_team": home,
                "away_team": away,
                "start_time_utc": game_date,
                "stadium": venue,
                "home_pitcher_id": hp.get("id"),
                "away_pitcher_id": ap.get("id"),
                "home_pitcher_name": hp.get("fullName", "TBD"),
                "away_pitcher_name": ap.get("fullName", "TBD"),
            })
    return games


def get_confirmed_lineups(game_id):
    """
    Returns {home: [{id, name, batting_order, position, bat_side}], away: [...]}
    or None if lineups not yet posted.
    """
    url = f"https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[data_fetcher] Boxscore fetch failed for {game_id}: {e}")
        return None

    result = {}
    for side in ("home", "away"):
        team_data = data.get("teams", {}).get(side, {})
        batting_order = team_data.get("battingOrder", [])
        if not batting_order:
            return None  # Lineups not posted yet

        players_data = team_data.get("players", {})
        players = []
        for i, pid in enumerate(batting_order):
            pkey = f"ID{pid}"
            pinfo = players_data.get(pkey, {})
            person = pinfo.get("person", {})
            bat_side = pinfo.get("batSide", {}).get("code", "R")
            players.append({
                "id": pid,
                "name": person.get("fullName", f"Player {pid}"),
                "batting_order": i + 1,
                "position": pinfo.get("position", {}).get("abbreviation", ""),
                "bat_side": bat_side,
            })
        result[side] = players
    return result


def get_probable_pitchers(game_id):
    """Returns {home_pitcher_id, away_pitcher_id} from schedule endpoint."""
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&gamePk={game_id}&hydrate=probablePitcher"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[data_fetcher] Probable pitchers fetch failed for {game_id}: {e}")
        return None

    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g["gamePk"] == game_id:
                hp = g["teams"]["home"].get("probablePitcher", {})
                ap = g["teams"]["away"].get("probablePitcher", {})
                return {
                    "home_pitcher_id": hp.get("id"),
                    "away_pitcher_id": ap.get("id"),
                    "home_pitcher_name": hp.get("fullName", "TBD"),
                    "away_pitcher_name": ap.get("fullName", "TBD"),
                    "home_pitcher_throws": hp.get("pitchHand", {}).get("code", "R"),
                    "away_pitcher_throws": ap.get("pitchHand", {}).get("code", "R"),
                }
    return None


def _get_pitcher_hand(pitcher_id):
    """Fetch pitcher throwing hand from MLB people endpoint."""
    if not pitcher_id:
        return "R"
    url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        people = data.get("people", [])
        if people:
            return people[0].get("pitchHand", {}).get("code", "R")
    except Exception:
        pass
    return "R"


# ---------------------------------------------------------------------------
# Player stats via MLB Stats API (reliable, no scraping)
# ---------------------------------------------------------------------------

def _fetch_mlb_hitting_stats(season, limit=500):
    """Fetch hitting stats from MLB Stats API. Returns list of split dicts."""
    all_splits = []
    offset = 0
    while True:
        url = (
            f"https://statsapi.mlb.com/api/v1/stats"
            f"?stats=season&group=hitting&season={season}&sportId=1"
            f"&limit={limit}&offset={offset}"
        )
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            splits = data.get("stats", [{}])[0].get("splits", [])
            if not splits:
                break
            all_splits.extend(splits)
            if len(splits) < limit:
                break
            offset += limit
        except Exception as e:
            print(f"[data_fetcher] MLB hitting stats page failed at offset {offset}: {e}")
            break
    return all_splits


def _fetch_mlb_pitching_stats(season, limit=500):
    """Fetch pitching stats from MLB Stats API. Returns list of split dicts."""
    all_splits = []
    offset = 0
    while True:
        url = (
            f"https://statsapi.mlb.com/api/v1/stats"
            f"?stats=season&group=pitching&season={season}&sportId=1"
            f"&limit={limit}&offset={offset}"
        )
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            splits = data.get("stats", [{}])[0].get("splits", [])
            if not splits:
                break
            all_splits.extend(splits)
            if len(splits) < limit:
                break
            offset += limit
        except Exception as e:
            print(f"[data_fetcher] MLB pitching stats page failed at offset {offset}: {e}")
            break
    return all_splits


def _safe_ip(ip_str):
    """Convert IP string like '177.2' to float innings (177.667)."""
    try:
        v = float(ip_str)
        whole = int(v)
        frac = v - whole  # .1 = 1/3, .2 = 2/3
        return whole + (frac * 10 / 3.0)
    except (TypeError, ValueError):
        return 0.0


def get_batter_statcast(season=None):
    """
    Returns DataFrame of batter stats for the given season (or current year).
    Cached once per day. Uses MLB Stats API.
    """
    season = season or date.today().year
    cache_key = f"batter_stats_mlb_{season}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    splits = _fetch_mlb_hitting_stats(season)
    if not splits:
        print(f"[data_fetcher] No batting data returned for {season}")
        return pd.DataFrame()

    rows = []
    for s in splits:
        stat = s.get("stat", {})
        player = s.get("player", {})
        team = s.get("team", {})
        ab = int(stat.get("atBats", 0))
        pa = int(stat.get("plateAppearances", 0))
        hr = int(stat.get("homeRuns", 0))
        so = int(stat.get("strikeOuts", 0))
        avg = float(stat.get("avg", 0))
        slg = float(stat.get("slg", 0))
        ops = float(stat.get("ops", 0))
        obp = float(stat.get("obp", 0))

        hits = int(stat.get("hits", 0))
        runs = int(stat.get("runs", 0))
        rbi = int(stat.get("rbi", 0))
        doubles = int(stat.get("doubles", 0))
        triples = int(stat.get("triples", 0))
        total_bases = int(stat.get("totalBases", 0))
        g = int(stat.get("gamesPlayed", 0))

        iso = slg - avg
        hr_fb = hr / (ab * 0.35) if ab > 0 else 0.0
        k_rate = so / pa if pa > 0 else 0.0

        rows.append({
            "Name": player.get("fullName", ""),
            "mlbID": player.get("id", 0),
            "Tm": team.get("abbreviation", ""),
            "Team": team.get("name", ""),
            "G": g,
            "PA": pa,
            "AB": ab,
            "H": hits,
            "2B": doubles,
            "3B": triples,
            "HR": hr,
            "R": runs,
            "RBI": rbi,
            "TB": total_bases,
            "SO": so,
            "BA": avg,
            "OBP": obp,
            "SLG": slg,
            "OPS": ops,
            "ISO": iso,
            "hr_fb_ratio": hr_fb,
            "k_rate": k_rate,
            "barrel_rate": float("nan"),
            "hard_hit_pct": float("nan"),
            "wRC+": (ops / 0.728) * 100 if ops > 0 else 100.0,
        })

    df = pd.DataFrame(rows)
    print(f"[data_fetcher] Loaded {len(df)} batters for {season} via MLB Stats API")
    _save_cache(cache_key, df)
    return df


def get_pitcher_statcast(season=None):
    """
    Returns DataFrame of pitcher stats for the given season.
    Cached once per day. Uses MLB Stats API.
    """
    season = season or date.today().year
    cache_key = f"pitcher_stats_mlb_{season}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    splits = _fetch_mlb_pitching_stats(season)
    if not splits:
        print(f"[data_fetcher] No pitching data returned for {season}")
        return pd.DataFrame()

    rows = []
    for s in splits:
        stat = s.get("stat", {})
        player = s.get("player", {})
        team = s.get("team", {})
        ip = _safe_ip(stat.get("inningsPitched", "0"))
        hr = int(stat.get("homeRuns", 0))
        so = int(stat.get("strikeOuts", 0))
        bb = int(stat.get("baseOnBalls", 0))
        era = float(stat.get("era", 0))
        gs = int(stat.get("gamesStarted", 0))
        k9 = float(stat.get("strikeoutsPer9Inn", 0))
        kbb = float(stat.get("strikeoutWalkRatio", 0))
        go_ao = float(stat.get("groundOutsToAirouts", 0))
        whip = float(stat.get("whip", 0))

        # Derived fields
        pitcher_hr_fb = hr / (ip * 1.2) if ip > 0 else 0.0
        fb_rate = 1.0 / (1.0 + go_ao) if go_ao > 0 else 0.45

        rows.append({
            "Name": player.get("fullName", ""),
            "mlbID": player.get("id", 0),
            "Tm": team.get("abbreviation", ""),
            "Team": team.get("name", ""),
            "G": int(stat.get("gamesPitched", 0)),
            "GS": gs,
            "IP": ip,
            "ERA": era,
            "FIP": era,  # ERA as FIP proxy
            "xFIP": era,
            "HR": hr,
            "SO": so,
            "BB": bb,
            "WHIP": whip,
            "K/9": k9,
            "K9": k9,
            "SO9": k9,
            "K/BB": kbb,
            "SO/W": kbb,
            "pitcher_hr_fb": pitcher_hr_fb,
            "fly_ball_rate": fb_rate,
            "FB%": fb_rate,
            "pitcher_hard_hit_allowed": float("nan"),
        })

    df = pd.DataFrame(rows)
    print(f"[data_fetcher] Loaded {len(df)} pitchers for {season} via MLB Stats API")
    _save_cache(cache_key, df)
    return df


# ---------------------------------------------------------------------------
# Weather (OpenWeatherMap)
# ---------------------------------------------------------------------------

def get_weather(stadium_name, game_time_utc=None):
    """
    Returns {temp_f, wind_speed_mph, wind_dir_degrees, wind_dir_label}
    Falls back to neutral defaults if API key missing or call fails.
    """
    neutral = {
        "temp_f": 70.0,
        "wind_speed_mph": 0.0,
        "wind_dir_degrees": 0,
        "wind_dir_label": "calm",
    }

    if not CFG.owm_api_key or CFG.owm_api_key == "your_openweathermap_key_here":
        return neutral

    coords = CFG.get_stadium_coords(stadium_name)
    if not coords:
        return neutral

    lat, lon = coords
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={CFG.owm_api_key}&units=imperial"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        wind = data.get("wind", {})
        main = data.get("main", {})

        wind_deg = wind.get("deg", 0)
        return {
            "temp_f": main.get("temp", 70.0),
            "wind_speed_mph": wind.get("speed", 0.0),
            "wind_dir_degrees": wind_deg,
            "wind_dir_label": _deg_to_label(wind_deg),
        }
    except Exception as e:
        print(f"[data_fetcher] Weather fetch failed for {stadium_name}: {e}")
        return neutral


def _deg_to_label(deg):
    dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    idx = round(deg / 22.5) % 16
    return dirs[idx]


# ---------------------------------------------------------------------------
# Park Factors
# ---------------------------------------------------------------------------

def get_park_factors(season=None):
    """
    Returns DataFrame of park factors. Tries pybaseball first,
    falls back to a sensible default set.
    Cached once per day.
    """
    season = season or date.today().year
    cache_key = f"park_factors_{season}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    # Try pybaseball's park factors if available
    try:
        from pybaseball import team_batting
        # team_batting doesn't have park factors directly, so we use a fallback
        raise ImportError("Use fallback")
    except Exception:
        pass

    # Fallback: hard-coded park HR factors (FanGraphs-derived, approximate)
    # Values >100 = hitter-friendly, <100 = pitcher-friendly, normalized to 1.0 scale
    park_factors = {
        "Coors Field": 1.38,
        "Great American Ball Park": 1.18,
        "Yankee Stadium": 1.15,
        "Globe Life Field": 1.12,
        "Citizens Bank Park": 1.10,
        "Wrigley Field": 1.08,
        "Fenway Park": 1.07,
        "Guaranteed Rate Field": 1.06,
        "Minute Maid Park": 1.05,
        "Dodger Stadium": 1.04,
        "Truist Park": 1.03,
        "Rogers Centre": 1.03,
        "Angel Stadium": 1.02,
        "Oriole Park at Camden Yards": 1.02,
        "Target Field": 1.01,
        "Citi Field": 1.00,
        "PNC Park": 0.99,
        "American Family Field": 0.99,
        "Busch Stadium": 0.98,
        "Comerica Park": 0.97,
        "Chase Field": 0.97,
        "Progressive Field": 0.96,
        "Kauffman Stadium": 0.96,
        "T-Mobile Park": 0.95,
        "Nationals Park": 0.95,
        "loanDepot park": 0.94,
        "Petco Park": 0.93,
        "Oracle Park": 0.90,
        "Tropicana Field": 0.92,
        "Oakland Coliseum": 0.88,
        "RingCentral Coliseum": 0.88,
    }
    df = pd.DataFrame(
        [{"stadium": k, "hr_factor": v} for k, v in park_factors.items()]
    )
    _save_cache(cache_key, df)
    return df


def get_park_factor_for_stadium(stadium_name, season=None):
    """Returns HR park factor float for a given stadium name."""
    df = get_park_factors(season)
    match = df[df["stadium"] == stadium_name]
    if not match.empty:
        return float(match.iloc[0]["hr_factor"])
    return 1.0  # neutral default


# ---------------------------------------------------------------------------
# Odds (The Odds API) - optional
# ---------------------------------------------------------------------------

def get_odds(sport="baseball_mlb", markets="h2h,totals"):
    """
    Returns odds data dict. Skips gracefully if API key not set.
    """
    if not CFG.odds_api_key or CFG.odds_api_key == "your_odds_api_key_here":
        return {}

    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
        f"?apiKey={CFG.odds_api_key}&regions=us&markets={markets}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[data_fetcher] Odds fetch failed: {e}")
        return {}


def _get_mlb_event_ids():
    """Fetch current MLB event IDs from The Odds API."""
    if not CFG.odds_api_key or CFG.odds_api_key == "your_odds_api_key_here":
        return []

    url = (
        f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events"
        f"?apiKey={CFG.odds_api_key}"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        events = resp.json()
        if not isinstance(events, list):
            return []
        return events
    except Exception as e:
        print(f"[data_fetcher] Event IDs fetch failed: {e}")
        return []


def get_player_props(market="batter_home_runs"):
    """
    Pull player prop lines from The Odds API.
    Player props require event-level endpoints (not sport-level).
    market: "batter_home_runs" or "pitcher_strikeouts"
    Returns list of dicts with player_name, over_line, over_odds, under_odds, etc.
    Prefers DraftKings, falls back to FanDuel, then others.
    """
    if not CFG.odds_api_key or CFG.odds_api_key == "your_odds_api_key_here":
        return []

    # Step 1: get event IDs
    events = _get_mlb_event_ids()
    if not events:
        print(f"[data_fetcher] No MLB events found for player props.")
        return []

    print(f"[data_fetcher] Found {len(events)} MLB events, fetching {market} props...")

    from ev_calculator import american_to_implied_prob

    PREFERRED_BOOKS = ["draftkings", "fanduel", "betmgm"]
    results = []

    # Step 2: for each event, fetch player prop odds
    for event_info in events:
        event_id = event_info.get("id", "")
        home_team = event_info.get("home_team", "")
        away_team = event_info.get("away_team", "")

        if not event_id:
            continue

        url = (
            f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds"
            f"?apiKey={CFG.odds_api_key}&regions=us&markets={market}&oddsFormat=american"
        )
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            event_data = resp.json()
        except Exception as e:
            # Skip this event silently (may not have props yet)
            continue

        bookmakers = sorted(
            event_data.get("bookmakers", []),
            key=lambda b: (0 if b.get("key", "") in PREFERRED_BOOKS else 1)
        )

        seen_players = set()

        for bookmaker in bookmakers:
            bk_key = bookmaker.get("key", "")
            for mkt in bookmaker.get("markets", []):
                if mkt.get("key") != market:
                    continue
                outcomes = mkt.get("outcomes", [])
                for outcome in outcomes:
                    player = outcome.get("description", outcome.get("name", ""))
                    side = outcome.get("name", "")
                    price = outcome.get("price")
                    line = outcome.get("point")

                    if not player or player in seen_players:
                        continue

                    if side == "Over" and price is not None and line is not None:
                        try:
                            impl_prob = american_to_implied_prob(int(price))
                        except (ValueError, TypeError):
                            continue

                        # Find matching under
                        under_odds = None
                        for o2 in outcomes:
                            if (o2.get("description", o2.get("name", "")) == player
                                    and o2.get("name") == "Under"):
                                under_odds = o2.get("price")

                        results.append({
                            "player_name": player,
                            "home_team": home_team,
                            "away_team": away_team,
                            "market": market,
                            "over_line": line,
                            "over_odds": int(price),
                            "under_odds": int(under_odds) if under_odds else None,
                            "implied_over_prob": round(impl_prob, 4),
                            "bookmaker": bk_key,
                        })
                        seen_players.add(player)

    print(f"[data_fetcher] Pulled {len(results)} {market} prop lines")
    return results


def get_bullpen_stats(season=None):
    """
    Pull relief pitcher ERA by team from MLB Stats API.
    Returns dict: {team_name: bullpen_era}
    """
    season = season or date.today().year
    cache_key = f"bullpen_stats_{season}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    all_splits = []
    offset = 0
    while True:
        url = (
            f"https://statsapi.mlb.com/api/v1/stats"
            f"?stats=season&group=pitching&season={season}&sportId=1"
            f"&limit=500&offset={offset}"
        )
        try:
            resp = requests.get(url, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            splits = data.get("stats", [{}])[0].get("splits", [])
            if not splits:
                break
            all_splits.extend(splits)
            if len(splits) < 500:
                break
            offset += 500
        except Exception as e:
            print(f"[data_fetcher] Bullpen stats page failed at offset {offset}: {e}")
            break

    # Aggregate: relievers = GS < 5, IP >= 5
    team_era_weighted = {}
    team_ip = {}

    for s in all_splits:
        stat = s.get("stat", {})
        team_name = s.get("team", {}).get("name", "")
        gs = int(stat.get("gamesStarted", 0))
        era = stat.get("era")
        ip_str = stat.get("inningsPitched", "0")

        if gs > 5 or not team_name or era is None:
            continue
        try:
            era_f = float(era)
            # Parse IP: "45.1" means 45 + 1/3
            ip_f = float(ip_str)
            whole = int(ip_f)
            frac = ip_f - whole
            ip_actual = whole + (frac * 10 / 3.0)
            if ip_actual < 5:
                continue
            team_era_weighted.setdefault(team_name, 0.0)
            team_ip.setdefault(team_name, 0.0)
            team_era_weighted[team_name] += era_f * ip_actual
            team_ip[team_name] += ip_actual
        except (TypeError, ValueError):
            continue

    result = {}
    for team in team_era_weighted:
        if team_ip[team] > 0:
            result[team] = round(team_era_weighted[team] / team_ip[team], 2)

    print(f"[data_fetcher] Loaded bullpen ERA for {len(result)} teams ({season})")
    _save_cache(cache_key, result)
    return result


def get_draft_actuals(date_str):
    """
    Pull actual game stats for a given date from MLB Stats API and apply
    Underdog scoring. Returns list of dicts:
      {name, team, position, actual_fp, fp_breakdown, game_id}
    Returns [] gracefully on failure. Cached per-date.
    """
    from market_underdog_draft import (
        classify_position, HITTER_SCORING, PITCHER_SCORING,
    )

    cache_key = f"actuals_{date_str}"
    cached = _load_cache(cache_key)
    if cached is not None:
        return cached

    schedule_url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={date_str}&hydrate=boxscore"
    )
    try:
        resp = requests.get(schedule_url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[data_fetcher] Schedule fetch failed for {date_str}: {e}")
        return []

    results = []

    for d in data.get("dates", []):
        for g in d.get("games", []):
            status = g.get("status", {}).get("abstractGameState", "")
            if status != "Final":
                continue
            game_id = g.get("gamePk")
            home_name = g.get("teams", {}).get("home", {}).get("team", {}).get("name", "")
            away_name = g.get("teams", {}).get("away", {}).get("team", {}).get("name", "")

            boxscore = g.get("boxscore")
            if not boxscore:
                # Fall back to per-game boxscore endpoint
                try:
                    bx_resp = requests.get(
                        f"https://statsapi.mlb.com/api/v1/game/{game_id}/boxscore",
                        timeout=15,
                    )
                    bx_resp.raise_for_status()
                    boxscore = bx_resp.json()
                except Exception as e:
                    print(f"[data_fetcher] Boxscore fetch failed for {game_id}: {e}")
                    continue

            for side, team_name in (("home", home_name), ("away", away_name)):
                team_data = boxscore.get("teams", {}).get(side, {})
                players = team_data.get("players", {})

                for _, pinfo in players.items():
                    person = pinfo.get("person", {})
                    name = person.get("fullName", "")
                    if not name:
                        continue
                    pos_abbr = pinfo.get("position", {}).get("abbreviation", "")
                    pos_class = classify_position(pos_abbr)
                    if pos_class is None:
                        continue

                    stats = pinfo.get("stats", {})

                    if pos_class == "P":
                        pitch = stats.get("pitching", {}) or {}
                        ip_str = pitch.get("inningsPitched")
                        if ip_str is None or ip_str == "":
                            continue
                        ip = _safe_ip(ip_str)
                        if ip <= 0:
                            continue
                        so = int(pitch.get("strikeOuts", 0) or 0)
                        er = int(pitch.get("earnedRuns", 0) or 0)
                        wins = int(pitch.get("wins", 0) or 0)
                        qs = 1 if (ip >= 6.0 and er <= 3) else 0

                        ip_fp = ip * PITCHER_SCORING["inning_pitched"]
                        k_fp = so * PITCHER_SCORING["strikeout"]
                        er_fp = er * PITCHER_SCORING["earned_run"]
                        win_fp = wins * PITCHER_SCORING["win"]
                        qs_fp = qs * PITCHER_SCORING["quality_start"]
                        total_fp = ip_fp + k_fp + er_fp + win_fp + qs_fp

                        results.append({
                            "name": name,
                            "team": team_name,
                            "position": "P",
                            "actual_fp": round(total_fp, 2),
                            "fp_breakdown": {
                                "ip": round(ip_fp, 2),
                                "k": round(k_fp, 2),
                                "er": round(er_fp, 2),
                                "win": round(win_fp, 2),
                                "qs": round(qs_fp, 2),
                            },
                            "game_id": game_id,
                        })
                    else:
                        bat = stats.get("batting", {}) or {}
                        ab = int(bat.get("atBats", 0) or 0)
                        bb_count = int(bat.get("baseOnBalls", 0) or 0)
                        hbp = int(bat.get("hitByPitch", 0) or 0)
                        if ab == 0 and bb_count == 0 and hbp == 0:
                            continue
                        h = int(bat.get("hits", 0) or 0)
                        d2 = int(bat.get("doubles", 0) or 0)
                        t3 = int(bat.get("triples", 0) or 0)
                        hr = int(bat.get("homeRuns", 0) or 0)
                        rbi = int(bat.get("rbi", 0) or 0)
                        runs = int(bat.get("runs", 0) or 0)
                        sb = int(bat.get("stolenBases", 0) or 0)
                        singles = max(0, h - d2 - t3 - hr)

                        single_fp = singles * HITTER_SCORING["single"]
                        double_fp = d2 * HITTER_SCORING["double"]
                        triple_fp = t3 * HITTER_SCORING["triple"]
                        hr_fp = hr * HITTER_SCORING["home_run"]
                        bb_fp = bb_count * HITTER_SCORING["walk"]
                        hbp_fp = hbp * HITTER_SCORING["hbp"]
                        rbi_fp = rbi * HITTER_SCORING["rbi"]
                        run_fp = runs * HITTER_SCORING["run"]
                        sb_fp = sb * HITTER_SCORING["stolen_base"]
                        total_fp = (single_fp + double_fp + triple_fp + hr_fp
                                    + bb_fp + hbp_fp + rbi_fp + run_fp + sb_fp)

                        results.append({
                            "name": name,
                            "team": team_name,
                            "position": pos_class,
                            "actual_fp": round(total_fp, 2),
                            "fp_breakdown": {
                                "hits": round(single_fp + double_fp + triple_fp, 2),
                                "hr": round(hr_fp, 2),
                                "walks": round(bb_fp + hbp_fp, 2),
                                "rbi": round(rbi_fp, 2),
                                "runs": round(run_fp, 2),
                                "sb": round(sb_fp, 2),
                            },
                            "game_id": game_id,
                        })

    print(f"[data_fetcher] Loaded {len(results)} actual draft player lines for {date_str}")
    _save_cache(cache_key, results)
    return results


def get_odds_for_markets(sport="baseball_mlb", markets="h2h,totals,spreads"):
    """
    Pull odds for MLB games from The Odds API.
    Returns the raw list of events with nested bookmaker/market data.
    Gracefully returns [] if API key not set or request fails.
    """
    if not CFG.odds_api_key or CFG.odds_api_key == "your_odds_api_key_here":
        return []

    url = (
        f"https://api.the-odds-api.com/v4/sports/{sport}/odds/"
        f"?apiKey={CFG.odds_api_key}&regions=us&markets={markets}&oddsFormat=american"
    )
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        print(f"[data_fetcher] Odds for markets fetch failed: {e}")
        return []
