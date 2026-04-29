"""
Module 1: Configuration loader.
Loads .env, config.yaml, and optional weights.json.
Exposes a single CFG object used by all other modules.
"""

import os
import json
from pathlib import Path

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# Load .env
load_dotenv(BASE_DIR / ".env")

# Load config.yaml
with open(BASE_DIR / "config.yaml", "r") as f:
    _yaml = yaml.safe_load(f)

# Load weights.json if it exists (calibrated backtest weights), else equal weights
_weights_path = BASE_DIR / "weights.json"
if _weights_path.exists():
    with open(_weights_path, "r") as f:
        _weights = json.load(f)
else:
    _weights = {
        "barrel_rate": 1.0,
        "hr_fb_ratio": 1.0,
        "iso": 1.0,
        "hard_hit_pct": 1.0,
        "platoon_factor": 1.0,
        "pitcher_hr_fb": 1.0,
        "pitcher_fly_ball_rate": 1.0,
        "pitcher_xfip": 1.0,
        "pitcher_hard_hit_allowed": 1.0,
        "park_hr_factor": 1.0,
        "wind_bonus": 1.0,
        "temp_bonus": 1.0,
        "batting_order_position": 1.0,
        "barrel_rate_14d": 1.2,
        "hard_hit_pct_14d": 1.0,
        "form_trend": 0.8,
        "pitcher_recent_hr_per_ip": 1.2,
        "pitcher_recent_hard_hit": 0.8,
        "pitcher_high_workload": 1.0,
        "pitch_type_edge": 1.0,
    }


class _Config:
    """Single configuration object for the entire pipeline."""

    def __init__(self):
        # API keys
        self.owm_api_key = os.getenv("OWM_API_KEY", "")
        self.telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.odds_api_key = os.getenv("THE_ODDS_API_KEY", "")

        # Bet sizing
        self.bet_high = _yaml["bet_sizing"]["high_confidence"]
        self.bet_standard = _yaml["bet_sizing"]["standard"]
        self.bet_lottery = _yaml["bet_sizing"]["lottery"]

        # Brief times
        self.morning_time = _yaml["brief_times"]["morning"]
        self.followup_time = _yaml["brief_times"]["followup"]

        # Lineup checking
        self.lineup_check_interval = _yaml["lineup_check_interval_minutes"]
        self.lineup_window_start = _yaml["lineup_check_window"]["start"]
        self.lineup_window_end = _yaml["lineup_check_window"]["end"]

        # Parlay settings
        self.sharp_legs = _yaml["parlay"]["sharp_legs"]
        self.lottery_legs = _yaml["parlay"]["lottery_legs"]
        self.target_pitchers_per_day = _yaml["parlay"]["target_pitchers_per_day"]
        self.batters_per_stack = _yaml["parlay"]["batters_per_stack"]

        # Backtesting
        self.bt_start_year = _yaml["backtesting"]["start_year"]
        self.bt_end_year = _yaml["backtesting"]["end_year"]
        self.bt_holdout_year = _yaml["backtesting"]["holdout_year"]

        # Weights
        self.weights = _weights

        # Paths
        # NOTE: Railway has an ephemeral filesystem. All writes to cache/,
        # state/, outputs/ will be lost on redeploy. For persistent storage,
        # these need to be backed by a volume mount or external database
        # (e.g. Railway Volume, S3, or a Postgres JSON column).
        # Current writes that need persistence:
        #   - cache/*.pkl           (API response caches, can be re-fetched)
        #   - state/daily_state_*.json (lineup tracking, ephemeral by design)
        #   - outputs/clv_log.json  (CLV tracking — NEEDS persistent storage)
        #   - outputs/sharp_brief.txt, hr_list.txt (regenerated each run)
        #   - outputs/backtest/     (backtest results — nice to keep but regenerable)
        #   - weights.json          (calibrated model weights — commit to repo instead)
        #   - weights_totals.json   (calibrated totals weights — commit to repo instead)
        self.base_dir = BASE_DIR
        self.cache_dir = BASE_DIR / "cache"
        self.state_dir = BASE_DIR / "state"
        self.outputs_dir = BASE_DIR / "outputs"
        self.backtest_dir = BASE_DIR / "outputs" / "backtest"

        # Ensure directories exist
        self.cache_dir.mkdir(exist_ok=True)
        self.state_dir.mkdir(exist_ok=True)
        self.outputs_dir.mkdir(exist_ok=True)
        self.backtest_dir.mkdir(exist_ok=True)

    # Stadium coordinates (lat, lon) for weather lookups
    STADIUM_COORDS = {
        "Chase Field": (33.4455, -112.0667),
        "Truist Park": (33.8907, -84.4677),
        "Oriole Park at Camden Yards": (39.2838, -76.6218),
        "Fenway Park": (42.3467, -71.0972),
        "Wrigley Field": (41.9484, -87.6553),
        "Guaranteed Rate Field": (41.8299, -87.6338),
        "Great American Ball Park": (39.0974, -84.5065),
        "Progressive Field": (41.4962, -81.6852),
        "Coors Field": (39.7561, -104.9942),
        "Comerica Park": (42.3390, -83.0485),
        "Minute Maid Park": (29.7573, -95.3555),
        "Kauffman Stadium": (39.0517, -94.4803),
        "Angel Stadium": (33.8003, -117.8827),
        "Dodger Stadium": (34.0739, -118.2400),
        "loanDepot park": (25.7781, -80.2196),
        "American Family Field": (43.0280, -87.9712),
        "Target Field": (44.9817, -93.2776),
        "Citi Field": (40.7571, -73.8458),
        "Yankee Stadium": (40.8296, -73.9262),
        "Oakland Coliseum": (37.7516, -122.2005),
        "Citizens Bank Park": (39.9061, -75.1665),
        "PNC Park": (40.4469, -80.0058),
        "Petco Park": (32.7076, -117.1570),
        "Oracle Park": (37.7786, -122.3893),
        "T-Mobile Park": (47.5914, -122.3325),
        "Busch Stadium": (38.6226, -90.1928),
        "Tropicana Field": (27.7682, -82.6534),
        "Globe Life Field": (32.7473, -97.0845),
        "Rogers Centre": (43.6414, -79.3894),
        "Nationals Park": (38.8730, -77.0074),
        # Aliases
        "RingCentral Coliseum": (37.7516, -122.2005),
    }

    # Park run factors (for totals model, FanGraphs-derived, 1.0 = neutral)
    PARK_RUN_FACTORS = {
        "Coors Field": 1.30,
        "Great American Ball Park": 1.12,
        "Yankee Stadium": 1.08,
        "Chase Field": 1.07,
        "Citizens Bank Park": 1.06,
        "Wrigley Field": 1.05,
        "Globe Life Field": 1.04,
        "Fenway Park": 1.03,
        "Guaranteed Rate Field": 1.02,
        "Minute Maid Park": 1.02,
        "Dodger Stadium": 1.01,
        "Truist Park": 1.01,
        "Rogers Centre": 1.01,
        "Angel Stadium": 1.00,
        "Oriole Park at Camden Yards": 1.00,
        "Target Field": 1.00,
        "Citi Field": 0.99,
        "PNC Park": 0.98,
        "American Family Field": 0.98,
        "Busch Stadium": 0.97,
        "Comerica Park": 0.97,
        "Progressive Field": 0.96,
        "Kauffman Stadium": 0.96,
        "T-Mobile Park": 0.95,
        "Nationals Park": 0.95,
        "loanDepot park": 0.94,
        "Petco Park": 0.93,
        "Tropicana Field": 0.93,
        "Oracle Park": 0.91,
        "Oakland Coliseum": 0.89,
        "RingCentral Coliseum": 0.89,
    }

    def get_stadium_coords(self, stadium_name):
        """Return (lat, lon) for a stadium, or None if not found."""
        return self.STADIUM_COORDS.get(stadium_name)

    def get_park_run_factor(self, stadium_name):
        """Return run factor for a stadium (1.0 = neutral)."""
        return self.PARK_RUN_FACTORS.get(stadium_name, 1.0)


CFG = _Config()
