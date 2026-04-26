"""
EV Calculator.
Given a model probability and book's American odds, calculates expected value,
Kelly criterion fraction, and suggested bet size.
"""

import os
import sys


def _validate_american_odds(american_odds):
    """Validate that odds look like American format, not decimal."""
    if american_odds is None:
        raise ValueError("Odds cannot be None")
    odds = float(american_odds)
    if odds == 0:
        return 0
    if -3 < odds < 3 and odds != 0:
        raise ValueError(
            f"Odds value {odds} looks like decimal odds, not American. "
            "American odds are integers like -150 or +130."
        )
    if odds > 50000 or odds < -50000:
        print(f"[ev_calculator] WARNING: extreme odds value {odds}", file=sys.stderr)
    return odds


def american_to_implied_prob(american_odds):
    """Convert American odds to implied probability (0-1)."""
    odds = _validate_american_odds(american_odds)
    if odds == 0:
        return 0.5
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return abs(odds) / (abs(odds) + 100.0)


def american_to_decimal(american_odds):
    """Convert American odds to decimal payout (includes stake)."""
    odds = _validate_american_odds(american_odds)
    if odds == 0:
        return 2.0  # even money
    if odds > 0:
        return (odds / 100.0) + 1.0
    else:
        return (100.0 / abs(odds)) + 1.0


def calculate_ev(model_prob, american_odds, stake=1.0):
    """
    Calculate expected value of a bet.
    EV = (model_prob * net_payout) - ((1 - model_prob) * stake)
    Returns EV per dollar staked.
    """
    decimal_odds = american_to_decimal(american_odds)
    net_payout = (decimal_odds - 1.0) * stake  # profit if win
    ev = (model_prob * net_payout) - ((1.0 - model_prob) * stake)
    return ev / stake  # normalize to per-dollar


def kelly_fraction(model_prob, american_odds):
    """
    Kelly criterion fraction (full Kelly, capped at 0.25 for safety).
    Returns fraction of bankroll to bet. Returns 0 if edge is negative.
    """
    decimal_odds = american_to_decimal(american_odds)
    b = decimal_odds - 1.0  # net odds (profit per $1 wagered)
    if b <= 0:
        return 0.0
    q = 1.0 - model_prob
    f = (model_prob * b - q) / b
    if f <= 0:
        return 0.0
    return min(f, 0.25)


def suggested_bet_size(ev_per_dollar, kelly, bankroll=100.0):
    """
    Given EV and Kelly fraction, return suggested bet size in dollars.
    Minimum $1. Maximum $50 if backtest weights exist, $5 otherwise.
    """
    weights_exist = os.path.exists(
        os.path.join(os.path.dirname(__file__), "weights.json")
    )
    max_bet = 50.0 if weights_exist else 5.0
    if kelly <= 0 or ev_per_dollar <= 0:
        return 1.0
    raw = kelly * bankroll
    return max(1.0, min(max_bet, round(raw, 0)))
