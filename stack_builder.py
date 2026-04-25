"""
Module 5: Stack Builder.
Takes scored batters, builds parlay recommendations.
"""

from collections import defaultdict

from config import CFG


def _pitcher_proneness_score(scored_batters_vs_pitcher):
    """
    Score a pitcher's HR-proneness based on the batters facing them.
    Uses average of pitcher_hr_fb, pitcher_fly_ball_rate, and inverse xFIP
    across the batters scored against this pitcher.
    """
    if not scored_batters_vs_pitcher:
        return 0.0

    hr_fb_sum = 0.0
    fb_rate_sum = 0.0
    xfip_sum = 0.0
    n = len(scored_batters_vs_pitcher)

    for b in scored_batters_vs_pitcher:
        hr_fb_sum += b.get("pitcher_hr_fb", 0.11)
        fb_rate_sum += b.get("pitcher_fly_ball_rate", 0.34)
        # Higher xFIP = worse pitcher = more HR prone (invert for scoring)
        xfip_sum += b.get("pitcher_xfip", 4.20)

    avg_hr_fb = hr_fb_sum / n
    avg_fb_rate = fb_rate_sum / n
    avg_xfip = xfip_sum / n

    # Composite: weight HR/FB most heavily, xFIP as inverse quality
    score = (avg_hr_fb * 3.0) + (avg_fb_rate * 1.0) + ((avg_xfip - 3.5) * 0.1)
    return score


def build_stacks(scored_batters):
    """
    Build parlay recommendations from scored batters.

    Returns dict:
        sharp_parlay: list of 3 batters (best correlated stack, STRONG/STANDARD only)
        lottery_parlay: list of 6 batters (can mix pitchers)
        floor_list: top 10 batters by score (names + teams only)
        target_pitchers: list of {pitcher_name, proneness_score, stack: [batters]}
    """
    if not scored_batters:
        return {
            "sharp_parlay": [],
            "lottery_parlay": [],
            "floor_list": [],
            "target_pitchers": [],
        }

    sharp_legs = CFG.sharp_legs
    lottery_legs = CFG.lottery_legs
    target_count = CFG.target_pitchers_per_day
    batters_per = CFG.batters_per_stack

    # Group batters by opponent pitcher
    by_pitcher = defaultdict(list)
    for b in scored_batters:
        by_pitcher[b["opponent_pitcher"]].append(b)

    # Score each pitcher's proneness
    pitcher_scores = []
    for pitcher_name, batters in by_pitcher.items():
        if pitcher_name == "TBD":
            continue
        proneness = _pitcher_proneness_score(batters)
        # Sort batters by score descending
        batters_sorted = sorted(batters, key=lambda x: x["score"], reverse=True)
        pitcher_scores.append({
            "pitcher_name": pitcher_name,
            "proneness_score": round(proneness, 4),
            "stack": batters_sorted[:batters_per],
            "all_batters": batters_sorted,
        })

    # Rank pitchers by proneness
    pitcher_scores.sort(key=lambda x: x["proneness_score"], reverse=True)
    target_pitchers = pitcher_scores[:target_count]

    # --- Sharp parlay: best 3-leg correlated stack ---
    # Find the best single-pitcher stack with STRONG/STANDARD batters
    sharp_parlay = []
    for ps in target_pitchers:
        eligible = [b for b in ps["stack"] if b["tier"] in ("STRONG", "STANDARD")]
        if len(eligible) >= sharp_legs:
            sharp_parlay = eligible[:sharp_legs]
            break
    # If no single pitcher has enough STRONG/STANDARD, take best available
    if len(sharp_parlay) < sharp_legs and target_pitchers:
        best_stack = target_pitchers[0]["stack"]
        sharp_parlay = best_stack[:sharp_legs]

    # --- Lottery parlay: best 6-leg stack ---
    # Start with top target pitcher's stack, fill from other pitchers if needed
    lottery_parlay = []
    used_names = set()

    for ps in target_pitchers:
        for b in ps["all_batters"]:
            if b["name"] not in used_names and len(lottery_parlay) < lottery_legs:
                lottery_parlay.append(b)
                used_names.add(b["name"])
        if len(lottery_parlay) >= lottery_legs:
            break

    # If still short, fill from remaining scored batters
    if len(lottery_parlay) < lottery_legs:
        for b in scored_batters:
            if b["name"] not in used_names and len(lottery_parlay) < lottery_legs:
                lottery_parlay.append(b)
                used_names.add(b["name"])

    # --- Floor guy list: top 10 by score ---
    floor_list = [
        {"name": b["name"], "team": b["team"]}
        for b in scored_batters[:10]
    ]

    return {
        "sharp_parlay": sharp_parlay,
        "lottery_parlay": lottery_parlay,
        "floor_list": floor_list,
        "target_pitchers": [
            {
                "pitcher_name": ps["pitcher_name"],
                "proneness_score": ps["proneness_score"],
                "stack": [b["name"] for b in ps["stack"]],
            }
            for ps in target_pitchers
        ],
    }
