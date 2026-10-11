"""Overall 0-100 team rating: weighted percentile ranks against the other teams
at an event, built only from verified match and skills data.

Components a team has no data for count as average (50) rather than being
dropped, so missing evidence never helps. Small samples are shrunk toward the
field average before ranking, and the overall is then pulled toward 50 by how
much evidence there is - so 5-0 at one event doesn't outrank 62% over 38
matches at four events. Experience rewards having competed more.
"""
import statistics
from dataclasses import dataclass, field

import numpy as np

from . import db, feedback
from .predictor import ELO_BASE, RIDGE, _phi

# (key, label, weight, higher_is_better)
COMPONENTS = [
    ("scoring", "Scoring", 0.20, True),
    ("experience", "Experience", 0.20, True),
    ("impact", "Match impact", 0.15, True),
    ("auton", "Autonomous", 0.10, True),
    ("driving", "Driving", 0.10, True),
    ("consistency", "Consistency", 0.10, True),
    ("winrate", "Win rate", 0.10, True),
    ("defense", "Defense", 0.05, True),
]
SHRINK_MATCHES = 8   # a team with this many matches counts half its own number, half the field average
# How far the overall may sit from 50: evidence / (evidence + EVIDENCE_K), where each extra
# competition counts like EVENT_MATCHES matches - doing well at 10 events beats doing well at 1.
EVIDENCE_K = 12
EVENT_MATCHES = 8
TIERS = [(80, "Elite"), (65, "Strong"), (45, "Solid"), (30, "Developing"), (0, "Building")]


def tier(score):
    if score is None:
        return "Unrated"
    return next(name for cut, name in TIERS if score >= cut)


@dataclass
class TeamRating:
    team_id: int
    number: str
    name: str
    raw: dict = field(default_factory=dict)       # component -> raw value (None = no data)
    pct: dict = field(default_factory=dict)       # component -> 0..100 percentile at this event
    contrib: dict = field(default_factory=dict)   # component -> points added to overall
    overall: float | None = None
    overall_raw: float | None = None              # before the sample-size pull toward 50
    overall_base: float | None = None             # before live feedback
    live: dict | None = None                      # current live-feedback status (feedback.current_status)
    reliability: float = 0.0                      # share of the raw score's distance from 50 that's kept
    tier: str = "Unrated"
    profile: str = "No data yet"
    confidence: str = "low"
    rank: int | None = None                       # among rated teams at the event
    stats: dict = field(default_factory=dict)     # display values: opr, dpr, ccwm, elo, swing, ...
    strengths: list = field(default_factory=list)
    weaknesses: list = field(default_factory=list)


def _dpr(matches, teams, mean_half):
    """Ridge fit of points *allowed* (opponent score) per team."""
    idx = {t: i for i, t in enumerate(teams)}
    A, y = [], []
    for m in matches:
        for side, opp in (("red", "blue"), ("blue", "red")):
            r = np.zeros(len(teams))
            for t in (m[f"{side}1"], m[f"{side}2"]):
                if t in idx:
                    r[idx[t]] = 1
            A.append(r); y.append(m[f"{opp}_score"])
    if not A:
        return {}
    A, y = np.array(A), np.array(y, float)
    x = np.linalg.solve(A.T @ A + RIDGE * np.eye(len(teams)), A.T @ y + RIDGE * mean_half)
    return {t: float(x[idx[t]]) for t in teams}


def _percentiles(values: dict):
    """Rank-based percentile (0 = lowest at this event, 100 = highest)."""
    vals = {k: v for k, v in values.items() if v is not None}
    if not vals:
        return {}
    if len(vals) == 1:
        return {k: 50.0 for k in vals}
    xs = list(vals.values())
    n = len(xs)
    return {k: 100 * (sum(x < v for x in xs) + 0.5 * (sum(x == v for x in xs) - 1)) / (n - 1) for k, v in vals.items()}


def _profile(r: TeamRating):
    p = r.pct
    n = r.stats.get("matches", 0)
    if r.overall is None:
        return "No data yet"
    if n == 0:
        return "Skills only"
    if n < 3:
        return "Limited data"
    g = lambda k: p.get(k, 50)
    if g("scoring") >= 70 and g("impact") >= 70 and g("auton") >= 60:
        return "Complete package"
    if g("scoring") >= 75:
        return "High scorer"
    if g("auton") >= 75 and g("scoring") < 60:
        return "Auton specialist"
    if g("defense") >= 75 and g("scoring") < 60:
        return "Defensive anchor"
    if g("consistency") <= 30 and g("scoring") >= 50:
        return "Boom or bust"
    if g("driving") >= 75:
        return "Skilled driver"
    if r.overall < 30:
        return "Rebuilding"
    return "Balanced"


def _describe(key, r: TeamRating, good: bool):
    s = r.stats
    if key == "scoring":
        return f"{'Strong' if good else 'Low'} scorer: ~{s['opr']:.1f} pts/match"
    if key == "impact":
        return f"{'Moves' if good else 'Rarely moves'} the margin: CCWM {s['ccwm']:+.1f}"
    if key == "auton":
        return f"{'Strong' if good else 'Weak'} autonomous: {s['auton']} prog. skills"
    if key == "driving":
        return f"{'Strong' if good else 'Weak'} driving: {s['driver']} driver skills"
    if key == "consistency":
        return f"{'Consistent' if good else 'Inconsistent'}: alliance score swings ±{s['swing']:.1f}"
    if key == "winrate":
        return f"{'Wins' if good else 'Loses'} often: {s['winrate']:.0%} win rate"
    if key == "experience":
        return f"{'Experienced' if good else 'Little experience'}: {s['events']} event(s), {s['matches']} matches this season"
    if key == "defense":
        return f"{'Holds' if good else 'Concedes'} opponents to ~{s['dpr']:.1f} pts"
    return key


def rate_event(conn, event_id, model):
    ev = db.one(conn, "SELECT season_id FROM events WHERE id=?", (event_id,))
    teams = db.event_teams(conn, event_id)
    ids = [t["id"] for t in teams]
    season = ev["season_id"] if ev else None
    matches = db.scored_matches(conn, season_id=season) if season is not None else db.scored_matches(conn, event_id=event_id)
    all_teams = sorted({t for m in matches for t in (m["red1"], m["red2"], m["blue1"], m["blue2"]) if t})
    dpr = _dpr(matches, all_teams, model.mean_alliance / 2) if all_teams else {}

    # best skills scores this season (any event)
    sk = {}
    for row in db.rows(conn, f"SELECT team_id, type, max(score) s FROM skills WHERE team_id IN ({','.join('?' * len(ids)) or 'NULL'}) "
                             "GROUP BY team_id, type", ids):
        if row["s"]:
            sk[(row["team_id"], row["type"])] = row["s"]

    out = {}
    for t in teams:
        tid = t["id"]
        mine = [m for m in matches if tid in (m["red1"], m["red2"], m["blue1"], m["blue2"])]
        scores, margins, wins = [], [], 0.0
        for m in mine:
            c, o = ("red", "blue") if tid in (m["red1"], m["red2"]) else ("blue", "red")
            scores.append(m[f"{c}_score"]); margins.append(m[f"{c}_score"] - m[f"{o}_score"])
            wins += 1 if margins[-1] > 0 else 0.5 if margins[-1] == 0 else 0
        n = len(mine)
        r = TeamRating(tid, t["number"], t["name"] or "")
        opr = model.opr.get(tid) if n else None
        d = dpr.get(tid) if n else None
        swing = statistics.pstdev(scores) if n >= 3 else None
        half = n // 2
        momentum = (np.mean(margins[half:]) - np.mean(margins[:half])) if n >= 4 else None
        r.stats = dict(matches=n, opr=opr, dpr=d, ccwm=(opr - d) if n else None,
                       elo=round(model.elo.get(tid, ELO_BASE)), swing=swing,
                       winrate=wins / n if n else None, auton=sk.get((tid, "programming")),
                       driver=sk.get((tid, "driver")), events=len({m["event_id"] for m in mine}),
                       momentum=momentum)
        # win rate: Bayesian average with SHRINK_MATCHES phantom matches at 50%
        wr_adj = (wins + 0.5 * SHRINK_MATCHES) / (n + SHRINK_MATCHES) if n else None
        r.raw = dict(scoring=opr, impact=r.stats["ccwm"], auton=r.stats["auton"], driving=r.stats["driver"],
                     consistency=-swing if swing is not None else None, winrate=wr_adj,
                     defense=-d if d is not None else None, momentum=momentum,
                     experience=r.stats["events"] + n / 1000 if n else None)
        r.confidence = "high" if n >= 8 else "medium" if n >= 3 else "low"
        out[tid] = r

    # shrink match-based numbers toward the field average in proportion to sample size
    for key in ("scoring", "impact", "defense", "consistency"):
        vals = [r.raw[key] for r in out.values() if r.raw.get(key) is not None]
        if not vals:
            continue
        mean = float(np.mean(vals))
        for r in out.values():
            if r.raw.get(key) is not None:
                n = r.stats["matches"]
                r.raw[key] = mean + (r.raw[key] - mean) * n / (n + SHRINK_MATCHES)

    for key in [c[0] for c in COMPONENTS] + ["momentum"]:
        for tid, p in _percentiles({tid: r.raw.get(key) for tid, r in out.items()}).items():
            out[tid].pct[key] = p

    for r in out.values():
        if any(k in r.pct for k, *_ in COMPONENTS if k != "experience"):
            r.contrib = {k: w * r.pct.get(k, 50.0) for k, _, w, _ in COMPONENTS}
            r.overall_raw = sum(r.contrib.values())
            evidence = (r.stats["matches"] + EVENT_MATCHES * max(0, r.stats["events"] - 1)
                        + 2 * sum(r.stats[x] is not None for x in ("auton", "driver")))
            r.reliability = evidence / (evidence + EVIDENCE_K)
            r.overall = round(50 + (r.overall_raw - 50) * r.reliability)
        r.tier = tier(r.overall)
        r.profile = _profile(r)
        for k, *_ in COMPONENTS:
            if k in r.pct and r.stats["matches"] >= 2 or k in ("auton", "driving") and k in r.pct:
                if r.pct[k] >= 75:
                    r.strengths.append(_describe(k, r, True))
                elif r.pct[k] <= 25:
                    r.weaknesses.append(_describe(k, r, False))

    # live feedback (e.g. "robot broken today") scales the overall; verified stats are untouched
    for tid, st in feedback.current_status(conn, event_id).items():
        r = out.get(tid)
        if r is None:
            continue
        r.live = st
        if r.overall is not None:
            r.overall_base = r.overall
            r.overall = round(r.overall * st["mult"]) if st["effect"] != "out" else r.overall
            r.tier = tier(r.overall)

    rated = sorted((r for r in out.values() if r.overall is not None and not (r.live and r.live["effect"] == "out")),
                   key=lambda r: -r.overall)
    for i, r in enumerate(rated, 1):
        r.rank = i
    return out


def pick_list(ratings: dict, my_id, model):
    """Rank partners for `my_id`: overall strength plus how well they cover our weak components."""
    me = ratings.get(my_id)
    if not me:
        return []
    my_opr = me.stats["opr"] if me.stats["opr"] is not None else model.mean_alliance / 2
    need = {k: max(0.0, 50 - me.pct.get(k, 50)) / 50 for k, *_ in COMPONENTS}
    picks = []
    for tid, r in ratings.items():
        if tid == my_id or r.overall is None or (r.live and r.live["effect"] == "out"):
            continue
        fit_w = {k: w * (1 + 2 * need[k]) for k, _, w, _ in COMPONENTS if k in r.pct}
        fit = sum(fit_w[k] * r.pct[k] for k in fit_w) / sum(fit_w.values())
        score = 0.6 * r.overall + 0.4 * fit
        their_opr = r.stats["opr"] if r.stats["opr"] is not None else model.mean_alliance / 2
        if r.live:
            their_opr *= r.live["mult"]
        exp = my_opr + their_opr
        p_typical = _phi((exp - model.mean_alliance) / (np.sqrt(2) * model.sigma))
        covers = [label for k, label, *_ in COMPONENTS if need[k] > 0.2 and r.pct.get(k, 0) >= 65]
        picks.append(dict(rating=r, score=round(score), fit=round(fit), exp_points=exp,
                          p_vs_typical=p_typical, covers=covers))
    return sorted(picks, key=lambda p: -p["score"])
