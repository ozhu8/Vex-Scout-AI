"""Highlight badges ("Strong offense", "Fast robot", "Breaks down", ...) from two
sources: verified stats (percentiles at this event) and scouts' quick taps."""
from collections import Counter

from . import db
from .scouting import NEG_TAGS, POS_TAGS

# badge label -> (stats rule: component, threshold) ; scouting tags that support it
GOOD = {
    "Strong offense": (("scoring", 70), ["Strong offense", "Scores quickly"]),
    "Strong defense": (("defense", 70), ["Good defense", "Strong drivetrain / hard to push"]),
    "Fast robot": (None, ["Fast robot", "Scores quickly"]),
    "Reliable robot": (("consistency", 70), ["Mechanically reliable"]),
    "Reliable auton": (("auton", 75), ["Reliable autonomous"]),
    "Skilled driver": (("driving", 75), ["Skilled driver"]),
    "Strong endgame": (None, ["Strong endgame"]),
    "Hard to push": (None, ["Strong drivetrain / hard to push"]),
    "Good teamwork": (None, ["Good teamwork"]),
}
BAD = {
    "Weak offense": (("scoring", 20), ["Weak offense", "Scores slowly"]),
    "Weak defense": (("defense", 20), ["Weak defense", "Gets pushed around"]),
    "Slow robot": (None, ["Slow robot", "Scores slowly"]),
    "Breaks down": (None, ["Breaks down"]),
    "Inconsistent": (("consistency", 20), []),
    "Unreliable auton": (None, ["Unreliable autonomous"]),
    "Inexperienced driver": (None, ["Inexperienced driver"]),
    "Weak endgame": (None, ["Weak endgame"]),
    "Poor teamwork": (None, ["Poor teamwork"]),
}
OPP = {"Strong offense": "Weak offense", "Strong defense": "Weak defense", "Fast robot": "Slow robot",
       "Reliable robot": "Breaks down", "Reliable auton": "Unreliable auton", "Skilled driver": "Inexperienced driver",
       "Strong endgame": "Weak endgame", "Good teamwork": "Poor teamwork"}


def scouted(conn, team_id, event_id):
    """Tag counts and auton attempt totals from quick-tap observations. Conflicting
    notes still count here - opposing badges are settled by vote in team_traits."""
    tags, auton = Counter(), {}
    for o in db.observations_for(conn, team_id, event_id):
        s = o["structured"]
        tags.update(s.get("tags_pos", []) + s.get("tags_neg", []))
        for side, (w, n) in (s.get("auton_attempts") or {}).items():
            a = auton.setdefault(side, [0, 0])
            a[0] += w; a[1] += n
    return tags, auton


def team_traits(conn, rating, event_id):
    """[(label, good: bool, source)] - stats need a meaningful sample; tags need more
    votes for than against."""
    tags, auton = scouted(conn, rating.team_id, event_id)
    enough = rating.stats["matches"] >= 4
    out = {}

    def consider(table, good):
        for label, (rule, support) in table.items():
            votes = sum(tags[t] for t in support)
            src = []
            if rule:
                k, cut = rule
                p = rating.pct.get(k)
                skills = k in ("auton", "driving")
                if p is not None and (enough or skills) and (p >= cut if good else p <= cut):
                    src.append("stats")
            if votes:
                src.append(f"scouted {votes}×")
            if src:
                out[label] = (good, ", ".join(src), votes)

    consider(GOOD, True)
    consider(BAD, False)
    tried = sum(n for _, n in auton.values())
    if tried >= 3:
        rate = sum(w for w, _ in auton.values()) / tried
        if rate >= 0.75:
            out["Reliable auton"] = (True, f"auton {rate:.0%} of {tried}", 99)
        elif rate <= 0.4:
            out["Unreliable auton"] = (False, f"auton {rate:.0%} of {tried}", 99)
    # opposing badges: keep whichever side the scouts voted for more
    for g, b in OPP.items():
        if g in out and b in out:
            drop = b if out[g][2] >= out[b][2] else g
            out.pop(drop)
    order = list(GOOD) + list(BAD)
    res = [(label, good, src) for label, (good, src, _) in sorted(out.items(), key=lambda kv: order.index(kv[0]))]
    # track record this season (more competitions = more proven)
    ev, n = rating.stats["events"], rating.stats["matches"]
    src = f"{ev} competition(s), {n} matches this season"
    if ev >= 3:
        res.insert(0, (f"Experienced · {ev} events", True, src))
    elif ev == 1:
        res.insert(0, ("Only 1 event so far", False, src))
    elif ev == 0:
        res.insert(0, ("No matches yet", False, "no scored matches this season"))
    if rating.live:  # today's status beats season-long impressions
        res.insert(0, ("Live: " + rating.live["label"], bool(rating.live["good"]), rating.live["text"]))
    return res


def auton_summary(conn, team_id, event_id):
    _, auton = scouted(conn, team_id, event_id)
    return {side: tuple(v) for side, v in auton.items() if v[1]}


assert set(t for _, s in GOOD.values() for t in s) <= set(POS_TAGS)
assert set(t for _, s in BAD.values() for t in s) <= set(NEG_TAGS)
