"""Synthetic event for offline demos (no RobotEvents token needed).

Everything here is clearly marked source='synthetic' and uses negative ids so it
can never collide with real RobotEvents records.
"""
import random

from . import db

DEMO_SKU = "DEMO-SYNTHETIC"
DEMO_EVENT_ID = -1
PRIOR_EVENT_ID = -2


def load_demo(seed: int = 7, n_teams: int = 24, quals_per_team: int = 8):
    rng = random.Random(seed)
    conn = db.connect()
    for eid in (DEMO_EVENT_ID, PRIOR_EVENT_ID):
        conn.execute("DELETE FROM matches WHERE event_id=?", (eid,))
        conn.execute("DELETE FROM event_teams WHERE event_id=?", (eid,))
        conn.execute("DELETE FROM skills WHERE event_id=?", (eid,))
        conn.execute("DELETE FROM rankings WHERE event_id=?", (eid,))
    conn.execute("""INSERT OR REPLACE INTO events (id, sku, name, program, season_id, season_name, start, end, source, fetched_at)
                    VALUES (?,?,?,?,?,?,?,?, 'synthetic', ?)""",
                 (DEMO_EVENT_ID, DEMO_SKU, "Demo Regional (synthetic data)", "V5RC", -100, "Demo season",
                  "2026-11-14", "2026-11-14", db.now()))
    conn.execute("""INSERT OR REPLACE INTO events (id, sku, name, program, season_id, season_name, start, end, source, fetched_at)
                    VALUES (?,?,?,?,?,?,?,?, 'synthetic', ?)""",
                 (PRIOR_EVENT_ID, "DEMO-PRIOR", "Demo Early-Season Scrimmage (synthetic)", "V5RC", -100, "Demo season",
                  "2026-10-03", "2026-10-03", db.now()))

    letters = "ABCDXYZ"
    teams = []
    for i in range(n_teams):
        tid = -(1000 + i)
        number = f"{rng.randint(100, 99999)}{rng.choice(letters)}"
        strength = rng.gauss(30, 9)           # hidden true scoring contribution
        auton = max(0, rng.gauss(6, 3))
        reliability = min(1, max(0.6, rng.gauss(0.9, 0.08)))
        teams.append(dict(id=tid, number=number, strength=strength, auton=auton, rel=reliability))
        conn.execute("""INSERT OR REPLACE INTO teams (id, number, name, organization, location, grade, source, fetched_at)
                        VALUES (?,?,?,?,?,?, 'synthetic', ?)""",
                     (tid, number, f"Demo Team {i + 1}", f"Demo School {i % 9 + 1}", "Demo City", "High School", db.now()))
        conn.execute("INSERT INTO event_teams VALUES (?,?)", (DEMO_EVENT_ID, tid))

    def score(alliance):
        s = 0.0
        for t in alliance:
            if rng.random() > t["rel"]:        # breakdown
                s += t["strength"] * 0.3
            else:
                s += t["strength"] + t["auton"] * 0.5
        return max(0, round(s + rng.gauss(0, 6)))

    mid = -10000

    def play(event_id, pool, n_matches, start_hour):
        nonlocal mid
        for k in range(n_matches):
            four = rng.sample(pool, 4)
            red, blue = four[:2], four[2:]
            rs, bs = score(red), score(blue)
            # auton bonus
            ra = sum(t["auton"] for t in red); ba = sum(t["auton"] for t in blue)
            rs += 8 if ra > ba else 0; bs += 8 if ba > ra else 0
            mid -= 1
            ts = f"2026-{'10-03' if event_id == PRIOR_EVENT_ID else '11-14'}T{start_hour + k // 12:02d}:{(k % 12) * 5:02d}:00"
            conn.execute("""INSERT INTO matches (id, event_id, division_id, round, instance, matchnum, name, scheduled, started,
                              red1, red2, blue1, blue2, red_score, blue_score, scored, source, fetched_at)
                            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,'synthetic',?)""",
                         (mid, event_id, None, 2, 1, k + 1, f"Qualifier #{k + 1}", ts, ts,
                          red[0]["id"], red[1]["id"], blue[0]["id"], blue[1]["id"], rs, bs, db.now()))

    # Half the teams attended an earlier scrimmage -> season history.
    play(PRIOR_EVENT_ID, teams[: n_teams // 2 + 4], 30, 9)
    play(DEMO_EVENT_ID, teams, n_teams * quals_per_team // 4, 9)

    for t in teams:
        conn.execute("INSERT INTO skills VALUES (?,?,?,?,?,?)",
                     (DEMO_EVENT_ID, t["id"], "driver", max(0, round(t["strength"] * 1.6 + rng.gauss(0, 6))), 3, None))
        conn.execute("INSERT INTO skills VALUES (?,?,?,?,?,?)",
                     (DEMO_EVENT_ID, t["id"], "programming", max(0, round(t["auton"] * 4 + rng.gauss(0, 5))), 3, None))
    conn.commit()
    _compute_rankings(conn, DEMO_EVENT_ID)
    return DEMO_EVENT_ID


def _compute_rankings(conn, event_id):
    stats = {}
    for m in db.scored_matches(conn, event_id=event_id):
        for side, opp, mine, theirs in (("red", "blue", m["red_score"], m["blue_score"]),
                                        ("blue", "red", m["blue_score"], m["red_score"])):
            for tid in (m[f"{side}1"], m[f"{side}2"]):
                s = stats.setdefault(tid, dict(w=0, l=0, t=0, pts=0, n=0))
                s["n"] += 1; s["pts"] += mine
                if mine > theirs: s["w"] += 1
                elif mine < theirs: s["l"] += 1
                else: s["t"] += 1
    ordered = sorted(stats.items(), key=lambda kv: (-(2 * kv[1]["w"] + kv[1]["t"]), -kv[1]["pts"]))
    for rank, (tid, s) in enumerate(ordered, 1):
        conn.execute("INSERT OR REPLACE INTO rankings VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (event_id, tid, rank, s["w"], s["l"], s["t"], 2 * s["w"] + s["t"], None, None,
                      round(s["pts"] / s["n"], 1)))
    conn.commit()
