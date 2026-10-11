"""RobotEvents API v2 client and ingestion into SQLite.

Every response is cached in the api_cache table, so restarts and repeat imports
don't spend the API's rate limit; if the API fails, the last cached copy is used.

Docs: https://www.robotevents.com/api/v2
"""
import json
import time
from datetime import datetime, timezone

import requests

from . import db
from .config import ROBOTEVENTS_BASE_URL, ROBOTEVENTS_TOKEN

BASE = ROBOTEVENTS_BASE_URL


class RobotEventsError(RuntimeError):
    pass


HOUR = 3600


class RobotEvents:
    def __init__(self, token: str = ROBOTEVENTS_TOKEN, ttl: float = 6 * HOUR, conn=None):
        """ttl: seconds a cached response stays fresh. Live match data uses a short ttl."""
        if not token:
            raise RobotEventsError("ROBOTEVENTS_TOKEN is not set (add it to .env)")
        self.ttl = ttl
        self.conn = conn or db.connect()
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/json"})

    def _get(self, path, params=None):
        key = path + "?" + json.dumps(params or {}, sort_keys=True)
        hit = db.one(self.conn, "SELECT fetched_at, body FROM api_cache WHERE key=?", (key,))
        if hit:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(hit["fetched_at"])).total_seconds()
            if age < self.ttl:
                return json.loads(hit["body"])
        try:
            body = self._fetch(path, params)
        except (RobotEventsError, requests.RequestException):
            if hit:  # API down or rate limited: serve the last good copy
                return json.loads(hit["body"])
            raise
        self.conn.execute("INSERT OR REPLACE INTO api_cache VALUES (?,?,?)", (key, db.now(), json.dumps(body)))
        self.conn.commit()
        return body

    def _fetch(self, path, params=None):
        for attempt in range(5):
            r = self.s.get(BASE + path, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
                continue
            if r.status_code >= 400:
                raise RobotEventsError(f"GET {path} -> {r.status_code}: {r.text[:200]}")
            return r.json()
        raise RobotEventsError(f"GET {path}: rate limited")

    def paged(self, path, params=None):
        params = dict(params or {}, per_page=250, page=1)
        while True:
            body = self._get(path, params)
            yield from body.get("data", [])
            meta = body.get("meta", {})
            if meta.get("current_page", 1) >= meta.get("last_page", 1):
                return
            params["page"] += 1

    def event_by_sku(self, sku):
        data = self._get("/events", {"sku[]": sku}).get("data", [])
        if not data:
            raise RobotEventsError(f"No event with SKU {sku}")
        return data[0]

    def team_events(self, team_id, season=None):
        return list(self.paged(f"/teams/{team_id}/events", {"season[]": season} if season else None))

    def search_events(self, name=None, season=None, start=None, end=None):
        params = {}
        if season: params["season[]"] = season
        if start: params["start"] = start
        if end: params["end"] = end
        events = list(self.paged("/events", params))
        if name:
            events = [e for e in events if name.lower() in e["name"].lower()]
        return events


# --- ingestion ---------------------------------------------------------------

def _loc(t):
    loc = t.get("location") or {}
    return ", ".join(x for x in (loc.get("city"), loc.get("region"), loc.get("country")) if x)


def _upsert_team(conn, t):
    conn.execute("""INSERT INTO teams (id, number, name, organization, location, grade, source, fetched_at)
                    VALUES (?,?,?,?,?,?, 'robotevents', ?)
                    ON CONFLICT(id) DO UPDATE SET number=excluded.number, name=excluded.name,
                      organization=excluded.organization, location=excluded.location,
                      grade=excluded.grade, fetched_at=excluded.fetched_at""",
                 (t["id"], t["number"], t.get("team_name"), t.get("organization"), _loc(t), t.get("grade"), db.now()))


def _upsert_event(conn, e):
    conn.execute("""INSERT INTO events (id, sku, name, program, season_id, season_name, start, end, venue, location, source, fetched_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?, 'robotevents', ?)
                    ON CONFLICT(id) DO UPDATE SET name=excluded.name, program=COALESCE(excluded.program, events.program),
                      season_id=COALESCE(excluded.season_id, events.season_id),
                      season_name=COALESCE(excluded.season_name, events.season_name),
                      start=COALESCE(excluded.start, events.start), end=COALESCE(excluded.end, events.end),
                      venue=COALESCE(excluded.venue, events.venue), location=COALESCE(excluded.location, events.location),
                      fetched_at=excluded.fetched_at""",
                 (e["id"], e.get("sku") or e.get("code"), e.get("name"),
                  (e.get("program") or {}).get("code"), (e.get("season") or {}).get("id"),
                  (e.get("season") or {}).get("name"), e.get("start"), e.get("end"),
                  (e.get("location") or {}).get("venue"), _loc(e), db.now()))


def _upsert_match(conn, m, event_id):
    red = next((a for a in m["alliances"] if a["color"] == "red"), {"teams": [], "score": None})
    blue = next((a for a in m["alliances"] if a["color"] == "blue"), {"teams": [], "score": None})

    def ids(a):
        t = [x["team"]["id"] for x in a.get("teams", []) if x.get("team")]
        return (t + [None, None])[:2]

    r1, r2 = ids(red)
    b1, b2 = ids(blue)
    # RobotEvents reports unplayed matches with 0-0 scores; trust `scored` when present.
    scored = bool(m.get("scored")) or (m.get("started") is not None and (red.get("score") or blue.get("score")))
    conn.execute("""INSERT INTO matches (id, event_id, division_id, round, instance, matchnum, name, scheduled, started,
                      red1, red2, blue1, blue2, red_score, blue_score, scored, source, fetched_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'robotevents', ?)
                    ON CONFLICT(id) DO UPDATE SET scheduled=excluded.scheduled, started=excluded.started,
                      red_score=excluded.red_score, blue_score=excluded.blue_score, scored=excluded.scored,
                      fetched_at=excluded.fetched_at""",
                 (m["id"], event_id, (m.get("division") or {}).get("id"), m.get("round"), m.get("instance"),
                  m.get("matchnum"), m.get("name"), m.get("scheduled"), m.get("started"),
                  r1, r2, b1, b2, red.get("score"), blue.get("score"), int(bool(scored)), db.now()))


def ingest_event(sku: str, with_history: bool = True, log=print):
    """Pull an event's teams, matches, rankings and skills; optionally each team's
    other matches from the same season (for pre-event intelligence)."""
    api = RobotEvents()
    conn = api.conn  # one connection, so cache writes and ingest writes don't lock each other
    ev = api.event_by_sku(sku)
    _upsert_event(conn, ev)
    eid = ev["id"]
    log(f"Event {ev['name']} ({sku}), id {eid}")

    teams = list(api.paged(f"/events/{eid}/teams"))
    for t in teams:
        _upsert_team(conn, t)
        conn.execute("INSERT OR IGNORE INTO event_teams VALUES (?,?)", (eid, t["id"]))
    log(f"  {len(teams)} teams")

    n = 0
    for div in ev.get("divisions") or []:
        for m in api.paged(f"/events/{eid}/divisions/{div['id']}/matches"):
            _upsert_match(conn, m, eid); n += 1
        for r in api.paged(f"/events/{eid}/divisions/{div['id']}/rankings"):
            conn.execute("""INSERT OR REPLACE INTO rankings VALUES (?,?,?,?,?,?,?,?,?,?)""",
                         (eid, r["team"]["id"], r.get("rank"), r.get("wins"), r.get("losses"), r.get("ties"),
                          r.get("wp"), r.get("ap"), r.get("sp"), r.get("average_points")))
    log(f"  {n} matches")

    for s in api.paged(f"/events/{eid}/skills"):
        conn.execute("INSERT OR REPLACE INTO skills VALUES (?,?,?,?,?,?)",
                     (eid, s["team"]["id"], s.get("type"), s.get("score"), s.get("attempts"), s.get("rank")))
    conn.commit()

    season = (ev.get("season") or {}).get("id")
    if with_history and season:
        for i, t in enumerate(teams, 1):
            try:
                hist = list(api.paged(f"/teams/{t['id']}/matches", {"season[]": season}))
            except RobotEventsError as e:
                log(f"  history {t['number']}: {e}")
                continue
            for m in hist:
                other = m.get("event") or {}
                if other.get("id") and other["id"] != eid:
                    conn.execute("""INSERT OR IGNORE INTO events (id, sku, name, season_id, source, fetched_at)
                                    VALUES (?,?,?,?, 'robotevents', ?)""",
                                 (other["id"], other.get("code") or f"EV-{other['id']}", other.get("name"), season, db.now()))
                    _upsert_match(conn, m, other["id"])
            try:  # best skills runs elsewhere this season feed the auton/driver ratings
                for sk in api.paged(f"/teams/{t['id']}/skills", {"season[]": season}):
                    other = sk.get("event") or {}
                    if not other.get("id") or other["id"] == eid:
                        continue
                    conn.execute("""INSERT OR IGNORE INTO events (id, sku, name, season_id, source, fetched_at)
                                    VALUES (?,?,?,?, 'robotevents', ?)""",
                                 (other["id"], other.get("code") or f"EV-{other['id']}", other.get("name"), season, db.now()))
                    conn.execute("INSERT OR REPLACE INTO skills VALUES (?,?,?,?,?,?)",
                                 (other["id"], t["id"], sk.get("type"), sk.get("score"), sk.get("attempts"), sk.get("rank")))
            except RobotEventsError as e:
                log(f"  skills {t['number']}: {e}")
            if i % 10 == 0:
                conn.commit()
                log(f"  season history: {i}/{len(teams)} teams")
        conn.commit()
    return eid


def refresh_event_matches(event_id: int, min_interval: float = 60):
    """Re-pull just matches + rankings for a live event. Responses younger than
    min_interval seconds are reused, so repeated clicks don't hit the API."""
    api = RobotEvents(ttl=min_interval)
    conn = api.conn
    ev = api._get(f"/events/{event_id}")
    _upsert_event(conn, ev)
    for div in ev.get("divisions") or []:
        for m in api.paged(f"/events/{event_id}/divisions/{div['id']}/matches"):
            _upsert_match(conn, m, event_id)
        for r in api.paged(f"/events/{event_id}/divisions/{div['id']}/rankings"):
            conn.execute("INSERT OR REPLACE INTO rankings VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (event_id, r["team"]["id"], r.get("rank"), r.get("wins"), r.get("losses"), r.get("ties"),
                          r.get("wp"), r.get("ap"), r.get("sp"), r.get("average_points")))
    for s in api.paged(f"/events/{event_id}/skills"):
        conn.execute("INSERT OR REPLACE INTO skills VALUES (?,?,?,?,?,?)",
                     (event_id, s["team"]["id"], s.get("type"), s.get("score"), s.get("attempts"), s.get("rank")))
    conn.commit()


# --- finding events ------------------------------------------------------------

def _event_row(e):
    loc = e.get("location") or {}
    coords = loc.get("coordinates") or {}
    return dict(id=e["id"], sku=e["sku"], name=e["name"], start=(e.get("start") or "")[:10],
                city=loc.get("city"), region=loc.get("region"), country=loc.get("country"),
                venue=loc.get("venue"), lat=coords.get("lat"), lon=coords.get("lon"), level=e.get("level"))


def season_events(season_id, ttl=6 * 3600):
    """Every event in a season (a few API calls, then served from the cache)."""
    return [_event_row(e) for e in RobotEvents(ttl=ttl).paged("/events", {"season[]": season_id})]


def team_info(team_id, ttl=24 * 3600):
    t = RobotEvents(ttl=ttl)._get(f"/teams/{team_id}")
    loc = t.get("location") or {}
    c = loc.get("coordinates") or {}
    return dict(number=t["number"], name=t.get("team_name"), city=loc.get("city"), region=loc.get("region"),
                lat=c.get("lat"), lon=c.get("lon"))


def miles(lat1, lon1, lat2, lon2):
    import math
    if None in (lat1, lon1, lat2, lon2):
        return None
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(a))
