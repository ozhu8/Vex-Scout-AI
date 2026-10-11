"""SQLite storage. Verified data (teams, matches, skills, rankings) is kept
separate from subjective scouting observations and from model predictions."""
import json
import sqlite3
from datetime import datetime, timezone

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY,            -- RobotEvents team id (negative = synthetic)
    number TEXT UNIQUE NOT NULL,
    name TEXT, organization TEXT, location TEXT, grade TEXT,
    source TEXT NOT NULL DEFAULT 'robotevents',
    fetched_at TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    sku TEXT UNIQUE NOT NULL,
    name TEXT, program TEXT, season_id INTEGER, season_name TEXT,
    start TEXT, end TEXT,
    source TEXT NOT NULL DEFAULT 'robotevents',
    fetched_at TEXT
);
CREATE TABLE IF NOT EXISTS event_teams (
    event_id INTEGER NOT NULL, team_id INTEGER NOT NULL,
    PRIMARY KEY (event_id, team_id)
);
CREATE TABLE IF NOT EXISTS matches (
    id INTEGER PRIMARY KEY,
    event_id INTEGER NOT NULL,
    division_id INTEGER,
    round INTEGER, instance INTEGER, matchnum INTEGER,
    name TEXT,
    scheduled TEXT, started TEXT,
    red1 INTEGER, red2 INTEGER, blue1 INTEGER, blue2 INTEGER,
    red_score INTEGER, blue_score INTEGER,
    scored INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'robotevents',
    fetched_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_matches_event ON matches(event_id);
CREATE TABLE IF NOT EXISTS skills (
    event_id INTEGER NOT NULL, team_id INTEGER NOT NULL,
    type TEXT NOT NULL,                -- driver | programming
    score INTEGER, attempts INTEGER, rank INTEGER,
    PRIMARY KEY (event_id, team_id, type)
);
CREATE TABLE IF NOT EXISTS rankings (
    event_id INTEGER NOT NULL, team_id INTEGER NOT NULL,
    rank INTEGER, wins INTEGER, losses INTEGER, ties INTEGER,
    wp INTEGER, ap INTEGER, sp INTEGER, average_points REAL,
    PRIMARY KEY (event_id, team_id)
);
CREATE TABLE IF NOT EXISTS scouting_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL,
    event_id INTEGER,
    raw_text TEXT NOT NULL,
    structured_json TEXT,              -- extracted attributes (see scouting.py)
    category TEXT,
    confidence REAL,                   -- 0..1, how sure the scout/extractor is
    match_ref TEXT,
    author TEXT,
    source TEXT NOT NULL DEFAULT 'human',
    status TEXT NOT NULL DEFAULT 'active',   -- active | conflicting | superseded
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS live_feedback (   -- day-of updates ("robot broken today") that adjust ratings
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    summary TEXT,
    effect TEXT NOT NULL,                  -- see feedback.EFFECTS
    duration TEXT NOT NULL,                -- today | until_removed
    expires_at TEXT,                       -- UTC; NULL = until removed
    status TEXT NOT NULL DEFAULT 'active', -- active | removed
    author TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    revisions_json TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE IF NOT EXISTS ai_reports (      -- saved Claude reports, reused until the stats move a lot
    kind TEXT NOT NULL,                    -- team | match
    event_id INTEGER NOT NULL,
    subject TEXT NOT NULL,                 -- team number, or "RED1,RED2 vs BLUE1,BLUE2"
    snapshot_json TEXT NOT NULL,           -- the numbers the report was based on
    body_json TEXT NOT NULL,
    model TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (kind, event_id, subject)
);
CREATE TABLE IF NOT EXISTS api_cache (      -- raw RobotEvents responses, to stay under the rate limit
    key TEXT PRIMARY KEY,
    fetched_at TEXT NOT NULL,
    body TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER,
    match_id INTEGER,
    red_teams TEXT NOT NULL, blue_teams TEXT NOT NULL,   -- JSON lists of team numbers
    red_score_pred REAL, blue_score_pred REAL, red_win_prob REAL,
    details_json TEXT,
    model_version TEXT,
    created_at TEXT NOT NULL,
    actual_red INTEGER, actual_blue INTEGER
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    for table, col in (("events", "venue"), ("events", "location"), ("ai_reports", "discussion_json")):  # added later
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")
        except sqlite3.OperationalError:
            pass
    return conn


def rows(conn, sql, params=()):
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def one(conn, sql, params=()):
    r = conn.execute(sql, params).fetchone()
    return dict(r) if r else None


# --- lookups -----------------------------------------------------------------

def team_by_number(conn, number: str):
    return one(conn, "SELECT * FROM teams WHERE upper(number) = upper(?)", (number.strip(),))


def team_numbers(conn, ids):
    if not ids:
        return {}
    q = ",".join("?" * len(ids))
    return {r["id"]: r["number"] for r in rows(conn, f"SELECT id, number FROM teams WHERE id IN ({q})", list(ids))}


def list_events(conn):
    return rows(conn, "SELECT * FROM events ORDER BY start DESC")


def event_teams(conn, event_id):
    return rows(conn, """
        SELECT t.*, r.rank, r.wins, r.losses, r.ties, r.wp, r.ap, r.sp, r.average_points
        FROM event_teams et JOIN teams t ON t.id = et.team_id
        LEFT JOIN rankings r ON r.event_id = et.event_id AND r.team_id = t.id
        WHERE et.event_id = ?
        ORDER BY COALESCE(r.rank, 9999), t.number""", (event_id,))


def scored_matches(conn, event_id=None, team_id=None, season_id=None):
    sql = """SELECT m.*, e.season_id, e.name AS event_name, e.sku FROM matches m
             JOIN events e ON e.id = m.event_id WHERE m.scored = 1"""
    p = []
    if event_id is not None:
        sql += " AND m.event_id = ?"; p.append(event_id)
    if season_id is not None:
        sql += " AND e.season_id = ?"; p.append(season_id)
    if team_id is not None:
        sql += " AND ? IN (m.red1, m.red2, m.blue1, m.blue2)"; p.append(team_id)
    sql += " ORDER BY COALESCE(m.started, m.scheduled, e.start), m.round, m.instance, m.matchnum, m.id"
    # RobotEvents sometimes lists the same match twice under different ids; count it once
    out, seen = [], set()
    for m in rows(conn, sql, p):
        k = (m["event_id"], m["name"], m["red1"], m["red2"], m["blue1"], m["blue2"], m["red_score"], m["blue_score"])
        if k not in seen:
            seen.add(k)
            out.append(m)
    return out


def skills_for(conn, team_id, event_id=None):
    if event_id is not None:
        return rows(conn, "SELECT * FROM skills WHERE team_id=? AND event_id=?", (team_id, event_id))
    return rows(conn, "SELECT * FROM skills WHERE team_id=?", (team_id,))


def observations_for(conn, team_id, event_id=None, include_inactive=False):
    sql = "SELECT * FROM scouting_observations WHERE team_id = ?"
    p = [team_id]
    if event_id is not None:
        sql += " AND (event_id = ? OR event_id IS NULL)"; p.append(event_id)
    if not include_inactive:
        sql += " AND status != 'superseded'"
    sql += " ORDER BY created_at DESC"
    out = rows(conn, sql, p)
    for o in out:
        o["structured"] = json.loads(o["structured_json"]) if o.get("structured_json") else {}
    return out


def insert_observation(conn, team_id, event_id, raw_text, structured, category, confidence,
                       match_ref=None, author=None, source="human", status="active"):
    cur = conn.execute(
        """INSERT INTO scouting_observations
           (team_id, event_id, raw_text, structured_json, category, confidence, match_ref, author, source, status, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (team_id, event_id, raw_text, json.dumps(structured), category, confidence,
         match_ref, author, source, status, now()))
    conn.commit()
    return cur.lastrowid


def set_observation_status(conn, obs_id, status):
    conn.execute("UPDATE scouting_observations SET status=? WHERE id=?", (status, obs_id))
    conn.commit()


def delete_observations(conn, ids):
    """Hard-delete notes; returns the deleted rows so the UI can offer undo."""
    if not ids:
        return []
    q = ",".join("?" * len(ids))
    gone = rows(conn, f"SELECT * FROM scouting_observations WHERE id IN ({q})", list(ids))
    conn.execute(f"DELETE FROM scouting_observations WHERE id IN ({q})", list(ids))
    conn.commit()
    return gone


def restore_observations(conn, saved_rows):
    for r in saved_rows:
        cols = ",".join(r)
        conn.execute(f"INSERT OR REPLACE INTO scouting_observations ({cols}) VALUES ({','.join('?' * len(r))})", list(r.values()))
    conn.commit()
