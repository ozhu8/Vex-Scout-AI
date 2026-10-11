"""Live feedback: day-of updates like "80001C robot is broken today".

Claude turns a message into (team, effect, duration). The newest active update
for a team decides its current status, so "fixed now" overrides "broken".
Effects adjust ratings, predictions and picks, never the verified match data.
Every edit is kept in revisions_json; removal is a soft delete that can be undone.
"""
import json
import re
from datetime import datetime, time, timedelta, timezone
from typing import List, Literal, Optional

import anthropic
from pydantic import BaseModel, Field

from . import db
from .config import MODEL

# effect -> (label, multiplier on expected points / overall, good?)
EFFECTS = {
    "out": ("Not competing", 0.0, False),
    "broken": ("Robot broken", 0.55, False),
    "degraded": ("Having problems", 0.85, False),
    "auton": ("Auton unreliable", 0.92, False),
    "repaired": ("Fixed / back to normal", 1.0, True),
    "improved": ("Upgraded / improved", 1.10, True),
    "info": ("Note only", 1.0, None),
}
Effect = Literal["out", "broken", "degraded", "auton", "repaired", "improved", "info"]
# one-tap buttons in the UI: (button label, effect)
QUICK = [("Robot down", "broken"), ("Having problems", "degraded"), ("Auton unreliable", "auton"),
         ("Repaired", "repaired"), ("Upgraded", "improved"), ("Not competing", "out")]
DURATIONS = {"today": "Today only", "until_removed": "Until removed"}


class Update(BaseModel):
    team_number: str = Field(description="Team the update is about, exactly as listed")
    effect: Effect = Field(description="out = withdrawn/not playing; broken = can't play properly; degraded = partial "
                                       "problem (jams, slower, one mechanism down); repaired = fixed / back to normal; "
                                       "improved = upgraded or clearly playing better; info = no effect on performance")
    summary: str = Field(description="Short neutral restatement, under 12 words")
    duration: Literal["today", "until_removed"] = Field(
        description="today if the message says today/this match/right now, otherwise until_removed")


class Parsed(BaseModel):
    updates: List[Update]


def parse(text: str, team_numbers: List[str]) -> List[Update]:
    try:
        resp = anthropic.Anthropic().messages.parse(
            model=MODEL, max_tokens=4000, output_config={"effort": "low"},
            system="You read live updates from a VEX Robotics tournament and record how each named team is affected. "
                   "Only use teams from the list. One update per team mentioned.",
            messages=[{"role": "user", "content": f"Teams at this event: {', '.join(team_numbers)}\n\nUpdate: {text}"}],
            output_format=Parsed)
        if resp.stop_reason != "refusal" and resp.parsed_output:
            return [u for u in resp.parsed_output.updates if u.team_number.upper() in {t.upper() for t in team_numbers}]
    except Exception:
        pass
    return parse_fallback(text, team_numbers)


def parse_fallback(text, team_numbers):
    """Keyword version for when Claude is unreachable."""
    t = text.lower()
    teams = [n for n in team_numbers if re.search(rf"\b{re.escape(n.lower())}\b", t)]
    effect = ("out" if re.search(r"withdr|not (here|competing|playing)|dq|disqualif|no.?show", t) else
              "repaired" if re.search(r"\b(fixed|repaired|back to normal|working again)\b", t) else
              "broken" if re.search(r"\b(broke|broken|dead|died|not working|won'?t move)\b", t) else
              "degraded" if re.search(r"\b(jam|issue|problem|slow|stall|disconnect)", t) else
              "improved" if re.search(r"\b(upgrad|new (intake|mech|robot)|improv)", t) else "info")
    duration = "today" if re.search(r"\b(today|right now|this match)\b", t) else "until_removed"
    return [Update(team_number=n, effect=effect, summary=text.strip()[:80], duration=duration) for n in teams]


def _expiry(duration):
    if duration != "today":
        return None
    local_midnight = datetime.combine(datetime.now().date() + timedelta(days=1), time.min).astimezone()
    return local_midnight.astimezone(timezone.utc).isoformat(timespec="seconds")


def insert(conn, event_id, team_number, effect, duration, text, summary=None, author=None):
    team = db.team_by_number(conn, team_number)
    cur = conn.execute("""INSERT INTO live_feedback (event_id, team_id, raw_text, summary, effect, duration, expires_at,
                            author, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                       (event_id, team["id"], text, summary or text[:80], effect, duration, _expiry(duration), author,
                        db.now(), db.now()))
    conn.commit()
    return cur.lastrowid


def add(conn, event_id, text, team_numbers, author=None, effect=None, duration=None):
    """Parse and store in one step. effect/duration override what was read from the text."""
    saved = []
    for u in parse(text, team_numbers):
        eff, dur = effect or u.effect, duration or u.duration
        fid = insert(conn, event_id, u.team_number, eff, dur, text, u.summary, author)
        saved.append(dict(id=fid, team=u.team_number, effect=eff, duration=dur, summary=u.summary))
    return saved


def revise(conn, fid, text=None, effect=None, duration=None, summary=None):
    row = db.one(conn, "SELECT * FROM live_feedback WHERE id=?", (fid,))
    revs = json.loads(row["revisions_json"])
    revs.append({k: row[k] for k in ("raw_text", "summary", "effect", "duration", "updated_at")})
    text = text if text is not None else row["raw_text"]
    duration = duration or row["duration"]
    conn.execute("""UPDATE live_feedback SET raw_text=?, summary=?, effect=?, duration=?, expires_at=?, updated_at=?,
                    revisions_json=? WHERE id=?""",
                 (text, summary or row["summary"], effect or row["effect"], duration,
                  _expiry(duration) if duration != row["duration"] else row["expires_at"], db.now(), json.dumps(revs), fid))
    conn.commit()


def set_status(conn, fid, status):
    conn.execute("UPDATE live_feedback SET status=?, updated_at=? WHERE id=?", (status, db.now(), fid))
    conn.commit()


def is_expired(row):
    return bool(row["expires_at"]) and datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc)


def list_for_event(conn, event_id):
    out = db.rows(conn, """SELECT f.*, t.number FROM live_feedback f JOIN teams t ON t.id=f.team_id
                           WHERE f.event_id=? ORDER BY f.created_at DESC, f.id DESC""", (event_id,))
    current = current_status(conn, event_id, rows=out)
    for r in out:
        r["revisions"] = json.loads(r["revisions_json"])
        r["expired"] = is_expired(r)
        r["in_effect"] = current.get(r["team_id"], {}).get("id") == r["id"]
    return out


def current_status(conn, event_id, rows=None):
    """{team_id: newest active, unexpired, non-'info' update} - the one that counts."""
    rows = rows if rows is not None else db.rows(
        conn, "SELECT * FROM live_feedback WHERE event_id=? ORDER BY created_at DESC, id DESC", (event_id,))
    cur = {}
    for r in rows:  # newest first
        if r["status"] == "active" and not is_expired(r) and r["effect"] != "info" and r["team_id"] not in cur:
            label, mult, good = EFFECTS[r["effect"]]
            cur[r["team_id"]] = dict(id=r["id"], effect=r["effect"], label=label, mult=mult, good=good,
                                     summary=r["summary"], text=r["raw_text"])
    return cur
