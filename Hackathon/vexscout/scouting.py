"""Turn free-text scouting notes into structured attributes with Claude
(structured outputs), with a keyword fallback when the API is unavailable."""
import re
from typing import List, Literal, Optional

import anthropic
from pydantic import BaseModel, Field

from . import db
from .config import MODEL

POS_TAGS = ["Strong offense", "Scores quickly", "Fast robot", "Skilled driver", "Reliable autonomous",
            "Mechanically reliable", "Good defense", "Strong drivetrain / hard to push", "Strong endgame", "Good teamwork"]
NEG_TAGS = ["Weak offense", "Scores slowly", "Slow robot", "Inexperienced driver", "Unreliable autonomous",
            "Breaks down", "Weak defense", "Gets pushed around", "Weak endgame", "Poor teamwork"]
PosTag = Literal[tuple(POS_TAGS)]
NegTag = Literal[tuple(NEG_TAGS)]

Category = Literal["autonomous", "driver", "defense", "reliability", "design", "strategy", "match_result", "other"]


class AutonAttempts(BaseModel):
    side: Literal["left", "right"]
    worked: int
    tried: int


class Observation(BaseModel):
    team_number: str = Field(description="VEX team number the note is about, e.g. '1234A'")
    category: Category
    summary: str = Field(description="One-sentence neutral restatement of the observation")
    auton_points: Optional[int] = Field(None, description="Points scored in autonomous, if stated")
    auton_consistent: Optional[bool] = None
    scoring_speed: Optional[int] = Field(None, description="1 (slow) to 5 (very fast), only if the note supports it")
    defense_capable: Optional[bool] = None
    reliability_issue: Optional[bool] = Field(None, description="True if a jam, disconnect, tip, or breakdown was seen")
    mechanisms: List[str] = Field(default_factory=list, description="Robot mechanisms mentioned (intake, lift, etc.)")
    match_ref: Optional[str] = Field(None, description="Match the note refers to, e.g. 'Q12'")
    tags_pos: List[PosTag] = Field(default_factory=list, description="Strength tags the note clearly supports")
    tags_neg: List[NegTag] = Field(default_factory=list, description="Weakness tags the note clearly supports")
    auton_attempts: List[AutonAttempts] = Field(default_factory=list, description="Only when the note gives counts, e.g. '9/10 on left side'")
    confidence: float = Field(description="0-1: how directly observed and specific the note is. Hearsay or vague ≈0.3, directly watched with specifics ≈0.8")
    is_hearsay: bool = Field(description="True if the scout did not see it themselves")


class Extraction(BaseModel):
    observations: List[Observation]


SYSTEM = """You convert VEX Robotics scouting notes into structured records.
Only fill a field when the note supports it; leave it null otherwise - never guess numbers.
A single note can describe several teams; emit one observation per team.
Mark second-hand information ("I heard", "someone said") as hearsay with low confidence."""


def extract(text: str, default_team: Optional[str] = None) -> Extraction:
    hint = f"\nIf no team is named, the note is about team {default_team}." if default_team else ""
    client = anthropic.Anthropic()
    resp = client.messages.parse(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": f"Scouting note:{hint}\n\n{text}"}],
        output_format=Extraction,
    )
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        raise RuntimeError("extraction declined or empty")
    return resp.parsed_output


def extract_fallback(text: str, team_number: str) -> Extraction:
    t = text.lower()
    issue = r"(jam|broke|broken|disconnect|tipp|stall|died|dead|fell)"
    negated = re.search(r"\b(no|never|without|zero)\s+(\w+\s+){0,2}" + issue, t)
    rel = None if negated else (True if re.search(r"\b" + issue, t) else None)
    if negated:
        rel = False
    cat = "reliability" if rel is not None else "autonomous" if "auton" in t else "defense" if "defen" in t else "other"
    pts = re.search(r"(\d+)\s*(?:pts|points)", t)
    return Extraction(observations=[Observation(
        team_number=team_number, category=cat, summary=text.strip()[:200],
        auton_points=int(pts.group(1)) if pts and cat == "autonomous" else None,
        reliability_issue=rel, confidence=0.4,
        is_hearsay=bool(re.search(r"\b(heard|apparently|someone said|rumou?r)", t)))])


CONFLICT_KEYS = ("reliability_issue", "auton_consistent", "defense_capable")


def record(conn, text: str, event_id: Optional[int], team_number: Optional[str] = None, author: str = None):
    """Extract, store, and flag conflicts with earlier active observations.
    Returns list of dicts describing what was saved."""
    used_llm = True
    try:
        ext = extract(text, team_number)
    except Exception as e:  # API down, no credentials, etc. - keep the app usable
        if not team_number:
            raise ValueError(f"Could not reach Claude ({e}); pick a team so the note can be saved as-is") from e
        ext, used_llm = extract_fallback(text, team_number), False

    saved = []
    for o in ext.observations:
        team = db.team_by_number(conn, o.team_number) or (db.team_by_number(conn, team_number) if team_number else None)
        if not team:
            saved.append(dict(team=o.team_number, error="team not in database"))
            continue
        structured = o.model_dump(exclude={"team_number", "confidence", "category", "match_ref"})
        # same shape as quick taps, so spoken/typed notes feed the badges and auton tally too
        structured["auton_attempts"] = {a.side: [min(a.worked, a.tried), a.tried] for a in o.auton_attempts if a.tried}
        conf = o.confidence * (0.5 if o.is_hearsay else 1.0)
        conflicts = []
        for prev in db.observations_for(conn, team["id"], event_id):
            ps = prev["structured"]
            for k in CONFLICT_KEYS:
                if structured.get(k) is not None and ps.get(k) is not None and ps[k] != structured[k]:
                    conflicts.append(dict(id=prev["id"], field=k, earlier=ps[k], now=structured[k],
                                          earlier_text=prev["raw_text"], earlier_at=prev["created_at"]))
        status = "conflicting" if conflicts else "active"
        oid = db.insert_observation(conn, team["id"], event_id, text if len(ext.observations) == 1 else o.summary,
                                    structured, o.category, round(conf, 2), o.match_ref, author,
                                    "human" if used_llm else "human (unparsed)", status)
        for c in conflicts:
            db.set_observation_status(conn, c["id"], "conflicting")
        saved.append(dict(id=oid, team=team["number"], category=o.category, summary=o.summary,
                          confidence=round(conf, 2), structured=structured, conflicts=conflicts,
                          extracted_by="claude" if used_llm else "keyword fallback"))
    return saved


# --- quick-tap scouting --------------------------------------------------------

OPPOSITE = dict(zip(POS_TAGS, NEG_TAGS)) | dict(zip(NEG_TAGS, POS_TAGS))


def record_tags(conn, event_id, team_number, pos, neg, auton=None, author=None):
    """Save one quick-tap observation. auton: {"left": [worked, tried], "right": [...]}.
    Tags are also mapped onto the fields the predictor and conflict check use."""
    team = db.team_by_number(conn, team_number)
    if not team:
        raise ValueError(f"Unknown team {team_number}")
    tags = set(pos) | set(neg)
    pick = lambda yes, no: True if yes in tags else False if no in tags else None
    auton = {k: v for k, v in (auton or {}).items() if v and v[1]}
    tried = sum(v[1] for v in auton.values())
    worked = sum(min(v[0], v[1]) for v in auton.values())
    structured = dict(
        tags_pos=list(pos), tags_neg=list(neg), auton_attempts=auton,
        reliability_issue=pick("Breaks down", "Mechanically reliable"),
        scoring_speed=5 if "Scores quickly" in tags else 1 if "Scores slowly" in tags else None,
        defense_capable=pick("Good defense", "Weak defense"),
        auton_consistent=(worked / tried >= 0.7) if tried >= 2 else pick("Reliable autonomous", "Unreliable autonomous"),
    )
    parts = [", ".join(pos), ", ".join(neg)] + [f"auton {side}: {w}/{n} worked" for side, (w, n) in auton.items()]
    text = "Tapped: " + "; ".join(p for p in parts if p)
    conflicts = []
    for prev in db.observations_for(conn, team["id"], event_id):
        for k in CONFLICT_KEYS:
            if structured.get(k) is not None and prev["structured"].get(k) is not None and prev["structured"][k] != structured[k]:
                conflicts.append(dict(id=prev["id"], field=k, earlier=prev["structured"][k], now=structured[k],
                                      earlier_text=prev["raw_text"]))
    for c in conflicts:
        db.set_observation_status(conn, c["id"], "conflicting")
    category = "reliability" if structured["reliability_issue"] is not None else "autonomous" if auton else "other"
    oid = db.insert_observation(conn, team["id"], event_id, text, structured, category, 0.8,
                                author=author, source="human (tapped)", status="conflicting" if conflicts else "active")
    return dict(id=oid, text=text, conflicts=conflicts)


def recheck_conflicts(conn, team_id, event_id):
    """After notes are deleted or restored, re-derive 'conflicting' flags from the notes that remain."""
    obs = [o for o in db.observations_for(conn, team_id, event_id) if o["status"] in ("active", "conflicting")]
    for o in obs:
        clash = any(o["structured"].get(k) is not None and p["structured"].get(k) is not None
                    and p["structured"][k] != o["structured"][k]
                    for p in obs if p["id"] != o["id"] for k in CONFLICT_KEYS)
        want = "conflicting" if clash else "active"
        if o["status"] != want:
            db.set_observation_status(conn, o["id"], want)


def delete_notes(conn, ids, event_id):
    gone = db.delete_observations(conn, ids)
    for team_id in {g["team_id"] for g in gone}:
        recheck_conflicts(conn, team_id, event_id)
    return gone
