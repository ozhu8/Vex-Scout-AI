"""Claude-written analysis on top of the verified stats: a scouting report per
team and a match read for two alliances. Structured outputs, so the UI gets
fields it can lay out rather than free text.

Claude sees the same numbers the dashboard shows (ratings, percentiles, recent
matches, scouting notes, the statistical prediction) and is told to stay
grounded in them - its match call is shown next to the model's, not instead of it.
"""
import json
from typing import List, Literal, Optional

import anthropic
from pydantic import BaseModel, Field

from . import db, ratings
from .config import GEMINI_API_KEY, GEMINI_MODEL, MODEL, OPENAI_API_KEY, OPENAI_MODEL, RULES_PATH


class Claim(BaseModel):
    point: str = Field(description="The claim, under 20 words")
    evidence: str = Field(description="The specific numbers or notes behind it")
    match_refs: List[str] = Field(default_factory=list,
                                  description="Refs (e.g. 'M3') of the listed matches that support it; [] if it rests on aggregates")


class TeamReport(BaseModel):
    strongest_capability: Claim = Field(description="The single thing this team does best, per the data")
    biggest_uncertainty: Claim = Field(description="What the data can't tell us yet, or the weakest-supported part of the read")
    recommended_action: str = Field(description="One concrete thing the reader should do about this team, under 30 words")
    summary: str = Field(description="2 sentences max")
    strengths: List[Claim] = Field(description="Up to 3")
    weaknesses: List[Claim] = Field(description="Up to 3")
    how_to_beat: List[str] = Field(description="Up to 3 concrete tactics against this team")
    as_partner: List[str] = Field(description="Up to 3 notes on working with this team: roles, auton split, risks")
    scout_next: List[str] = Field(description="Up to 3 things to watch for to fill gaps in the data")
    certainty: Literal["low", "medium", "high"] = Field(
        description="How sure you are of this read overall, considering sample size, recency and conflicting notes")


class MatchAnalysis(BaseModel):
    decisive_factor: str = Field(description="The one factor most likely to decide this match, grounded in the data, under 30 words")
    biggest_uncertainty: str = Field(description="What could most change the outcome or what the data can't tell us, under 30 words")
    recommended_action: str = Field(description="The one thing the reader's alliance should do, under 30 words")
    predicted_winner: Literal["red", "blue", "toss-up"]
    red_win_prob: float = Field(description="0-1. Start from the statistical model and move it only for evidence you can name")
    red_score: int
    blue_score: int
    vs_model: str = Field(description="One sentence: agree with the statistical model, or why you moved from it")
    reasoning: List[Claim] = Field(description="Up to 4 factors behind the call")
    red_game_plan: List[str] = Field(description="Up to 3")
    blue_game_plan: List[str] = Field(description="Up to 3")
    upset_risks: List[str] = Field(description="Up to 3")
    certainty: Literal["low", "medium", "high"]


SYSTEM = """You are a VEX V5 Robotics Competition scout and strategist helping a student team at a tournament.

You get verified data (match results, OPR/DPR/CCWM, Elo, skills scores, percentiles against the other teams at this event) and subjective scouting notes. Base every claim on that data and cite the numbers. Say when the evidence is thin - teams with few season matches, no skills runs, or no notes - and lower your confidence instead of guessing. Scouting notes are subjective: weigh them by their confidence, and treat hearsay or conflicting notes cautiously.

Glossary: OPR = points a team adds to its alliance per match; DPR = points its opponents score; CCWM = OPR - DPR (effect on margin); Swing = std. dev. of its alliance scores; Auton/Driver = best programming/driver skills scores this season; percentiles are vs. teams at this event (50 = middle); ml_red_win_prob = a logistic-regression model's red win chance, trained on this season's matches.

A live_update is a same-day report from the user's team (e.g. a robot broke) - it overrides what the season data suggests about current performance, and the overall/expected numbers you see already include its multiplier.

Grounding rules:
- When a claim rests on particular matches, put their refs (e.g. "M3", "80001C-M2") in match_refs. Only use refs that appear in the data.
- Score swing (std. dev.) describes how spread out a team's scores are. It does not tell you when or how often a bad match will happen - never claim a poor match is "due", "likely within the next few", or similar.
- Don't extrapolate trends from fewer than 4 matches.
- Use the ranking wording exactly as given in rank_text; don't restate it as "#X of N".
- "certainty" is your confidence in the read, which is separate from data_confidence (how many matches back the numbers). If they differ, say why in biggest_uncertainty.
- Be concise: lists have at most 3 items, each a single sentence.

Use the game rules notes for game-specific tactics. If they don't cover something, give general V5RC advice and don't invent scoring values. Write briefly and concretely: robot roles, auton coordination, scoring priorities, defense decisions."""


def _client():
    return anthropic.Anthropic()


def _call(user: str, schema, system=None):
    resp = _client().beta.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=system or SYSTEM,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": user}],
        output_format=schema,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("Claude declined this request")
    if resp.stop_reason == "max_tokens" or resp.parsed_output is None:
        raise RuntimeError("Claude's answer was cut off or empty - try again")
    return resp.parsed_output


def _rules():
    try:
        return RULES_PATH.read_text()
    except OSError:
        return "(no rules file)"


def ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def rank_text(r, n_rated, n_registered):
    if r.rank is None:
        return f"not rated yet ({n_rated} of {n_registered} registered teams have enough data)"
    return f"{ordinal(r.rank)} among {n_rated} rated teams · {n_registered} registered"


def team_facts(conn, event_id, r: ratings.TeamRating, n_matches=10, ref_prefix="", ranking=""):
    """Everything the dashboard knows about one team, as plain data."""
    t = db.one(conn, "SELECT * FROM teams WHERE id=?", (r.team_id,))
    s = r.stats
    ms = db.scored_matches(conn, team_id=r.team_id)[::-1][:n_matches]
    nums = db.team_numbers(conn, {x for m in ms for x in (m["red1"], m["red2"], m["blue1"], m["blue2"]) if x})
    recent = []
    for i, m in enumerate(ms, 1):
        c, o = ("red", "blue") if r.team_id in (m["red1"], m["red2"]) else ("blue", "red")
        recent.append(dict(ref=f"{ref_prefix}M{i}", event=m["event_name"], match=m["name"],
                           partner=[nums.get(x, "?") for x in (m[f"{c}1"], m[f"{c}2"]) if x and x != r.team_id],
                           opponents=[nums.get(x, "?") for x in (m[f"{o}1"], m[f"{o}2"]) if x],
                           score=f"{m[c + '_score']}-{m[o + '_score']}"))
    notes = [dict(note=o["raw_text"], category=o["category"], confidence=o["confidence"], status=o["status"],
                  match=o["match_ref"]) for o in db.observations_for(conn, r.team_id, event_id)]
    rnd = lambda v: None if v is None else round(v, 1)
    return dict(
        team=r.number, name=r.name, organization=t["organization"], location=t["location"],
        overall=r.overall, tier=r.tier, profile=r.profile, rank_text=ranking, data_confidence=r.confidence,
        season_matches=s["matches"], events_played=s["events"],
        opr=rnd(s["opr"]), dpr=rnd(s["dpr"]), ccwm=rnd(s["ccwm"]), elo=s["elo"], swing=rnd(s["swing"]),
        win_rate=None if s["winrate"] is None else round(s["winrate"], 2),
        best_auton_skills=s["auton"], best_driver_skills=s["driver"],
        percentiles={k: round(v) for k, v in r.pct.items()},
        recent_matches=recent, scouting_notes=notes,
        live_update=(dict(status=r.live["label"], message=r.live["text"], rating_multiplier=r.live["mult"])
                     if r.live else None),
    )


def team_prompt(conn, event_id, r, field_size, my=None, n_rated=None):
    facts = team_facts(conn, event_id, r, ranking=rank_text(r, n_rated or field_size, field_size))
    ask = (f"Scouting report on team {r.number} for a tournament with {field_size} registered teams.\n\n"
           f"<team_data>\n{json.dumps(facts, default=str)}\n</team_data>\n")
    if my is not None and my.team_id != r.team_id:
        ask += (f"\nThe reader is team {my.number}. Their own data, for comparing and for the partner advice:\n"
                f"<our_team>\n{json.dumps(team_facts(conn, event_id, my, 5, ref_prefix=my.number + '-'), default=str)}\n</our_team>\n")
    return ask + f"\n<game_rules>\n{_rules()}\n</game_rules>"


def match_prompt(conn, event_id, red, blue, stats_pred, R, our_number=None):
    """red/blue: lists of TeamRating. stats_pred: predictor.predict output."""
    n_rated = sum(1 for x in R.values() if x.overall is not None)
    facts = lambda r: team_facts(conn, event_id, r, 6, ref_prefix=r.number + "-", ranking=rank_text(r, n_rated, len(R)))
    data = dict(red=[facts(r) for r in red], blue=[facts(r) for r in blue])
    model = {k: stats_pred[k] for k in ("red_score", "blue_score", "red_win_prob", "score_sd", "p_points_model",
                                        "p_elo", "confidence", "factors", "data_gaps", "ml_red_win_prob") if k in stats_pred}
    ask = ("Prediction and analysis of this qualification match.\n\n"
           f"<alliances>\n{json.dumps(data, default=str)}\n</alliances>\n\n"
           f"<statistical_model>\n{json.dumps(model)}\n</statistical_model>\n\n"
           f"<game_rules>\n{_rules()}\n</game_rules>")
    if our_number:
        ask += f"\n\nThe reader is team {our_number}; make their alliance's game plan the most specific."
    return ask


def team_report(conn, event_id, r, field_size, my=None, n_rated=None):
    return _call("Write this report.\n\n" + team_prompt(conn, event_id, r, field_size, my, n_rated), TeamReport)


def match_analysis(conn, event_id, red, blue, stats_pred, R, our_number=None):
    return _call("Write this analysis.\n\n" + match_prompt(conn, event_id, red, blue, stats_pred, R, our_number), MatchAnalysis)


# --- collective intelligence: Claude x ChatGPT x Gemini ---------------------------
# No central decision maker. Every agent has the same role and the same vote:
# Round 1: each agent analyses the same data independently (no anchoring on each other).
# Round 2: each sees all drafts, challenges and concedes, and writes a REVISED analysis.
# Round 3: each agent votes on every part of every revised analysis (approval voting,
#          self-approval allowed) and gives a final win chance.
# The result is assembled by fixed rules, not by a model: a part is kept only with a
# majority of approvals; unanimous parts are consensus; split parts are listed as open
# disagreements; numbers are the median of the agents' final values.
# Agents that are missing a key or fail are dropped; with only Claude left, its draft is used.

AGENT_NAMES = {"claude": "Claude", "chatgpt": "ChatGPT", "gemini": "Gemini"}


class Challenge(BaseModel):
    to: str = Field(description="Whose draft you challenge: claude, chatgpt or gemini")
    claim: str = Field(description="The claim, briefly")
    why: str = Field(description="Why it's wrong, overstated or unsupported, citing the data")
    match_refs: List[str] = Field(default_factory=list)


class Position(BaseModel):
    endorse: List[str] = Field(description="Up to 3 points from any draft that the data clearly supports")
    challenges: List[Challenge] = Field(description="Up to 3 challenges to other drafts; [] if none")
    concessions: List[str] = Field(description="Up to 3 points where another draft changed your mind; [] if none")
    bottom_line: str = Field(description="Your revised one-sentence take")


class MatchPosition(Position):
    predicted_winner: Literal["red", "blue", "toss-up"]
    red_win_prob: float = Field(description="Your revised 0-1 estimate after reading the other drafts")


class TeamNegotiation(Position):
    revised: TeamReport = Field(description="Your full analysis, revised after reading the others")


class MatchNegotiation(MatchPosition):
    revised: MatchAnalysis = Field(description="Your full analysis, revised after reading the others")


class SlotVote(BaseModel):
    field: str = Field(description="The field being voted on, exactly as listed")
    approve: List[str] = Field(description="Agents whose version of this field you approve: any of the listed agents, "
                                           "including yourself; approve every version you think is accurate and useful")


class Ballot(BaseModel):
    slot_votes: List[SlotVote] = Field(description="One entry per listed field")
    approved_items: List[str] = Field(description="Ids of the list items you approve keeping, e.g. 'strengths:claude:1'. "
                                                  "Approve only items the data supports; at most 3 per list")
    dissent: List[str] = Field(description="Up to 2 short notes on what you still disagree with and why; [] if none")


class MatchBallot(Ballot):
    red_win_prob: float = Field(description="Your final 0-1 red win chance")


def _agent_system(name, others):
    return SYSTEM.replace(
        "You are a VEX V5 Robotics Competition scout and strategist helping a student team at a tournament.",
        f"You are {AGENT_NAMES[name]}, one of {1 + len(others)} equal analysts "
        f"({', '.join(AGENT_NAMES[o] for o in [name] + others)}) scouting for a student VEX V5 team. There is no "
        "moderator: the panel's answer comes from everyone's votes. Form your own view from the data; agree when the "
        "data supports another analyst, push back when it doesn't.")


def available_agents():
    return ["claude"] + (["chatgpt"] if OPENAI_API_KEY else []) + (["gemini"] if GEMINI_API_KEY else [])


def openai_ready():
    return bool(OPENAI_API_KEY)


def _gpt(user: str, schema, system=SYSTEM):
    from openai import OpenAI
    resp = OpenAI(api_key=OPENAI_API_KEY).responses.parse(
        model=OPENAI_MODEL, instructions=system, input=user, text_format=schema)
    if resp.output_parsed is None:
        raise RuntimeError("ChatGPT returned no structured answer")
    return resp.output_parsed


def _inline_schema(schema):
    """Pydantic JSON schema with $refs inlined and defaults removed, for Gemini's schema subset."""
    js = schema.model_json_schema()
    defs = js.pop("$defs", {})

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return walk(dict(defs[node["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in node.items() if k not in ("default", "title")}
        if isinstance(node, list):
            return [walk(x) for x in node]
        return node
    return walk(js)


def _gemini(user: str, schema, system=SYSTEM):
    from google import genai
    from google.genai import types
    resp = genai.Client(api_key=GEMINI_API_KEY).models.generate_content(
        model=GEMINI_MODEL, contents=user,
        config=types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                           response_json_schema=_inline_schema(schema)))
    if not resp.text:
        raise RuntimeError("Gemini returned no answer")
    return schema.model_validate_json(resp.text)


def _ask(agent, user, schema, system):
    if agent == "claude":
        return _call(user, schema, system)
    if agent == "chatgpt":
        return _gpt(user, schema, system)
    return _gemini(user, schema, system)


def _parallel(jobs):
    """jobs: {agent: zero-arg callable}. Returns ({agent: result}, {agent: error})."""
    from concurrent.futures import ThreadPoolExecutor
    ok, err = {}, {}
    with ThreadPoolExecutor(max_workers=len(jobs) or 1) as ex:
        futs = {a: ex.submit(f) for a, f in jobs.items()}
        for a, f in futs.items():
            try:
                ok[a] = f.result()
            except Exception as e:
                err[a] = f"{type(e).__name__}: {str(e)[:200]}"
    return ok, err


CERTAINTY = ["low", "medium", "high"]


def _median(vals):
    vals = sorted(vals)
    n = len(vals)
    return vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2


def _fields(schema):
    """Split a report schema into slot fields (one value), list fields, and the rest."""
    import typing
    slots, lists = [], []
    for name, f in schema.model_fields.items():
        if name in ("certainty", "predicted_winner", "red_win_prob", "red_score", "blue_score"):
            continue
        (lists if typing.get_origin(f.annotation) in (list, List) else slots).append(name)
    return slots, lists


def _text(v):
    return v["point"] if isinstance(v, dict) and "point" in v else str(v)


def _ballot_prompt(revised, slots, lists):
    parts = []
    for fld in slots:
        parts.append(f"FIELD {fld}:\n" + "\n".join(f"  [{a}] {_text(getattr(r, fld) if not isinstance(getattr(r, fld), BaseModel) else getattr(r, fld).model_dump())}"
                                                 for a, r in revised.items()))
    for fld in lists:
        items = []
        for a, r in revised.items():
            for i, it in enumerate(getattr(r, fld)):
                items.append(f"  {fld}:{a}:{i} - {_text(it.model_dump() if isinstance(it, BaseModel) else it)}")
        parts.append(f"LIST {fld}:\n" + ("\n".join(items) or "  (none)"))
    return "\n\n".join(parts)


def _assemble(schema, revised, ballots, kind):
    """Build the panel's report from the votes. No agent decides; the rules do."""
    agents = list(revised)
    need = len(agents) // 2 + 1                     # majority of the agents taking part
    slots, lists = _fields(schema)
    out, votes, disputes, consensus = {}, [], [], []

    for fld in slots:
        tally = {a: [v for v, b in ballots.items() for sv in b.slot_votes if sv.field == fld and a in sv.approve]
                 for a in agents}
        win = max(agents, key=lambda a: (len(tally[a]), -agents.index(a)))
        out[fld] = getattr(revised[win], fld)
        point = _text(out[fld].model_dump() if isinstance(out[fld], BaseModel) else out[fld])
        backers = tally[win]
        votes.append(dict(point=f"{fld.replace('_', ' ').capitalize()}: {point}", agree=backers,
                          disagree=[a for a in ballots if a not in backers]))
        if len(backers) == len(ballots):
            consensus.append(point)
        elif len(backers) < need:
            disputes.append(dict(issue=f"No majority on the {fld.replace('_', ' ')}",
                                 sides="; ".join(f"{AGENT_NAMES[a]}'s version backed by "
                                                 f"{', '.join(AGENT_NAMES[x] for x in tally[a]) or 'no one'}" for a in agents)))

    for fld in lists:
        cands = [(f"{fld}:{a}:{i}", it) for a in agents for i, it in enumerate(getattr(revised[a], fld))]
        scored = []
        for cid, it in cands:
            app = [v for v, b in ballots.items() if cid in b.approved_items]
            scored.append((len(app), cid, it, app))
        scored.sort(key=lambda x: -x[0])
        kept = [x for x in scored if x[0] >= need][:3]
        out[fld] = [x[2] for x in kept]
        for n, cid, it, app in kept:
            if n == len(ballots) and len(consensus) < 4:
                consensus.append(_text(it.model_dump() if isinstance(it, BaseModel) else it))
        for n, cid, it, app in scored:
            if 0 < n < need and len(disputes) < 3 and fld in ("strengths", "weaknesses", "reasoning", "upset_risks"):
                disputes.append(dict(issue=_text(it.model_dump() if isinstance(it, BaseModel) else it),
                                     sides=f"Backed by {', '.join(AGENT_NAMES[a] for a in app)}; "
                                           f"not by {', '.join(AGENT_NAMES[a] for a in ballots if a not in app)}"))

    cert = [CERTAINTY.index(r.certainty) for r in revised.values()]
    out["certainty"] = CERTAINTY[int(round(_median(cert)))]
    probs = None
    if kind == "match":
        probs = {a: min(max(getattr(b, "red_win_prob", revised[a].red_win_prob), 0.0), 1.0) for a, b in ballots.items()}
        p = _median(list(probs.values()))
        out["red_win_prob"] = round(p, 3)
        out["predicted_winner"] = "toss-up" if 0.45 <= p <= 0.55 else ("red" if p > 0.5 else "blue")
        out["red_score"] = int(round(_median([r.red_score for r in revised.values()])))
        out["blue_score"] = int(round(_median([r.blue_score for r in revised.values()])))
    return schema(**out), dict(votes=votes, consensus=consensus[:4], still_disagree=disputes[:3],
                               ai_probs=probs, ai_median=round(out["red_win_prob"], 3) if probs else None)


def _collective(base_prompt, draft_schema, negotiation_schema, ballot_schema, kind, on_stage=None):
    """Returns (final, discussion). kind: 'team' or 'match'. No moderator: the result comes from the votes."""
    stage = on_stage or (lambda m: None)
    agents = available_agents()
    disc = dict(version=4, agents={a: dict(model={"claude": MODEL, "chatgpt": OPENAI_MODEL, "gemini": GEMINI_MODEL}[a],
                                           status="pending") for a in agents})
    for a in AGENT_NAMES:
        if a not in agents:
            disc["agents"][a] = dict(status="no key", model=None)
    what = "report" if kind == "team" else "analysis"
    whats = "reports" if kind == "team" else "analyses"
    others = lambda a, pool: [o for o in pool if o != a]

    # Round 1 - independent drafts
    stage(f"Round 1 · {', '.join(AGENT_NAMES[a] for a in agents)} each writing an independent {what}…")
    drafts, err = _parallel({a: (lambda a=a: _ask(a, f"Write this {what}.\n\n" + base_prompt, draft_schema,
                                                  _agent_system(a, others(a, agents)))) for a in agents})
    for a in agents:
        disc["agents"][a]["status"] = "ok" if a in drafts else "failed"
        if a in err:
            disc["agents"][a]["error"] = err[a]
    if "claude" not in drafts and not drafts:
        raise RuntimeError(f"No agent could draft: {err}")
    disc["drafts"] = {a: d.model_dump() for a, d in drafts.items()}
    live = list(drafts)
    if len(live) == 1:
        disc["skipped"] = f"Only {AGENT_NAMES[live[0]]} was available, so this is its analysis alone."
        return drafts[live[0]], disc

    try:
        # Round 2 - negotiate and revise
        stage(f"Round 2 · agents reading each other's {whats}, challenging and revising…")
        all_drafts = "\n\n".join(f"<draft by=\"{a}\">\n{drafts[a].model_dump_json()}\n</draft>" for a in live)
        negs, err2 = _parallel({a: (lambda a=a: _ask(
            a, base_prompt + f"\n\n{all_drafts}\n\nYour own draft is the one by \"{a}\". Read the others. Endorse what "
               "the data supports, challenge what it doesn't (say whose draft), concede where another analyst is right, "
               f"and write your revised full {what}.", negotiation_schema,
            _agent_system(a, others(a, live)))) for a in live})
        for a, e in err2.items():
            disc["agents"][a].update(status="dropped in round 2", error=e)
        if len(negs) < 2:
            raise RuntimeError("fewer than two agents completed round 2")
        disc["positions"] = {a: n.model_dump(exclude={"revised"}) for a, n in negs.items()}
        revised = {a: n.revised for a, n in negs.items()}
        live = list(revised)

        # Round 3 - every agent votes; no one decides alone
        stage("Round 3 · every agent voting on every part of the revised " + whats + "…")
        slots, lists = _fields(draft_schema)
        options = _ballot_prompt(revised, slots, lists)
        vote_note = ("\n\nAlso give your final red win chance." if kind == "match" else "")
        ballots, err3 = _parallel({a: (lambda a=a: _ask(
            a, base_prompt + f"\n\nThe panel's revised {whats}, split into parts:\n\n{options}\n\nVote. For each FIELD, "
               "approve every agent's version you think is accurate and useful (yours included). For each LIST, approve "
               f"the items the data supports, at most 3 per list.{vote_note}", ballot_schema,
            _agent_system(a, others(a, live)))) for a in live})
        for a, e in err3.items():
            disc["agents"][a].update(status="dropped in round 3", error=e)
        if len(ballots) < 2:
            raise RuntimeError("fewer than two agents voted")
        disc["ballots"] = {a: b.model_dump() for a, b in ballots.items()}
        final, tallies = _assemble(draft_schema, revised, ballots, kind)
        disc.update(tallies)
        disc["rationale"] = (f"No moderator: each of the {len(ballots)} agents voted on every part. A part is kept with a "
                             f"majority ({len(revised) // 2 + 1} of {len(revised)}); numbers are the median of the agents' values.")
        disc["dissent"] = {a: b.dissent for a, b in ballots.items() if b.dissent}
        return final, disc
    except Exception as e:
        disc["error"] = f"{type(e).__name__}: {e}"
        return drafts.get("claude") or next(iter(drafts.values())), disc


# --- saved reports -------------------------------------------------------------
# A report is reused until the team's numbers move by at least one of these.
THRESHOLDS = dict(matches=3, events=1, overall=5, opr=3.0, notes=2)


def snapshot(conn, event_id, r):
    """The numbers a team report depends on."""
    s = r.stats
    notes = db.one(conn, "SELECT count(*) n FROM scouting_observations WHERE team_id=? AND (event_id=? OR event_id IS NULL) "
                         "AND status != 'superseded'", (r.team_id, event_id))["n"]
    return dict(matches=s["matches"], events=s["events"], overall=r.overall, opr=None if s["opr"] is None else round(s["opr"], 1),
                notes=notes, live=r.live["effect"] if r.live else None, auton=s["auton"], driver=s["driver"])


def changes(old, new):
    """Human-readable list of significant differences between two snapshots ([] = still current)."""
    out = []
    for k, label in (("matches", "matches"), ("events", "competitions"), ("notes", "scouting notes")):
        if (new.get(k) or 0) - (old.get(k) or 0) >= THRESHOLDS[k]:
            out.append(f"{new[k] - (old.get(k) or 0):+d} {label}")
    for k, label in (("overall", "overall"), ("opr", "OPR")):
        a, b = old.get(k), new.get(k)
        if (a is None) != (b is None) or (a is not None and abs(b - a) >= THRESHOLDS[k]):
            out.append(f"{label} {a if a is not None else '–'} → {b if b is not None else '–'}")
    if old.get("live") != new.get("live"):
        out.append("live status changed")
    for k, label in (("auton", "auton skills"), ("driver", "driver skills")):
        if old.get(k) != new.get(k):
            out.append(f"new {label} score")
    return out


def load_saved(conn, kind, event_id, subject, schema):
    row = db.one(conn, "SELECT * FROM ai_reports WHERE kind=? AND event_id=? AND subject=?", (kind, event_id, subject))
    if not row:
        return None
    try:
        return dict(report=schema.model_validate_json(row["body_json"]), snapshot=json.loads(row["snapshot_json"]),
                    created_at=row["created_at"],
                    discussion=json.loads(row["discussion_json"]) if row.get("discussion_json") else None)
    except ValueError:  # schema changed since it was saved
        return None


def save(conn, kind, event_id, subject, snap, report, discussion=None):
    conn.execute("""INSERT OR REPLACE INTO ai_reports (kind, event_id, subject, snapshot_json, body_json, model, created_at,
                    discussion_json) VALUES (?,?,?,?,?,?,?,?)""",
                 (kind, event_id, subject, json.dumps(snap), report.model_dump_json(), MODEL, db.now(),
                  json.dumps(discussion) if discussion else None))
    conn.commit()


def saved_team_report(conn, event_id, r, my=None):
    """(saved, what_changed) - saved is None if there's no report yet."""
    subject = r.number + (f"|for {my.number}" if my is not None and my.team_id != r.team_id else "")
    saved = load_saved(conn, "team", event_id, subject, TeamReport)
    return saved, (changes(saved["snapshot"], snapshot(conn, event_id, r)) if saved else [])


def new_team_report(conn, event_id, r, field_size, my=None, n_rated=None, on_stage=None):
    """Panel of agents: independent drafts, negotiation, then a vote - no moderator. Returns (report, discussion)."""
    subject = r.number + (f"|for {my.number}" if my is not None and my.team_id != r.team_id else "")
    base = team_prompt(conn, event_id, r, field_size, my, n_rated)
    rep, disc = _collective(base, TeamReport, TeamNegotiation, Ballot, "team", on_stage)
    save(conn, "team", event_id, subject, snapshot(conn, event_id, r), rep, disc)
    return rep, disc


def _match_subject(red, blue):
    return ",".join(r.number for r in red) + " vs " + ",".join(r.number for r in blue)


def _match_snapshot(conn, event_id, red, blue, stats_pred):
    return dict(teams={r.number: snapshot(conn, event_id, r) for r in red + blue},
                red_win_prob=stats_pred["red_win_prob"])


def saved_match_analysis(conn, event_id, red, blue, stats_pred):
    saved = load_saved(conn, "match", event_id, _match_subject(red, blue), MatchAnalysis)
    if not saved:
        return None, []
    new = _match_snapshot(conn, event_id, red, blue, stats_pred)
    diff = [f"{t}: {c}" for t, snap in new["teams"].items() for c in changes(saved["snapshot"]["teams"].get(t, {}), snap)]
    if abs(new["red_win_prob"] - saved["snapshot"]["red_win_prob"]) >= 0.08:
        diff.append(f"model win chance {saved['snapshot']['red_win_prob']:.0%} → {new['red_win_prob']:.0%}")
    return saved, diff


def new_match_analysis(conn, event_id, red, blue, stats_pred, R, our_number=None, on_stage=None):
    """Panel of agents, no moderator; the panel's win chance is the median of their final votes. Returns (analysis, discussion)."""
    base = match_prompt(conn, event_id, red, blue, stats_pred, R, our_number)
    rep, disc = _collective(base, MatchAnalysis, MatchNegotiation, MatchBallot, "match", on_stage)
    save(conn, "match", event_id, _match_subject(red, blue), _match_snapshot(conn, event_id, red, blue, stats_pred), rep, disc)
    return rep, disc
