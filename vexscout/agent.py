"""Strategy copilot: Claude with tools over the scouting DB and the predictor.
The model must call predict_match for any numbers - it never invents probabilities."""
import json

import anthropic

from . import db, predictor, scouting
from .config import MODEL, RULES_PATH

SYSTEM = """You are VEX Scout AI, a scouting and strategy assistant for a VEX Robotics team at a tournament.

Ground every claim in tool results. Rules:
- Any score estimate or win probability must come from predict_match. Never invent numbers.
- Distinguish verified data (match results, skills, rankings) from scouting observations, which are subjective - say which is which and note confidence, hearsay, and conflicting observations.
- When evidence is thin (few matches, no scouting), say so and suggest what to scout next.
- For game-specific advice, check get_game_rules first. If the rules file doesn't cover it, say you're reasoning generally rather than from this season's rules.
- Strategy answers should be concrete: robot roles, auton coordination, scoring priorities, whether to play defense, risks and contingencies.
- Be concise; students read this between matches. Use short headers and bullets."""

TOOLS = [
    {"name": "get_team_profile",
     "description": "Team info, event ranking, skills, rating (Elo/OPR), and counts of matches and scouting notes.",
     "input_schema": {"type": "object", "properties": {"team_number": {"type": "string"}}, "required": ["team_number"]}},
    {"name": "get_recent_matches",
     "description": "A team's most recent scored matches this season (any event), newest first, with partners, opponents, and scores.",
     "input_schema": {"type": "object", "properties": {"team_number": {"type": "string"},
                                                      "limit": {"type": "integer", "default": 10}},
                      "required": ["team_number"]}},
    {"name": "get_scouting_observations",
     "description": "Human scouting notes for a team at this event, including confidence and conflict status.",
     "input_schema": {"type": "object", "properties": {"team_number": {"type": "string"}}, "required": ["team_number"]}},
    {"name": "predict_match",
     "description": "Run the statistical model for two alliances. Returns expected scores, win probability, confidence, key factors and data gaps.",
     "input_schema": {"type": "object", "properties": {
         "red_teams": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2},
         "blue_teams": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2}},
         "required": ["red_teams", "blue_teams"]}},
    {"name": "record_scout_observation",
     "description": "Save a scouting note the user reports (only when the user is telling you something they observed).",
     "input_schema": {"type": "object", "properties": {"team_number": {"type": "string"},
                                                      "observation": {"type": "string"}},
                      "required": ["team_number", "observation"]}},
    {"name": "get_game_rules",
     "description": "Search this season's game rules/scoring notes for a keyword or topic.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "list_event_teams",
     "description": "All teams at the event with rank, record, and model ratings - useful for alliance selection.",
     "input_schema": {"type": "object", "properties": {}}},
]


class Agent:
    def __init__(self, conn, event_id):
        self.conn, self.event_id = conn, event_id
        self._model = None
        self.client = anthropic.Anthropic()

    @property
    def model(self):
        if self._model is None:
            self._model = predictor.build_model(self.conn, self.event_id)
        return self._model

    # --- tools ---------------------------------------------------------------
    def _team(self, number):
        t = db.team_by_number(self.conn, number)
        if not t:
            raise ValueError(f"Team {number} not found")
        return t

    def get_team_profile(self, team_number):
        t = self._team(team_number)
        rank = db.one(self.conn, "SELECT * FROM rankings WHERE event_id=? AND team_id=?", (self.event_id, t["id"]))
        sk = db.skills_for(self.conn, t["id"], self.event_id)
        m = self.model
        return dict(team=t["number"], name=t["name"], organization=t["organization"], location=t["location"],
                    data_source=t["source"],
                    event_ranking=rank, skills={s["type"]: s["score"] for s in sk},
                    elo=round(m.elo.get(t["id"], predictor.ELO_BASE)),
                    opr_expected_points=round(m.opr.get(t["id"], m.mean_alliance / 2), 1),
                    scored_matches_this_season=m.n_matches.get(t["id"], 0),
                    scouting_notes=len(db.observations_for(self.conn, t["id"], self.event_id)))

    def get_recent_matches(self, team_number, limit=10):
        t = self._team(team_number)
        ms = db.scored_matches(self.conn, team_id=t["id"])[-limit:][::-1]
        nums = db.team_numbers(self.conn, {x for m in ms for x in (m["red1"], m["red2"], m["blue1"], m["blue2"]) if x})
        out = []
        for m in ms:
            color = "red" if t["id"] in (m["red1"], m["red2"]) else "blue"
            opp = "blue" if color == "red" else "red"
            mine, theirs = m[f"{color}_score"], m[f"{opp}_score"]
            out.append(dict(event=m["event_name"], match=m["name"], alliance=color,
                            partner=[nums.get(x) for x in (m[f"{color}1"], m[f"{color}2"]) if x and x != t["id"]],
                            opponents=[nums.get(x) for x in (m[f"{opp}1"], m[f"{opp}2"]) if x],
                            score=f"{mine}-{theirs}", result="W" if mine > theirs else "L" if mine < theirs else "T"))
        return out

    def get_scouting_observations(self, team_number):
        t = self._team(team_number)
        return [dict(id=o["id"], at=o["created_at"], text=o["raw_text"], category=o["category"],
                     confidence=o["confidence"], status=o["status"], match=o["match_ref"],
                     source=o["source"], attributes={k: v for k, v in o["structured"].items() if v not in (None, [], "")})
                for o in db.observations_for(self.conn, t["id"], self.event_id)]

    def predict_match(self, red_teams, blue_teams):
        return predictor.predict_numbers(self.conn, self.event_id, red_teams, blue_teams, model=self.model)

    def record_scout_observation(self, team_number, observation):
        return scouting.record(self.conn, observation, self.event_id, team_number, author="copilot chat")

    def get_game_rules(self, query):
        if not RULES_PATH.exists():
            return "No game rules file loaded (data/game_rules.md)."
        text = RULES_PATH.read_text()
        sections = [s for s in text.split("\n## ") if s.strip()]
        words = [w for w in query.lower().split() if len(w) > 2]
        hits = sorted(sections, key=lambda s: -sum(s.lower().count(w) for w in words))
        return "\n\n## ".join(hits[:3]) if hits else text[:3000]

    def list_event_teams(self):
        m = self.model
        return [dict(team=t["number"], rank=t["rank"], record=f"{t['wins']}-{t['losses']}-{t['ties']}" if t["wins"] is not None else None,
                     elo=round(m.elo.get(t["id"], predictor.ELO_BASE)),
                     opr=round(m.opr.get(t["id"], m.mean_alliance / 2), 1), matches=m.n_matches.get(t["id"], 0))
                for t in db.event_teams(self.conn, self.event_id)]

    def run_tool(self, name, args):
        fn = getattr(self, name, None)
        if name not in {t["name"] for t in TOOLS} or fn is None:
            raise ValueError(f"unknown tool {name}")
        return fn(**args)

    # --- loop ----------------------------------------------------------------
    def chat(self, messages, on_tool=None):
        """messages: API-format history ending with a user turn. Appends to it in
        place (keeping full content blocks) and returns the final text."""
        for _ in range(12):
            resp = self.client.beta.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=SYSTEM,
                tools=TOOLS,
                messages=messages,
                thinking={"type": "adaptive"},
                output_config={"effort": "medium"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "refusal":
                return "The model declined this request. Try rephrasing the question."
            if resp.stop_reason != "tool_use":
                return "".join(b.text for b in resp.content if b.type == "text")
            results = []
            for b in resp.content:
                if b.type != "tool_use":
                    continue
                if on_tool:
                    on_tool(b.name, b.input)
                try:
                    out = self.run_tool(b.name, b.input)
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out, default=str)})
                except Exception as e:
                    results.append({"type": "tool_result", "tool_use_id": b.id, "content": f"Error: {e}", "is_error": True})
            messages.append({"role": "user", "content": results})
        return "Stopped after too many tool calls - try a narrower question."
