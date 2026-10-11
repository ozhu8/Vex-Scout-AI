# VEX Scout AI

AI scouting and strategy assistant for VEX Robotics: RobotEvents data → SQLite → Elo + OPR predictor → Claude agent → Streamlit UI.

## Quick start

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
# edit .env: add ROBOTEVENTS_TOKEN and ANTHROPIC_API_KEY (or run `ant auth login`)
.venv/bin/python scripts/ingest.py demo              # offline synthetic event
.venv/bin/python scripts/ingest.py RE-V5RC-26-XXXX   # real event (+ each team's season history)
.venv/bin/streamlit run app.py
```

## Layout

| File | What it does |
|---|---|
| `vexscout/db.py` | SQLite schema. Verified data (teams, matches, skills, rankings), scouting observations, and predictions live in separate tables. |
| `vexscout/robotevents.py` | RobotEvents API v2 client, event ingest, live refresh. |
| `vexscout/demo_data.py` | Synthetic event (negative ids, `source='synthetic'`) so the demo works with no network. |
| `vexscout/predictor.py` | Margin-aware Elo + ridge OPR (expected points), skills-based cold-start prior, heuristic scouting multipliers, chronological backtest. |
| `vexscout/scouting.py` | Claude structured-output extraction of free-text notes; hearsay discount; conflict flagging; keyword fallback offline. |
| `vexscout/agent.py` | Strategy copilot: Claude tool loop over team profiles, matches, observations, `predict_match`, rules search. |
| `data/game_rules.md` | **Fill this in** with the current season's scoring/rules. The copilot searches it. |
| `app.py` | Streamlit: Dashboard, Team scouting, Match predictor (+ backtest), Strategy copilot. |

## Model

- **Win probability** = 0.65 × P(points model) + 0.35 × P(Elo). The points model is a normal approximation on the OPR score difference, with spread widened when teams have few matches.
- **Scouting adjustments** are hand-set multipliers (reliability issues −8%×confidence; scoring-speed rating ±4%), capped at ±25%. They're labelled as heuristic in every explanation. Conflicting or outdated notes are excluded.
- **Backtest**: each match is predicted using only matches that finished before it. Synthetic demo: 71% winner accuracy, Brier 0.193 (coin flip = 0.25).

## Offline behavior

With no RobotEvents token, use the demo event. With no Claude access, notes are saved via keyword extraction and the dashboard, predictor and backtest still work. Only the copilot chat needs Claude.
