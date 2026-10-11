"""Match prediction: Elo (win/loss strength) + ridge OPR (expected points),
with optional, explicitly-labelled scouting adjustments.

Everything here is fit only on matches passed in, so the backtest can replay
history chronologically without leakage.
"""
import json
import math
from dataclasses import dataclass, field

import numpy as np

from . import db
from .config import MODEL_VERSION

ELO_BASE, ELO_K = 1500.0, 32.0
RIDGE = 2.0                  # pull toward prior; ~2 matches worth of evidence
PRIOR_EVENT_WEIGHT = 0.6     # matches at other (earlier) events count less than this event
MIN_SIGMA = 8.0


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


@dataclass
class Model:
    elo: dict = field(default_factory=dict)
    opr: dict = field(default_factory=dict)
    n_matches: dict = field(default_factory=dict)
    sigma: float = 15.0
    mean_alliance: float = 50.0
    n_train: int = 0


def elo_expect(ra, rb):
    return 1 / (1 + 10 ** ((rb - ra) / 400))


def update_elo(elo, m):
    red = [t for t in (m["red1"], m["red2"]) if t]
    blue = [t for t in (m["blue1"], m["blue2"]) if t]
    ra = np.mean([elo.get(t, ELO_BASE) for t in red])
    rb = np.mean([elo.get(t, ELO_BASE) for t in blue])
    e = elo_expect(ra, rb)
    o = 1.0 if m["red_score"] > m["blue_score"] else 0.0 if m["red_score"] < m["blue_score"] else 0.5
    # margin-of-victory multiplier, damped so blowouts don't dominate
    mov = math.log(abs(m["red_score"] - m["blue_score"]) + 1) * 0.5 + 0.5
    d = ELO_K * mov * (o - e)
    for t in red: elo[t] = elo.get(t, ELO_BASE) + d
    for t in blue: elo[t] = elo.get(t, ELO_BASE) - d


def fit(matches, focus_event_id=None, skills_prior=None):
    """matches: chronological scored matches. skills_prior: {team_id: prior points}."""
    model = Model()
    if not matches:
        return model
    for m in matches:
        update_elo(model.elo, m)

    teams = sorted({t for m in matches for t in (m["red1"], m["red2"], m["blue1"], m["blue2"]) if t})
    idx = {t: i for i, t in enumerate(teams)}
    rows_, y, w = [], [], []
    for m in matches:
        wt = 1.0 if focus_event_id is None or m["event_id"] == focus_event_id else PRIOR_EVENT_WEIGHT
        for side in ("red", "blue"):
            r = np.zeros(len(teams))
            for t in (m[f"{side}1"], m[f"{side}2"]):
                if t:
                    r[idx[t]] = 1
                    model.n_matches[t] = model.n_matches.get(t, 0) + 1
            rows_.append(r); y.append(m[f"{side}_score"]); w.append(wt)
    A, y, w = np.array(rows_), np.array(y, float), np.array(w)
    model.mean_alliance = float(np.average(y, weights=w))
    prior = np.full(len(teams), model.mean_alliance / 2)
    for t, p in (skills_prior or {}).items():
        if t in idx:
            prior[idx[t]] = p
    AtW = A.T * w
    x = np.linalg.solve(AtW @ A + RIDGE * np.eye(len(teams)), AtW @ y + RIDGE * prior)
    model.opr = {t: float(x[idx[t]]) for t in teams}
    resid = y - A @ x
    model.sigma = max(MIN_SIGMA, float(np.sqrt(np.average(resid ** 2, weights=w))))
    model.n_train = len(matches)
    return model


def skills_prior(conn, event_id, model_for_calibration=None):
    """Cold-start prior from driver skills: linear map skills -> OPR, fit on teams
    with enough matches. Returns {} when there's not enough data to calibrate."""
    sk = {r["team_id"]: r["score"] for r in db.rows(conn, "SELECT team_id, score FROM skills WHERE event_id=? AND type='driver'", (event_id,))}
    if not model_for_calibration or len(sk) < 6:
        return {}
    pairs = [(sk[t], model_for_calibration.opr[t]) for t in sk
             if t in model_for_calibration.opr and model_for_calibration.n_matches.get(t, 0) >= 4]
    if len(pairs) < 6:
        return {}
    xs, ys = np.array(pairs, float).T
    b, a = np.polyfit(xs, ys, 1)
    return {t: float(a + b * s) for t, s in sk.items()}


# --- scouting adjustments ----------------------------------------------------

def scouting_adjustment(observations):
    """Heuristic multiplier on a team's expected points from active observations.
    Hand-set, not fitted - surfaced as such in explanations."""
    mult, notes = 1.0, []
    for o in observations:
        if o.get("status") != "active":
            continue
        s, c = o.get("structured") or {}, float(o.get("confidence") or 0.5)
        if s.get("reliability_issue"):
            mult -= 0.08 * c
            notes.append(f"reliability concern ({o['raw_text'][:60]}…)" if len(o['raw_text']) > 60 else f"reliability concern ({o['raw_text']})")
        rate = s.get("scoring_speed")
        if rate in (4, 5):
            mult += 0.04 * c
        elif rate in (1, 2):
            mult -= 0.04 * c
    return max(0.75, min(1.25, mult)), notes


# --- prediction --------------------------------------------------------------

def predict(model: Model, red, blue, obs_by_team=None, names=None, live=None):
    """red/blue: lists of team ids. live: {team_id: feedback.current_status entry}.
    Returns dict with scores, probability, factors."""
    obs_by_team = obs_by_team or {}
    live = live or {}
    names = names or {}
    factors, gaps = [], []

    def side(ids):
        total, parts = 0.0, []
        for t in ids:
            base = model.opr.get(t, model.mean_alliance / 2)
            mult, notes = scouting_adjustment(obs_by_team.get(t, []))
            if t in live:
                mult *= live[t]["mult"]
                notes = notes + [f"live: {live[t]['label'].lower()} ({live[t]['summary']})"]
            total += base * mult
            n = model.n_matches.get(t, 0)
            parts.append(dict(team=names.get(t, t), opr=round(base, 1), adj=round(mult, 2),
                              elo=round(model.elo.get(t, ELO_BASE)), matches=n, notes=notes))
            if n < 3:
                gaps.append(f"{names.get(t, t)} has only {n} scored matches - estimate leans on the prior")
        return total, parts

    rs, rparts = side(red)
    bs, bparts = side(blue)
    min_n = min([model.n_matches.get(t, 0) for t in red + blue] or [0])
    # inflate spread when evidence is thin
    thin = 1 + 1.5 / (1 + min_n)
    sd = math.sqrt(2) * model.sigma * thin
    p_score = _phi((rs - bs) / sd)
    ra = np.mean([model.elo.get(t, ELO_BASE) for t in red])
    rb = np.mean([model.elo.get(t, ELO_BASE) for t in blue])
    p_elo = elo_expect(ra, rb)
    p = 0.65 * p_score + 0.35 * p_elo

    factors.append(f"Expected points (OPR sum{', scouting-adjusted' if any(x['adj'] != 1 for x in rparts + bparts) else ''}): red {rs:.0f} vs blue {bs:.0f}")
    factors.append(f"Alliance Elo: red {ra:.0f} vs blue {rb:.0f} (→ {p_elo:.0%} red by Elo alone)")
    for t in red + blue:
        if t in live:
            factors.append(f"Live update - {names.get(t, t)}: {live[t]['label']} (\"{live[t]['text'][:80]}\") → expected points ×{live[t]['mult']}")
    for part in rparts + bparts:
        if part["adj"] != 1 and not any(n.startswith("live:") for n in part["notes"]):
            factors.append(f"{part['team']}: scouting multiplier ×{part['adj']} (heuristic) - " + "; ".join(part["notes"] or ["scoring-speed rating"]))
    if abs(p_score - p_elo) > 0.2:
        factors.append(f"Points model ({p_score:.0%}) and Elo ({p_elo:.0%}) disagree - treat as low confidence")

    margin = abs(p - 0.5)
    conf = "low" if (min_n < 3 or abs(p_score - p_elo) > 0.2) else "high" if (min_n >= 6 and margin > 0.25) else "medium"
    return dict(
        red_score=round(rs, 1), blue_score=round(bs, 1),
        red_win_prob=round(p, 3), blue_win_prob=round(1 - p, 3),
        score_sd=round(sd / math.sqrt(2), 1),
        p_points_model=round(p_score, 3), p_elo=round(p_elo, 3),
        confidence=conf, factors=factors, data_gaps=gaps,
        teams=dict(red=rparts, blue=bparts),
        model_version=MODEL_VERSION, trained_on_matches=model.n_train,
    )


# --- convenience wrappers used by the app and agent --------------------------

def build_model(conn, event_id):
    ev = db.one(conn, "SELECT season_id FROM events WHERE id=?", (event_id,))
    matches = db.scored_matches(conn, season_id=ev["season_id"]) if ev and ev["season_id"] is not None \
        else db.scored_matches(conn, event_id=event_id)
    base = fit(matches, focus_event_id=event_id)
    prior = skills_prior(conn, event_id, base)
    return fit(matches, focus_event_id=event_id, skills_prior=prior) if prior else base


def predict_numbers(conn, event_id, red_numbers, blue_numbers, model=None, save=True, match_id=None):
    resolve = {}
    for n in red_numbers + blue_numbers:
        t = db.team_by_number(conn, n)
        if not t:
            raise ValueError(f"Unknown team {n}")
        resolve[n] = t["id"]
    red = [resolve[n] for n in red_numbers]
    blue = [resolve[n] for n in blue_numbers]
    model = model or build_model(conn, event_id)
    obs = {t: db.observations_for(conn, t, event_id) for t in red + blue}
    names = {v: k for k, v in resolve.items()}
    from . import feedback
    out = predict(model, red, blue, obs, names, live=feedback.current_status(conn, event_id))
    if save:
        cur = conn.execute("""INSERT INTO predictions (event_id, match_id, red_teams, blue_teams, red_score_pred,
                                blue_score_pred, red_win_prob, details_json, model_version, created_at)
                              VALUES (?,?,?,?,?,?,?,?,?,?)""",
                           (event_id, match_id, json.dumps(red_numbers), json.dumps(blue_numbers), out["red_score"],
                            out["blue_score"], out["red_win_prob"], json.dumps(out), MODEL_VERSION, db.now()))
        conn.commit()
        out["prediction_id"] = cur.lastrowid
    return out


def backtest(conn, event_id, warmup=8):
    """Replay the event chronologically. For each match, fit only on matches that
    finished before it (season history + earlier matches at this event)."""
    ev = db.one(conn, "SELECT season_id FROM events WHERE id=?", (event_id,))
    season = db.scored_matches(conn, season_id=ev["season_id"]) if ev["season_id"] is not None else []
    history = [m for m in season if m["event_id"] != event_id]
    event_matches = db.scored_matches(conn, event_id=event_id)
    results = []
    for i, m in enumerate(event_matches):
        train = history + event_matches[:i]
        if len(train) < warmup:
            continue
        model = fit(train, focus_event_id=event_id)
        p = predict(model, [m["red1"], m["red2"]], [m["blue1"], m["blue2"]])
        actual = 1.0 if m["red_score"] > m["blue_score"] else 0.0 if m["red_score"] < m["blue_score"] else 0.5
        results.append(dict(match=m["name"], p_red=p["red_win_prob"], actual=actual,
                            pred_red=p["red_score"], pred_blue=p["blue_score"],
                            red=m["red_score"], blue=m["blue_score"]))
    if not results:
        return dict(n=0)
    p = np.array([r["p_red"] for r in results]); a = np.array([r["actual"] for r in results])
    decided = a != 0.5
    acc = float(np.mean((p[decided] > 0.5) == (a[decided] == 1))) if decided.any() else float("nan")
    brier = float(np.mean((p - a) ** 2))
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    logloss = float(-np.mean(a * np.log(pc) + (1 - a) * np.log(1 - pc)))
    mae = float(np.mean([abs(r["pred_red"] - r["red"]) + abs(r["pred_blue"] - r["blue"]) for r in results]) / 2)
    bins = []
    for lo in np.arange(0, 1, 0.2):
        sel = (p >= lo) & (p < lo + 0.2 + (1e-9 if lo >= 0.8 else 0))
        if sel.any():
            bins.append(dict(bin=f"{lo:.0%}-{lo + 0.2:.0%}", n=int(sel.sum()),
                             predicted=round(float(p[sel].mean()), 3), observed=round(float(a[sel].mean()), 3)))
    return dict(n=len(results), accuracy=round(acc, 3), brier=round(brier, 3), log_loss=round(logloss, 3),
                score_mae=round(mae, 1), baseline_brier=0.25, calibration=bins, matches=results)
