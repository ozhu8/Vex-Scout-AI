"""Machine-learned win probability: L2-regularised logistic regression on this
season's matches.

Each training example is one match, described only by what was known *before*
it was played (walk-forward), so the model never learns from the future:
    elo_gap       alliance Elo difference / 100
    opr_gap       alliance OPR-sum difference / 10
    winrate_gap   difference in (smoothed) win rate
    exp_gap       difference in log(1 + matches played)
The label is whether red won. Accuracy is checked on the latest 30% of matches,
which the model is not trained on.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from . import db
from .predictor import ELO_BASE, fit, update_elo

FEATURES = ["Elo gap", "OPR gap", "Win-rate gap", "Experience gap"]
REFIT_EVERY = 10      # refit OPR on prior matches every N matches while building the dataset
MIN_TRAIN = 40        # below this the model isn't trusted and is left out of the blend
L2 = 1.0


@dataclass
class MLModel:
    w: np.ndarray = None
    b: float = 0.0
    mu: np.ndarray = None
    sd: np.ndarray = None
    n_train: int = 0
    test: dict = field(default_factory=dict)   # held-out accuracy / Brier for ML and the stats model
    usable: bool = False


def _sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _train(X, y, l2=L2, steps=3000, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):  # plain gradient descent; the problem is small and convex
        p = _sigmoid(Z @ w + b)
        g = Z.T @ (p - y) / len(y) + l2 * w / len(y)
        w -= lr * g
        b -= lr * float(np.mean(p - y))
    return w, b, mu, sd


def _features(red, blue, elo, opr, wins, played, mean_half):
    def side(ids):
        ids = [t for t in ids if t]
        e = np.mean([elo.get(t, ELO_BASE) for t in ids]) if ids else ELO_BASE
        o = sum(opr.get(t, mean_half) for t in ids)
        wr = np.mean([(wins.get(t, 0) + 1) / (played.get(t, 0) + 2) for t in ids]) if ids else 0.5
        ex = np.mean([math.log1p(played.get(t, 0)) for t in ids]) if ids else 0
        return e, o, wr, ex
    r, b = side(red), side(blue)
    return [(r[0] - b[0]) / 100, (r[1] - b[1]) / 10, r[2] - b[2], r[3] - b[3]]


def build_dataset(matches):
    """Walk forward through the season, recording each match's pre-match features."""
    X, y, stats_p = [], [], []
    elo, wins, played = {}, {}, {}
    opr, mean_half = {}, 25.0
    for i, m in enumerate(matches):
        if i and i % REFIT_EVERY == 0:
            prior = fit(matches[:i])
            opr, mean_half = prior.opr, prior.mean_alliance / 2
        red, blue = [m["red1"], m["red2"]], [m["blue1"], m["blue2"]]
        if m["red_score"] != m["blue_score"] and i >= REFIT_EVERY:
            X.append(_features(red, blue, elo, opr, wins, played, mean_half))
            y.append(1.0 if m["red_score"] > m["blue_score"] else 0.0)
            # what the stats model's Elo half would have said, for a fair comparison
            ra = np.mean([elo.get(t, ELO_BASE) for t in red if t]); rb = np.mean([elo.get(t, ELO_BASE) for t in blue if t])
            stats_p.append(1 / (1 + 10 ** ((rb - ra) / 400)))
        update_elo(elo, m)
        rw = 1 if m["red_score"] > m["blue_score"] else 0.5 if m["red_score"] == m["blue_score"] else 0
        for t in red:
            if t: wins[t] = wins.get(t, 0) + rw; played[t] = played.get(t, 0) + 1
        for t in blue:
            if t: wins[t] = wins.get(t, 0) + 1 - rw; played[t] = played.get(t, 0) + 1
    return np.array(X, float), np.array(y, float), np.array(stats_p, float)


def train(conn, event_id):
    ev = db.one(conn, "SELECT season_id FROM events WHERE id=?", (event_id,))
    matches = db.scored_matches(conn, season_id=ev["season_id"]) if ev and ev["season_id"] is not None else []
    out = MLModel()
    if len(matches) < MIN_TRAIN:
        return out
    X, y, stats_p = build_dataset(matches)
    if len(y) < MIN_TRAIN or len(set(y)) < 2:
        return out
    # held-out check: train on the earliest 70%, test on the latest 30%
    cut = int(len(y) * 0.7)
    w, b, mu, sd = _train(X[:cut], y[:cut])
    p_test = _sigmoid(((X[cut:] - mu) / sd) @ w + b)
    yt, sp = y[cut:], stats_p[cut:]
    out.test = dict(n=int(len(yt)),
                    ml_acc=float(np.mean((p_test > .5) == (yt == 1))), ml_brier=float(np.mean((p_test - yt) ** 2)),
                    elo_acc=float(np.mean((sp > .5) == (yt == 1))), elo_brier=float(np.mean((sp - yt) ** 2)))
    # final model uses every match
    out.w, out.b, out.mu, out.sd = _train(X, y)
    out.n_train = int(len(y))
    out.usable = out.test["ml_brier"] < 0.25  # better than a coin flip on unseen matches
    return out


def predict(ml: MLModel, red, blue, model, ratings_by_id, live=None):
    """Red win probability for team ids, using today's numbers (live updates scale OPR)."""
    if ml is None or ml.w is None:
        return None
    live = live or {}
    opr = {t: model.opr.get(t, model.mean_alliance / 2) * (live[t]["mult"] if t in live else 1) for t in red + blue}
    wins = {t: (ratings_by_id[t].stats["winrate"] or 0) * ratings_by_id[t].stats["matches"]
            for t in red + blue if t in ratings_by_id}
    played = {t: ratings_by_id[t].stats["matches"] for t in red + blue if t in ratings_by_id}
    x = np.array(_features(red, blue, model.elo, opr, wins, played, model.mean_alliance / 2), float)
    return float(_sigmoid(((x - ml.mu) / ml.sd) @ ml.w + ml.b))


def weights(ml: MLModel):
    """Feature weights in standard units, for showing what the model learned."""
    if ml is None or ml.w is None:
        return []
    return sorted(zip(FEATURES, ml.w.tolist()), key=lambda kv: -abs(kv[1]))
