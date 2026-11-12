"""Three-state possession model: H, A, D, decoded per half with Viterbi.

Emissions are log-odds built from three sources of evidence; a switching cost between states
makes the decoder prefer few, long possessions -- which is what spellF1 rewards and what the
definition of possession ("controlled", loose balls stay with the last team) asks for.

Every constant below is a prior chosen from football and from the physics of the tracking, not
fitted to the ground truth (see RULES.md).
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import FPS, HALVES, rolling

H, A, D = 0, 1, 2

PARAMS = dict(
    # --- ball-proximity evidence (H vs A) ---------------------------------------------------
    control_radius=1.5,    # m: a player this close to the ball can be controlling it
    proximity_scale=1.5,   # m: evidence decays as exp(-(d - radius)/scale) beyond the radius
    gap_clip=3.0,          # m: |dA - dH| beyond this adds nothing
    proximity_weight=1.0,  # log-odds per frame at full strength (gap_clip metres of advantage)
    # --- team-shape evidence (H vs A), from the self-supervised classifier -----------------
    shape_weight_ball=0.03,     # per frame, when the ball is tracked
    shape_weight_noball=0.08,   # per frame, when it is not
    # --- dead-ball evidence (D vs alive) ----------------------------------------------------
    speed_ref=1.5,         # m/s: mean player speed at which speed says nothing
    speed_weight=2.0,      # log-odds per frame per m/s below speed_ref
    ball_fast=6.0,         # m/s: a ball moving this fast in the last 2 s means play is live
    ball_fast_weight=1.5,
    out_margin=0.5,        # m beyond the lines
    out_weight=4.0,
    tag_dead_weight=10.0,  # before a tagged kickoff
    # --- switching costs (log-prob) ---------------------------------------------------------
    switch_team=30.0,      # H <-> A
    switch_dead=12.0,      # alive <-> D
)


def proximity_evidence(dist, p):
    """Per-frame log-odds H vs A from the nearest player of each team. 0 when unknown."""
    dh, da = dist[:, 0], dist[:, 1]
    dmin = np.minimum(dh, da)
    ok = np.isfinite(dmin)
    closeness = np.zeros_like(dmin)
    closeness[ok] = np.exp(-np.maximum(dmin[ok] - p["control_radius"], 0) / p["proximity_scale"])
    gap = np.clip(np.where(ok, da - dh, 0), -p["gap_clip"], p["gap_clip"])
    gap = np.nan_to_num(gap, nan=0.0, posinf=p["gap_clip"], neginf=-p["gap_clip"])
    return p["proximity_weight"] * closeness * gap / p["gap_clip"], ok


def pseudo_labels(dist, ball_raw_present, mean_speed):
    """Frames where the tracking alone says, unambiguously, who has the ball.

    One team within 1 m of the ball, the other further than 3 m, the ball really tracked (not
    interpolated) and the players moving like live play. No human decides any of these.
    """
    dh, da = dist[:, 0], dist[:, 1]
    live = mean_speed > 1.8
    yh = ball_raw_present & live & (dh < 1.0) & (da > 3.0)
    ya = ball_raw_present & live & (da < 1.0) & (dh > 3.0)
    y = np.full(len(dh), -1, dtype=np.int8)
    y[yh] = H
    y[ya] = A
    return y


def shape_probability(shape_df, y, half):
    """P(home in possession | team shapes), cross-fitted across halves.

    Trained on one half's pseudo-labels, applied to the other, so no frame is scored by a model
    that saw its own label.
    """
    X = shape_df.to_numpy(dtype=np.float32)
    p = np.full(len(X), 0.5, dtype=np.float32)
    for test_half in (1, 2):
        train = (half != test_half) & (half > 0) & (y >= 0)
        test = half == test_half
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                             max_leaf_nodes=31, l2_regularization=1.0,
                                             random_state=0)
        clf.fit(X[train], (y[train] == H).astype(int))
        p[test] = clf.predict_proba(X[test])[:, 1]
    return p


def dead_evidence(mean_speed, ball, ball_raw_present, length, width, tag_dead, p):
    """Per-frame log-odds D vs alive."""
    s = rolling(mean_speed, 2 * FPS)
    ev = p["speed_weight"] * (p["speed_ref"] - np.nan_to_num(s, nan=p["speed_ref"]))
    bs = np.where(ball_raw_present, ball[:, 2], 0.0)
    bs = np.nan_to_num(bs, nan=0.0)
    recent_fast = rolling(bs, 2 * FPS, "max") > p["ball_fast"]
    ev -= p["ball_fast_weight"] * recent_fast
    bx, by = ball[:, 0], ball[:, 1]
    m = p["out_margin"]
    with np.errstate(invalid="ignore"):
        out = ball_raw_present & ((bx < -m) | (bx > length + m) | (by < -m) | (by > width + m))
    ev += p["out_weight"] * out
    ev += p["tag_dead_weight"] * tag_dead
    return ev


def viterbi(E, switch_team, switch_dead):
    """Max-sum decoding over 3 states. E is (n, 3) log-emissions."""
    n = len(E)
    T = np.array([[0, -switch_team, -switch_dead],
                  [-switch_team, 0, -switch_dead],
                  [-switch_dead, -switch_dead, 0]], dtype=np.float64)
    score = E[0].astype(np.float64).copy()
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(1, n):
        cand = score[:, None] + T          # from (rows) -> to (cols)
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], np.arange(3)] + E[t]
    path = np.zeros(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def decode(prox, shape_logit, ball_present, dead_ev, p, halves=HALVES):
    """Labels for every frame: 'H', 'A', 'D' on graded frames, 'X' elsewhere."""
    n = len(prox)
    w_shape = np.where(ball_present, p["shape_weight_ball"], p["shape_weight_noball"])
    pos = prox + w_shape * shape_logit           # log-odds H vs A
    E = np.stack([pos / 2, -pos / 2, dead_ev], axis=1)
    labels = np.full(n, "X", dtype="<U1")
    for _, a, b in halves:
        path = viterbi(E[a:b + 1], p["switch_team"], p["switch_dead"])
        labels[a:b + 1] = np.array(["H", "A", "D"])[path]
    return labels
