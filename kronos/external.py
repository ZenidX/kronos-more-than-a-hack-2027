"""External tuning data: Metrica Sports' public sample games (github.com/metrica-sports/sample-data).

These are OTHER matches, with tracking and human-annotated event data, which RULES.md allows in any
amount. We turn their events into frame-level possession labels with the same semantics as the
challenge (H / A / D, loose balls stay with the last team), degrade their clean tracking so it looks
like the challenge's automatic tracking (long ball dropouts, lost players, noise), run exactly the
same pipeline, and score with the official evaluation/possession_metrics.py.

    git clone --depth 1 https://github.com/metrica-sports/sample-data external/sample-data
    python -m kronos.external                  # scores the current parameters
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as PM  # noqa: E402

SAMPLE_DIR = os.path.join(ROOT, "external", "sample-data", "data")
LENGTH, WIDTH = 105.0, 68.0
FPS = 25

# On-ball actions: the acting team has the ball when the event starts.
OWNING = {"PASS", "SET PIECE", "SHOT", "RECOVERY", "BALL LOST", "BALL OUT"}
# Actions that restart play after a stoppage.
RESTART = {"SET PIECE", "PASS", "SHOT", "RECOVERY"}
MAX_DEAD = 3 * 60 * FPS


def _read_team_csv(path):
    raw = pd.read_csv(path, skiprows=2)
    cols = list(raw.columns)
    period = raw["Period"].to_numpy()
    xy_cols = [c for c in cols[3:]]
    # columns come as PlayerN, Unnamed (y) pairs, then Ball, Unnamed (y)
    names = [c for c in xy_cols if not c.startswith("Unnamed")]
    arr = raw[xy_cols].to_numpy(dtype=np.float32)
    pairs = arr.reshape(len(raw), -1, 2)
    return period, names, pairs


def load_csv_game(k):
    folder = os.path.join(SAMPLE_DIR, f"Sample_Game_{k}")
    period, names_h, home = _read_team_csv(
        os.path.join(folder, f"Sample_Game_{k}_RawTrackingData_Home_Team.csv"))
    _, names_a, away = _read_team_csv(
        os.path.join(folder, f"Sample_Game_{k}_RawTrackingData_Away_Team.csv"))
    ball = home[:, names_h.index("Ball")]
    home = home[:, [i for i, nm in enumerate(names_h) if nm != "Ball"]]
    away = away[:, [i for i, nm in enumerate(names_a) if nm != "Ball"]]
    xy = np.concatenate([home, away], axis=1)
    team = np.array([0] * home.shape[1] + [1] * away.shape[1], dtype=np.int8)
    events = pd.read_csv(os.path.join(folder, f"Sample_Game_{k}_RawEventsData.csv"))
    return period, xy, ball, team, events


def to_metres(xy):
    out = xy.astype(np.float32).copy()
    out[..., 0] *= LENGTH
    out[..., 1] *= WIDTH
    return out


def speed(xy):
    v = np.zeros(xy.shape[:-1], dtype=np.float32)
    d = np.hypot(np.diff(xy[..., 0], axis=0), np.diff(xy[..., 1], axis=0)) * FPS
    v[1:] = d
    s = pd.DataFrame(v.reshape(len(v), -1)).rolling(5, center=True, min_periods=1).mean()
    return s.to_numpy(dtype=np.float32).reshape(v.shape)


def halves_from_period(period):
    out = []
    for h in (1, 2):
        idx = np.flatnonzero(period == h)
        if len(idx):
            out.append((h, int(idx[0]), int(idx[-1])))
    return tuple(out)


def gt_from_events(events, n, period):
    """Frame-level H/A/D/X from human event data, with the challenge's semantics."""
    ev = events.sort_values(["Start Frame", "End Frame"]).reset_index(drop=True)
    owner = np.full(n, -1, dtype=np.int8)
    change = np.full(n, -1, dtype=np.int8)
    for _, e in ev.iterrows():
        if e["Type"] in OWNING:
            f = int(e["Start Frame"]) - 1
            if 0 <= f < n:
                change[f] = 0 if e["Team"] == "Home" else 1
    cur = -1
    for i in range(n):
        if i > 0 and period[i] != period[i - 1]:
            cur = -1
        if change[i] >= 0:
            cur = change[i]
        owner[i] = cur

    dead = np.zeros(n, dtype=bool)
    starts = ev["Start Frame"].to_numpy() - 1
    for idx, e in ev.iterrows():
        sub = str(e["Subtype"]) if pd.notna(e["Subtype"]) else ""
        stop = None
        if e["Type"] == "BALL OUT":
            stop = int(e["End Frame"]) - 1
        elif e["Type"] == "SHOT" and ("-OUT" in sub or "GOAL" in sub):
            stop = int(e["End Frame"]) - 1
        elif e["Type"] == "FAULT RECEIVED":
            stop = int(e["Start Frame"]) - 1
        if stop is None:
            continue
        nxt = [int(starts[j]) for j in range(idx + 1, len(ev))
               if ev.loc[j, "Type"] in RESTART and starts[j] > stop]
        if nxt and nxt[0] - stop <= MAX_DEAD:
            dead[stop:nxt[0]] = True

    labels = np.full(n, "X", dtype="<U1")
    labels[owner == 0] = "H"
    labels[owner == 1] = "A"
    labels[dead & (owner >= 0)] = "D"
    # kickoff of each half: the first frames before the first event are dead-ball setup
    return labels


def degrade(xy, ball, rng, ball_missing_median=268, ball_present_mean=640,
            player_drop=0.08, player_noise=0.3, ball_noise=0.5):
    """Make clean tracking look like automatic tracking: long ball dropouts, lost players, noise."""
    n = len(ball)
    xy = xy + rng.normal(0, player_noise, xy.shape).astype(np.float32)
    ball = ball + rng.normal(0, ball_noise, ball.shape).astype(np.float32)
    keep = np.ones(n, dtype=bool)
    i = int(rng.exponential(ball_present_mean))
    while i < n:
        miss = int(rng.lognormal(np.log(ball_missing_median), 1.0))
        keep[i:i + miss] = False
        i += miss + int(rng.exponential(ball_present_mean))
    ball = ball.copy()
    ball[~keep] = np.nan
    # player dropouts in runs of ~2 s
    for s in range(xy.shape[1]):
        i = int(rng.exponential(50 / player_drop))
        while i < n:
            ln = int(rng.exponential(50))
            xy[i:i + ln, s] = np.nan
            i += ln + int(rng.exponential(50 / player_drop))
    return xy, ball


def build_game(k, seed=0, clean=False):
    period, xy, ball, team, events = load_csv_game(k)
    xy, ball = to_metres(xy), to_metres(ball)
    n = len(ball)
    gt = gt_from_events(events, n, period)
    # speeds from the clean positions: the provider's ATD speeds are already smoothed, so
    # differentiating noisy positions would invent 13 m/s jogs
    ps = speed(xy)
    bs = speed(ball[:, None, :])[:, 0]
    if not clean:
        xy, ball = degrade(xy, ball, np.random.default_rng(1000 * k + seed))
        ps = np.where(np.isnan(xy[..., 0]), np.nan, ps)
        bs = np.where(np.isnan(ball[:, 0]), np.nan, bs)
    players = np.concatenate([xy, ps[..., None]], axis=2)
    ball3 = np.concatenate([ball, bs[:, None]], axis=1)
    return dict(players=players, ball=ball3, team=team, gt=gt,
                halves=halves_from_period(period), name=f"game{k}" + ("" if clean else f"_s{seed}"))


def score(gt, pred):
    return PM.evaluate(PM.label_codes(gt), PM.label_codes(pred))


def evaluate_params(games, params=None):
    from kronos.run import predict
    out = []
    for g in games:
        pred = predict(g["players"], g["ball"], g["team"], LENGTH, WIDTH, g["halves"],
                       params=params)
        r = score(g["gt"], pred)
        out.append((g["name"], r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
    return out


if __name__ == "__main__":
    games = [build_game(1), build_game(2)]
    for g in games:
        lab = g["gt"]
        print(g["name"], {k: int((lab == k).sum()) for k in "HADX"}, g["halves"])
    for row in evaluate_params(games):
        print("%-10s spellF1 %.4f  macroF1 %.4f  poss %d vs gt %d" % row)
