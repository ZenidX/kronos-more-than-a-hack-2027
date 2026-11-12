"""External tuning data, part 2: SkillCorner open data (github.com/SkillCorner/opendata).

Broadcast tracking of OTHER matches (A-League 2024-25), so its noise is real rather than
simulated: the ball is detected on ~60 % of frames, in gaps, like the challenge's ATD.

Labels come from SkillCorner's phases of play: each phase names the team in possession, and the
gaps between phases are stoppages. That matches the challenge's semantics (team possession that
survives passes and loose balls; dead ball as its own state) better than the per-frame
`possession.group`, which only marks frames where a player is on the ball.

Tracking is 10 fps and centred on the pitch; it is shifted to the corner origin and resampled to
25 fps so the pipeline runs unchanged. Players use SkillCorner's extrapolated positions (the
challenge's tactical camera sees nearly all players); the ball is kept only when detected.

    git clone --depth 1 https://github.com/SkillCorner/opendata external/skillcorner
"""

import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC_DIR = os.path.join(ROOT, "external", "skillcorner", "data", "matches")
SRC_FPS, FPS = 10, 25


def match_ids():
    return sorted(d for d in os.listdir(SC_DIR) if d.isdigit()) if os.path.isdir(SC_DIR) else []


def _speed(xy):
    d = np.hypot(np.diff(xy[..., 0], axis=0), np.diff(xy[..., 1], axis=0)) * FPS
    v = np.concatenate([d[:1], d], axis=0)
    s = pd.DataFrame(v.reshape(len(v), -1)).rolling(5, center=True, min_periods=1).mean()
    return s.to_numpy(dtype=np.float32).reshape(v.shape)


def build_match(mid):
    folder = os.path.join(SC_DIR, mid)
    with open(os.path.join(folder, f"{mid}_match.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    L, W = float(meta["pitch_length"]), float(meta["pitch_width"])
    home_id = meta["home_team"]["id"]
    slots = {p["id"]: i for i, p in enumerate(meta["players"])}
    team = np.array([0 if p["team_id"] == home_id else 1 for p in meta["players"]], dtype=np.int8)

    frames, periods, rows_p, rows_b = [], [], [], []
    with open(os.path.join(folder, f"{mid}_tracking_extrapolated.jsonl")) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("period") not in (1, 2):
                continue
            frames.append(r["frame"])
            periods.append(r["period"])
            pos = np.full((len(slots), 2), np.nan, dtype=np.float32)
            for p in r.get("player_data") or []:
                i = slots.get(p["player_id"])
                if i is not None:
                    pos[i] = (p["x"] + L / 2, p["y"] + W / 2)
            rows_p.append(pos)
            b = r.get("ball_data") or {}
            if b.get("x") is not None and b.get("is_detected"):
                rows_b.append((b["x"] + L / 2, b["y"] + W / 2))
            else:
                rows_b.append((np.nan, np.nan))
    frames = np.array(frames)
    periods = np.array(periods)
    P10 = np.stack(rows_p)
    B10 = np.array(rows_b, dtype=np.float32)

    # resample each period to 25 fps on the source clock
    P, B, per, src = [], [], [], []
    for h in (1, 2):
        m = periods == h
        f = frames[m]
        t_new = np.arange(f[0] * FPS / SRC_FPS, f[-1] * FPS / SRC_FPS + 1e-9) * SRC_FPS / FPS
        idx = np.clip(np.searchsorted(f, t_new), 1, len(f) - 1)
        lo, hi = f[idx - 1], f[idx]
        w = ((t_new - lo) / np.maximum(hi - lo, 1)).astype(np.float32)[:, None]
        Pm, Bm = P10[m], B10[m]
        P.append(Pm[idx - 1] * (1 - w[..., None]) + Pm[idx] * w[..., None])
        # ball: interpolate only between two detections one source frame apart
        both = (hi - lo == 1)[:, None] & ~np.isnan(Bm[idx - 1]) & ~np.isnan(Bm[idx])
        Bi = np.where(both, Bm[idx - 1] * (1 - w) + Bm[idx] * w, np.nan)
        B.append(Bi.astype(np.float32))
        per.append(np.full(len(t_new), h))
        src.append(np.rint(t_new).astype(int))
    xy = np.concatenate(P)
    ball = np.concatenate(B)
    period = np.concatenate(per)
    src_frame = np.concatenate(src)

    players = np.concatenate([xy, _speed(xy)[..., None]], axis=2)
    bs = _speed(np.where(np.isnan(ball), 0, ball)[:, None, :])[:, 0]
    ball3 = np.concatenate([ball, np.where(np.isnan(ball[:, :1]), np.nan, bs[:, None])], axis=1)

    gt = labels_from_phases(os.path.join(folder, f"{mid}_phases_of_play.csv"),
                            src_frame, period, home_id)
    halves = []
    for h in (1, 2):
        idx = np.flatnonzero(period == h)
        halves.append((h, int(idx[0]), int(idx[-1])))
    return dict(players=players, ball=ball3.astype(np.float32), team=team, gt=gt,
                halves=tuple(halves), length=L, width=W, name=f"sc{mid}")


def labels_from_phases(path, src_frame, period, home_id):
    ph = pd.read_csv(path).sort_values("frame_start")
    n = len(src_frame)
    lab = np.full(n, "X", dtype="<U1")
    for h in (1, 2):
        p = ph[ph["period"] == h]
        if p.empty:
            continue
        m = period == h
        f = src_frame[m]
        out = np.full(len(f), "D", dtype="<U1")
        out[f < p["frame_start"].min()] = "X"
        out[f > p["frame_end"].max()] = "X"
        for _, r in p.iterrows():
            sel = (f >= r["frame_start"]) & (f < r["frame_end"])
            out[sel] = "H" if r["team_in_possession_id"] == home_id else "A"
        lab[m] = out
    return lab


if __name__ == "__main__":
    import time
    from kronos.external import score
    from kronos.run import predict
    t0 = time.time()
    g = build_match(match_ids()[0])
    print(g["name"], g["players"].shape, "ball", float(np.mean(~np.isnan(g["ball"][:, 0]))),
          {k: int((g["gt"] == k).sum()) for k in "HADX"}, g["halves"], "%.0fs" % (time.time() - t0))
    pred = predict(g["players"], g["ball"], g["team"], g["length"], g["width"], g["halves"])
    r = score(g["gt"], pred)
    print("spellF1 %.4f macroF1 %.4f poss %d vs gt %d" % (r["spell_f1_mean"], r["macro_f1"],
                                                         r["n_pred_spells"], r["n_gt_spells"]))
