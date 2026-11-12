"""Per-frame features from the ATD and the Smart Tagging.

All distances in metres, speeds in m/s, frames are video frames (= submission rows).
"""

import numpy as np
import pandas as pd

FPS = 25
HALVES = ((1, 11185, 90651), (2, 110808, 190197))  # graded blocks, from games.half_video_blocks


def rolling(x, window, fn="mean"):
    s = pd.Series(x)
    r = s.rolling(window, center=True, min_periods=1)
    return getattr(r, fn)().to_numpy()


def interpolate_ball(ball, max_gap=8):
    """Linear interpolation of x, y across gaps of at most `max_gap` frames."""
    out = ball.copy()
    for c in (0, 1):
        s = pd.Series(ball[:, c].astype(float))
        out[:, c] = s.interpolate(limit=max_gap, limit_area="inside").to_numpy()
    return out


def nearest_by_team(players, team, ball_xy):
    """(n, 2) min distance from the ball to each team's players; inf where undefined."""
    res = np.full((len(ball_xy), 2), np.inf, dtype=np.float32)
    cnt3 = np.zeros((len(ball_xy), 2), dtype=np.float32)
    for k in (0, 1):
        px, py = players[:, team == k, 0], players[:, team == k, 1]
        d = np.hypot(px - ball_xy[:, None, 0], py - ball_xy[:, None, 1])
        d = np.where(np.isnan(d), np.inf, d)
        res[:, k] = d.min(axis=1)
        cnt3[:, k] = (d < 3.0).sum(axis=1)
    return res, cnt3


def attack_direction(players, team, length, halves=HALVES):
    """+1 if home attacks towards x = length in that half, -1 otherwise. Keyed by half number.

    The team whose average position is deeper on the left defends the left goal.
    """
    out = {}
    for half, a, b in halves:
        hx = np.nanmean(players[a:b + 1, team == 0, 0])
        ax = np.nanmean(players[a:b + 1, team == 1, 0])
        out[half] = 1 if hx < ax else -1
    return out


def half_of(n, halves=HALVES):
    h = np.zeros(n, dtype=np.int8)
    for half, a, b in halves:
        h[a:b + 1] = half
    return h


def shape_features(players, team, length, width, direction, halves=HALVES):
    """Collective features of each team, in a frame where home attacks +x.

    No ball is used, so these exist on every frame the players are tracked.
    """
    n = len(players)
    half = half_of(n, halves)
    sign = np.ones(n, dtype=np.float32)
    for h, d in direction.items():
        sign[half == h] = d
    x = players[..., 0].copy()
    flip = sign < 0
    x[flip] = length - x[flip]
    y = players[..., 1]
    s = players[..., 2]
    feats = {}
    for k, name in ((0, "h"), (1, "a")):
        xs, ys, ss = x[:, team == k], y[:, team == k], s[:, team == k]
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                feats[f"{name}_cx"] = np.nanmean(xs, 1)
                feats[f"{name}_cy"] = np.nanmean(ys, 1)
                feats[f"{name}_sx"] = np.nanstd(xs, 1)
                feats[f"{name}_sy"] = np.nanstd(ys, 1)
                feats[f"{name}_maxx"] = np.nanmax(xs, 1)
                feats[f"{name}_minx"] = np.nanmin(xs, 1)
                feats[f"{name}_spd"] = np.nanmean(ss, 1)
                feats[f"{name}_n"] = (~np.isnan(xs)).sum(1).astype(np.float32)
    df = pd.DataFrame(feats)
    df = df.apply(lambda c: rolling(c.to_numpy(), FPS))
    for name in ("h", "a"):
        df[f"{name}_vx"] = np.gradient(rolling(df[f"{name}_cx"].to_numpy(), 2 * FPS)) * FPS
    df["d_cx"] = df["h_cx"] - df["a_cx"]
    df["d_sx"] = df["h_sx"] - df["a_sx"]
    df["d_sy"] = df["h_sy"] - df["a_sy"]
    df["d_spd"] = df["h_spd"] - df["a_spd"]
    df["d_vx"] = df["h_vx"] + df["a_vx"]  # both in the home-attacks-+x frame
    df["mid_x"] = (df["h_cx"] + df["a_cx"]) / 2
    return df


def smart_tag_dead_prior(events, n, halves=HALVES):
    """Frames that are certainly dead from the tagged restarts: before each kickoff."""
    """Frames that are certainly dead: from the start of a graded half to its tagged kickoff.

    The tagged 'Kick Off (Start)' interval is 5 s long and contains the kick in its first
    two thirds; we take the start of the interval as the safe end of the dead stretch.
    (GOALS tags span the whole sequence around a goal, minutes long, so they do not locate
    the stoppage and are not used here.)
    """
    dead = np.zeros(n, dtype=bool)
    for e in events:
        if e["code"].upper() != "SET PIECES":
            continue
        if str(e["tags"].get("Type", "")).lower() != "kick off (start)":
            continue
        for _, a, b in halves:
            if a <= e["start_frame"] <= b:
                dead[a:e["start_frame"]] = True
    return dead
