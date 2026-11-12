"""Load the tactical ATD into dense numpy arrays, cached next to the data.

Arrays (video frame index = row, 0..n-1):
    players  (n, S, 3) float32   x, y in metres (pitch frame), speed m/s; NaN when not tracked
    ball     (n, 3)    float32   x, y in metres, speed m/s; NaN when not tracked
    team     (S,)      int8      0 home, 1 away, per track slot

The cache is written to data/ (gitignored), so it never reaches the public repository.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402


def team_of_each_track(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home_id = team_ids[0]
    return np.array([0 if p.get("teamId") == home_id else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def parse_raw(raw_path, n_slots, n_frames):
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            vals = np.array(parts[1].replace(";", ",").split(","), dtype=np.float32)
            players[i] = vals.reshape(n_slots, 3)
            ball[i] = np.array(parts[2].split(",")[:3], dtype=np.float32)
    return players, ball


def load(game=GAME, data_dir=None, use_cache=True):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    cache = os.path.join(os.path.dirname(raw_path), "_kronos_atd_cache.npz")
    length, width = GAMES[game]["pitch"]
    if use_cache and os.path.exists(cache):
        z = np.load(cache)
        return z["players"], z["ball"], z["team"]
    team = team_of_each_track(meta_path)
    players, ball = parse_raw(raw_path, len(team), source_frames("tactical", game))
    for arr in (players[..., 0], ball[:, 0]):
        arr *= length
    for arr in (players[..., 1], ball[:, 1]):
        arr *= width
    np.savez(cache, players=players, ball=ball, team=team)
    return players, ball, team


if __name__ == "__main__":
    p, b, t = load(use_cache=False)
    print(p.shape, b.shape, t.shape, "home slots", int((t == 0).sum()), "away", int((t == 1).sum()))
    print("ball present", float(np.mean(~np.isnan(b[:, 0]))))
