"""Produce the Kronos submission from the provided data.

    python -m kronos.run                 # writes submission/Kronos.csv
    python evaluation/validate_submission.py submission/Kronos.csv

Deterministic: same data in, same file out.
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, GAMES  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tagging  # noqa: E402

from kronos import features as F  # noqa: E402
from kronos import model as M  # noqa: E402
from kronos.load import load  # noqa: E402

PARAMS_FILE = os.path.join(ROOT, "kronos", "params.json")


def current_params():
    """Defaults from model.PARAMS, overridden by kronos/params.json (tuned on external games)."""
    p = dict(M.PARAMS)
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as fh:
            p.update(json.load(fh)["params"])
    return p


def prepare(players, ball_raw, team, length, width, halves, tag_dead=None):
    """Everything that does not depend on the tunable parameters."""
    warnings.simplefilter("ignore", RuntimeWarning)
    n = len(ball_raw)
    if tag_dead is None:
        tag_dead = np.zeros(n, dtype=bool)
    ball_raw_present = ~np.isnan(ball_raw[:, 0])
    ball = F.interpolate_ball(ball_raw)
    dist, _ = F.nearest_by_team(players, team, ball[:, :2])
    mean_speed = np.nanmean(players[..., 2], axis=1)
    direction = F.attack_direction(players, team, length, halves)
    shape = F.shape_features(players, team, length, width, direction, halves)
    half = F.half_of(n, halves)
    y = M.pseudo_labels(dist, ball_raw_present, mean_speed)
    p_home = M.shape_probability(shape, y, half)
    shape_logit = np.log(np.clip(p_home, 1e-3, 1 - 1e-3) / np.clip(1 - p_home, 1e-3, 1))
    return dict(n=n, halves=halves, length=length, width=width, tag_dead=tag_dead,
                ball_raw=ball_raw, ball_raw_present=ball_raw_present,
                ball_present=~np.isnan(ball[:, 0]), dist=dist, mean_speed=mean_speed,
                direction=direction, half=half, y=y, shape_logit=shape_logit)


def finish(prep, params=None):
    """Labels from prepared evidence and a parameter set."""
    p = dict(current_params(), **(params or {}))
    prox, _ = M.proximity_evidence(prep["dist"], p)
    dead_ev = M.dead_evidence(prep["mean_speed"], prep["ball_raw"], prep["ball_raw_present"],
                              prep["length"], prep["width"], prep["tag_dead"], p)
    return M.decode(prox, prep["shape_logit"], prep["ball_present"], dead_ev, p, prep["halves"])


def predict(players, ball_raw, team, length, width, halves, tag_dead=None, params=None,
            return_info=False):
    """Labels for every frame of one match. Works on any tracking in the same array layout."""
    prep = prepare(players, ball_raw, team, length, width, halves, tag_dead)
    labels = finish(prep, params)
    if not return_info:
        return labels
    graded = prep["half"] > 0
    return labels, {
        "pseudo_labels": int((prep["y"] >= 0).sum()),
        "label_counts": {k: int((labels[graded] == k).sum()) for k in "HAD"},
        "possessions": count_possessions(labels),
        "direction": prep["direction"],
    }


def count_possessions(labels):
    c, prev = 0, None
    for lab in labels:
        if lab in "HA" and lab != prev:
            c += 1
        prev = lab if lab != "X" else prev
    return c


def main(out_path):
    players, ball_raw, team = load()
    length, width = GAMES[GAME]["pitch"]
    tag_dead = F.smart_tag_dead_prior(load_smart_tagging(), len(ball_raw))
    labels, info = predict(players, ball_raw, team, length, width, F.HALVES, tag_dead,
                           return_info=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines(f"{i},{lab}\n" for i, lab in enumerate(labels))
    return labels, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "submission", "Kronos.csv"))
    args = ap.parse_args()
    _, info = main(args.out)
    print(info)
    print("wrote", args.out)
