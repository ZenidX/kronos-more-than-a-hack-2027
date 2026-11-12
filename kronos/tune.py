"""Tune the decoder's parameters on EXTERNAL games only (Metrica's public sample games 1 and 2).

The challenge match and its public ground truth are never read here. Each external game is
degraded with two random seeds to look like automatic tracking (see external.degrade), and the
objective is the official spellF1, averaged over the four game-seed pairs.

    python -m kronos.tune --trials 80        # writes kronos/params.json

Random search in log space around the priors in model.PARAMS, seeded, so it is reproducible.
Game-level cross-check: the chosen parameters are also scored game by game, so a choice that only
helps one match shows up.
"""

import argparse
import json
import os
import time

import numpy as np

from kronos import model as M
from kronos.external import LENGTH, WIDTH, build_game, score
from kronos.run import PARAMS_FILE, finish, prepare
from kronos.skillcorner import build_match, match_ids

SEARCH = {  # name: (low, high, log-scale?)
    "control_radius": (0.8, 2.5, False),
    "proximity_scale": (0.5, 4.0, True),
    "proximity_weight": (0.3, 4.0, True),
    "shape_weight_ball": (0.0, 0.15, False),
    "shape_weight_noball": (0.0, 0.4, False),
    "speed_ref": (1.0, 2.2, False),
    "speed_weight": (0.5, 6.0, True),
    "ball_fast": (3.0, 10.0, False),
    "ball_fast_weight": (0.0, 4.0, False),
    "out_weight": (1.0, 10.0, True),
    "switch_team": (5.0, 80.0, True),
    "switch_dead": (3.0, 40.0, True),
}


def sample(rng):
    p = {}
    for k, (lo, hi, log) in SEARCH.items():
        p[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi)))) if log else float(rng.uniform(lo, hi))
    return p


def objective(preps, params):
    rows = []
    for name, prep, gt in preps:
        r = score(gt, finish(prep, params))
        rows.append((name, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]))
    return float(np.mean([r[1] for r in rows])), rows


def _prep_sc(mid):
    g = build_match(mid)
    return (g["name"], prepare(g["players"], g["ball"], g["team"], g["length"], g["width"],
                               g["halves"]), g["gt"])


def main(trials, seed):
    t0 = time.time()
    preps = []
    for k in (1, 2):
        for s in (0, 1):
            g = build_game(k, seed=s)
            preps.append((g["name"], prepare(g["players"], g["ball"], g["team"], LENGTH, WIDTH,
                                             g["halves"]), g["gt"]))
    ids = match_ids()
    tune_ids, hold_ids = ids[0::2], ids[1::2]   # alternate: 10 to tune, 10 held out
    preps += [_prep_sc(m) for m in tune_ids]
    holdout = [_prep_sc(m) for m in hold_ids]
    print("prepared %d tuning + %d held-out matches in %.0fs" % (len(preps), len(holdout),
                                                                 time.time() - t0))

    base = {k: M.PARAMS[k] for k in SEARCH}
    best_val, best_rows = objective(preps, base)
    best = base
    print("priors      spellF1 %.4f" % best_val)
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as fh:
            prev = {k: v for k, v in json.load(fh)["params"].items() if k in SEARCH}
        val, rows = objective(preps, dict(base, **prev))
        print("previous    spellF1 %.4f" % val)
        if val > best_val:
            best, best_val, best_rows = dict(base, **prev), val, rows
    rng = np.random.default_rng(seed)
    for t in range(trials):
        cand = sample(rng)
        val, rows = objective(preps, cand)
        if val > best_val:
            best, best_val, best_rows = cand, val, rows
            print("trial %3d   spellF1 %.4f  %s" % (t, val, {k: round(v, 3) for k, v in cand.items()}))
    # local refinement: perturb the best
    for t in range(trials // 2):
        cand = {}
        for k, (lo, hi, log) in SEARCH.items():
            v = best[k] * float(np.exp(rng.normal(0, 0.2))) if log else best[k] + rng.normal(0, 0.1 * (hi - lo))
            cand[k] = float(np.clip(v, lo, hi))
        val, rows = objective(preps, cand)
        if val > best_val:
            best, best_val, best_rows = cand, val, rows
            print("refine %3d  spellF1 %.4f" % (t, val))

    for r in best_rows:
        print("  %-10s spellF1 %.4f  macroF1 %.4f  poss %d vs gt %d" % r)
    base_hold, _ = objective(holdout, base)
    best_hold, hold_rows = objective(holdout, best)
    print("held-out SkillCorner: priors %.4f -> tuned %.4f" % (base_hold, best_hold))
    with open(PARAMS_FILE, "w") as fh:
        json.dump({"source": "kronos.tune on Metrica sample games 1-2 (degraded, seeds 0-1) "
                             "and SkillCorner open data matches %s" % ",".join(tune_ids),
                   "objective_spell_f1_mean": round(best_val, 4),
                   "heldout_matches": hold_ids,
                   "heldout_spell_f1_priors": round(base_hold, 4),
                   "heldout_spell_f1_tuned": round(best_hold, 4),
                   "trials": trials, "seed": seed,
                   "per_game": [dict(zip(("name", "spell_f1", "macro_f1", "n_pred", "n_gt"), r))
                                for r in best_rows],
                   "params": {k: round(v, 4) for k, v in best.items()}}, fh, indent=2)
    print("wrote", PARAMS_FILE, "in %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.trials, a.seed)
