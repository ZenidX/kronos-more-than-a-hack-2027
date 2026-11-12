"""Figures and animations for the README.

Everything is drawn from a HELD-OUT SkillCorner open-data match (MIT licence), never from the
challenge match: the rules forbid publishing data derived from it.

    python -m kronos.figures            # writes docs/img/*.png and docs/img/*.gif
"""

import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as PM  # noqa: E402
from kronos import model as M  # noqa: E402
from kronos.run import PARAMS_FILE, current_params, finish, prepare  # noqa: E402
from kronos.skillcorner import build_match, match_ids  # noqa: E402

OUT = os.path.join(ROOT, "docs", "img")
FPS = 25
C = {"H": "#d62728", "A": "#1f77b4", "D": "#9e9e9e", "X": "#ffffff"}
CMAP = ListedColormap([C["H"], C["A"], C["D"], C["X"]])
CODE = {"H": 0, "A": 1, "D": 2, "X": 3}
warnings.simplefilter("ignore")


def codes(labels):
    return np.array([CODE[x] for x in labels])


def strip(ax, labels, title):
    ax.imshow(codes(labels)[None, :], aspect="auto", cmap=CMAP, vmin=0, vmax=3,
              interpolation="nearest")
    ax.set_yticks([])
    ax.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=10)


def draw_pitch(ax, L, W):
    from matplotlib.patches import Circle, Rectangle
    ax.add_patch(Rectangle((0, 0), L, W, facecolor="#3f8f4f", edgecolor="white", zorder=0))
    ax.plot([L / 2, L / 2], [0, W], color="white", lw=1, zorder=1)
    ax.add_patch(Circle((L / 2, W / 2), 9.15, fill=False, edgecolor="white", lw=1, zorder=1))
    for x0 in (0, L - 16.5):
        ax.add_patch(Rectangle((x0, (W - 40.3) / 2), 16.5, 40.3, fill=False, edgecolor="white", lw=1))
    ax.set_xlim(-2, L + 2)
    ax.set_ylim(-2, W + 2)
    ax.set_aspect("equal")
    ax.axis("off")


def naive_labels(prep, p):
    """Per-frame decision with no switching cost: what the evidence says frame by frame."""
    prox, _ = M.proximity_evidence(prep["dist"], p)
    w = np.where(prep["ball_present"], p["shape_weight_ball"], p["shape_weight_noball"])
    pos = prox + w * prep["shape_logit"]
    dead = M.dead_evidence(prep["mean_speed"], prep["ball_raw"], prep["ball_raw_present"],
                           prep["length"], prep["width"], prep["tag_dead"], p)
    lab = np.where(dead > np.abs(pos) / 2, "D", np.where(pos >= 0, "H", "A")).astype("<U1")
    lab[prep["half"] == 0] = "X"
    return lab, prox, pos, dead


def pick_window(gt, length, start_min=600 * FPS):
    """A window with several possession changes and a stoppage, chosen by rule, not by eye."""
    c = codes(gt)
    change = np.r_[0, (np.diff(c) != 0).astype(int)]
    cs = np.cumsum(change)
    best, best_i = -1, start_min
    for i in range(start_min, len(gt) - length, FPS * 5):
        seg = gt[i:i + length]
        n = cs[i + length - 1] - cs[i]
        if (seg == "D").any() and (seg == "H").any() and (seg == "A").any() and n > best:
            best, best_i = n, i
    return best_i


def fig_signals(g, prep, p, lab, out):
    naive, prox, pos, dead = naive_labels(prep, p)
    i0 = pick_window(g["gt"], 150 * FPS)
    sl = slice(i0, i0 + 150 * FPS)
    t = np.arange(sl.stop - sl.start) / FPS
    fig, axes = plt.subplots(6, 1, figsize=(12, 8.5), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 2, 0.6, 0.6, 0.6]})
    bp = prep["ball_raw_present"][sl]
    axes[0].plot(t, prox[sl], color="black", lw=0.8)
    axes[0].fill_between(t, 0, prox[sl], where=prox[sl] > 0, color=C["H"], alpha=.5)
    axes[0].fill_between(t, 0, prox[sl], where=prox[sl] < 0, color=C["A"], alpha=.5)
    axes[0].fill_between(t, -1.2, 1.2, where=~bp, color="#ffe082", alpha=.5, step="mid",
                         label="balón no detectado")
    axes[0].set_ylabel("1. cercanía\nal balón")
    axes[0].legend(loc="upper right", fontsize=8)
    ps = 1 / (1 + np.exp(-prep["shape_logit"][sl]))
    axes[1].plot(t, ps, color="black", lw=0.8)
    axes[1].axhline(0.5, color="grey", lw=0.5)
    axes[1].fill_between(t, 0.5, ps, where=ps > 0.5, color=C["H"], alpha=.4)
    axes[1].fill_between(t, 0.5, ps, where=ps < 0.5, color=C["A"], alpha=.4)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("2. forma de\nlos equipos\nP(local)")
    axes[2].plot(t, dead[sl], color="black", lw=0.8)
    axes[2].fill_between(t, 0, dead[sl], where=dead[sl] > 0, color=C["D"], alpha=.7)
    axes[2].axhline(0, color="grey", lw=0.5)
    axes[2].set_ylabel("3. balón\nparado")
    for ax, (labels, name) in zip(axes[3:], ((naive[sl], "fotograma a\nfotograma"),
                                            (lab[sl], "Kronos\n(Viterbi)"),
                                            (g["gt"][sl], "verdad"))):
        ax.imshow(codes(labels)[None, :], aspect="auto", cmap=CMAP, vmin=0, vmax=3,
                  interpolation="nearest", extent=(0, t[-1], 0, 1))
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
    axes[-1].set_xlabel("segundos")
    fig.suptitle("Las tres señales y la decisión final (partido SkillCorner reservado, 150 s)",
                 fontsize=12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C[k]) for k in "HAD"]
    fig.legend(handles, ["H local", "A visitante", "D balón parado"], loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return i0


def anim_pitch(g, prep, lab, i0, out, seconds=30, step=2):
    L, W = g["length"], g["width"]
    P, B, team, gt = g["players"], g["ball"], g["team"], g["gt"]
    frames = list(range(i0, i0 + seconds * FPS, step))
    fig = plt.figure(figsize=(8, 6.4))
    ax = fig.add_axes([0.02, 0.28, 0.96, 0.66])
    draw_pitch(ax, L, W)
    home = ax.scatter([], [], s=60, color=C["H"], edgecolors="black", lw=.5, zorder=3)
    away = ax.scatter([], [], s=60, color=C["A"], edgecolors="black", lw=.5, zorder=3)
    ball = ax.scatter([], [], s=40, color="white", edgecolors="black", lw=1, zorder=4)
    title = ax.set_title("", fontsize=11)
    sl = slice(frames[0], frames[-1] + 1)
    axk = fig.add_axes([0.12, 0.16, 0.84, 0.05])
    axg = fig.add_axes([0.12, 0.08, 0.84, 0.05])
    for a, labels, name in ((axk, lab[sl], "Kronos"), (axg, gt[sl], "verdad")):
        a.imshow(codes(labels)[None, :], aspect="auto", cmap=CMAP, vmin=0, vmax=3,
                 interpolation="nearest", extent=(0, len(labels), 0, 1))
        a.set_yticks([])
        a.set_xticks([])
        a.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=10)
    cur = [a.axvline(0, color="black", lw=2) for a in (axk, axg)]
    names = {"H": "LOCAL", "A": "VISITANTE", "D": "BALÓN PARADO"}

    def update(k):
        f = frames[k]
        xy = P[f, :, :2]
        ok = ~np.isnan(xy[:, 0])
        home.set_offsets(xy[ok & (team == 0)])
        away.set_offsets(xy[ok & (team == 1)])
        ball.set_offsets(B[f:f + 1, :2] if not np.isnan(B[f, 0]) else np.empty((0, 2)))
        title.set_text("Kronos: %s   ·   verdad: %s%s" % (
            names.get(lab[f], "—"), names.get(gt[f], "—"),
            "" if not np.isnan(B[f, 0]) else "   (balón no detectado)"))
        for c in cur:
            c.set_xdata([f - frames[0]] * 2)
        return home, away, ball, title, *cur

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 * step / FPS, blit=False)
    anim.save(out, writer=PillowWriter(fps=FPS // step))
    plt.close(fig)


def fig_switching_cost(out):
    """Toy example: why a switching cost beats per-frame decisions on spellF1."""
    rng = np.random.default_rng(3)
    truth = np.array(list("H" * 300 + "D" * 80 + "A" * 260 + "H" * 220 + "D" * 60 + "A" * 280), dtype="<U1")
    noisy = truth.copy()
    flip = rng.random(len(truth)) < 0.18
    alt = {"H": "A", "A": "H", "D": "H"}
    noisy[flip] = [alt[x] for x in truth[flip]]
    E = np.full((len(truth), 3), -1.0)
    for k, s in enumerate("HAD"):
        E[noisy == s, k] = 1.0
    smooth = np.array(["H", "A", "D"])[M.viterbi(E, 12.0, 12.0)]
    full = np.array(list("X" * 10), dtype="<U1")
    rows = []
    for name, lab in (("verdad", truth), ("fotograma a fotograma", noisy),
                      ("con coste de cambio (Viterbi)", smooth)):
        g = np.r_[full, truth, full]
        pr = np.r_[full, lab, full]
        r = PM.evaluate(PM.label_codes(g), PM.label_codes(pr))
        rows.append((name, lab, r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"]))
    fig, axes = plt.subplots(3, 1, figsize=(11, 3.2), sharex=True)
    for ax, (name, lab, mf, sf, npos) in zip(axes, rows):
        ax.imshow(codes(lab)[None, :], aspect="auto", cmap=CMAP, vmin=0, vmax=3,
                  interpolation="nearest")
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
        ax.set_title("macroF1 %.2f   ·   spellF1 %.2f   ·   %d posesiones" % (mf, sf, npos),
                     fontsize=9, loc="right")
    axes[-1].set_xlabel("fotogramas (ejemplo sintético)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def fig_tuning(out):
    with open(PARAMS_FILE) as fh:
        meta = json.load(fh)
    tuned = meta["params"]
    rows = []
    for mid in meta["heldout_matches"]:
        g = build_match(mid)
        prep = prepare(g["players"], g["ball"], g["team"], g["length"], g["width"], g["halves"])
        s = []
        for p in (dict(M.PARAMS), tuned):
            r = PM.evaluate(PM.label_codes(g["gt"]), PM.label_codes(finish(prep, p)))
            s.append(r["spell_f1_mean"])
        rows.append((mid, *s))
    rows.sort(key=lambda r: r[2])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    y = np.arange(len(rows))
    ax.barh(y - 0.2, [r[1] for r in rows], height=0.4, color="#bdbdbd", label="valores iniciales (a mano)")
    ax.barh(y + 0.2, [r[2] for r in rows], height=0.4, color="#2e7d32", label="valores ajustados")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.set_xlabel("spellF1 (la métrica del ranking)")
    ax.set_title("10 partidos que el ajuste nunca vio: media %.3f → %.3f" % (
        np.mean([r[1] for r in rows]), np.mean([r[2] for r in rows])))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    p = current_params()
    with open(PARAMS_FILE) as fh:
        mid = json.load(fh)["heldout_matches"][0]
    g = build_match(mid)
    prep = prepare(g["players"], g["ball"], g["team"], g["length"], g["width"], g["halves"])
    lab = finish(prep, p)
    i0 = fig_signals(g, prep, p, lab, os.path.join(OUT, "senales.png"))
    anim_pitch(g, prep, lab, i0, os.path.join(OUT, "partido.gif"))
    fig_switching_cost(os.path.join(OUT, "coste_de_cambio.png"))
    fig_tuning(os.path.join(OUT, "ajuste_reservados.png"))
    print("figures in", OUT, "match", mid)


if __name__ == "__main__":
    main()
