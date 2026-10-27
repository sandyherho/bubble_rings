"""Animation 2. Four rings released together at 5, 10, 20 and 30 m.

Planar thin-ring model with the full core closure; all rings carry the
reference gas volume, radius and circulation at their release depth.
Each column is a side view in which the ring appears as its diameter;
faint bars mark its position every 4 s, so the column of bars traces the
widening.  The gray outline is the same ring with its gas volume held at
V0.  Horizontal and vertical scales differ (see the axes).  Rings stop
at 0.5 m, where the model is ended.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

from bubble_rings.anim import dark, FG, fig_to_rgb, write_gif, \
    water_background, HALO
from bubble_rings import reduced as rd
from bubble_rings.scenario import PRM

D0S = [5.0, 10.0, 20.0, 30.0]
COLS = ["#9fe7ff", "#7fd3c4", "#ffc36a", "#ff8a5c"]
runs = [rd.integrate(replace(PRM, d0=d), mode="full") for d in D0S]
turn = [rd.integrate(replace(PRM, d0=d), mode="turner") for d in D0S]
T_END = max(o["T"] for o in runs) + 6.0
TF = np.arange(0.0, T_END, 1.5)
XM = 0.75


def at(o, t):
    """Radius and depth at time t (held at the end once surfaced)."""
    tt = min(t, o["T"])
    return np.interp(tt, o["t"], o["R"]), np.interp(tt, o["t"], o["d"])


dark()
frames = []
for t in TF:
    fig = plt.figure(figsize=(7.6, 5.2), dpi=100)
    for j, (o, ot, d0, col) in enumerate(zip(runs, turn, D0S, COLS)):
        ax = fig.add_axes([0.08 + 0.225 * j, 0.10, 0.20, 0.74])
        water_background(ax, [-XM, XM, 31.0, -0.6], -0.6, 31.0)
        ax.axhspan(-0.6, 0.0, color="#1c6f8f", alpha=0.6)
        ax.axhline(0.0, color="#c8eefa", lw=1.0)
        for tk in np.arange(0.0, min(t, o["T"]) + 1e-9, 4.0):
            R, d = at(o, tk)
            ax.plot([-R, R], [d, d], color=col, lw=1.0, alpha=0.6)
        Rt, dt_ = at(ot, t)
        ax.plot([-Rt, Rt], [dt_, dt_], color="0.65", lw=1.0, ls=(0, (2, 2)))
        R, d = at(o, t)
        ax.plot([-R, R], [d, d], color=col, lw=3.2,
                solid_capstyle="round", path_effects=HALO)
        ax.plot([-R, R], [d, d], "o", color="white", ms=2.5)
        ax.set_xlim(-XM, XM)
        ax.set_ylim(31.0, -0.6)
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_xlabel("x (m)", fontsize=8)
        if j == 0:
            ax.set_ylabel("depth (m)")
        else:
            ax.set_yticklabels([])
        ax.tick_params(labelsize=7)
        if t >= o["T"]:
            state = "at 0.5 m"
        else:
            state = f"{np.interp(t, o['t'], o['U']):.2f} m/s"
        ax.text(0.5, 1.012, f"$d_0$ = {d0:g} m\nR = {100 * R:4.1f} cm, "
                f"{state}", transform=ax.transAxes, ha="center",
                va="bottom", fontsize=7.5)
    fig.text(0.5, 0.985, f"same ring released at four depths   t = {t:5.1f} s",
             ha="center", va="top", fontsize=10, color=FG)
    frames.append(fig_to_rgb(fig))
    plt.close(fig)
frames += [frames[-1]] * 14
print(write_gif("anim02_release", frames, fps=12, colors=200))
