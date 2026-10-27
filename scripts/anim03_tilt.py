"""Animation 3. Tilted rings level themselves.

Four rings released together at 10 m with their axes tilted by 15, 30,
45 and 60 deg, followed with the filament model (64 nodes) to 0.5 m.
Left: side view of the whole column (axes not to scale) with the paths
of the centers.  Right: close-ups that follow each ring at true scale,
seen from 15 deg above the horizontal so the ring appears as an ellipse;
the arrow is the ring axis and the dotted line the vertical.  Horizontal
impulse is conserved while buoyancy adds vertical impulse, so sin(theta)
falls as (R0/R)^2 and each ring drifts a finite distance sideways.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from bubble_rings.anim import dark, FG, fig_to_rgb, write_gif, \
    water_background, HALO
from bubble_rings.experiments import ascent
from bubble_rings.filament import ring_diagnostics
from bubble_rings.io_utils import save_cache, load_cache
from bubble_rings.scenario import PRM

TH = [15.0, 30.0, 45.0, 60.0]
COLS = ["#9fe7ff", "#7fd3c4", "#ffc36a", "#ff8a5c"]
cache = load_cache("anim03")
if cache is None:
    cache = {}
    for th in TH:
        o = ascent(PRM, theta0=np.radians(th), N=64, dt0=4e-3,
                   t_step=0.25, keep=True)
        cache[f"X_{th:g}"] = o["X"]
        cache[f"t_{th:g}"] = o["t"]
    save_cache("anim03", **cache)

el = np.radians(15.0)
nf = max(len(cache[f"t_{th:g}"]) for th in TH)
HWZ, HHZ = 0.32, 0.24          # half window of the close-ups (m)


def screen(X, c):
    """Camera 15 deg above the horizontal: screen coordinates about c."""
    Y = X - c
    return Y[:, 0], Y[:, 2] * np.cos(el) - Y[:, 1] * np.sin(el)


dark()
frames = []
for i in range(nf):
    fig = plt.figure(figsize=(8.4, 6.2), dpi=100)
    ax = fig.add_axes([0.06, 0.08, 0.30, 0.84])
    water_background(ax, [-0.4, 3.0, 10.4, -0.4], -0.4, 10.4)
    ax.axhspan(-0.4, 0.0, color="#1c6f8f", alpha=0.6)
    ax.axhline(0.0, color="#c8eefa", lw=1.0)
    for k, (th, col) in enumerate(zip(TH, COLS)):
        Xs = cache[f"X_{th:g}"]
        j = min(i, len(Xs) - 1)
        X = Xs[j]
        cs = Xs[: j + 1].mean(1)
        ax.plot(cs[:, 0], -cs[:, 2], color=col, lw=0.8, alpha=0.7)
        sx, sy = screen(X, np.zeros(3))
        ax.plot(sx, -sy, color=col, lw=1.4)
        dg = ring_diagnostics(PRM, X)
        # close-up following this ring, true scale
        zx = fig.add_axes([0.42 + 0.29 * (k % 2), 0.53 - 0.43 * (k // 2),
                           0.27, 0.38])
        c = dg["c"]
        water_background(zx, [-HWZ, HWZ, -HHZ, HHZ], -c[2] - HHZ,
                         -c[2] + HHZ)
        zx.axvline(0.0, color="0.45", lw=0.6, ls=":")
        sx, sy = screen(X, c)
        back = (X - c)[:, 1] > 0
        for sel, a_ in ((back, 0.45), (~back, 1.0)):
            zx.plot(np.where(sel, sx, np.nan), np.where(sel, sy, np.nan),
                    color=col, lw=2.6, alpha=a_, solid_capstyle="round",
                    path_effects=HALO)
        n = dg["n"]
        zx.annotate("", (0.18 * n[0], 0.18 * n[2] * np.cos(el)), (0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="white",
                                    lw=1.0))
        zx.set_xlim(-HWZ, HWZ)
        zx.set_ylim(-HHZ, HHZ)
        zx.set_aspect("equal")
        zx.set_xticks([])
        zx.set_yticks([])
        zx.text(0.03, 0.96, rf"$\theta_0$ = {th:g}$^\circ$", color=col,
                path_effects=HALO,
                transform=zx.transAxes, va="top", fontsize=9)
        zx.text(0.97, 0.96, rf"$\theta$ = {np.degrees(dg['theta']):4.1f}"
                rf"$^\circ$" + f"\nR = {100 * dg['R']:4.1f} cm"
                f"\nx = {c[0]:4.2f} m", color=FG, path_effects=HALO,
                transform=zx.transAxes, va="top", ha="right", fontsize=8,
                family="monospace")
        if k == 3:
            zx.plot([HWZ - 0.13, HWZ - 0.03], [-HHZ + 0.03] * 2, color=FG,
                    lw=1.5)
            zx.text(HWZ - 0.08, -HHZ + 0.04, "10 cm", ha="center",
                    va="bottom", fontsize=7.5)
    t = cache[f"t_{TH[-1]:g}"][min(i, len(cache[f't_{TH[-1]:g}']) - 1)]
    ax.set_xlim(-0.4, 3.0)
    ax.set_ylim(10.4, -0.4)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("depth (m)")
    fig.text(0.06, 0.955, "tilted rings level themselves", fontsize=11,
             color=FG)
    fig.text(0.42, 0.955, f"t = {t:4.1f} s   close-ups follow each ring "
             "(true scale, arrow = ring axis)", fontsize=9, color=FG)
    frames.append(fig_to_rgb(fig))
    plt.close(fig)
frames += [frames[-1]] * 14
print(write_gif("anim03_tilt", frames, fps=12, colors=220))
