"""Figure 0. Geometry and notation.

(a) A diver at depth d0 releases a ring of gas; as it rises the ambient
pressure p = p_atm + rho g d falls, the gas expands and the ring widens,
fastest near the surface.  Ring outlines are the computed reference
ascent (full closure) every 4 s, drawn at true scale horizontally and
with the vertical axis compressed (see axes).
(b) Cross-section of the ring: radius R, core radius a, unit normal n,
circulation Gamma about the core, swirl speed u_c = Gamma/(2 pi a) at the
core surface, buoyancy per unit length rho g pi a^2, and the drift D of
the core relative to the liquid that balances it.
(c) A tilted ring: the impulse I = rho pi Gamma R^2 n gains only vertical
components, so its horizontal part is conserved.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch

from bubble_rings.io_utils import write_csv
from bubble_rings.plotting import setup, OKABE_ITO as C, panel_label, save
from bubble_rings import reduced as rd
from bubble_rings.scenario import PRM

o = rd.integrate(PRM, mode="full")
write_csv("fig00a_outlines", {"t_s": o["t"], "depth_m": o["d"],
                              "R_m": o["R"]})


def arrow(ax, p, q, col="k", lw=1.0, ms=8, **kw):
    """Straight arrow from p to q."""
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms,
                                 color=col, lw=lw, **kw))


setup()
fig = plt.figure(figsize=(6.8, 3.9))
ax = fig.add_axes([0.07, 0.12, 0.30, 0.80])
ax.axhspan(-0.6, 0.0, color="#d6eef7")
ax.axhline(0.0, color=C["blue"], lw=1.0)
for tk in np.arange(0.0, o["T"], 4.0):
    R = np.interp(tk, o["t"], o["R"])
    d = np.interp(tk, o["t"], o["d"])
    ax.add_patch(Ellipse((0, d), 2 * R, 0.12, fill=False, lw=1.3,
                         color=C["blue"]))
ax.plot(0.0, PRM.d0 + 0.45, marker="v", ms=7, color="0.3")
ax.text(0.03, PRM.d0 + 0.45, "release by diver", fontsize=7, va="center")
ax.set_xlim(-0.42, 0.42)
ax.set_ylim(PRM.d0 + 0.8, -0.6)
ax.set_xlabel(r"horizontal position (m)")
ax.set_ylabel(r"depth $d$ (m)")
for dd in (0.0, 5.0, 10.0):
    ax.text(0.41, dd - 0.15, rf"$p={PRM.p_inf(dd) / 1e5:.2f}$ bar",
            ha="right", va="bottom", fontsize=7)
panel_label(ax, "(a)", "ascent")

ax = fig.add_axes([0.43, 0.12, 0.30, 0.80])
R, a = 1.0, 0.16
for s in (-1, 1):
    ax.add_patch(Circle((s * R, 0), a, color="#cfe8f3", ec=C["blue"],
                        lw=1.2))
    th = np.linspace(0.3, 2 * np.pi - 0.3, 60)
    rr = 0.30
    ax.plot(s * R + rr * np.cos(th), rr * np.sin(th) * s, color=C["vermil"],
            lw=0.9)
    arrow(ax, (s * R + rr * np.cos(th[-2]), rr * np.sin(th[-2]) * s),
          (s * R + rr * np.cos(th[-1]), rr * np.sin(th[-1]) * s),
          col=C["vermil"], ms=7)
ax.plot([-R, R], [0, 0], color="0.5", lw=0.6, ls="--")
ax.text(0.0, -0.08, r"$2R$", ha="center", va="top")
ax.annotate("", (R + a, -0.55), (R - a, -0.55),
            arrowprops=dict(arrowstyle="<->", lw=0.7))
ax.text(R, -0.62, r"$2a$", ha="center", va="top")
arrow(ax, (0, 0.05), (0, 0.85), col="k", lw=1.2)
ax.text(0.05, 0.80, r"$\mathbf{n}$, $U$", va="top")
arrow(ax, (-R, -0.62), (-R, -0.27), col=C["green"])
ax.text(-R + 0.05, -0.60, r"$\rho g\pi a^2$", color=C["green"], fontsize=8)
arrow(ax, (R + 0.2, 0.0), (R + 0.6, 0.0), col=C["purple"])
ax.text(R + 0.40, 0.06, r"$\mathbf{D}$", color=C["purple"], ha="center")
ax.text(R + 0.18, 0.35, r"$\Gamma$, $u_c=\Gamma/2\pi a$", color=C["vermil"],
        fontsize=8)
ax.text(-1.45, -1.05, r"$p_g = p - \rho\Gamma^2/(8\pi^2a^2) + \gamma/a$",
        fontsize=7.5)
ax.text(-1.45, -1.22, r"$p_g V = $ const, $V = 2\pi^2 R a^2$", fontsize=7.5)
ax.set_xlim(-1.5, 1.7)
ax.set_ylim(-1.35, 1.0)
ax.set_aspect("equal")
ax.axis("off")
panel_label(ax, "(b)", "core and forces", dy=0.98)

ax = fig.add_axes([0.76, 0.12, 0.23, 0.80])
th0 = np.radians(35)
n = np.array([np.sin(th0), np.cos(th0)])
e = np.array([np.cos(th0), -np.sin(th0)])
ax.plot([-e[0], e[0]], [-e[1], e[1]], color=C["blue"], lw=2.4,
        solid_capstyle="round")
arrow(ax, (0, 0), 1.2 * n, col="k", lw=1.2)
arrow(ax, (0, 0), (1.2 * n[0], 0), col=C["orange"], lw=1.0)
arrow(ax, (1.2 * n[0], 0), 1.2 * n, col=C["green"], lw=1.0)
ax.text(1.2 * n[0] + 0.05, 1.2 * n[1], r"$\mathbf{I}$", va="center")
ax.text(0.35, -0.12, r"$I_h$ const", color=C["orange"], fontsize=8)
ax.text(1.2 * n[0] + 0.05, 0.4, r"$dI_z/dt=\rho gV$", color=C["green"],
        fontsize=8)
ax.plot([0, 0], [0, 0.9], color="0.6", lw=0.6, ls="--")
th = np.linspace(np.pi / 2 - th0, np.pi / 2, 20)
ax.plot(0.45 * np.cos(th), 0.45 * np.sin(th), color="0.3", lw=0.7)
ax.text(0.12, 0.5, r"$\theta$", fontsize=8)
ax.text(-1.0, -1.05, r"$\sin\theta=\sin\theta_0\,(R_0/R)^2$", fontsize=8)
ax.set_xlim(-1.1, 1.4)
ax.set_ylim(-1.2, 1.3)
ax.set_aspect("equal")
ax.axis("off")
panel_label(ax, "(c)", "tilted ring", dy=0.98)
print(save(fig, "fig00_schematic"))
