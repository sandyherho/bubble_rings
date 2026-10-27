"""Figure 3. Self-leveling of tilted rings.

Rings released at 10 m with their axis tilted by theta0 = 15, 30, 45
and 60 deg from the vertical, followed with the three-dimensional
filament model (64 nodes, symbols) and the planar thin-ring model (full
closure, lines).
(a) Tilt against depth.
(b) Collapse of the tilt: sin(theta)/sin(theta0) against R/R0 for the
filament rings (open symbols), with the horizontal-impulse law (R0/R)^2
(gray line).  Filled symbols: filament rings at 30 and 60 deg with the
axial hydrostatic tension removed (transverse buoyancy only, the closure
of Chang and Llewellyn Smith 2018), which follow R0/R (gray dashed).
(c) Horizontal travel by 0.5 m against initial tilt for release depths
of 5, 10 and 20 m: exact ideal-gas quadrature (lines), planar model with
full closure (open symbols), filament (filled symbols, 10 m).
(d) Radius of each tilted filament ring relative to the level ring at
the same depth.  The thin-ring theory predicts R(d) independent of tilt.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

from bubble_rings.experiments import ascent
from bubble_rings.io_utils import write_csv, save_cache, load_cache
from bubble_rings.plotting import (setup, SEQ, panel_label, legend_below,
                                   handles, save)
from bubble_rings import reduced as rd
from bubble_rings.scenario import PRM

TH = [15.0, 30.0, 45.0, 60.0]
cache = load_cache("fig03")
if cache is None:
    out = {}
    for th in [0.0] + TH:
        o = ascent(PRM, theta0=np.radians(th), N=64, dt0=4e-3, t_step=0.5)
        for k in ("t", "d", "R", "theta", "cx"):
            out[f"{k}_{th:g}"] = o[k]
    for th in (30.0, 60.0):
        o = ascent(PRM, theta0=np.radians(th), N=64, dt0=4e-3, t_step=0.5,
                   hydro_tension=False)
        for k in ("t", "d", "R", "theta", "cx"):
            out[f"{k}_cls{th:g}"] = o[k]
    save_cache("fig03", **out)
    cache = out

red = {th: rd.integrate(PRM, theta0=np.radians(th), mode="full")
       for th in [0.0] + TH}
THG = np.linspace(0, 80, 33)
lat = {}
for d0 in (5.0, 10.0, 20.0):
    p = replace(PRM, d0=d0)
    lat[d0] = np.array([rd.lateral_offset_exact(p, np.radians(t))
                        for t in THG])
lat_full = {d0: np.array([rd.integrate(replace(PRM, d0=d0),
                                       theta0=np.radians(t),
                                       mode="full")["x"][-1] for t in TH])
            for d0 in (5.0, 10.0, 20.0)}

for th in (30.0, 60.0):
    write_csv(f"fig03b_transverse_only_{th:g}deg",
              {k: cache[f"{k}_cls{th:g}"] for k in ("t", "d", "R", "theta",
                                                    "cx")})
for th in TH:
    write_csv(f"fig03ab_filament_{th:g}deg",
              {k: cache[f"{k}_{th:g}"] for k in ("t", "d", "R", "theta",
                                                 "cx")})
write_csv("fig03c_lateral", {"theta0_deg": THG, "x_5m": lat[5.0],
                             "x_10m": lat[10.0], "x_20m": lat[20.0]})

setup()
c = SEQ[1:]
fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.6))
ax = axs[0, 0]
for th, col in zip(TH, c):
    o = red[th]
    ax.plot(np.degrees(o["theta"]), o["d"], color=col)
    ax.plot(np.degrees(cache[f"theta_{th:g}"][::6]),
            cache[f"d_{th:g}"][::6], "o", ms=3, mfc="none", color=col)
ax.set_ylim(PRM.d0, 0)
ax.set_xlabel(r"tilt $\theta$ (deg)")
ax.set_ylabel(r"depth $d$ (m)")
panel_label(ax, "(a)", "tilt")

ax = axs[0, 1]
rg = np.linspace(1, 4, 50)
ax.plot(rg, rg ** -2, color="0.6", lw=2.4)
ax.plot(rg, rg ** -1, color="0.6", lw=1.6, ls="--")
for th, col in ((30.0, c[1]), (60.0, c[3])):
    R, t = cache[f"R_cls{th:g}"], cache[f"theta_cls{th:g}"]
    ax.plot(R[::4] / PRM.R0, np.sin(t[::4]) / np.sin(np.radians(th)), "o",
            ms=2.6, color=col)
for th, col in zip(TH, c):
    R, t = cache[f"R_{th:g}"], cache[f"theta_{th:g}"]
    ax.plot(R[::4] / PRM.R0, np.sin(t[::4]) / np.sin(np.radians(th)), "o",
            ms=2.6, mfc="none", color=col)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([1, 2, 3, 4])
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r"$R/R_0$")
ax.set_ylabel(r"$\sin\theta/\sin\theta_0$")
panel_label(ax, "(b)", "self-leveling law")

ax = axs[1, 0]
for d0, col, mk in zip((5.0, 10.0, 20.0), SEQ, ("s", "o", "^")):
    ax.plot(THG, lat[d0], color=col)
    ax.plot(TH, lat_full[d0], mk, ms=4, mfc="none", color=col)
ax.plot(TH, [cache[f"cx_{th:g}"][-1] for th in TH], "o", ms=3,
        color=SEQ[1])
ax.set_xlabel(r"initial tilt $\theta_0$ (deg)")
ax.set_ylabel(r"horizontal travel (m)")
panel_label(ax, "(c)", "lateral drift")

ax = axs[1, 1]
d0r, R0r = cache["d_0"], cache["R_0"]
for th, col in zip(TH, c):
    d, R = cache[f"d_{th:g}"], cache[f"R_{th:g}"]
    ok = (d >= d0r.min()) & (d < PRM.d0 - 0.05)
    Rl = np.interp(d[ok], d0r[::-1], R0r[::-1])
    ax.plot(np.abs(R[ok] / Rl - 1), d[ok], color=col)
ax.set_xscale("log")
ax.set_xlim(1e-9, 1e-4)
ax.set_ylim(PRM.d0, 0)
ax.set_xlabel(r"$|R_\theta/R_{\rm level}-1|$")
ax.set_ylabel(r"depth $d$ (m)")
panel_label(ax, "(d)", "tilt invariance of R(d)")
for ax in axs.flat:
    ax.minorticks_on()
fig.subplots_adjust(wspace=0.38, hspace=0.45, bottom=0.15)
h, lab = handles([rf"$\theta_0={t:g}^\circ$" for t in TH], c)
h2, lab2 = handles(["5 m", "10 m", "20 m"], SEQ[:3],
                   markers=["s", "o", "^"])
legend_below(fig, axs[1, 1], h, lab, ncol=2, pad=0.015)
legend_below(fig, axs[1, 0], h2, [f"$d_0=${s}" for s in lab2],
             ncol=3, pad=0.015)
print(save(fig, "fig03_tilt"))
