"""Figure 4. Stability margins of the gas core.

(a) Axisymmetric growth rate of a hollow vortex with surface tension,
(sigma/Omega)^2 = [(1 - x^2)/We - 1] x K1(x)/K0(x), x = k a, for swirl
Weber numbers We = rho Gamma^2/(4 pi^2 gamma a) of 0.25 to 2, over
the band 0 < k a < 1 outside which every mode is neutral.  Positive
values are unstable; swirl stabilizes every wavelength once We > 1,
the condition of Pedley (1967, 1968).
(b) Viscous capillary margin of Pedley (1968, Eq. 6.5),
M = We [1 - exp(-3 a^2/(8 nu t))]^2, along the ascent from release
depths of 5, 10, 15, 20 and 40 m.  Vorticity diffusing from the core
surface lowers the swirl there, and the core becomes capillary-unstable
where M crosses one (gray line).
(c) Smallest circulation that keeps each margin above one over the whole
ascent, against release depth: capillary (We), core gravity (Fr),
Pedley's viscous margin (M), and swirl head u_c^2/(2g) at least the ring
diameter, for 10 mL of gas (solid); the swirl-head bound is also shown
for 5 and 20 mL.  The
reference circulation is the gray line.
(d) Swirl head over ring diameter along the ascent, same release
depths as (b).  A uniform core can
be held in hydrostatic balance over a vertical extent below the swirl
head, so a ratio above one admits any tilt.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
from scipy.optimize import brentq

from bubble_rings.core import capillary_growth, pedley_margin, pedley_breakup
from bubble_rings.io_utils import write_csv
from bubble_rings.plotting import (setup, SEQ, panel_label, legend_below,
                                   handles, save)
from bubble_rings import reduced as rd
from bubble_rings.scenario import PRM

WES = [0.25, 0.5, 0.75, 1.0, 2.0]
x = np.linspace(1e-3, 1.0, 400)
D0S = [5.0, 10.0, 15.0, 20.0, 40.0]
runs = {d0: rd.integrate(replace(PRM, d0=d0), mode="full", n_out=4000)
        for d0 in D0S}


def margin(name, p):
    """Minimum over the ascent of one stability margin."""
    o = rd.integrate(p, mode="full", t_max=1e7, n_out=800)
    if name == "We":
        return o["We"].min()
    if name == "Fr":
        return o["Fr"].min()
    if name == "M":
        return pedley_margin(p, o["t"], o["a"]).min()
    return (o["hs"] / (2 * o["R"])).min()


def gamma_c(name, p):
    """Circulation at which the margin is one (bisection in log Gamma)."""
    def f(lg):
        """Log of the margin at circulation exp(lg)."""
        return np.log(margin(name, replace(p, Gamma=np.exp(lg))))
    return float(np.exp(brentq(f, np.log(5e-4), np.log(2.0), xtol=1e-6)))


DG = np.array([1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0])
crit = {}
for name in ("We", "Fr", "M", "hs"):
    crit[name] = np.array([gamma_c(name, replace(PRM, d0=d)) for d in DG])
for v in (5e-6, 20e-6):
    crit[f"hs_{v * 1e6:g}"] = np.array(
        [gamma_c("hs", replace(PRM, d0=d, V0=v)) for d in DG])
write_csv("fig04a_dispersion", {"ka": x, **{f"We{w:g}": capillary_growth(x, w)
                                            for w in WES}})
write_csv("fig04c_gamma_c", {"d0_m": DG, "Gc_We": crit["We"],
                             "Gc_Fr": crit["Fr"], "Gc_M": crit["M"],
                             "Gc_hs_10mL": crit["hs"],
                             "Gc_hs_5mL": crit["hs_5"],
                             "Gc_hs_20mL": crit["hs_20"]})
DB = np.arange(2.0, 40.5, 1.0)
brk = np.array([pedley_breakup(replace(PRM, d0=d),
                               rd.integrate(replace(PRM, d0=d), mode="full",
                                            n_out=4000)) for d in DB])
write_csv("fig04b_breakup", {"d0_m": DB, "t_breakup_s": brk[:, 0],
                             "depth_breakup_m": brk[:, 1]})
for d0 in D0S:
    o = runs[d0]
    write_csv(f"fig04bd_margins_{d0:g}m",
              {"t_s": o["t"], "depth_m": o["d"], "We": o["We"],
               "Fr": o["Fr"], "M_pedley": pedley_margin(PRM, o["t"], o["a"]),
               "hs_over_2R": o["hs"] / (2 * o["R"]), "eps": o["eps"]})

setup()
fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.6))
ax = axs[0, 0]
for w, col in zip(WES, SEQ):
    ax.plot(x, capillary_growth(x, w), color=col)
ax.axhline(0, color="0.6", lw=0.6)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(-1.6, 2.0)
ax.set_xlabel(r"axial wavenumber $ka$")
ax.set_ylabel(r"$(\sigma/\Omega)^2$")
panel_label(ax, "(a)", "hollow-core capillary waves")

ax = axs[0, 1]
for d0, col in zip(D0S, SEQ):
    o = runs[d0]
    q = replace(PRM, d0=d0)
    ax.plot(pedley_margin(q, o["t"], o["a"]), o["d"], color=col)
ax.axvline(1.0, color="0.6", lw=0.8)
ax.set_xscale("log")
ax.set_xlim(5e-3, 1e3)
ax.set_ylim(41, 0)
ax.set_xlabel(r"Pedley margin $M$")
ax.set_ylabel(r"depth $d$ (m)")
panel_label(ax, "(b)", "core margins")

ax = axs[1, 0]
ax.plot(DG, crit["We"], color=SEQ[0], marker="o", ms=3)
ax.plot(DG, crit["Fr"], color=SEQ[1], marker="s", ms=3)
ax.plot(DG, crit["M"], color=SEQ[3], marker="D", ms=3)
ax.plot(DG, crit["hs"], color=SEQ[2], marker="^", ms=3)
ax.plot(DG, crit["hs_5"], color=SEQ[2], ls="--")
ax.plot(DG, crit["hs_20"], color=SEQ[2], ls=":")
ax.axhline(PRM.Gamma, color="0.6", lw=0.8)
ax.set_yscale("log")
ax.set_xlabel(r"release depth $d_0$ (m)")
ax.set_ylabel(r"critical circulation $\Gamma_c$ (m$^2$ s$^{-1}$)")
panel_label(ax, "(c)", "required circulation")

ax = axs[1, 1]
for d0, col in zip(D0S, SEQ):
    o = runs[d0]
    ax.plot(o["hs"] / (2 * o["R"]), o["d"], color=col)
ax.axvline(1.0, color="0.6", lw=0.6)
ax.set_xscale("log")
ax.set_xticks([1, 2, 3, 4])
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_ylim(41, 0)
ax.set_xlabel(r"swirl head over diameter $u_c^2/(4gR)$")
ax.set_ylabel(r"depth $d$ (m)")
panel_label(ax, "(d)", "hydrostatic core margin")
for a in axs.flat:
    a.minorticks_on()
fig.subplots_adjust(wspace=0.36, hspace=0.62, bottom=0.12)
h, lab = handles([f"{w:g}" for w in WES], SEQ[:5])
lab[0] = "We = " + lab[0]
legend_below(fig, axs[0, 0], h, lab, ncol=5, pad=0.015)
h, lab = handles([f"{d:g}" for d in D0S], SEQ[:5])
lab[0] = "$d_0$ = 5"
lab[-1] = "40 m"
legend_below(fig, axs[0, 1], h, lab, ncol=5, pad=0.015)
h, lab = handles(["We = 1", "Fr$_c$ = 1", "$M$ = 1", r"$u_c^2/2g=2R$",
                  "5 mL", "20 mL"], [SEQ[0], SEQ[1], SEQ[3], SEQ[2], SEQ[2],
                                     SEQ[2]], ["-", "-", "-", "-", "--", ":"],
                 ["o", "s", "D", "^", None, None])
legend_below(fig, axs[1, 0], h, lab, ncol=3, pad=0.015)
h, lab = handles([f"{d:g}" for d in D0S], SEQ[:5])
lab[0] = "$d_0$ = 5"
lab[-1] = "40 m"
legend_below(fig, axs[1, 1], h, lab, ncol=5, pad=0.015)
print(save(fig, "fig04_core"))
