"""Figure 2. Dependence on release depth.

Level rings with the reference gas volume, radius and circulation are
released at depths of 1 to 40 m and followed to 0.5 m.
(a) Radius at 0.5 m over R0: full closure, exact ideal-gas solution,
constant gas volume.
(b) Radius gain with Boyle expansion over the gain at constant volume
against the release pressure number Pi = rho g d0 / p_atm; the frozen-
logarithm prediction ((1 + Pi)/Pi) ln(1 + Pi) is universal.
(c) Ascent time: full closure, exact ideal-gas quadrature, frozen-log
closed form, constant volume.
(d) Flaring: expansion per meter of rise at 0.5 m over its value at
release.  The frozen-log law gives p0 / p(0.5 m).
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

from bubble_rings.io_utils import write_csv
from bubble_rings.plotting import (setup, SEQ, panel_label, legend_below,
                                   handles, save)
from bubble_rings import reduced as rd
from bubble_rings.scenario import PRM

D0 = np.concatenate([np.linspace(1.0, 5.0, 9), np.linspace(6.0, 40.0, 18)])
cols = {k: [] for k in ("Rf", "Ri", "Rt", "Tf", "Ti", "Tfz", "Tt", "flf",
                        "flt", "gf", "gi", "Pi")}
for d0 in D0:
    p = replace(PRM, d0=d0)
    f = rd.integrate(p, mode="full")
    t = rd.integrate(p, mode="turner")
    Ri = rd.R_exact(p, p.d_end)

    def slope(o):
        """dR/dh of a level ring at the first and last sample."""
        s = p.g * o["V"] / (2 * np.pi * p.Gamma * o["R"]) / o["U"]
        return s[-1] / s[0]
    cols["Rf"].append(f["R"][-1] / p.R0)
    cols["Ri"].append(Ri / p.R0)
    cols["Rt"].append(t["R"][-1] / p.R0)
    cols["Tf"].append(f["T"])
    cols["Ti"].append(rd.ascent_time_exact(p))
    cols["Tfz"].append(rd.ascent_time(p))
    cols["Tt"].append(t["T"])
    cols["flf"].append(slope(f))
    cols["flt"].append(slope(t))
    cols["gf"].append((f["R"][-1] - p.R0) / (t["R"][-1] - p.R0))
    # exact ideal over the constant-volume radius with the same running log
    cols["gi"].append((Ri - p.R0) / (t["R"][-1] - p.R0))
    cols["Pi"].append(p.rho * p.g * (d0 - p.d_end) / p.p_inf(p.d_end))
cols = {k: np.asarray(v) for k, v in cols.items()}
# the frozen-log ratio uses the pressure at the end depth as reference
Pi = cols["Pi"]
fl_fz = PRM.p_inf(D0) / PRM.p_inf(PRM.d_end)
write_csv("fig02_release", {"d0_m": D0, "Pi_end": Pi,
                            "Rend_full_over_R0": cols["Rf"],
                            "Rend_exact_over_R0": cols["Ri"],
                            "Rend_constV_over_R0": cols["Rt"],
                            "gain_full": cols["gf"],
                            "gain_exact": cols["gi"],
                            "gain_frozen": rd.turner_ratio(Pi),
                            "T_full_s": cols["Tf"], "T_exact_s": cols["Ti"],
                            "T_frozen_s": cols["Tfz"],
                            "T_constV_s": cols["Tt"],
                            "flare_full": cols["flf"],
                            "flare_constV": cols["flt"],
                            "flare_frozen": fl_fz})

setup()
c = SEQ
fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.4))
ax = axs[0, 0]
ax.plot(D0, cols["Rf"], color=c[0])
ax.plot(D0, cols["Ri"], color=c[1], ls="--")
ax.plot(D0, cols["Rt"], color=c[3], ls=":")
ax.set_xlabel(r"release depth $d_0$ (m)")
ax.set_ylabel(r"$R(0.5\,\mathrm{m})/R_0$")
panel_label(ax, "(a)", "radius near the surface")

ax = axs[0, 1]
Pg = np.logspace(np.log10(Pi.min()), np.log10(Pi.max()), 200)
ax.plot(Pi, cols["gf"], color=c[0])
ax.plot(Pi, cols["gi"], color=c[1], ls="--")
ax.plot(Pg, rd.turner_ratio(Pg), color=c[2], ls="-.")
ax.set_xscale("log")
ax.set_xlabel(r"$\Pi=\rho g (d_0-d_e)/p(d_e)$")
ax.set_ylabel(r"$\Delta R_{\rm Boyle}/\Delta R_{V_0}$")
panel_label(ax, "(b)", "gain over constant volume")

ax = axs[1, 0]
ax.plot(D0, cols["Tf"], color=c[0])
ax.plot(D0, cols["Ti"], color=c[1], ls="--")
ax.plot(D0, cols["Tfz"], color=c[2], ls="-.")
ax.plot(D0, cols["Tt"], color=c[3], ls=":")
ax.set_xlabel(r"release depth $d_0$ (m)")
ax.set_ylabel(r"ascent time $T$ (s)")
panel_label(ax, "(c)", "ascent time")

ax = axs[1, 1]
ax.plot(D0, cols["flf"], color=c[0])
ax.plot(D0, fl_fz, color=c[2], ls="-.")
ax.plot(D0, cols["flt"], color=c[3], ls=":")
ax.axhline(1.0, color="0.6", lw=0.6)
ax.set_xlabel(r"release depth $d_0$ (m)")
ax.set_ylabel(r"$(dR/dh)_{0.5\,\mathrm{m}}/(dR/dh)_{d_0}$")
panel_label(ax, "(d)", "flaring near the surface")
for ax in axs.flat:
    ax.minorticks_on()
fig.subplots_adjust(wspace=0.36, hspace=0.45, bottom=0.14)
h, lab = handles(["full closure", "exact ideal gas", "frozen-log closed form",
                  "constant volume"], c[:4], ["-", "--", "-.", ":"])
legend_below(fig, axs[1, :], h, lab, ncol=4, pad=0.02)
print(save(fig, "fig02_release"))
