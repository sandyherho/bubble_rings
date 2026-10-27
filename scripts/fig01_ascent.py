"""Figure 1. Ascent of the reference ring.

A level ring (R0 = 5 cm, 10 mL of gas, Gamma = 0.06 m^2/s) released at
10 m and followed to 0.5 m with the planar thin-ring model.
(a) Ring radius against depth: full core closure (swirl suction and
surface tension), the exact ideal-gas solution in exponential integrals,
the frozen-logarithm Boyle law R0 + ell_B ln(p0/p), and a ring whose gas
volume is held at V0 (constant buoyancy).
(b) Expansion per meter of rise, dR/dh, for the same cases; the frozen-
log law gives ell_B rho g / p, which grows toward the surface as 1/p.
(c) Rise speed.
(d) Gas volume V/V0 and core radius a/a0.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from bubble_rings.io_utils import write_csv
from bubble_rings.plotting import (setup, SEQ, panel_label, legend_below,
                                   handles, save)
from bubble_rings import reduced as rd
from bubble_rings.scenario import PRM

full = rd.integrate(PRM, mode="full")
ideal = rd.integrate(PRM, mode="ideal")
turn = rd.integrate(PRM, mode="turner")
L0 = rd.L_release(PRM)
dg = np.linspace(PRM.d_end, PRM.d0, 400)
R_ex = rd.R_exact(PRM, dg)
R_fz = rd.R_log(PRM, dg)


def dRdh(o):
    """Expansion per meter of rise of a level ring, (dR/dt) / U."""
    return PRM.g * o["V"] / (2 * np.pi * PRM.Gamma * o["R"]) / o["U"]


# exact ideal-gas slope: L dR = ell_B L0 dlambda, dlambda/dh = rho g / p
lam = rd.lam_of_d(PRM, dg)
L_ex = L0 + 1.5 * np.log(R_ex / PRM.R0) - 0.5 * lam
slope_ex = rd.boyle_length(PRM) * L0 / L_ex * PRM.rho * PRM.g / PRM.p_inf(dg)
slope_fz = rd.boyle_length(PRM) * PRM.rho * PRM.g / PRM.p_inf(dg)

write_csv("fig01a_radius", {"depth_m": full["d"], "R_full_m": full["R"],
                            "R_turner_m": np.interp(full["d"][::-1],
                                                    turn["d"][::-1],
                                                    turn["R"][::-1])[::-1]})
write_csv("fig01a_closed", {"depth_m": dg, "R_exact_m": R_ex,
                            "R_frozen_m": R_fz})
write_csv("fig01b_slope", {"depth_m": dg, "dRdh_exact": slope_ex,
                           "dRdh_frozen": slope_fz})
write_csv("fig01cd_full", {"t_s": full["t"], "depth_m": full["d"],
                           "U_ms": full["U"], "V_m3": full["V"],
                           "a_m": full["a"]})

setup()
fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.6))
c = SEQ
ax = axs[0, 0]
ax.plot(100 * full["R"], full["d"], color=c[0])
ax.plot(100 * R_ex, dg, color=c[1], ls="--")
ax.plot(100 * R_fz, dg, color=c[2], ls="-.")
ax.plot(100 * turn["R"], turn["d"], color=c[3], ls=":")
ax.set_xlabel(r"ring radius $R$ (cm)")
ax.set_ylabel(r"depth $d$ (m)")
panel_label(ax, "(a)", "radius")

ax = axs[0, 1]
ax.plot(100 * dRdh(full), full["d"], color=c[0])
ax.plot(100 * slope_ex, dg, color=c[1], ls="--")
ax.plot(100 * slope_fz, dg, color=c[2], ls="-.")
ax.plot(100 * dRdh(turn), turn["d"], color=c[3], ls=":")
ax.set_xlabel(r"expansion per meter of rise $dR/dh$ (cm m$^{-1}$)")
panel_label(ax, "(b)", "flaring")

ax = axs[1, 0]
ax.plot(full["U"], full["d"], color=c[0])
ax.plot(ideal["U"], ideal["d"], color=c[1], ls="--")
ax.plot(turn["U"], turn["d"], color=c[3], ls=":")
ax.set_xlabel(r"rise speed $U$ (m s$^{-1}$)")
ax.set_ylabel(r"depth $d$ (m)")
panel_label(ax, "(c)", "speed")

ax = axs[1, 1]
ax.plot(full["V"] / PRM.V0, full["d"], color=c[0])
ax.plot(full["a"] / PRM.a0, full["d"], color=c[0], ls=(0, (5, 1.5)))
ax.plot(turn["a"] / PRM.a0, turn["d"], color=c[3], ls=":")
ax.set_xlabel(r"$V/V_0$ (solid), $a/a_0$ (dashed, dotted)")
panel_label(ax, "(d)", "gas volume and core")
for ax in axs.flat:
    ax.set_ylim(PRM.d0, 0)
    ax.minorticks_on()
fig.subplots_adjust(wspace=0.28, hspace=0.42, bottom=0.14)
h, lab = handles(["full closure", "exact ideal gas", "frozen-log Boyle law",
                  "constant volume"], c[:4], ["-", "--", "-.", ":"])
legend_below(fig, axs[1, :], h, lab, ncol=4, pad=0.02)
print(save(fig, "fig01_ascent"))
