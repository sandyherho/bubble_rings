"""Figure 5. Wobble of a wavy ring.

Rings released level at 10 m with random bending and breathing waves in
azimuthal modes n = 2..8 (rms 2 mm per mode, 1.4 mm in each of the
radial and axial components; four seeds),
followed to 0.5 m with the filament model (128 nodes).  Solid: gas
volume follows the hydrostatic pressure; dashed: gas volume held at V0.
(a) Total wave amplitude A = (sum_n A_n^2)^(1/2) over its initial value
against R/R0, with the slope -1/2 of conserved wave action (gray).
(b) Amplitude of each mode for the first seed (Boyle).
(c) Least-squares exponent of A_n ~ R^beta for each mode (symbols:
seed mean, bars: seed range) and of the total amplitude (right).
(d) Normalized spectral entropy of the mode energies against depth.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

from bubble_rings.experiments import ascent, wave_modes
from bubble_rings.io_utils import write_csv, save_cache, load_cache
from bubble_rings.plotting import (setup, SEQ, panel_label, legend_below,
                                   handles, save)
from bubble_rings.scenario import PRM

SEEDS = [3, 5, 8, 13]
AMP = 0.002
CASES = [(s, b) for b in (True, False) for s in SEEDS]


def job(case):
    """One filament ascent; returns the sampled diagnostics."""
    s, b = case
    o = ascent(PRM, modes=wave_modes(s, AMP), N=128, dt0=2e-3,
               t_step=0.25, boyle=b)
    return {k: o[k] for k in ("t", "d", "R", "E", "H", "dz", "a")}


cache = load_cache("fig05")
if cache is None:
    with Pool(2) as pool:
        res = pool.map(job, CASES)
    cache = {}
    for (s, b), o in zip(CASES, res):
        for k, v in o.items():
            cache[f"{k}_{s}_{int(b)}"] = v
    save_cache("fig05", **cache)

NS = np.arange(2, 25)
NM = 7                      # modes 2..8 carry the initial waves
fits = {}
for s, b in CASES:
    key = f"{s}_{int(b)}"
    lR = np.log(cache[f"R_{key}"])
    A = np.sqrt(2 * cache[f"E_{key}"][:, :NM])
    fits[key] = np.array([np.polyfit(lR, np.log(A[:, j]), 1)[0]
                          for j in range(NM)]
                         + [np.polyfit(lR, 0.5 * np.log(
                             cache[f"E_{key}"][:, :NM].sum(1)), 1)[0]])
    write_csv(f"fig05_run_seed{s}_{'boyle' if b else 'constV'}",
              {"t_s": cache[f"t_{key}"], "depth_m": cache[f"d_{key}"],
               "R_m": cache[f"R_{key}"], "H": cache[f"H_{key}"],
               "dz_m": cache[f"dz_{key}"],
               **{f"A{n}_m": A[:, j] for j, n in enumerate(NS[:NM])}})
FB = np.array([fits[f"{s}_1"] for s in SEEDS])
FC = np.array([fits[f"{s}_0"] for s in SEEDS])
write_csv("fig05c_exponents", {"mode": np.r_[NS[:NM], 0],
                               "boyle_mean": FB.mean(0),
                               "boyle_min": FB.min(0),
                               "boyle_max": FB.max(0),
                               "constV_mean": FC.mean(0),
                               "constV_min": FC.min(0),
                               "constV_max": FC.max(0)})

setup()
fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.6))
ax = axs[0, 0]
for i, s in enumerate(SEEDS):
    for b, ls in ((1, "-"), (0, "--")):
        key = f"{s}_{b}"
        A = np.sqrt(cache[f"E_{key}"][:, :NM].sum(1))
        ax.plot(cache[f"R_{key}"] / PRM.R0, A / A[0], color=SEQ[i], ls=ls,
                lw=1.0)
rg = np.linspace(1, 3.8, 20)
ax.plot(rg, rg ** -0.5, color="0.6", lw=2.4, zorder=0)
panel_label(ax, "(a)", "total wave amplitude")
ax.set_ylabel(r"$A/A(0)$")

ax = axs[0, 1]
key = f"{SEEDS[0]}_1"
A = np.sqrt(2 * cache[f"E_{key}"][:, :NM])
for j in range(NM):
    ax.plot(cache[f"R_{key}"] / PRM.R0, 1e3 * A[:, j],
            color=plt.cm.viridis(j / (NM - 1)), lw=1.0)
ax.set_ylabel(r"mode amplitude $A_n$ (mm)")

panel_label(ax, "(b)", "modes, first seed")
for ax in axs[0]:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$R/R_0$")
axs[0, 1].set_yticks([0.2, 0.5, 1, 2, 5])
axs[0, 1].yaxis.set_major_formatter(ScalarFormatter())
axs[0, 1].yaxis.set_minor_formatter(NullFormatter())
axs[0, 0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axs[0, 0].yaxis.set_major_formatter(ScalarFormatter())
axs[0, 0].yaxis.set_minor_formatter(NullFormatter())

ax = axs[1, 0]
xs = np.r_[NS[:NM], NS[NM - 1] + 1.6]
for F, dx, mk, fc in ((FB, -0.12, "o", "0.2"), (FC, 0.12, "s", "none")):
    m = F.mean(0)
    ax.errorbar(xs + dx, m, yerr=[m - F.min(0), F.max(0) - m], fmt=mk,
                ms=4, mfc=fc, color="0.2", capsize=2, lw=0.9)
ax.axhline(-0.5, color="0.6", lw=1.2)
ax.set_xticks(xs)
ax.set_xticklabels([str(n) for n in NS[:NM]] + ["all"])
ax.set_xlabel(r"azimuthal mode $n$")
ax.set_ylabel(r"exponent $\beta$ in $A_n\propto R^{\beta}$")
panel_label(ax, "(c)", "decay exponents")

ax = axs[1, 1]
for i, s in enumerate(SEEDS):
    for b, ls in ((1, "-"), (0, "--")):
        key = f"{s}_{b}"
        ax.plot(cache[f"H_{key}"], cache[f"d_{key}"], color=SEQ[i], ls=ls,
                lw=1.0)
ax.set_xlim(0, 1)
ax.set_ylim(PRM.d0 + 0.2, 0)
ax.set_xlabel(r"spectral entropy $H$")
ax.set_ylabel(r"depth $d$ (m)")
panel_label(ax, "(d)", "mode-energy spread")
for a in axs[1]:
    a.minorticks_on()
axs[1, 0].tick_params(axis="x", which="minor", bottom=False, top=False)
fig.subplots_adjust(wspace=0.38, hspace=0.62, bottom=0.16)
h, lab = handles([f"{n}" for n in NS[:NM]],
                 [plt.cm.viridis(j / (NM - 1)) for j in range(NM)])
lab[0] = "n = 2"
legend_below(fig, axs[0, 1], h, lab, ncol=7, pad=0.015)
h, lab = handles([f"seed {s}" for s in SEEDS] + ["Boyle", "constant V"],
                 SEQ[:4] + ["0.3", "0.3"], ["-"] * 5 + ["--"])
h2, lab2 = handles(["Boyle (c)", "constant V (c)"], ["0.2", "0.2"],
                   ["none", "none"], ["o", "s"])
h2[0].set_markerfacecolor("0.2")
legend_below(fig, axs[1, :], h + h2, lab + lab2, ncol=4, pad=0.02)
print(save(fig, "fig05_wobble"))
