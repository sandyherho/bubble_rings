"""Figure 6. Verification.

(a) Speed of a circular filament against the hollow-ring speed
Gamma/(4 pi R)[ln(8R/a) - 1/2 + 1/We], relative error against node
number for slenderness a/R = 0.1, 0.03 and 0.01, with the near-singular
subtraction (solid) and with the plain trapezoidal rule (dashed).  The
plateau of the subtracted rule is the O(eps^2 ln eps) difference between
the regularized ring and the thin-ring asymptote.
(b) Temporal convergence of RK4: maximum node displacement error of a
wavy ring (64 nodes) after 0.4 s against the step, relative to a run
with dt = 6.25e-5 s; the gray line has slope 4.
(c) Kelvin impulse balance of a wavy ring tilted by 30 deg:
|I(t) - I(0) - rho g z_hat int V dt| / |I(t)|, vertical and horizontal
components, with V integrated by the composite Simpson rule on 0.01 s
samples.
(d) Filament against the planar thin-ring model after rising from 10 to
7 m, relative difference in radius and difference in arrival time over
the ascent time, against the release slenderness a0/R0; the gray line
has slope 2.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, NullFormatter
from scipy.integrate import cumulative_simpson

from bubble_rings.core import ring_speed
from bubble_rings.experiments import ascent, wave_modes
from bubble_rings.filament import Filament, circle, ring_diagnostics
from bubble_rings.io_utils import write_csv, save_cache, load_cache
from bubble_rings.plotting import (setup, SEQ, panel_label, legend_below,
                                   handles, save)
from bubble_rings import reduced as rd
from bubble_rings.scenario import PRM

EPS = [0.1, 0.03, 0.01]
NN = [16, 32, 64, 128, 256, 512]
DTS = np.array([4e-3, 2e-3, 1e-3, 5e-4])
EPS_D = np.array([0.02, 0.04, 0.08, 0.16])


def p_eps(e):
    """Return the reference parameters with slenderness a0/R0 = e."""
    return replace(PRM, V0=2 * np.pi ** 2 * PRM.R0 ** 3 * e * e)


cache = load_cache("fig06")
if cache is None:
    out = {}
    for e in EPS:
        p = p_eps(e)
        for sub in (1, 0):
            err = []
            for N in NN:
                f = Filament(p, circle(N, p.R0, (0, 0, -p.d0)),
                             buoyancy=False, subtract=bool(sub))
                u, st = f.velocity(f.X)
                err.append(abs(u[0, :, 2].mean()
                               / ring_speed(p, p.R0, st["a"][0]) - 1))
            out[f"speed_{e:g}_{sub}"] = np.array(err)

    X0 = circle(64, PRM.R0, (0, 0, -PRM.d0), modes=wave_modes(3, 0.002))

    def run_dt(dt):
        """Wavy ring to 0.4 s with a fixed step."""
        f = Filament(PRM, X0)
        f.run(0.4, dt=dt)
        return f.X[0]
    ref = run_dt(6.25e-5)
    out["dt_err"] = np.array([np.abs(run_dt(dt) - ref).max() for dt in DTS])

    th = np.radians(30.0)
    X0 = circle(128, PRM.R0, (0, 0, -PRM.d0), (np.sin(th), 0, np.cos(th)),
                wave_modes(5, 0.002))
    f = Filament(PRM, X0)
    rec = {"t": [], "I": [], "V": []}

    def grab(fl):
        """Store impulse and gas volume."""
        rec["t"].append(fl.t)
        rec["I"].append(fl.impulse()[0])
        rec["V"].append(fl.state()["V"][0])
    grab(f)
    f.run_sampled(np.arange(0.01, 5.0 + 1e-9, 0.01), dt0=2e-3,
                  callback=grab)
    t = np.array(rec["t"])
    imp = np.array(rec["I"])
    V = np.array(rec["V"])
    iv = cumulative_simpson(V, x=t, initial=0.0)
    res = imp - imp[0]
    res[:, 2] -= PRM.rho * PRM.g * iv
    nI = np.linalg.norm(imp, axis=1)
    out["imp_t"] = t
    out["imp_z"] = np.abs(res[:, 2]) / nI
    out["imp_h"] = np.hypot(res[:, 0], res[:, 1]) / nI

    dR, dT = [], []
    for e in EPS_D:
        p = p_eps(e)
        o = ascent(p, N=64, dt0=min(4e-3, 4e-3 * (e / PRM.eps0) ** 2),
                   t_step=0.02, d_end=7.0)
        r = rd.integrate(p, mode="full", d_end=7.0)
        # radius and time when each model first reaches 7 m
        tf = np.interp(7.0, o["d"][::-1], o["t"][::-1])
        Rf = np.interp(7.0, o["d"][::-1], o["R"][::-1])
        dR.append(abs(Rf / r["R"][-1] - 1))
        dT.append(abs(tf / r["T"] - 1))
    out["dR"] = np.array(dR)
    out["dT"] = np.array(dT)

    # node-number convergence of the wavy ring at 3 s
    conv = []
    for N in (64, 128, 256):
        X0 = circle(N, PRM.R0, (0, 0, -PRM.d0), modes=wave_modes(3, 0.002))
        f = Filament(PRM, X0)
        f.run_sampled([3.0], dt0=1e-3 if N == 256 else 2e-3, scale=False)
        dg = ring_diagnostics(PRM, f.X[0])
        conv.append([dg["R"], dg["c"][2], np.sqrt(dg["E"][:7].sum())])
    out["Nconv"] = np.array(conv)
    save_cache("fig06", **out)
    cache = out

write_csv("fig06a_speed", {"N": np.array(NN, float),
                           **{f"eps{e:g}_{'sub' if s else 'plain'}":
                              cache[f"speed_{e:g}_{s}"]
                              for e in EPS for s in (1, 0)}})
write_csv("fig06b_rk4", {"dt_s": DTS, "max_err_m": cache["dt_err"]})
write_csv("fig06c_impulse", {"t_s": cache["imp_t"], "rel_z": cache["imp_z"],
                             "rel_h": cache["imp_h"]})
write_csv("fig06d_reduced", {"eps0": EPS_D, "rel_dR": cache["dR"],
                             "rel_dT": cache["dT"]})
write_csv("fig06e_nodes", {"N": np.array([64.0, 128.0, 256.0]),
                           "R_m": cache["Nconv"][:, 0],
                           "z_m": cache["Nconv"][:, 1],
                           "A_m": cache["Nconv"][:, 2]})

setup()
fig, axs = plt.subplots(2, 2, figsize=(6.6, 5.6))
ax = axs[0, 0]
for e, col in zip(EPS, SEQ):
    ax.plot(NN, cache[f"speed_{e:g}_1"], color=col, marker="o", ms=3)
    ax.plot(NN, cache[f"speed_{e:g}_0"], color=col, ls="--", marker="s",
            ms=3, mfc="none")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("nodes $N$")
ax.set_ylabel("relative speed error")
panel_label(ax, "(a)", "ring speed")

ax = axs[0, 1]
ax.plot(DTS, cache["dt_err"], color=SEQ[0], marker="o", ms=3)
ax.plot(DTS, cache["dt_err"][0] * (DTS / DTS[0]) ** 4, color="0.6", lw=1.2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(DTS)
ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r"time step $\Delta t$ (s)")
ax.set_ylabel("max node error (m)")
panel_label(ax, "(b)", "RK4 order")

ax = axs[1, 0]
ax.plot(cache["imp_t"][1:], cache["imp_z"][1:], color=SEQ[1])
ax.plot(cache["imp_t"][1:], cache["imp_h"][1:], color=SEQ[3])
ax.set_yscale("log")
ax.set_xlabel(r"time $t$ (s)")
ax.set_ylabel("relative impulse residual")
panel_label(ax, "(c)", "Kelvin impulse balance")

ax = axs[1, 1]
ax.plot(EPS_D, cache["dR"], color=SEQ[2], marker="o", ms=3)
ax.plot(EPS_D, cache["dT"], color=SEQ[4], marker="s", ms=3)
ax.plot(EPS_D, cache["dR"][-1] * (EPS_D / EPS_D[-1]) ** 2, color="0.6",
        lw=1.2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks(EPS_D)
ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel(r"release slenderness $a_0/R_0$")
ax.set_ylabel("filament vs planar model")
panel_label(ax, "(d)", "thin-ring limit")
fig.subplots_adjust(wspace=0.38, hspace=0.70, bottom=0.12)
h, lab = handles([f"{e:g}" for e in EPS] + ["subtracted", "plain"],
                 SEQ[:3] + ["0.3", "0.3"], ["-"] * 4 + ["--"])
lab[0] = "a/R = 0.1"
legend_below(fig, axs[0, 0], h, lab, ncol=5, pad=0.015)
h, lab = handles(["vertical", "horizontal"], [SEQ[1], SEQ[3]])
legend_below(fig, axs[1, 0], h, lab, ncol=2, pad=0.015)
h, lab = handles(["radius", "arrival time"], [SEQ[2], SEQ[4]],
                 markers=["o", "s"])
legend_below(fig, axs[1, 1], h, lab, ncol=2, pad=0.015)
print(save(fig, "fig06_verification"))
