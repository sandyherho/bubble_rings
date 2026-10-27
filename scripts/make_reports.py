"""Generate the plain-text reports.

Numbers are recomputed here from the library or read back from the CSV
files written by the figure scripts, so every value quoted in a report
can be traced to a file under outputs/.  The sensitivity report runs its
own small set of planar-model integrations.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import os
from dataclasses import replace

import numpy as np
from scipy.optimize import brentq

from bubble_rings import reduced as rd
from bubble_rings.core import (gas_constant, core_radius, weber, froude,
                               swirl_head, swirl_speed, capillary_growth)
from bubble_rings.filament import F_beta, _F_exact
from bubble_rings.io_utils import DATADIR, write_report, write_csv
from bubble_rings.scenario import PRM


def csv(stem):
    """Read a CSV written by io_utils.write_csv as a structured array."""
    return np.genfromtxt(os.path.join(DATADIR, f"{stem}.csv"),
                         delimiter=",", names=True)


def wrap(text, width=68):
    """Wrap prose to report lines indented two spaces."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append("  " + cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append("  " + cur)
    return lines


def js(values, spec):
    """Join numbers with a format spec."""
    return ", ".join(format(float(v), spec) for v in values)


def dots(label, value, width=52):
    """Dotted leader line."""
    return "  " + label + " " + "." * max(2, width - len(label)) + " " + value


P = PRM
L0 = rd.L_release(P)
lB = rd.boyle_length(P)
full = rd.integrate(P, mode="full")
ideal = rd.integrate(P, mode="ideal")
turn = rd.integrate(P, mode="turner")
a0 = P.a0
t_th = a0 ** 2 / P.kappa_gas
t_p = P.p0 / (P.rho * P.g * full["U"][0])

# ------------------------------------------------------------- parameters
write_report("parameters", "Symbols, units, and reference values", [
    ("units", wrap(
        "SI throughout: m, s, Pa, m^3 (mL where stated), m^2 s^-1 for "
        "circulation. z points up and depth is d = -z. The ring is "
        "followed from its release depth d0 to d_e = 0.5 m, where the "
        "free surface lies more than twice the largest ring radius above "
        "the center.")),
    ("reference values (illustrative, not calibrated)", [
        dots("sea water density rho", f"{P.rho} kg m^-3"),
        dots("gravity g", f"{P.g} m s^-2"),
        dots("surface tension gamma", f"{P.gamma} N m^-1"),
        dots("surface pressure p_atm", f"{P.p_atm} Pa"),
        dots("release depth d0", f"{P.d0} m"),
        dots("ring radius R0", f"{P.R0} m"),
        dots("gas volume at release V0", f"{P.V0 * 1e6:g} mL"),
        dots("circulation Gamma", f"{P.Gamma} m^2 s^-1"),
        dots("polytropic exponent", f"{P.poly:g} (isothermal)"),
        dots("end depth d_e", f"{P.d_end} m"),
    ]),
    ("derived at release", [
        dots("core radius a0 = (V0/(2 pi^2 R0))^(1/2)", f"{1e3 * a0:.3f} mm"),
        dots("slenderness a0/R0", f"{P.eps0:.4f}"),
        dots("ln(8R0/a0) - 1/2 = L0", f"{L0:.4f}"),
        dots("Boyle length ell_B = 2 V0 p0/(rho Gamma^2 L0)",
             f"{100 * lB:.2f} cm"),
        dots("release pressure number rho g d0/p_atm", f"{P.Pi:.4f}"),
        dots("swirl speed u_c = Gamma/(2 pi a0)",
             f"{swirl_speed(P, a0):.3f} m s^-1"),
        dots("swirl Weber number", f"{weber(P, a0):.1f}"),
        dots("core Froude number", f"{froude(P, a0):.1f}"),
        dots("swirl head u_c^2/(2g)", f"{swirl_head(P, a0):.3f} m"),
        dots("swirl pressure deficit / p0",
             f"{swirl_head(P, a0) * P.rho * P.g / P.p0:.4f}"),
        dots("circulation Reynolds number Gamma/nu", f"{P.Re:.3g}"),
        dots("rise speed at release", f"{full['U'][0]:.3f} m s^-1"),
        dots("thermal time a0^2/kappa_gas", f"{t_th:.3f} s"),
        dots("pressure time p0/(rho g U0)", f"{t_p:.1f} s"),
        dots("ratio (isothermal if << 1)", f"{t_th / t_p:.4f}"),
    ]),
])

# ------------------------------------------------------------- closed forms
Rex = rd.R_exact(P, P.d_end)
Tex = rd.ascent_time_exact(P)
blocks = [
    ("Kelvin impulse balance", wrap(
        "A massless bubble in unbounded liquid transfers its whole "
        "buoyancy to the Kelvin impulse of the liquid, dI/dt = rho g V "
        "z_hat, with V(t) free to change (Benjamin and Ellis 1966; "
        "Benjamin 1987). For a thin planar ring I = rho pi Gamma R^2 n. "
        "Hence J = R^2 n obeys dJ/dt = g V/(pi Gamma) z_hat: |J| = R^2 "
        "grows with the time integral of V, and the horizontal part of J "
        "is conserved, sin(theta) R^2 = sin(theta0) R0^2 for any tilt. "
        "For a level ring of constant volume this is the law a^2 = a0^2 + "
        "g V t/(2 pi^2 Gamma_P) of Pedley (1968, Eq. 2.10), whose "
        "circulation is Gamma_P = Gamma/(2 pi), and of Turner (1957).")),
    ("speed of a hollow ring with surface tension", wrap(
        "The Hamiltonian H = (1/2) rho Gamma^2 R [ln(8R/a) - 2] + "
        "4 pi^2 gamma R a + 2 pi^2 p R a^2 + G(V), G the free energy of "
        "the gas, is stationary in a at the core balance p_g = p - rho "
        "Gamma^2/(8 pi^2 a^2) + gamma/a. U = dH/dI at fixed Gamma then "
        "gives U = Gamma/(4 pi R) [ln(8R/a) - 1/2 + 1/We], We = rho "
        "Gamma^2/(4 pi^2 gamma a); the hollow constant 1/2 of Hicks is "
        "recovered for gamma = 0. With isothermal gas the balance times "
        "a^2 is a quadratic in a with one positive root. The same core "
        "balance, with adiabatic gas, appears as Eq. 53 of Ruban and "
        "Rasmussen (2003).")),
    ("earlier treatments of hydrostatic volume change", wrap(
        "Pedley (1968, section 5) wrote the impulse equation with Boyle's "
        "law, da^2/dt = C1 (1 + h0/d)/(1 + h/d), noted that it is not "
        "readily integrable, and integrated it numerically for one "
        "laboratory bubble released about 3.4 m below the surface; the "
        "radius differed from the constant-volume value by just under 7 "
        "percent after 40 s, which he judged nearly negligible. Ruban and "
        "Rasmussen (2003) included an adiabatic gas in a variational "
        "model of a vertically moving bubble ring and integrated it "
        "numerically. Neither gives a closed form. The exact solution "
        "above, the universal gain ((1 + Pi)/Pi) ln(1 + Pi) and its "
        "growth with release depth, under which the correction dominates "
        "at diving depths, are new as far as the authors know.")),
    ("frozen-logarithm solution (ideal isothermal gas)", wrap(
        "With L frozen, L dR = -(2 g V/Gamma^2) dd and V = V0 p0/p give "
        "R = R0 + ell_B ln(p0/p), ell_B = 2 V0 p0/(rho Gamma^2 L). "
        "The same law holds for any tilt, because the tilt cancels from "
        "the radius equation. Over a constant-volume ring the radius gain "
        "is ((1 + Pi)/Pi) ln(1 + Pi), Pi = rho g (d0 - d_e)/p(d_e); the "
        "expansion per meter of rise, ell_B rho g/p, grows as 1/p.")),
    ("exact solution with the running logarithm", wrap(
        "For the ideal gas L = L0 + (3/2) ln r - lambda/2 with r = R/R0 "
        "and lambda = ln(p0/p), so L dR = ell_B L0 dlambda is linear in "
        "lambda(r): b dlambda/dr + lambda/2 = L0 + (3/2) ln r, b = ell_B "
        "L0/R0. With alpha = 1/(2b): lambda(r) = 2 L0 (1 - exp(-alpha "
        "(r - 1))) + 3 [ln r - exp(-alpha r)(Ei(alpha r) - Ei(alpha))]. "
        "Ascent time and lateral travel follow by one quadrature.")),
    ("reference ring, values", [
        dots("R(0.5 m), full closure ODE", f"{100 * full['R'][-1]:.3f} cm"),
        dots("R(0.5 m), ideal-gas ODE", f"{100 * ideal['R'][-1]:.4f} cm"),
        dots("R(0.5 m), exact Ei solution", f"{100 * Rex:.4f} cm"),
        dots("R(0.5 m), frozen-log law",
             f"{100 * rd.R_log(P, P.d_end):.3f} cm"),
        dots("R(0.5 m), constant volume", f"{100 * turn['R'][-1]:.3f} cm"),
        dots("ascent time, full / exact / frozen",
             f"{full['T']:.2f} / {Tex:.2f} / {rd.ascent_time(P):.2f} s"),
        dots("ascent time, constant volume", f"{turn['T']:.2f} s"),
        dots("lateral travel, theta0 = 30 deg (exact)",
             f"{rd.lateral_offset_exact(P, np.radians(30)):.3f} m"),
        dots("tilt at 0.5 m, theta0 = 30 deg",
             f"{np.degrees(rd.theta_law(P, Rex, np.radians(30))):.2f} deg"),
    ]),
    ("hollow-core capillary waves", wrap(
        "Linearizing about the potential vortex outside a gas column of "
        "radius a with surface tension gives, for azimuthal mode m and "
        "axial wavenumber k, (omega - m Omega)^2 = [Omega^2 + gamma (k^2 "
        "a^2 + m^2 - 1)/(rho a^3)] / G, G = -K_m(ka)/(ka K_m'(ka)) > 0. "
        "Only m = 0 can be unstable, and only if We < 1 - (ka)^2, so swirl "
        "stabilizes every wavelength once We > 1. Without swirl the "
        "classical gas-column result is recovered. These conditions are "
        "those of Pedley (1967), applied to the toroidal bubble by Pedley "
        "(1968, Eqs. 6.2 to 6.4); they are rederived here only to fix "
        "notation.")),
    ("viscous breakup criterion (Pedley 1968)", wrap(
        "Vorticity diffusing from the free-slip core surface lowers the "
        "swirl there to v(b) = Gamma_P/b [1 - exp(-3 b^2/(8 nu t))] "
        "(Pedley 1968, Eq. 6.1), so the capillary margin becomes M = We "
        "[1 - exp(-3 a^2/(8 nu t))]^2 and the core turns unstable when M "
        "= 1 (Eq. 6.5). Evaluated along the present trajectories, with "
        "the core radius a(t) from the Boyle closure, it gives a breakup "
        "time and depth for each release depth (figure 4b).")),
]
write_report("closed_forms", "Closed forms and their values", blocks)

# ------------------------------------------------------------- verification
# Pedley (1968), section 5: Walters and Davidson's bubble, Gamma_P = 50
# cm^2/s, V = 21 cm^3, a0 = 5 cm, fresh water, released d/3 below the
# surface with d = 10.30 m the water-column height of the atmosphere.
PP = replace(P, rho=1000.0, gamma=0.074, R0=0.05, V0=21e-6,
             Gamma=2 * np.pi * 50e-4, d0=10.30 / 3,
             p_atm=1000.0 * P.g * 10.30, d_end=-50.0)
ped_i = rd.integrate(PP, mode="ideal", t_max=40.0)
ped_t = rd.integrate(PP, mode="turner", t_max=40.0)
ped_dev = 100 * (ped_i["R"][-1] / ped_t["R"][-1] - 1)
write_csv("verification_pedley1968", {
    "t_s": ped_i["t"], "R_boyle_m": ped_i["R"],
    "R_constV_m": np.interp(ped_i["t"], ped_t["t"], ped_t["R"])})
o_id = ideal
err_exact = np.max(np.abs(o_id["R"] - rd.R_exact(P, o_id["d"])) / o_id["R"])
fz = rd.integrate(P, mode="ideal", L=L0)
err_fz = abs(fz["R"][-1] / rd.R_log(P, P.d_end) - 1)
err_T = abs(fz["T"] / rd.ascent_time(P) - 1)
th = np.radians(30)
ot = rd.integrate(P, mode="ideal", theta0=th)
err_lat = abs(ot["x"][-1] / rd.lateral_offset_exact(P, th) - 1)
err_th = np.max(np.abs(np.sin(ot["theta"])
                       - np.sin(th) * (P.R0 / ot["R"]) ** 2))
err_tiltR = abs(ot["R"][-1] / ideal["R"][-1] - 1)
# quadratic root against bracketing
C = gas_constant(P)
aq = core_radius(P, C, 2 * np.pi * P.R0, P.p0)


def balance(a):
    """Core pressure balance at release, zero at the core radius."""
    return (P.p0 - P.rho * P.Gamma ** 2 / (8 * np.pi ** 2 * a * a)
            + P.gamma / a - C / (2 * np.pi ** 2 * P.R0 * a * a))


ab = brentq(balance, 1e-4, 1e-2, xtol=1e-16, rtol=1e-15)
Fb = [abs(F_beta(b) / _F_exact(b) - 1) for b in (3e-6, 1e-3, 0.03, 0.5, 2)]
sp = csv("fig06a_speed")
rk = csv("fig06b_rk4")
order = np.diff(np.log(rk["max_err_m"])) / np.diff(np.log(rk["dt_s"]))
imp = csv("fig06c_impulse")
red = csv("fig06d_reduced")
ordR = np.diff(np.log(red["rel_dR"])) / np.diff(np.log(red["eps0"]))
nod = csv("fig06e_nodes")
x = np.linspace(1e-3, 0.999, 500)
write_report("verification", "Numerical verification", [
    ("independent algorithms", wrap(
        "The planar model integrates the Kelvin impulse balance with an "
        "adaptive eighth-order Runge-Kutta method. The closed forms are "
        "evaluated by root finding and adaptive quadrature. The filament "
        "model integrates the regularized Biot-Savart law with a local "
        "Kutta-Joukowski drift and never uses the impulse balance, so its "
        "agreement with the planar model and with the impulse identity is "
        "an independent check.")),
    ("closed forms against the planar ODE", [
        dots("exact Ei radius, max relative error", f"{err_exact:.2e}"),
        dots("frozen-log radius at 0.5 m", f"{err_fz:.2e}"),
        dots("frozen-log ascent time", f"{err_T:.2e}"),
        dots("exact lateral travel, 30 deg", f"{err_lat:.2e}"),
        dots("sin(theta) law, max abs error", f"{err_th:.2e}"),
        dots("tilted vs level radius at 0.5 m", f"{err_tiltR:.2e}"),
        dots("core quadratic vs bracketing root", f"{abs(aq / ab - 1):.2e}"),
        dots("F(beta) table vs quadrature, max", f"{max(Fb):.2e}"),
        dots("Pedley (1968) bubble, Boyle vs constant V, 40 s",
             f"{ped_dev:+.1f} % (Pedley: just under 7 %)"),
        dots("same, core radius b0", f"{1e3 * PP.a0:.2f} mm (0.46 cm)"),
        dots("capillary growth at We = 1, max",
             f"{capillary_growth(x, 1.0).max():.2e}"),
    ]),
    ("filament", [
        "  ring speed, relative error at N = 16 .. 512 (subtracted):",
        *[f"    a/R = {e:<5}: " + js(sp[f"eps{e[0]}{e[2:]}_sub"], ".2e")
          for e in ("0.1", "0.03", "0.01")],
        "  same, plain trapezoidal rule:",
        *[f"    a/R = {e:<5}: " + js(sp[f"eps{e[0]}{e[2:]}_plain"], ".2e")
          for e in ("0.1", "0.03", "0.01")],
        "  RK4 node error vs dt " + js(rk["max_err_m"], ".2e") + " m",
        "    observed orders " + js(order, ".2f"),
        dots("impulse residual, vertical, max over 5 s",
             f"{imp['rel_z'].max():.2e}"),
        dots("impulse residual, horizontal, max",
             f"{imp['rel_h'].max():.2e}"),
        "  filament vs planar at 7 m, eps0 = " + js(red["eps0"], "g"),
        "    radius " + js(red["rel_dR"], ".2e") + "; orders "
        + js(ordR, ".2f"),
        "    arrival time " + js(red["rel_dT"], ".2e"),
        "  wavy ring at 3 s, N = 64 / 128 / 256:",
        "    R (m) " + js(nod["R_m"], ".8f"),
        "    z (m) " + js(nod["z_m"], ".6f"),
        "    wave amplitude (mm) " + js(1e3 * nod["A_m"], ".4f"),
    ]),
    ("reading these numbers", wrap(
        "The closed forms agree with the ODE to its tolerance. With the "
        "near-singular part subtracted, the filament reproduces the ring "
        "speed at every node number; the remaining plateau is the O(eps^2 "
        "ln eps) difference between the regularized ring and the thin-ring "
        "asymptote, while the plain rule needs node spacing below the core "
        "parameter. RK4 converges at fourth order until roundoff. The "
        "impulse residual measures the time integration and the Simpson "
        "rule for int V dt. Filament and planar model differ at O(eps^2), "
        "the order neglected by thin-ring theory.")),
])

# ------------------------------------------------------------- figure notes
r2 = csv("fig02_release")
c4 = csv("fig04c_gamma_c")
ex = csv("fig05c_exponents")
i10 = np.argmin(np.abs(r2["d0_m"] - 10))
lines2 = ["  d0 (m) | R/R0 full | exact | const V | gain | T (s) | flare"]
for i in range(0, len(r2), 3):
    row = (r2["d0_m"][i], r2["Rend_full_over_R0"][i],
           r2["Rend_exact_over_R0"][i], r2["Rend_constV_over_R0"][i],
           r2["gain_full"][i], r2["T_full_s"][i], r2["flare_full"][i])
    lines2.append("  {:6.1f} | {:9.3f} | {:5.3f} | {:7.3f} | {:4.2f} |"
                  " {:5.1f} | {:5.2f}".format(*row))
lines4 = ["  d0 (m) | We = 1 | Fr = 1 | M = 1  | head 10 mL | 5 mL"
          "   | 20 mL (m^2/s)"]
for i in range(len(c4)):
    lines4.append(f"  {c4['d0_m'][i]:6.1f} | {c4['Gc_We'][i]:.4f} |"
                  f" {c4['Gc_Fr'][i]:.4f} | {c4['Gc_M'][i]:.4f} |"
                  f" {c4['Gc_hs_10mL'][i]:.4f}     |"
                  f" {c4['Gc_hs_5mL'][i]:.4f} | {c4['Gc_hs_20mL'][i]:.4f}")
bk = csv("fig04b_breakup")
ok = ~np.isfinite(bk["t_breakup_s"])
lines4b = [dots("deepest release reaching 0.5 m intact",
                f"{bk['d0_m'][ok].max():.0f} m (1 m grid)")]
lines4b += ["  d0 (m) | breakup time (s) | breakup depth (m)"]
for i in range(len(bk)):
    if np.isfinite(bk["t_breakup_s"][i]) and bk["d0_m"][i] % 5 == 0:
        lines4b.append(f"  {bk['d0_m'][i]:6.0f} | {bk['t_breakup_s'][i]:16.1f}"
                       f" | {bk['depth_breakup_m'][i]:17.1f}")
lat = csv("fig03c_lateral")
lines5 = ["  mode | Boyle mean [min, max] | constant V mean [min, max]"]
for i in range(len(ex)):
    m = "all" if ex["mode"][i] == 0 else f"{int(ex['mode'][i])}"
    row = [ex[k][i] for k in ("boyle_mean", "boyle_min", "boyle_max",
                              "constV_mean", "constV_min", "constV_max")]
    lines5.append(f"  {m:>4} | " + "{:+.3f} [{:+.3f}, {:+.3f}] | "
                  "{:+.3f} [{:+.3f}, {:+.3f}]".format(*row))
write_report("figure_notes", "Figure notes and tabulated values", [
    ("purpose", wrap(
        "Figures carry no in-panel numbers. Values that would otherwise "
        "be printed inside an axes are recorded here; every panel is also "
        "available as CSV under outputs/data.")),
    ("figure 1, reference ascent from 10 m", [
        dots("R at 0.5 m, full / constant volume",
             f"{100 * full['R'][-1]:.2f} / {100 * turn['R'][-1]:.2f} cm"),
        dots("dR/dh at 10 m and 0.5 m, full",
             "{:.3f}, {:.3f} cm/m".format(
                 *(100 * P.g * full['V'][[0, -1]] / (2 * np.pi * P.Gamma
                                                     * full['R'][[0, -1]])
                   / full['U'][[0, -1]]))),
        dots("rise speed at 10 m and 0.5 m",
             f"{full['U'][0]:.3f}, {full['U'][-1]:.3f} m/s"),
        dots("V/V0 and a/a0 at 0.5 m",
             f"{full['V'][-1] / P.V0:.3f}, {full['a'][-1] / a0:.3f}"),
    ]),
    ("figure 2, release depth", lines2 + [
        dots("gain at 10 m, full / frozen closed form",
             f"{r2['gain_full'][i10]:.3f} / {r2['gain_frozen'][i10]:.3f}")]),
    ("figure 3, tilt", [
        "  lateral travel by 0.5 m (exact ideal gas),",
        "  theta0 = 15, 30, 45, 60 deg:",
        *[f"    d0 = {d} m: " + ", ".join(
            f"{np.interp(t, lat['theta0_deg'], lat[f'x_{d}m']):.2f}"
            for t in (15, 30, 45, 60)) + " m" for d in (5, 10, 20)],
        *wrap("With the axial hydrostatic tension removed the filament "
              "follows R sin(theta) = const (the closure of Chang and "
              "Llewellyn Smith 2018) instead of R^2 sin(theta) = const."),
    ]),
    ("figure 4, critical circulation", lines4),
    ("figure 4, Pedley viscous breakup, reference ring", lines4b),
    ("figure 5, wobble decay exponents beta in A_n ~ R^beta", lines5),
])

# ------------------------------------------------------------- sensitivity
cases = [("reference", P), ("V0 = 5 mL", replace(P, V0=5e-6)),
         ("V0 = 20 mL", replace(P, V0=20e-6)),
         ("R0 = 3 cm", replace(P, R0=0.03)),
         ("R0 = 8 cm", replace(P, R0=0.08)),
         ("Gamma = 0.03", replace(P, Gamma=0.03)),
         ("Gamma = 0.12", replace(P, Gamma=0.12)),
         ("adiabatic n = 1.4", replace(P, poly=1.4)),
         ("fresh water", replace(P, rho=998.0, gamma=0.072)),
         ("release 20 m", replace(P, d0=20.0))]
sl = ["  case              | R_e (cm) | T (s) | gain | flare | x30 (m) |"
      " hs/2R min"]
for name, p in cases:
    o = rd.integrate(p, mode="full")
    t_ = rd.integrate(p, mode="turner")
    o30 = rd.integrate(p, mode="full", theta0=np.radians(30))
    s = p.g * o["V"] / (2 * np.pi * p.Gamma * o["R"]) / o["U"]
    sl.append(f"  {name:<17} | {100 * o['R'][-1]:8.2f} | {o['T']:5.1f} |"
              f" {(o['R'][-1] - p.R0) / (t_['R'][-1] - p.R0):4.2f} |"
              f" {s[-1] / s[0]:5.2f} | {o30['x'][-1]:7.2f} |"
              f" {(o['hs'] / (2 * o['R'])).min():8.2f}")
write_report("sensitivity", "Sensitivity of headline numbers", [
    ("planar model, full closure, end depth 0.5 m", sl),
    ("reading", wrap(
        "Absolute radii, times and drifts scale with the unmeasured gas "
        "volume, radius and circulation. The gain over a constant-volume "
        "ring depends mainly on the release pressure number, and the "
        "self-leveling law sin(theta) R^2 = const holds in every case "
        "because it follows from the impulse balance alone.")),
])
print("reports written")

# ------------------------------------------------------------- references
REF_RINGS = [
    "  Turner JS (1957) Buoyant vortex rings. Proc R Soc Lond A 239:61.",
    "    https://doi.org/10.1098/rspa.1957.0022",
    "  Walters JK, Davidson JF (1963) The initial motion of a gas bubble",
    "    formed in an inviscid liquid. Part 2. J Fluid Mech 17:321.",
    "    https://doi.org/10.1017/S0022112063001373",
    "  Pedley TJ (1967) The stability of rotating flows with a",
    "    cylindrical free surface. J Fluid Mech 30:127.",
    "    https://doi.org/10.1017/S0022112067001338",
    "  Pedley TJ (1968) The toroidal bubble. J Fluid Mech 32:97.",
    "    https://doi.org/10.1017/S0022112068000601",
    "  Lundgren TS, Mansour NN (1991) Vortex ring bubbles. J Fluid Mech",
    "    224:177. https://doi.org/10.1017/S0022112091001702",
    "  Vasel-Be-Hagh AR, Carriveau R, Ting DS-K (2015) A balloon",
    "    bursting underwater. J Fluid Mech 769:522.",
    "    https://doi.org/10.1017/jfm.2015.126",
    "  Vasel-Be-Hagh AR, Carriveau R, Ting DS-K, Turner JS (2015) Drag",
    "    of buoyant vortex rings. Phys Rev E 92:043024.",
    "    https://doi.org/10.1103/PhysRevE.92.043024",
    "  Ruban VP, Rasmussen JJ (2003) Toroidal bubbles with circulation",
    "    in ideal hydrodynamics: a variational approach. Phys Rev E",
    "    68:056301. https://doi.org/10.1103/PhysRevE.68.056301",
    "  Cheng M, Lou J, Lim TT (2013) Motion of a bubble ring in a",
    "    viscous fluid. Phys Fluids 25:067104.",
    "    https://doi.org/10.1063/1.4811407",
    "  Chang C, Llewellyn Smith SG (2018) The motion of a buoyant",
    "    vortex filament. J Fluid Mech 857:R1.",
    "    https://doi.org/10.1017/jfm.2018.795",
]
REF_NUM = [
    "  Benjamin TB, Ellis AT (1966) The collapse of cavitation bubbles",
    "    and the pressures thereby produced against solid boundaries.",
    "    Phil Trans R Soc Lond A 260:221.",
    "    https://doi.org/10.1098/rsta.1966.0046",
    "  Benjamin TB (1987) Hamiltonian theory for motions of bubbles in",
    "    an infinite liquid. J Fluid Mech 181:349.",
    "    https://doi.org/10.1017/S002211208700212X",
    "  Moore DW, Saffman PG (1972) The motion of a vortex filament with",
    "    axial flow. Phil Trans R Soc Lond A 272:403.",
    "    https://doi.org/10.1098/rsta.1972.0055",
    "  Widnall SE, Bliss DB, Tsai C-Y (1974) The instability of short",
    "    waves on a vortex ring. J Fluid Mech 66:35.",
    "    https://doi.org/10.1017/S0022112074000048",
    "  Leonard A (1985) Computing three-dimensional incompressible",
    "    flows with vortex elements. Annu Rev Fluid Mech 17:523.",
    "    https://doi.org/10.1146/annurev.fl.17.010185.002515",
    "  Saffman PG (1992) Vortex Dynamics. Cambridge University Press.",
    "    https://doi.org/10.1017/CBO9780511624063",
    "  Hou TY, Lowengrub JS, Shelley MJ (1994) Removing the stiffness",
    "    from interfacial flows with surface tension. J Comput Phys",
    "    114:312. https://doi.org/10.1006/jcph.1994.1170",
]
write_report("references",
             "References",
             [("bubble rings and buoyant vortex rings", REF_RINGS),
              ("impulse, filaments, numerics", REF_NUM)])

write_report("open_items", "Open items before submission", [
    ("prior work and what is new", wrap(
        "Pedley (1968) gives the impulse law for a toroidal bubble, the "
        "hollow-core speed, the viscous diffusion of vorticity from the "
        "core, the capillary stability criterion (from Pedley 1967) and a "
        "breakup time; he also integrated the Boyle-law impulse equation "
        "numerically for one laboratory bubble and found a 7 percent "
        "effect, which the present model reproduces (verification.txt). "
        "Ruban and Rasmussen (2003) treat an adiabatic gas core "
        "variationally and numerically; their Eq. 53 is the core balance "
        "used here. Cheng, Lou and Lim (2013) simulate viscous bubble "
        "rings at Re = 30 to 100 with constant gas volume and show "
        "breakup through azimuthal surface-tension instability of short "
        "waves (k a near 2), the band the filament model projects out. "
        "Chang and Llewellyn Smith (2018) solve a tilted buoyant filament "
        "and obtain R sin(theta) = const; the present closure for a "
        "massless gas core gives R^2 sin(theta) = const. Claimed as new: "
        "the exact Boyle solution and universal gain, their growth with "
        "release depth, the tilt law and lateral drift, the swirl-head "
        "bound, the R^-1/2 decay of long bending waves, and Pedley's "
        "breakup criterion evaluated along Boyle trajectories.")),
    ("physics left out", wrap(
        "Inviscid liquid: circulation is held fixed, although viscous "
        "diffusion from the free-slip core surface and turbulence decay "
        "it in practice; the breakup depths of figure 4b use Pedley's "
        "estimate of that decay at the core surface only. No "
        "gas dissolution. Uniform core cross-section along the filament, "
        "justified while the vertical extent of the core stays below the "
        "swirl head (figure 4d). No core-scale dynamics: bending waves "
        "with k a > 1, where the Widnall-type short-wave instability lives, "
        "are projected out, so the breakup itself is not computed. Free-"
        "surface "
        "interaction above 0.5 m is not modeled.")),
    ("parameters", wrap(
        "Gas volume, radius and circulation of diver-blown rings are "
        "illustrative. The results stated as ratios (gain over constant "
        "volume, flaring, the tilt law, the R^-1/2 wobble decay) do not "
        "depend on them; absolute radii, times and drifts do. A pool test "
        "with a camera at two depths would fix R0, V0 and Gamma and test "
        "the tilt law directly.")),
])
print("static reports written")
