"""Planar thin-ring model and its closed-form solutions.

A massless gas bubble in an unbounded liquid transfers its whole
buoyancy to the Kelvin impulse of the liquid,

    dI/dt = rho g V(t) z_hat,

whether or not its volume changes.  For a thin planar ring of radius R
and unit normal n the impulse is I = rho pi Gamma R^2 n, so with
J = I / (rho pi Gamma) = R^2 n

    dJ/dt = g V / (pi Gamma) z_hat,      dc/dt = U n,

where c is the ring center and U the self-induced speed of the hollow
ring.  The horizontal part of J is conserved, which levels a tilted
ring, and |J| = R^2 grows with the time integral of the gas volume.
V is fixed by the core closure of :mod:`bubble_rings.core` at the
hydrostatic pressure of the center.

With the logarithm in U frozen at L and an ideal isothermal gas,
V = V0 p0 / p, the equations integrate in closed form in the log-pressure
coordinate lambda = ln(p0 / p):

    R = R0 + ell_B lambda,          ell_B = 2 V0 p0 / (rho Gamma^2 L),
    sin(theta) = sin(theta0) (R0 / R)^2,

for any initial tilt theta0, and the ascent time of a level ring is

    T = 4 pi p0 / (Gamma L rho g)
        [R0 (1 - e^-lam) + ell_B (1 - (1 + lam) e^-lam)].

Without freezing the logarithm, L = L0 + (3/2) ln(R/R0) - lambda/2 for
the ideal gas, and L dR = ell_B L0 dlambda is linear in lambda(R).  With
r = R/R0, b = ell_B L0 / R0 and alpha = 1/(2b) it integrates exactly to

    lambda(r) = 2 L0 (1 - e^{-alpha (r-1)})
                + 3 [ln r - e^{-alpha r} (Ei(alpha r) - Ei(alpha))],

which holds for every initial tilt, because the tilt drops out of
L dR = -(2 g V / Gamma^2) dd.  Ascent time and lateral travel then
follow by one quadrature each.
"""

import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
from scipy.special import expi

from .core import gas_constant, core_radius, ring_speed, weber, froude, \
    swirl_head

__all__ = ["integrate", "L_release", "boyle_length", "R_log", "R_turner",
           "ascent_time", "theta_law", "lateral_offset", "lam_of_d",
           "flare_ratio", "turner_ratio", "lam_of_r", "R_exact",
           "ascent_time_exact", "lateral_offset_exact"]


def L_release(prm):
    """ln(8 R0 / a0) - 1/2 at release."""
    return float(np.log(8 * prm.R0 / prm.a0) - 0.5)


def lam_of_d(prm, d):
    """Log-pressure coordinate ln(p0 / p(d))."""
    return np.log(prm.p0 / prm.p_inf(d))


def boyle_length(prm, L=None):
    """Boyle length 2 V0 p0 / (rho Gamma^2 L) (m)."""
    L = L_release(prm) if L is None else L
    return 2 * prm.V0 * prm.p0 / (prm.rho * prm.Gamma ** 2 * L)


def R_log(prm, d, L=None):
    """Frozen-log ring radius at depth d for an ideal isothermal gas."""
    return prm.R0 + boyle_length(prm, L) * lam_of_d(prm, d)


def R_turner(prm, d, L=None):
    """Frozen-log ring radius at depth d for a constant gas volume V0."""
    L = L_release(prm) if L is None else L
    return prm.R0 + 2 * prm.g * prm.V0 * (prm.d0 - np.asarray(d)) / (
        prm.Gamma ** 2 * L)


def turner_ratio(Pi):
    """Boyle over constant-volume radius gain, ((1+Pi)/Pi) ln(1+Pi)."""
    Pi = np.asarray(Pi, float)
    return (1 + Pi) / Pi * np.log1p(Pi)


def flare_ratio(prm, d):
    """Expansion rate dR/dh at depth d over its value at release, p0/p."""
    return prm.p0 / prm.p_inf(d)


def ascent_time(prm, d_end=None, L=None):
    """Frozen-log ascent time of a level ring from d0 to d_end (s)."""
    L = L_release(prm) if L is None else L
    d_end = prm.d_end if d_end is None else d_end
    lam = float(lam_of_d(prm, d_end))
    lb = boyle_length(prm, L)
    pre = 4 * np.pi * prm.p0 / (prm.Gamma * L * prm.rho * prm.g)
    return pre * (prm.R0 * (1 - np.exp(-lam))
                  + lb * (1 - (1 + lam) * np.exp(-lam)))


def theta_law(prm, R, theta0):
    """Tilt (rad) from conservation of horizontal impulse."""
    return np.arcsin(np.sin(theta0) * (prm.R0 / np.asarray(R)) ** 2)


def lateral_offset(prm, theta0, d_end=None, L=None):
    """Frozen-log horizontal travel (m) of a ring released with tilt theta0."""
    L = L_release(prm) if L is None else L
    d_end = prm.d_end if d_end is None else d_end
    lam_e = float(lam_of_d(prm, d_end))
    lb = boyle_length(prm, L)
    jh = prm.R0 ** 2 * np.sin(theta0)

    def f(lam):
        R = prm.R0 + lb * lam
        return jh / np.sqrt(R ** 4 - jh ** 2) * prm.p0 / (
            prm.rho * prm.g) * np.exp(-lam)
    return quad(f, 0.0, lam_e, epsabs=0, epsrel=1e-12, limit=200)[0]


def integrate(prm, theta0=0.0, mode="full", L=None, d_end=None,
              rtol=1e-11, atol=1e-14, t_max=600.0, n_out=2000):
    """Integrate the planar-ring equations from release to d_end.

    mode:
      "full"   core closure with swirl and surface tension, U with 1/We;
      "ideal"  V = V0 (p0/p)^(1/n), U with ln(8R/a) - 1/2;
      "turner" V = V0 fixed, U with ln(8R/a) - 1/2.
    L: if given, U = Gamma L / (4 pi R) with the logarithm frozen.
    Returns a dict of time series sampled at n_out times.
    """
    d_end = prm.d_end if d_end is None else d_end
    C = gas_constant(prm) if mode == "full" else None

    def state(y):
        J, c = y[:3], y[3:]
        jn = np.sqrt(J @ J)
        R = np.sqrt(jn)
        n = J / jn
        d = -c[2]
        p = float(prm.p_inf(d))
        if mode == "full":
            a = float(core_radius(prm, C, 2 * np.pi * R, p))
            V = 2 * np.pi ** 2 * R * a * a
        else:
            V = prm.V0 * ((prm.p0 / p) ** (1 / prm.poly)
                          if mode == "ideal" else 1.0)
            a = np.sqrt(V / (2 * np.pi ** 2 * R))
        if L is not None:
            U = prm.Gamma * L / (4 * np.pi * R)
        else:
            U = float(ring_speed(prm, R, a, tension=(mode == "full")))
        return R, n, d, a, V, U, p

    def rhs(t, y):
        R, n, d, a, V, U, p = state(y)
        dJ = np.array([0.0, 0.0, prm.g * V / (np.pi * prm.Gamma)])
        return np.concatenate([dJ, U * n])

    def surf(t, y):
        return -y[5] - d_end
    surf.terminal = True
    surf.direction = -1

    J0 = prm.R0 ** 2 * np.array([np.sin(theta0), 0.0, np.cos(theta0)])
    y0 = np.concatenate([J0, [0.0, 0.0, -prm.d0]])
    sol = solve_ivp(rhs, (0, t_max), y0, method="DOP853", rtol=rtol,
                    atol=atol, events=surf, dense_output=True)
    T = sol.t_events[0][0] if len(sol.t_events[0]) else sol.t[-1]
    t = np.linspace(0, T, n_out)
    Y = sol.sol(t)
    out = {k: np.empty(n_out) for k in
           ("R", "d", "a", "V", "U", "theta", "x", "p")}
    for i in range(n_out):
        R, n, d, a, V, U, p = state(Y[:, i])
        out["R"][i], out["d"][i], out["a"][i] = R, d, a
        out["V"][i], out["U"][i], out["p"][i] = V, U, p
        out["theta"][i] = np.arctan2(np.hypot(n[0], n[1]), n[2])
        out["x"][i] = Y[3, i]
    out["t"] = t
    out["J"] = Y[:3].T
    out["T"] = T
    out["nfev"] = sol.nfev
    out["eps"] = out["a"] / out["R"]
    out["We"] = weber(prm, out["a"])
    out["Fr"] = froude(prm, out["a"])
    out["hs"] = swirl_head(prm, out["a"])
    return out


def _b_alpha(prm):
    """Return (L0, b, alpha) of the exact ideal-gas solution."""
    L0 = L_release(prm)
    b = boyle_length(prm, L0) * L0 / prm.R0
    return L0, b, 0.5 / b


def lam_of_r(prm, r):
    """Exact log-pressure lambda at which the radius is r R0 (ideal gas)."""
    L0, b, al = _b_alpha(prm)
    r = np.asarray(r, float)
    return (2 * L0 * (1 - np.exp(-al * (r - 1)))
            + 3 * (np.log(r) - np.exp(-al * r) * (expi(al * r) - expi(al))))


def R_exact(prm, d):
    """Exact running-log ring radius at depth d (ideal isothermal gas)."""
    lam = np.atleast_1d(lam_of_d(prm, d)).astype(float)
    out = np.empty_like(lam)
    for i, la in enumerate(lam):
        if la == 0.0:
            out[i] = prm.R0
            continue
        hi = 2.0
        while lam_of_r(prm, hi) < la:
            hi *= 2.0
        out[i] = prm.R0 * brentq(lambda r: lam_of_r(prm, r) - la, 1.0, hi,
                                 xtol=1e-15, rtol=1e-15)
    return out if np.ndim(d) else float(out[0])


def ascent_time_exact(prm, d_end=None):
    """Exact ascent time of a level ring (ideal isothermal gas, s).

    dt = 4 pi R p dR / (Gamma rho g ell_B L0) along the exact R(lambda).
    """
    d_end = prm.d_end if d_end is None else d_end
    L0, b, al = _b_alpha(prm)
    lb = boyle_length(prm, L0)
    r_e = R_exact(prm, d_end) / prm.R0

    def f(r):
        return r * np.exp(-lam_of_r(prm, r))
    integral = quad(f, 1.0, r_e, epsabs=0, epsrel=1e-12, limit=200)[0]
    return (4 * np.pi * prm.R0 ** 2 * prm.p0
            / (prm.Gamma * prm.rho * prm.g * lb * L0) * integral)


def lateral_offset_exact(prm, theta0, d_end=None):
    """Exact horizontal travel of a tilted ring (ideal isothermal gas, m).

    dx = tan(theta) |dd| with sin(theta) = sin(theta0) (R0/R)^2 and
    |dd| = p dlambda / (rho g), dlambda = L dR / (ell_B L0).
    """
    d_end = prm.d_end if d_end is None else d_end
    L0, b, al = _b_alpha(prm)
    lb = boyle_length(prm, L0)
    r_e = R_exact(prm, d_end) / prm.R0
    s0 = np.sin(theta0)

    def f(r):
        lam = lam_of_r(prm, r)
        L = L0 + 1.5 * np.log(r) - 0.5 * lam
        s = s0 / r ** 2
        return (s / np.sqrt(1 - s * s) * prm.p0 * np.exp(-lam)
                / (prm.rho * prm.g) * L * prm.R0 / (lb * L0))
    return quad(f, 1.0, r_e, epsabs=0, epsrel=1e-12, limit=200)[0]
