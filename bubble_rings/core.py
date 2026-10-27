"""Closure of the hollow, gas-filled vortex core.

The core is a thin tube of radius a and length ell holding a fixed
amount of gas.  Outside it the liquid swirls irrotationally with
circulation Gamma, so the liquid pressure at the core surface is lowered
by the swirl dynamic pressure rho Gamma^2 / (8 pi^2 a^2).  Gas pressure,
surface tension and swirl balance as

    p_g = p_inf - rho Gamma^2 / (8 pi^2 a^2) + gamma / a,

and the gas follows p_g V^n = C with V = pi a^2 ell (n = 1 is
isothermal).  The same balance, with an adiabatic gas, defines the core
volume in the variational model of Ruban and Rasmussen (2003, Eq. 53).
For n = 1 the balance multiplied by a^2 is a quadratic in a with one
positive root, which is used directly.  The same closure gives the core
diagnostics: the swirl Weber number, which decides capillary stability
of a hollow vortex (Pedley 1967, 1968), the core Froude number, which
measures the swirl against gravity across the core, and the swirl
velocity head, which bounds the vertical extent over which a uniform
core can be held in hydrostatic balance.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import k0e, k1e

__all__ = ["swirl_dp", "gas_constant", "core_radius", "weber", "froude",
           "swirl_head", "swirl_speed", "core_constant", "ring_speed",
           "rm_delta", "capillary_growth", "pedley_margin",
           "pedley_breakup"]


def swirl_dp(prm, a):
    """Swirl pressure deficit at the core surface (Pa)."""
    return prm.rho * prm.Gamma ** 2 / (8.0 * np.pi ** 2 * np.asarray(a) ** 2)


def gas_constant(prm, swirl=True, tension=True):
    """Return C = p_g0 V0^n for the release state.

    The release state is defined by V0 and R0 (so a0 follows from
    V0 = 2 pi^2 R0 a0^2) at the hydrostatic pressure of the release depth.
    """
    a0 = prm.a0
    pg0 = prm.p0
    if swirl:
        pg0 -= swirl_dp(prm, a0)
    if tension:
        pg0 += prm.gamma / a0
    return pg0 * prm.V0 ** prm.poly


def core_radius(prm, C, ell, p_inf, swirl=True, tension=True):
    """Core radius a (m) for gas constant C, tube length ell, pressure p_inf.

    ``ell`` and ``p_inf`` may be arrays of equal shape.  For an
    isothermal gas the positive root of

        p_inf a^2 + gamma a - (rho Gamma^2/(8 pi^2) + C/(pi ell)) = 0

    is returned; otherwise the balance is solved by bracketing.
    """
    ell = np.asarray(ell, float)
    p_inf = np.asarray(p_inf, float)
    s = 1.0 if swirl else 0.0
    t = 1.0 if tension else 0.0
    if prm.poly == 1.0:
        B = s * prm.rho * prm.Gamma ** 2 / (8 * np.pi ** 2) + C / (np.pi * ell)
        gam = t * prm.gamma
        disc = np.sqrt(gam ** 2 + 4.0 * p_inf * B)
        # rationalized root, free of cancellation when gamma^2 << p B
        return 2.0 * B / (gam + disc)
    out = np.empty(np.broadcast(ell, p_inf).shape)
    for idx, (el, p) in enumerate(np.broadcast(ell, p_inf)):
        def f(a):
            v = np.pi * a * a * el
            return (p - s * swirl_dp(prm, a) + t * prm.gamma / a
                    - C / v ** prm.poly)
        a_iso = np.sqrt((C ** (1 / prm.poly)) / (np.pi * el * p))
        out.flat[idx] = brentq(f, 1e-3 * a_iso, 1e3 * a_iso, xtol=1e-15,
                               rtol=1e-14)
    return out if out.ndim else float(out)


def weber(prm, a):
    """Swirl Weber number rho Gamma^2 / (4 pi^2 gamma a)."""
    return prm.rho * prm.Gamma ** 2 / (4 * np.pi ** 2 * prm.gamma
                                       * np.asarray(a))


def froude(prm, a):
    """Core Froude number Omega^2 a / g = Gamma^2 / (4 pi^2 g a^3)."""
    return prm.Gamma ** 2 / (4 * np.pi ** 2 * prm.g * np.asarray(a) ** 3)


def swirl_speed(prm, a):
    """Swirl speed at the core surface Gamma / (2 pi a) (m s^-1)."""
    return prm.Gamma / (2 * np.pi * np.asarray(a))


def swirl_head(prm, a):
    """Swirl velocity head u_c^2 / (2 g) (m)."""
    return swirl_speed(prm, a) ** 2 / (2 * prm.g)


def core_constant(prm, a, tension=True):
    """Core constant c in U = Gamma/(4 pi R) [ln(8R/a) - c].

    Hollow core: c = 1/2 (Hicks); surface tension lowers it by 1/We.
    """
    c = 0.5 * np.ones_like(np.asarray(a, float))
    if tension:
        c = c - 1.0 / weber(prm, a)
    return c


def ring_speed(prm, R, a, tension=True):
    """Self-induced speed of a thin hollow ring (m s^-1)."""
    return prm.Gamma / (4 * np.pi * R) * (np.log(8 * R / a)
                                          - core_constant(prm, a, tension))


def rm_delta(prm, a, tension=True):
    """Rosenhead-Moore core parameter giving the hollow-ring speed.

    A circular filament with the regularized kernel moves at
    Gamma/(4 pi R) [ln(8R/delta) - 1]; matching ring_speed gives
    delta = a exp(c - 1).
    """
    return np.asarray(a) * np.exp(core_constant(prm, a, tension) - 1.0)


def capillary_growth(x, We):
    """Axisymmetric growth rate (sigma/Omega)^2 of a hollow vortex.

    x = k a is the axial wavenumber and We the swirl Weber number.
    Negative values are neutral oscillations.  Instability requires
    We < 1 - x^2, the axisymmetric condition of Pedley (1967), restated
    for a toroidal bubble by Pedley (1968, Eq. 6.2).  With We -> infinity the
    swirl alone is stabilizing; with Omega -> 0 the classical gas-column
    result sigma^2 = gamma/(rho a^3) x (1 - x^2) K1/K0 is recovered.
    """
    x = np.asarray(x, float)
    return ((1.0 - x * x) / We - 1.0) * x * k1e(x) / k0e(x)


def pedley_margin(prm, t, a):
    """Viscous capillary margin of Pedley (1968, Eq. 6.5) along an ascent.

    Vorticity diffusing from the free-slip core surface lowers the swirl
    there to v(b) = Gamma_P/b [1 - exp(-3 b^2/(8 nu t))], Gamma_P =
    Gamma/(2 pi), so the stability parameter rho b v^2/gamma becomes

        M(t) = We [1 - exp(-3 a^2 / (8 nu t))]^2,

    and the core turns unstable to axisymmetric capillary disturbances
    when M falls below one.  ``t`` (s) is time since release and ``a``
    (m) the core radius at that time.
    """
    t = np.asarray(t, float)
    a = np.asarray(a, float)
    with np.errstate(divide="ignore", over="ignore"):
        damp = 1.0 - np.exp(-3.0 * a * a / (8.0 * prm.nu * t))
    damp = np.where(t > 0, damp, 1.0)
    return weber(prm, a) * damp ** 2


def pedley_breakup(prm, out):
    """First time (s) and depth (m) at which pedley_margin falls below one.

    ``out`` is a trajectory from :func:`reduced.integrate`.  Returns
    (inf, nan) when the ring reaches the end depth intact.
    """
    M = pedley_margin(prm, out["t"], out["a"])
    bad = np.nonzero(M < 1.0)[0]
    if len(bad) == 0:
        return np.inf, np.nan
    i = bad[0]
    if i == 0:
        return 0.0, float(out["d"][0])
    # linear interpolation of ln M between the bracketing samples
    f = (0.0 - np.log(M[i - 1])) / (np.log(M[i]) - np.log(M[i - 1]))
    tb = out["t"][i - 1] + f * (out["t"][i] - out["t"][i - 1])
    return float(tb), float(np.interp(tb, out["t"], out["d"]))
