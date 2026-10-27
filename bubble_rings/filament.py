"""Three-dimensional buoyant vortex-filament model of bubble rings.

Each ring is a closed curve X(alpha), alpha in [0, 2 pi), carried by N
Lagrangian nodes, with a hollow gas core of uniform radius a(t).  A node
moves with

    dX/dt = u_RM(X) + D,

where u_RM is the Rosenhead-Moore regularized Biot-Savart velocity of
all rings,

    u_RM(x) = Gamma/(4 pi) oint X_alpha' x (x - X')
              / (|x - X'|^2 + delta^2)^(3/2) dalpha',

with delta = a exp(c - 1) so that a circular ring moves at the hollow-
ring speed Gamma/(4 pi R) [ln(8R/a) - c] (see :func:`core.rm_delta`).

Buoyancy enters through the Kutta-Joukowski balance of the massless
core.  Per unit length the core carries the buoyancy rho g pi a^2 z_hat
and an axial tension T(s) set by the hydrostatic variation of the
pressure difference across the core wall, T = -rho g pi a^2 (z - zbar),
whose gradient absorbs the tangential part of the buoyancy.  The core
drifts relative to the liquid with

    D = t x f / (rho Gamma)
      = (g pi a^2 / Gamma) [t x z_hat - (z - zbar) kappa b],

so that oint D x t ds integrates to the exact Kelvin impulse balance
dI/dt = rho g V z_hat for any shape.  Keeping only the transverse part
t x z_hat, as for a filament whose core can store axial momentum, gives
dI/dt = rho g pi a^2 oint (z_hat - t_z t) ds instead, which differs for
a tilted ring by a horizontal impulse source; a massless gas core cannot
store that momentum, and the tension term returns it to the liquid.

The self-induced integral is evaluated with the periodic trapezoidal
rule after subtracting its near-singular part,

    g_i(tau) = (X_alpha x X_alphaalpha)_i / 2
               * tau_p^2 / (q_i^2 tau_p^2 + delta^2)^(3/2),

with tau_p = 2 sin(tau/2) and q_i = |X_alpha|, whose integral over the
period is added back exactly through the tabulated function F(beta).
This keeps the quadrature accurate when the node spacing exceeds delta.
Derivatives are spectral and time stepping is classical RK4.  Thin-
filament theory holds only for bending waves long compared with the core,
k a << 1; shorter waves on a regularized filament are spuriously unstable
near k delta ~ 1 (the filament analogue of the short-wave instability of
Widnall, Bliss and Tsai, which a filament cannot represent faithfully).
The node velocities are therefore projected (Galerkin) onto the
azimuthal modes n <= n_c = ka_max ell0 / (2 pi a0) fixed at release.  As
the ring grows a/R falls, so these modes keep k a <= ka_max throughout.
Because positions and velocities share the same band, the projection
leaves the impulse balance exact and the RK4 order intact.

Only the normal part of the velocity moves the curve; its tangential
part merely relabels points.  The tangential velocity is therefore
replaced by the one that keeps the nodes equally spaced in arc length
(the construction of Hou, Lowengrub and Shelley for interfaces), with a
weak relaxation of any residual nonuniformity.  Every geometric quantity,
including the impulse, is unaffected by this choice.
"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

from .core import core_radius, rm_delta, gas_constant

__all__ = ["F_beta", "Filament", "circle", "ring_diagnostics",
           "equal_arc_resample"]


def _F_exact(beta):
    """Integral of tau_p^2 / (tau_p^2 + beta^2)^(3/2) dtau over one period."""
    # u = sin(tau/2): dtau = 2 du / sqrt(1 - u^2); u = sin(w) removes the
    # endpoint singularity, leaving a smooth integrand peaked at w = 0.
    def f(w):
        u = np.sin(w)
        return 16.0 * u * u / (4 * u * u + beta * beta) ** 1.5
    pts = [min(beta, 0.5)]
    return quad(f, 0.0, 0.5 * np.pi, points=pts, epsabs=0,
                epsrel=1e-13, limit=400)[0]


_LB = np.linspace(np.log(1e-9), np.log(4.0), 1601)
_FT = CubicSpline(_LB, [_F_exact(np.exp(x)) for x in _LB])


def F_beta(beta):
    """Tabulated F(beta), cubic spline in ln(beta)."""
    return _FT(np.log(beta))


def circle(N, R, center=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0),
           modes=(), phase0=0.0):
    """Nodes of a circle of radius R with optional wave perturbations.

    modes: iterable of (n, radial amplitude, axial amplitude, phase),
    amplitudes in meters.  The circle is oriented counterclockwise about
    ``normal`` so that positive circulation propels it along ``normal``.
    """
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    ref = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else \
        np.array([1.0, 0.0, 0.0])
    e2 = ref - (ref @ n) * n
    e2 /= np.linalg.norm(e2)
    e1 = np.cross(e2, n)
    al = 2 * np.pi * np.arange(N) / N + phase0
    r = R * np.ones(N)
    w = np.zeros(N)
    for (m, ar, aw, ph) in modes:
        r += ar * np.cos(m * al + ph)
        w += aw * np.sin(m * al + ph)
    X = (np.asarray(center, float)[None, :]
         + r[:, None] * (np.cos(al)[:, None] * e1 + np.sin(al)[:, None]
                         * e2) + w[:, None] * n)
    return equal_arc_resample(X) if modes else X


def equal_arc_resample(X, refine=16):
    """Re-sample a closed curve at equal arc length, spectrally.

    The trigonometric interpolant of X is evaluated on a grid ``refine``
    times finer, arc length is accumulated there, and the interpolant is
    evaluated at the parameters of N equally spaced arc lengths.
    """
    N = X.shape[0]
    Xh = np.fft.fft(X, axis=0) / N
    k = np.fft.fftfreq(N, 1.0 / N)
    M = refine * N
    af = 2 * np.pi * np.arange(M + 1) / M
    E = np.exp(1j * np.outer(af, k))
    Xa = (E * (1j * k)[None, :]) @ Xh
    q = np.linalg.norm(Xa.real, axis=1)
    s = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1]))
                        * (af[1] - af[0])])
    target = s[-1] * np.arange(N) / N
    an = np.interp(target, s, af)
    return (np.exp(1j * np.outer(an, k)) @ Xh).real


class Filament:
    """A set of buoyant hollow vortex rings in an unbounded liquid.

    Parameters
    ----------
    prm : Params
    X : array (M, N, 3) of node positions (z up, depth = -z)
    C : sequence of M gas constants p_g V^n (default: release state)
    buoyancy : include the Kutta-Joukowski buoyant drift
    hydro_tension : include the axial tension T = -rho g pi a^2 (z - zbar);
        False keeps only the transverse buoyancy t x z_hat, the closure
        of Chang and Llewellyn Smith (2018) for a core without axial flow
    boyle : evaluate the core closure at the local hydrostatic pressure;
        False holds it at the release pressure (constant gas volume)
    tension : include surface tension in the core closure and speed
    """

    def __init__(self, prm, X, C=None, buoyancy=True, tension=True,
                 subtract=True, truncate=True, equal_arc=True,
                 relax_time=0.05, ka_max=1.0, boyle=True,
                 hydro_tension=True):
        """Set up the rings and the spectral operators."""
        self.prm = prm
        self.X = np.array(X, float)
        if self.X.ndim == 2:
            self.X = self.X[None]
        self.M, self.N, _ = self.X.shape
        if C is None:
            C = [gas_constant(prm, tension=tension)] * self.M
        self.C = np.asarray(C, float)
        self.buoyancy = buoyancy
        self.tension = tension
        self.subtract = subtract
        self.boyle = boyle
        self.hydro_tension = hydro_tension
        self.equal_arc = equal_arc
        self.relax = 1.0
        self.relax_time = relax_time
        self.t = 0.0
        N = self.N
        self.h = 2 * np.pi / N
        self.k = np.fft.fftfreq(N, 1.0 / N)
        self.truncate = truncate
        self.ka_max = ka_max
        tau = self.h * np.arange(N)
        tp = 2 * np.sin(0.5 * tau)
        self._tp2 = tp * tp          # tau_p^2 as a function of j - i mod N
        idx = (np.arange(N)[None, :] - np.arange(N)[:, None]) % N
        self._idx = idx
        Xa, _ = self.derivs(self.X)
        ell, zbar, a, delta, q = self.core(self.X, Xa)
        self.nc = np.floor(ka_max * ell / (2 * np.pi * a))
        self.keep = (np.abs(self.k)[None, :] <= self.nc[:, None]).astype(
            float)
        if truncate:
            self.X = self._project(self.X)

    # ------------------------------------------------------------ geometry
    def derivs(self, X):
        """Spectral X_alpha and X_alphaalpha, shape (M, N, 3)."""
        Xh = np.fft.fft(X, axis=1)
        ik = 1j * self.k[None, :, None]
        Xa = np.fft.ifft(ik * Xh, axis=1).real
        Xaa = np.fft.ifft(ik * ik * Xh, axis=1).real
        return Xa, Xaa

    def core(self, X, Xa):
        """Per ring: length, mean height, core radius a and delta."""
        q = np.linalg.norm(Xa, axis=2)
        ell = q.sum(axis=1) * self.h
        zbar = (X[..., 2] * q).sum(axis=1) * self.h / ell
        p = self.prm.p_inf(-zbar) if self.boyle else \
            self.prm.p0 * np.ones_like(zbar)
        a = np.array([core_radius(self.prm, c, el, pp, tension=self.tension)
                      for c, el, pp in zip(self.C, ell, p)])
        delta = rm_delta(self.prm, a, tension=self.tension)
        return ell, zbar, a, delta, q

    # ------------------------------------------------------------ velocity
    def velocity(self, X):
        """Node velocities (M, N, 3) and the core state."""
        prm = self.prm
        G = prm.Gamma
        Xa, Xaa = self.derivs(X)
        ell, zbar, a, delta, q = self.core(X, Xa)
        P = X.reshape(-1, 3)
        Ta = Xa.reshape(-1, 3)
        dsrc = np.repeat(delta, self.N)
        D = P[:, None, :] - P[None, :, :]                  # x_i - X_j
        r2 = (D * D).sum(-1) + dsrc[None, :] ** 2
        w = r2 ** -1.5
        cr = np.cross(Ta[None, :, :], D)                    # X_a,j x (x_i-X_j)
        u = (G / (4 * np.pi)) * self.h * (cr * w[..., None]).sum(1)
        u = u.reshape(self.M, self.N, 3)
        if self.subtract:
            for m in range(self.M):
                bk = 0.5 * np.cross(Xa[m], Xaa[m])            # (N, 3)
                qi = q[m]
                den = (qi[:, None] ** 2 * self._tp2[self._idx]
                       + delta[m] ** 2) ** -1.5
                gsum = (self._tp2[self._idx] * den).sum(1) * self.h
                Fi = F_beta(delta[m] / qi) / qi ** 3
                u[m] += (G / (4 * np.pi)) * bk * (Fi - gsum)[:, None]
        if self.buoyancy:
            t = Xa / q[..., None]
            kb = np.cross(Xa, Xaa) / q[..., None] ** 3      # kappa b
            zh = np.array([0.0, 0.0, 1.0])
            dz = X[..., 2] - zbar[:, None]
            coef = prm.g * np.pi * a ** 2 / G
            drift = np.cross(t, zh)
            if self.hydro_tension:
                drift = drift - dz[..., None] * kb
            u += coef[:, None, None] * drift
        if self.equal_arc:
            u = self._equal_arc(u, Xa, q, ell)
        if self.truncate:
            u = self._project(u)
        return u, dict(ell=ell, zbar=zbar, a=a, delta=delta)

    def _project(self, Y):
        """Keep the azimuthal modes n <= n_c of each ring."""
        Yh = np.fft.fft(Y, axis=1) * self.keep[..., None]
        return np.fft.ifft(Yh, axis=1).real

    def _equal_arc(self, u, Xa, q, ell):
        """Replace the tangential velocity by the equal-arc-length one."""
        t = Xa / q[..., None]
        un = u - (u * t).sum(-1)[..., None] * t
        ik = 1j * self.k[None, :, None]
        dun = np.fft.ifft(ik * np.fft.fft(un, axis=1), axis=1).real
        sgrow = (t * dun).sum(-1)                         # t . d(u_n)/dalpha
        qbar = (ell / (2 * np.pi))[:, None]
        rhs = (sgrow.mean(1, keepdims=True) - sgrow
               + self.relax * (qbar - q) / self.relax_time)
        rh = np.fft.fft(rhs, axis=1)
        kk = self.k[None, :]
        Th = np.where(kk != 0, rh / (1j * np.where(kk != 0, kk, 1)), 0.0)
        T = np.fft.ifft(Th, axis=1).real
        return un + T[..., None] * t

    # ------------------------------------------------------------ stepping
    def step(self, dt):
        """One classical RK4 step."""
        X = self.X
        k1, _ = self.velocity(X)
        k2, _ = self.velocity(X + 0.5 * dt * k1)
        k3, _ = self.velocity(X + 0.5 * dt * k2)
        k4, _ = self.velocity(X + dt * k3)
        X = X + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.X = X
        self.t += dt

    def run(self, t_end, dt=None, every=None, callback=None, stop=None):
        """Advance to t_end with a fixed step.

        callback(self) is called every ``every`` steps; stop(self) -> bool
        ends the run early (checked every step).
        """
        dt = self.prm.dt if dt is None else dt
        nsteps = int(round((t_end - self.t) / dt))
        for i in range(nsteps):
            self.step(dt)
            if callback is not None and every and (i + 1) % every == 0:
                callback(self)
            if stop is not None and stop(self):
                break
        return self

    def run_sampled(self, t_out, dt0=None, callback=None, stop=None,
                    scale=True):
        """Advance through the output times t_out, calling callback(self).

        The fastest retained bending wave, n = n_c, has a frequency of
        order Gamma n_c^2 / (4 pi R^2), so the step is dt0 (ell/ell0)^2,
        at most 3 dt0, with ell the shortest ring; it is shortened to land
        on each output time exactly.
        """
        dt0 = self.prm.dt if dt0 is None else dt0
        ell0 = self.state()["ell"].min()
        for to in t_out:
            while self.t < to - 1e-12:
                f = (self.state()["ell"].min() / ell0) ** 2 if scale else 1.0
                dt = min(dt0 * min(f, 3.0), to - self.t)
                self.step(dt)
            if callback is not None:
                callback(self)
            if stop is not None and stop(self):
                break
        return self

    # ------------------------------------------------------------ diagnostics
    def impulse(self, X=None):
        """Kelvin impulse per ring (rho Gamma / 2) oint X x X_alpha dalpha."""
        X = self.X if X is None else X
        Xa, _ = self.derivs(X)
        return (0.5 * self.prm.rho * self.prm.Gamma * self.h
                * np.cross(X, Xa).sum(axis=1))

    def state(self):
        """Core state of the current configuration."""
        Xa, _ = self.derivs(self.X)
        ell, zbar, a, delta, q = self.core(self.X, Xa)
        V = np.pi * a ** 2 * ell
        return dict(ell=ell, zbar=zbar, a=a, delta=delta, V=V,
                    spacing=q.max(1) / q.min(1))


def ring_diagnostics(prm, X, nmax=24):
    """Shape diagnostics of one ring (N, 3).

    Returns the impulse radius, the unit normal, the tilt from vertical
    (rad), the arc-weighted center, the vertical extent of the core, and
    the energy of radial and axial deviations from the best circle in
    azimuthal modes n = 2..nmax together with their normalized spectral
    entropy.
    """
    N = X.shape[0]
    h = 2 * np.pi / N
    k = np.fft.fftfreq(N, 1.0 / N)
    Xa = np.fft.ifft(1j * k[:, None] * np.fft.fft(X, axis=0), axis=0).real
    q = np.linalg.norm(Xa, axis=1)
    ell = q.sum() * h
    c = (X * q[:, None]).sum(0) * h / ell
    J = 0.5 * h * np.cross(X, Xa).sum(0)
    R = np.sqrt(np.linalg.norm(J) / np.pi)
    n = J / np.linalg.norm(J)
    theta = np.arctan2(np.hypot(n[0], n[1]), n[2])
    ref = np.array([0.0, 1.0, 0.0]) if abs(n[1]) < 0.9 else \
        np.array([1.0, 0.0, 0.0])
    e2 = ref - (ref @ n) * n
    e2 /= np.linalg.norm(e2)
    e1 = np.cross(e2, n)
    Y = X - c
    phi = np.unwrap(np.arctan2(Y @ e2, Y @ e1))
    rr = np.hypot(Y @ e1, Y @ e2) - R
    ww = Y @ n
    # Fourier coefficients in the geometric azimuth, trapezoid in phi
    dphi = np.gradient(phi) if phi[-1] > phi[0] else -np.gradient(phi)
    ns = np.arange(2, nmax + 1)
    E = np.empty(len(ns))
    for i, m in enumerate(ns):
        e = np.exp(-1j * m * phi)
        cr = (rr * e * dphi).sum() / np.pi
        cw = (ww * e * dphi).sum() / np.pi
        E[i] = 0.5 * (abs(cr) ** 2 + abs(cw) ** 2)
    p = E / E.sum() if E.sum() > 0 else np.ones_like(E) / len(E)
    H = -(p * np.log(np.where(p > 0, p, 1.0))).sum() / np.log(len(E))
    return dict(R=R, n=n, theta=theta, c=c, dz=np.ptp(X[:, 2]), E=E,
                modes=ns, H=H, ell=ell)
