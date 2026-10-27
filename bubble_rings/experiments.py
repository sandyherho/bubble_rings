"""Filament runs shared by several scripts."""

import numpy as np

from .filament import Filament, circle, ring_diagnostics

__all__ = ["ascent", "wave_modes"]


def wave_modes(seed, amp, nmin=2, nmax=8):
    """Random bending and breathing waves of rms amplitude ``amp`` (m)."""
    rng = np.random.default_rng(seed)
    out = []
    for m in range(nmin, nmax + 1):
        out.append((m, amp * rng.standard_normal() / np.sqrt(2),
                    amp * rng.standard_normal() / np.sqrt(2),
                    rng.uniform(0, 2 * np.pi)))
    return out


def ascent(prm, theta0=0.0, modes=(), N=None, dt0=None, t_step=0.25,
           keep=False, scale=True, t_max=200.0, d_end=None, **kw):
    """Release one ring at prm.d0 and follow it to d_end.

    Returns a dict of time series sampled every t_step seconds with the
    filament diagnostics, plus node snapshots when ``keep``.
    """
    N = prm.n_nodes if N is None else N
    d_end = prm.d_end if d_end is None else d_end
    nrm = (np.sin(theta0), 0.0, np.cos(theta0))
    X0 = circle(N, prm.R0, (0.0, 0.0, -prm.d0), nrm, modes)
    f = Filament(prm, X0, **kw)
    rec = {k: [] for k in ("t", "R", "theta", "cx", "cy", "cz", "dz", "H",
                           "a", "V", "ell", "Iz", "Ix", "E")}
    snaps = []

    def grab(fl):
        dg = ring_diagnostics(prm, fl.X[0])
        st = fl.state()
        imp = fl.impulse()[0]
        for k, v in (("t", fl.t), ("R", dg["R"]), ("theta", dg["theta"]),
                     ("cx", dg["c"][0]), ("cy", dg["c"][1]),
                     ("cz", dg["c"][2]), ("dz", dg["dz"]), ("H", dg["H"]),
                     ("a", st["a"][0]), ("V", st["V"][0]),
                     ("ell", st["ell"][0]), ("Iz", imp[2]), ("Ix", imp[0]),
                     ("E", dg["E"])):
            rec[k].append(v)
        if keep:
            snaps.append(fl.X[0].copy())

    def stop(fl):
        return -fl.state()["zbar"].max() <= d_end

    grab(f)
    t_out = np.arange(t_step, t_max + 1e-9, t_step)
    f.run_sampled(t_out, dt0=dt0, callback=grab, stop=stop, scale=scale)
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["d"] = -out["cz"]
    if keep:
        out["X"] = np.asarray(snaps)
    return out
