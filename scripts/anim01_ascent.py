"""Animation 1. A wavy bubble ring rising from 10 m.

The reference ring (R0 = 5 cm, 10 mL, Gamma = 0.06 m^2/s) is released at
10 m with random bending and breathing waves (rms 2.5 mm per mode,
n = 2..8) and followed with the filament model (128 nodes) to
0.5 m.  The camera follows the ring center and turns slowly about the
vertical; the projection is orthographic, so lengths in the image plane
are at the scale bar.  The core is drawn at its true diameter 2a and
colored by its height above the mean level of the whole core (mm), which
shows the bending waves.  Points are passive
tracers advected by the Rosenhead-Moore velocity of the ring (RK4,
0.01 s); each streak joins a tracer's positions 0.25 s apart, so its
length and its color both give the distance the water moved in that
time (5 cm in 0.25 s is 0.2 m/s).  Tracers that leave the window are
re-seeded at random above it.  The inset traces R against depth.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from bubble_rings.anim import (dark, WAVE, TRACE, CORE, fig_to_rgb,
                               write_gif, water_background, scalebar,
                               HALO)
from bubble_rings.experiments import wave_modes
from bubble_rings.filament import Filament, circle, ring_diagnostics
from bubble_rings.io_utils import save_cache, load_cache
from bubble_rings.scenario import PRM

N, DT_TR, T_FR = 128, 0.01, 0.25
HW, HH = 0.32, 0.30          # half width and half height of the window (m)
NTR = 2400
rng = np.random.default_rng(7)


def tracer_velocity(P, X, delta):
    """Rosenhead-Moore velocity of one ring at points P (M, 3)."""
    k = np.fft.fftfreq(X.shape[0], 1.0 / X.shape[0])
    Xa = np.fft.ifft(1j * k[:, None] * np.fft.fft(X, axis=0), axis=0).real
    D = P[:, None, :] - X[None, :, :]
    w = ((D * D).sum(-1) + delta ** 2) ** -1.5
    cr = np.cross(Xa[None, :, :], D)
    return (PRM.Gamma / (4 * np.pi) * (2 * np.pi / X.shape[0])
            * (cr * w[..., None]).sum(1))


def seed(n, zc, top=False):
    """Random tracers in the window around center height zc."""
    x = rng.uniform(-HW * 1.3, HW * 1.3, n)
    y = rng.uniform(-HW * 1.3, HW * 1.3, n)
    lo = zc + HH * (0.9 if top else -1.0)
    z = rng.uniform(lo, zc + HH * 1.4, n)
    return np.c_[x, y, z]


cache = load_cache("anim01")
if cache is None:
    f = Filament(PRM, circle(N, PRM.R0, (0, 0, -PRM.d0),
                             modes=wave_modes(3, 0.0025)))
    P = seed(NTR, -PRM.d0)
    frames_X, frames_P, frames_S, info = [], [], [], []
    t_next = 0.0
    while True:
        st = f.state()
        X0, d0 = f.X[0].copy(), st["delta"][0]
        if f.t >= t_next - 1e-9:
            dg = ring_diagnostics(PRM, f.X[0])
            frames_X.append(X0)
            v = tracer_velocity(P, X0, d0)
            frames_P.append(P.copy())
            frames_S.append(np.linalg.norm(v, axis=1))
            info.append([f.t, dg["R"], dg["c"][2], st["V"][0], st["a"][0]])
            t_next += T_FR
            if -dg["c"][2] <= PRM.d_end:
                break
        f.run_sampled([f.t + DT_TR], dt0=2e-3, scale=False)
        X1 = f.X[0]
        Xm = 0.5 * (X0 + X1)
        k1 = tracer_velocity(P, X0, d0)
        k2 = tracer_velocity(P + 0.5 * DT_TR * k1, Xm, d0)
        k3 = tracer_velocity(P + 0.5 * DT_TR * k2, Xm, d0)
        k4 = tracer_velocity(P + DT_TR * k3, X1, d0)
        P = P + DT_TR / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        zc = (X1[:, 2]).mean()
        out = (P[:, 2] < zc - HH * 1.1) | (np.abs(P[:, 0]) > HW * 1.5) | \
            (np.abs(P[:, 1]) > HW * 1.5)
        if out.any():
            P[out] = seed(out.sum(), zc, top=True)
    cache = dict(X=np.array(frames_X), P=np.array(frames_P),
                 S=np.array(frames_S), info=np.array(info))
    save_cache("anim01", **cache)

X, P, S, info = cache["X"], cache["P"], cache["S"], cache["info"]
nf = len(X)
dark()
W, H, DPI = 7.2, 5.4, 100
DNORM = Normalize(0.0, 5.0)      # tracer path in one frame (cm)
ZNORM = Normalize(-10.0, 10.0)    # core height about its mean (mm)
frames = []
Rtr, dtr = info[:, 1], -info[:, 2]
for i in range(nf):
    t, R, zc, V, a = info[i]
    az = np.radians(25 + 40 * i / nf)
    el = np.radians(22.0)
    # orthographic camera: e_x horizontal, e_y up in the image, e_v depth
    ex = np.array([np.cos(az), np.sin(az), 0.0])
    ev = np.array([-np.sin(az) * np.cos(el), np.cos(az) * np.cos(el),
                   -np.sin(el)])
    ey = np.cross(ev, ex)
    c = np.array([X[i][:, 0].mean(), X[i][:, 1].mean(), zc])
    fig = plt.figure(figsize=(W, H), dpi=DPI)
    ax = fig.add_axes([0.0, 0.0, 0.74, 1.0])
    ext = [-HW, HW, -HH * 1.25, HH * 1.25]
    water_background(ax, [ext[0], ext[1], ext[2], ext[3]],
                     -zc - HH * 1.25, -zc + HH * 1.25)
    if -zc - HH * 1.25 < 0:          # the sea surface enters the window
        ys = 0.0 - zc
        ax.axhspan(ys, ext[3], color="#1c6f8f", alpha=0.55, zorder=1)
        ax.axhline(ys, color="#c8eefa", lw=1.2, zorder=1)
    Q = (P[i] - c)
    qx, qy, qv = Q @ ex, Q @ ey, Q @ ev
    Y = X[i] - c
    yx, yy, yv = Y @ ex, Y @ ey, Y @ ev
    back = qv > 0
    px_per_m = 0.74 * W * DPI / (2 * HW)
    lw = max(2 * a * px_per_m * 72 / DPI, 0.6)
    # each streak is the straight segment from a tracer's position one
    # frame (0.25 s) earlier to its present position, in the water's
    # frame; its color is that same distance, so length and color agree
    path = np.zeros(len(P[i]))
    if i > 0:
        Q0 = P[i - 1] - c
        path = np.linalg.norm(P[i] - P[i - 1], axis=1)
        jump = path > 0.08              # re-seeded tracers, not motion
        path[jump] = 0.0
        p0x, p0y = Q0 @ ex, Q0 @ ey
        for sel, zo in ((back & ~jump, 2), (~back & ~jump, 5)):
            segs = np.stack([np.c_[p0x[sel], p0y[sel]],
                             np.c_[qx[sel], qy[sel]]], 1)
            ax.add_collection(LineCollection(
                segs, colors=TRACE(DNORM(100 * path[sel])), linewidths=1.0,
                alpha=0.7, zorder=zo))
    for sel, zo in ((back, 2), (~back, 5)):
        ax.scatter(qx[sel], qy[sel], s=2.4,
                   c=TRACE(DNORM(100 * path[sel])), alpha=0.9, lw=0,
                   zorder=zo)
    seg = np.stack([np.c_[yx, yy], np.roll(np.c_[yx, yy], -1, 0)], 1)
    order = np.argsort(-0.5 * (yv + np.roll(yv, -1)))
    dz = Y[:, 2] - Y[:, 2].mean()
    col = WAVE(ZNORM(1e3 * dz))
    ax.add_collection(LineCollection(seg[order], colors="#bfe9ff",
                                     linewidths=lw + 7.0, alpha=0.10,
                                     zorder=3, capstyle="round"))
    ax.add_collection(LineCollection(seg[order], colors=col[order],
                                     linewidths=lw + 1.2, zorder=3,
                                     capstyle="round"))
    ax.add_collection(LineCollection(seg[order], colors=CORE(0.95),
                                     linewidths=max(lw * 0.35, 0.3),
                                     alpha=0.55, zorder=4,
                                     capstyle="round"))
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.axis("off")
    scalebar(ax, ext[0] + 0.03, ext[2] + 0.03, 0.10, "10 cm")
    ax.text(ext[0] + 0.03, ext[3] - 0.03,
            f"t = {t:5.1f} s\ndepth {-zc:5.2f} m", va="top", fontsize=9,
            family="monospace", path_effects=HALO)
    ax.text(ext[1] - 0.03, ext[3] - 0.03,
            f"R = {100 * R:4.1f} cm\nV = {1e6 * V:4.1f} mL\n"
            f"2a = {2e3 * a:3.1f} mm", va="top", ha="right", fontsize=9,
            family="monospace", path_effects=HALO)
    # inset: radius against depth, filled as the ring rises
    ia = fig.add_axes([0.80, 0.42, 0.17, 0.50])
    ia.plot(100 * Rtr, dtr, color="#33415a", lw=1.0)
    ia.plot(100 * Rtr[:i + 1], dtr[:i + 1], color="#9fe7ff", lw=1.4)
    ia.plot(100 * R, -zc, "o", ms=4, color="white")
    ia.set_ylim(PRM.d0, 0)
    ia.set_xlim(4, 20)
    ia.set_xlabel("R (cm)", fontsize=8)
    ia.set_ylabel("depth (m)", fontsize=8)
    ia.tick_params(labelsize=7)
    cb1 = fig.add_axes([0.80, 0.26, 0.17, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(ZNORM, WAVE), cax=cb1,
                      orientation="horizontal")
    cb.set_ticks([-10, -5, 0, 5, 10])
    cb.set_label("ring color: core height\nabove its mean (mm)",
                 fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    cb2 = fig.add_axes([0.80, 0.12, 0.17, 0.022])
    cb = fig.colorbar(plt.cm.ScalarMappable(DNORM, TRACE), cax=cb2,
                      orientation="horizontal")
    cb.set_ticks([0, 1, 2, 3, 4, 5])
    cb.set_label("streaks: water moved\nin the last 0.25 s (cm)",
                 fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    frames.append(fig_to_rgb(fig))
    plt.close(fig)
frames += [frames[-1]] * 12
print(write_gif("anim01_ascent", frames, fps=12, colors=255))
