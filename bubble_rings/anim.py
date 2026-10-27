"""Dark style, color maps, and looping GIF export.

Animations are rendered frame by frame with Matplotlib into RGB arrays and
written as palette GIFs with Pillow.  Every frame is a direct render of a
computed model state: ring nodes, core radii at true scale, and passive
tracers advected by the computed velocity field.  The only display
operations are Lambertian shading of the core tube (illumination only)
and the stated color scales.  The background gradient darkens with depth
for legibility and carries no quantity.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

matplotlib.use("Agg")

__all__ = ["dark", "BG", "FG", "SEA", "CORE", "WAVE", "TRACE", "fig_to_rgb",
           "write_gif", "ANIMDIR", "water_background", "scalebar", "HALO"]

_HERE = os.path.dirname(os.path.abspath(__file__))
ANIMDIR = os.path.join(os.path.dirname(_HERE), "outputs", "animations")

BG = "#03070d"
FG = "#d9e4ee"

# water seen from inside, deep to shallow: clear tropical sea blue
SEA = LinearSegmentedColormap.from_list("sea", [
    (0.00, "#03162a"), (0.20, "#072f52"), (0.40, "#0b4d7e"),
    (0.60, "#136ca3"), (0.80, "#1f87bb"), (1.00, "#35a1cc")])

# thin dark outline that keeps light marks legible on light water
HALO = [patheffects.withStroke(linewidth=2.2, foreground="#04182b",
                               alpha=0.55)]

# gas core: the silver of a bubble ring lit from above
CORE = LinearSegmentedColormap.from_list("core", [
    (0.00, "#1b3346"), (0.35, "#6f9db8"), (0.70, "#d5ebf5"),
    (1.00, "#ffffff")])

# signed height of the core about its mean level
WAVE = LinearSegmentedColormap.from_list("wave", [
    (0.00, "#2b6cff"), (0.30, "#58b7ff"), (0.50, "#e9f6ff"),
    (0.70, "#ffc36a"), (1.00, "#ff6a2b")])

# tracer speed, slow to fast
TRACE = LinearSegmentedColormap.from_list("trace", [
    (0.00, "#3f6f86"), (0.35, "#56c3e8"), (0.70, "#c9f4ff"),
    (1.00, "#fff3b0")])


def dark():
    """Apply the animation style."""
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": "#33415a",
        "xtick.color": FG, "ytick.color": FG, "font.family": "serif",
        "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "font.size": 9, "axes.linewidth": 0.6,
    })


def water_background(ax, extent, d_top, d_bot, d_fold=12.0):
    """Paint a vertical light gradient that darkens with depth (no data).

    The level falls as exp(-d/d_fold), the shape of daylight attenuation
    in clear water, so equal depths share one color in every animation.
    """
    d = np.linspace(d_top, d_bot, 512)
    lev = np.exp(-np.clip(d, 0.0, None) / d_fold)
    ax.imshow(lev[:, None], extent=extent, origin="upper", cmap=SEA,
              vmin=0, vmax=1, aspect="auto", interpolation="bilinear",
              zorder=0)


def scalebar(ax, x0, y0, length, label, color=FG):
    """Horizontal scale bar in data units."""
    ax.plot([x0, x0 + length], [y0, y0], color=color, lw=1.6,
            solid_capstyle="butt")
    ax.text(x0 + 0.5 * length, y0, label, color=color, ha="center",
            va="bottom", fontsize=8)


def fig_to_rgb(fig):
    """Rasterize a figure to an (h, w, 3) uint8 array."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def write_gif(stem, frames, fps=18, colors=224):
    """Write a looping, palette-quantized GIF and return its path.

    One adaptive palette is computed from a sample of frames and shared
    by all of them, which removes frame-to-frame palette flicker.
    """
    os.makedirs(ANIMDIR, exist_ok=True)
    path = os.path.join(ANIMDIR, f"{stem}.gif")
    pick = frames[:: max(1, len(frames) // 12)]
    strip = Image.fromarray(np.concatenate(pick, axis=0))
    pal = strip.quantize(colors=colors, method=Image.MEDIANCUT,
                         dither=Image.Dither.NONE)
    imgs = [Image.fromarray(f).quantize(palette=pal,
                                        dither=Image.Dither.NONE)
            for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], loop=0,
                 duration=int(round(1000 / fps)), optimize=True, disposal=2)
    return path
