"""Regenerate every figure, animation, data file, and report.

Filament ensembles cache to outputs/cache and are only re-plotted on
reruns; delete that directory to recompute everything.  The reports read
the CSV files written by the figure scripts.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["fig00_schematic.py", "fig01_ascent.py", "fig02_release.py",
         "fig03_tilt.py", "fig04_core.py", "fig05_wobble.py",
         "fig06_verification.py", "anim01_ascent.py", "anim02_release.py",
         "anim03_tilt.py", "make_reports.py"]

t0 = time.time()
for s in ORDER:
    print(f"--- {s}", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, s)],
                       capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"{s} failed")
print(f"--- done in {time.time() - t0:.1f} s")
