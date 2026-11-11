"""Run the v2 figure scripts given in FIGS (space-separated stems, default all) one after another; a failure in one does not stop the others.
Run:  python scripts/figures/v2/render.py      (FIGS="fig2_speed fig4_ptni" python scripts/figures/v2/render.py for a subset)"""
import os, sys, subprocess, time
from neuralce.paths import REPO
HERE = os.path.dirname(os.path.abspath(__file__))
ALL = ["fig2_speed", "fig3_gnn", "fig4_ptni", "fig5_stf_fenicr", "figS3_cluster_defs", "figS4_fenicr_order"]
for name in (os.environ.get("FIGS") or " ".join(ALL)).split():
    t0 = time.time(); print(f"===== {name} =====", flush=True)
    rc = subprocess.run([sys.executable, "-u", f"{HERE}/{name}.py"], cwd=REPO)
    print(f"===== {name}: exit {rc.returncode}, {time.time() - t0:.0f} s =====", flush=True)
