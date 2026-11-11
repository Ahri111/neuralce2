"""Collect the Disorder_GNN summaries into the benchmark-table row (3-seed ensemble, seed average, per seed).

Inputs : $RESULTS/cv/external/disorder_gnn/<system>/summary.json (env DGNN_DIR overrides the parent directory, e.g.
         $RUNS/external/disorder_gnn; env DGNN_SYSTEMS, comma list, overrides the systems, default ptni); a missing
         file prints "pending".  Only the PtNi summary is shipped.
Output : markdown table on stdout.
Run    : python scripts/baselines/disorder_gnn_table.py
"""
import json, os
from neuralce.paths import RESULTS

D = os.environ.get("DGNN_DIR", f"{RESULTS}/cv/external/disorder_gnn")
LABEL = {"ptni": "PtNi (121)", "stfo": "STFO (387)", "fenicr": "FeNiCr (49 x 108-atom test)"}
rows = []
for sysname in os.environ.get("DGNN_SYSTEMS", "ptni").split(","):
    label = LABEL.get(sysname, sysname)
    p = f"{D}/{sysname}/summary.json"
    if not os.path.exists(p):
        rows.append((label, "pending", "", "")); continue
    s = json.load(open(p))
    ens = f"{s['mae_mean']:.2f} ± {s['mae_std']:.2f}"
    seeds = "; ".join(f"seed{k}: {v['mean']:.2f} ± {v['std']:.2f}" for k, v in s["per_seed"].items())
    sa = f"{s['seed_average_mean']:.2f} ± {s['seed_average_std']:.2f}"
    extra = f"params {s['params']:,}; best_ep mean {s['epochs']['best_epoch_mean']:.0f}; wall {s['wall_time_sec']['total']/60:.0f} min ({s['wall_time_sec']['mean_per_run']:.0f} s/run); pooled RMSE {s['pooled_rmse_ens']:.2f}"
    if "comp_mae_ens" in s:
        extra += "; comp MAE " + ", ".join(f"{c}:{m:.2f}" for c, m in sorted(s["comp_mae_ens"].items(), key=lambda x: int(x[0])))
    if "mae_all_sizes_mean" in s:
        extra += f"; all-size fold MAE mean {s['mae_all_sizes_mean']:.2f}"
    rows.append((label, ens, sa, seeds + " | " + extra))
print("| System | Disorder_GNN Transformer, 3-seed ensemble (MAE meV/atom, mean ± fold std) | seed-average | per-seed / details |")
print("|---|---|---|---|")
for r in rows:
    print("| " + " | ".join(r) + " |")
