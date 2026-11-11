"""Label corrections of the STFO benchmark from the DFT re-check.

Rule (model independent): only labels whose ORIGINAL run ended non-FM are eligible; the FM re-relaxation (same
protocol, two relaxations) must be converged and lower by > 1 meV/site (DFT noise), then the LOWER energy replaces
the label.  FM labels that were re-checked because of a large model residual are NOT corrected (sensitivity list
only), so there is no selection bias towards the model.

Input  : --summary  csv of the DFT re-check with columns id, label_nonFM, converged, dE_meV, E_new
                    (default $RESULTS/dft_recheck/summary.csv; this raw re-check table is not part of the data set,
                    the resulting data/new_bench/label_corrections.json is shipped instead)
Outputs: --out      data/new_bench/label_corrections.json {id: new eV/site} (applied by train_nce.py, STFO_LABELFIX=1)
         --csv_out  data/new_bench/detailed_info_corrected.csv   (corrected copy of the label csv)
         --folds_out data/benchmark_folds_corrected.json         (corrected copy of the benchmark folds)
         --al_csv / --al_out  optional active-learning label csv (config_id, E_dft_eV) and its corrected copy;
                    skipped when the input does not exist (default $DATA/al_labels/al_labels.csv, not shipped).
Env    : STFO_FIX_RULE  nonfm (default) | all   (eligibility rule)
Run    : python scripts/train/stfo_label_corrections.py --summary <recheck summary.csv>
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
"""
import argparse, json, os, numpy as np, pandas as pd
from neuralce.paths import DATA, RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--summary", default=f"{RESULTS}/dft_recheck/summary.csv")
ap.add_argument("--out", default=f"{DATA}/new_bench/label_corrections.json")
ap.add_argument("--csv_in", default=f"{DATA}/new_bench/detailed_info.csv")
ap.add_argument("--csv_out", default=f"{DATA}/new_bench/detailed_info_corrected.csv")
ap.add_argument("--folds_in", default=f"{DATA}/benchmark_folds.json")
ap.add_argument("--folds_out", default=f"{DATA}/benchmark_folds_corrected.json")
ap.add_argument("--al_csv", default=f"{DATA}/al_labels/al_labels.csv")
ap.add_argument("--al_out", default=f"{DATA}/al_labels/al_labels_corrected.csv")
a = ap.parse_args()
for _p in (a.summary, a.csv_in, a.folds_in):                                              # the STFO inputs are not shipped: skip cleanly when they are missing
    if not os.path.exists(_p):
        print(f"[skip] {os.path.basename(__file__)}: {_p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under --summary / --csv_in / --folds_in ($NCE_RESULTS/dft_recheck, $NCE_DATA) to run", flush=True); raise SystemExit(0)
if "stfo" not in json.load(open(a.folds_in)):
    print(f"[skip] {os.path.basename(__file__)}: {a.folds_in} has no 'stfo' entry (STFO / Fe-Ni-Cr data are not public); provide it under --folds_in to run", flush=True); raise SystemExit(0)
s = pd.read_csv(a.summary)
elig = s.label_nonFM.astype(bool) if os.environ.get("STFO_FIX_RULE", "nonfm") == "nonfm" else np.ones(len(s), bool)
fix = s[elig & s.converged & (s.dE_meV < -1.0)]; keep = s[~(elig & s.converged & (s.dE_meV < -1.0))]
sens = s[(~elig) & s.converged & (s.dE_meV < -1.0)]
out = {"rule": "labels whose original run ended non-FM, re-relaxed FM (same protocol); converged and lower by > 1 meV/site -> lower value",
       "corrections": {r.id: float(r.E_new) for r in fix.itertuples()}, "shift_meV": {r.id: float(r.dE_meV) for r in fix.itertuples()},
       "unchanged": keep.id.tolist(), "sensitivity_only_FM_labels_lower": {r.id: float(r.dE_meV) for r in sens.itertuples()}}
json.dump(out, open(a.out, "w"), indent=1)
# corrected copies of the two label sources every baseline reads
df = pd.read_csv(a.csv_in); n_csv = 0
for sid, e in out["corrections"].items():
    m = df.id.astype(str) == sid; df.loc[m, "total_energy"] = float(e) * df.loc[m, "n_atoms"]; n_csv += int(m.sum())
df.to_csv(a.csv_out, index=False)
bf = json.load(open(a.folds_in)); n_js = 0
for it in bf["stfo"]["items"]:
    if it["id"] in out["corrections"]: it["y"] = float(out["corrections"][it["id"]]); n_js += 1
bf["stfo"]["note"] = bf["stfo"].get("note", "") + f" | labels corrected ({out['rule']}), {n_js} ids, data/new_bench/label_corrections.json"
json.dump(bf, open(a.folds_out, "w"), indent=1)
n_al = None
if os.path.exists(a.al_csv):
    al = pd.read_csv(a.al_csv); n_al = 0
    for sid, e in out["corrections"].items():
        if sid.startswith("AL_"):
            cid = sid[3:].rsplit("_", 1)[0]; m = al.config_id == cid; al.loc[m, "E_dft_eV"] = float(e) * 160.0; n_al += int(m.sum())
    al.to_csv(a.al_out, index=False)
print(f"wrote {a.out}, {a.csv_out} ({n_csv} rows changed), {a.folds_out} ({n_js} items changed)"
      + (f", {a.al_out} ({n_al} rows changed)" if n_al is not None else f"; {a.al_csv} not found, AL labels skipped"))
print("sensitivity (FM labels that also came out lower, NOT applied):", {r.id: round(float(r.dE_meV), 1) for r in sens.itertuples()})
print(f"{len(fix)} labels corrected (mean shift {fix.dE_meV.mean():.1f} meV/site): " + ", ".join(f"{r.id} {r.dE_meV:+.1f}" for r in fix.itertuples()))
print(f"{len(keep)} unchanged: " + ", ".join(keep.id.tolist()))
