"""What a pair-only model (EPI, Liu 2025; BIC-selected range) cannot see, in all three systems, without any ground-state
knowledge: (1) structures identical in the pair descriptor the fitted EPI actually uses, and their DFT energy spread;
(2) PtNi: a pair-only Hamiltonian is symmetric under Ni<->Pt (up to the linear point term), so the ordering energetics of
Pt3Ni and Ni3Pt must be mirror images -- DFT and experiment (Tc 670 K vs 853 K) are not.

Inputs : $RESULTS/cv/external/epi/{stfo_lab12, ptni, fenicr}/{features.npz, summary.json, oof.csv} (scripts/baselines/
         epi_cv.py), data/benchmark_folds.json, $RESULTS/cv/nipt_v3_nomax_sel/{features.npz, oof_nested_ensemble.npy}
         (neural CE PtNi nested CV).
Output : two markdown tables on stdout.
Run    : python scripts/baselines/epi_blindness.py
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
"""
import json, os
import numpy as np, pandas as pd
from neuralce.paths import DATA, RESULTS

EPI = f"{RESULTS}/cv/external/epi"
for _p in (f"{EPI}/stfo_lab12/features.npz", f"{EPI}/fenicr/features.npz"):              # the STFO / Fe-Ni-Cr EPI runs are not shipped: skip cleanly
    if not os.path.exists(_p):
        print(f"[skip] {os.path.basename(__file__)}: {_p} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under $NCE_RESULTS/cv/external/epi to run", flush=True); raise SystemExit(0)
print("## (1) degeneracy in the EPI descriptor\n\n| system | EPI range used | structures identical to another one | DFT spread inside groups: mean / max (meV/atom) | worst group |\n|---|---|---|---|---|")
for sysn, natoms in (("stfo_lab12", 160), ("ptni", 108), ("fenicr", None)):
    d = f"{EPI}/{sysn}"; z = np.load(f"{d}/features.npz", allow_pickle=True); s = json.load(open(f"{d}/summary.json"))
    o = pd.read_csv(f"{d}/oof.csv"); K = int(o.K.mode()[0]); ids = [str(i) for i in z["ids"]]; y = np.asarray(z["y"], float); npair = len(z["pairs"])
    cols = [z["X_point"], z["X_pair"][:, :K * npair]]
    if "X_spair" in z.files: cols += [z["X_spoint"], z["X_spair"][:, :K * npair]]          # chem+spin EPI (FeNiCr)
    X = np.round(np.hstack(cols), 6)
    nat = pd.read_csv(f"{d}/oof.csv").astype({"id": str}).groupby("id").n_atoms.first().reindex(ids).fillna(0).values
    if "n_atoms" in z.files: nat = np.asarray(z["n_atoms"])
    g = {}
    for i, x, n in zip(ids, X, nat): g.setdefault((int(n),) + tuple(x), []).append(i)
    deg = [v for v in g.values() if len(v) > 1]; yy = dict(zip(ids, y)); sp = [(max(yy[i] for i in v) - min(yy[i] for i in v)) * 1e3 for v in deg]
    w = deg[int(np.argmax(sp))] if deg else []
    print(f"| {sysn} | K={K} shells (<= {float(z['shell_d'][K-1]):.2f} A){' + spin pairs' if 'X_spair' in z.files else ''} | {sum(len(v) for v in deg)} / {len(ids)} | {np.mean(sp) if sp else 0:.1f} / {max(sp) if sp else 0:.1f} | "
          + ", ".join(f"{i} ({(yy[i]-min(yy[j] for j in w))*1e3:+.1f})" for i in sorted(w, key=lambda k: yy[k])[:4]) + " |")
print("   (features.npz keys of fenicr:", list(np.load(f"{EPI}/fenicr/features.npz", allow_pickle=True).files), ")")
# ---- (2) PtNi mirror asymmetry ----
it = pd.DataFrame(json.load(open(f"{DATA}/benchmark_folds.json"))["ptni"]["items"]); E_Ni, E_Pt = -617.84377 / 108, -652.72821 / 108
it["x"] = it.n_ni / 108; it["Hf"] = (it.y - (it.x * E_Ni + (1 - it.x) * E_Pt)) * 1e3
epi = pd.read_csv(f"{EPI}/ptni/oof.csv").groupby("id").pred.mean(); it["epi"] = (it.id.map(epi) - (it.x * E_Ni + (1 - it.x) * E_Pt)) * 1e3
CV = f"{RESULTS}/cv/nipt_v3_nomax_sel"
z = np.load(f"{CV}/features.npz", allow_pickle=True); conv = pd.Series(np.load(f"{CV}/oof_nested_ensemble.npy"), index=[str(i) for i in z["ids"]])
it["conv"] = (it.id.map(conv) - (it.x * E_Ni + (1 - it.x) * E_Pt)) * 1e3
print("\n## (2) PtNi: lowest formation enthalpy at mirrored compositions (meV/atom); a pair-only model forces the two columns of a row to be equal up to a linear term\n")
print("| n_Ni / 108 | DFT min Hf | mirror n_Ni | DFT min Hf (mirror) | DFT asymmetry | EPI asymmetry (its own minima) | conv asymmetry |\n|---|---|---|---|---|---|---|")
for n in sorted(set(it.n_ni)):
    m = 108 - n
    if n >= m or m not in set(it.n_ni): continue
    a, b = it[it.n_ni == n], it[it.n_ni == m]
    print(f"| {n} | {a.Hf.min():.1f} | {m} | {b.Hf.min():.1f} | {a.Hf.min()-b.Hf.min():+.1f} | {a.epi.min()-b.epi.min():+.1f} | {a.conv.min()-b.conv.min():+.1f} |")
