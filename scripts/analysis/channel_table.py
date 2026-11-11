"""Cluster channels of the cluster-convolution models used for the SHAP analysis (the models of scripts/analysis/shap_cluster_conv.py), same grouping
rule (group_of copied from there).  Per channel: number of clusters (orbits kept by the full-data selection), per cluster the number of sites, every
site-site distance, the largest one and, for fcc, the neighbour order of every site pair; model cutoffs and the species allowed on each sublattice letter.
Spin channels (FeNiCr): spin-pair features per neighbour shell.
Inputs:  models/nce/<tag>/model.pkl for nipt_v3_nomax_sel, stfo_dd_p10_w10_wd5_aug467_al1a_w0, fenicr_s2b_mm_c65_d1.
Outputs: results/figure_data/channel_table.json (read by scripts/figures/tableS2_cluster_catalogue.py for Supplementary Table S2) + printed table.
Run:     python scripts/analysis/channel_table.py"""
import os, json, itertools, numpy as np
from icet import ClusterSpace
from ase.neighborlist import neighbor_list
from neuralce.paths import MODELS as MODEL_DIR, RESULTS
from neuralce.nce.mc_model import load_model
MODELS = {"ptni": ("nipt_v3_nomax_sel", [(2.77, "pairs 1NN"), (3.92, "pairs 2NN")]), "stfo": ("stfo_dd_p10_w10_wd5_aug467_al1a_w0", None),
          "fenicr": ("fenicr_s2b_mm_c65_d1", [(2.55, "chem. pairs 1NN"), (3.60, "chem. pairs 2NN")])}
out = {}
for system, (tag, SH) in MODELS.items():
    if not os.path.exists(f"{MODEL_DIR}/nce/{tag}/model.pkl"):                           # the STFO / Fe-Ni-Cr models are not shipped: table the systems present
        print(f"[skip] channel_table.py: {MODEL_DIR}/nce/{tag}/model.pkl is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under $NCE_MODELS/nce/ to run", flush=True); continue
    mdl = load_model(f"{MODEL_DIR}/nce/{tag}/model.pkl"); cs = ClusterSpace(mdl["primitive"], list(mdl["cutoffs"]), mdl["chemical_symbols"]); od = cs.as_list
    keep = np.asarray(mdl["keep"]); K = int(mdl.get("K_chem", len(od))); prim = cs.primitive_structure
    i_, j_, d_ = neighbor_list("ijd", prim, 12.0); shells = np.unique(np.round(d_, 2))            # neighbour shells of the lattice (all sublattices together)
    def nn_order(d): return int(np.argmin(np.abs(shells - d)) + 1)
    def cluster_geom(entry):
        orb = cs.orbit_list.get_orbit(int(entry["orbit_index"])); cl = orb.representative_cluster
        pos = [prim.positions[s.index] + np.dot(s.unitcell_offset, prim.cell) for s in cl.lattice_sites]
        dd = [float(np.linalg.norm(a - b)) for a, b in itertools.combinations(pos, 2)]
        return len(pos), dd
    def group_of(k):
        if k >= K:
            s = (k - K) // len(mdl["spin_pairs"]); return f"spin pairs {s + 1}NN" if s < 2 else "other spin terms"
        d = od[k]; o = int(d["order"]); dist = 2 * float(d["radius"])
        if o == 0: return None
        if o == 1: return "singlets"
        if system == "stfo":
            if o == 2:
                if dist < 2.3: return "B-O pair 1.98 A  (Fe/Ti - V_O)"
                if dist < 3.1: return "O-O pair 2.8 A  (V_O-V_O, 90 deg)"
                if dist < 4.1: return "B-B 1NN and O-O 180 deg pairs (3.96 A)"
                return "longer pairs"
            return "triplets" if o == 3 else "quadruplets"
        if o == 2:
            for sd_, nm in SH:
                if abs(dist - sd_) < 0.2: return nm
            return ("chem. " if system == "fenicr" else "") + "pairs 3NN+"
        return "triplets" if o == 3 else "quadruplets"
    ch = {}
    for k in keep:
        k = int(k); g = group_of(k)
        if g is None or g == "singlets": continue
        c = ch.setdefault(g, dict(functions=0, orbits={}, spin_features=0))
        if k >= K: c["spin_features"] += 1; c["spin_shell"] = (k - K) // len(mdl["spin_pairs"]) + 1; continue
        c["functions"] += 1; e = od[k]; oi = int(e["orbit_index"])
        if oi not in c["orbits"]:
            n, dd = cluster_geom(e); c["orbits"][oi] = dict(sites=n, d_max=round(max(dd), 3), nn=sorted(nn_order(x) for x in dd) if system != "stfo" else None, sublattices=str(e.get("sublattices", "")),
                                                            d_pairs=sorted(round(x, 2) for x in dd))              # every site-site distance (Supplementary Table S2)
    info = {k: (mdl[k] if not hasattr(mdl[k], "tolist") else mdl[k].tolist()) for k in ("cutoffs", "spin", "spin_edges", "spin_pairs", "spin_species") if k in mdl}
    info["sublattice_species"] = {sl.symbol: list(sl.chemical_symbols) for sl in cs.get_sublattices(prim).active_sublattices}   # letter in "sublattices" -> species allowed there
    out[system] = dict(model=tag, info=info, shells=shells[:10].tolist(), channels={g: dict(n_clusters=len(c["orbits"]), n_functions=c["functions"], n_spin_features=c["spin_features"],
                       spin_shell=c.get("spin_shell"), clusters=list(c["orbits"].values())) for g, c in ch.items()})
    print(f"\n=== {system}  model {tag}  cutoffs {info.get('cutoffs')}  shells(A) {np.round(shells[:8], 2).tolist()}", {k: v for k, v in info.items() if k != "cutoffs"})
    for g, c in out[system]["channels"].items():
        cl = c["clusters"]; print(f"  {g:42s} clusters {c['n_clusters']:3d}  functions {c['n_functions']:3d}  spin features {c['n_spin_features']:2d}  spin shell {c['spin_shell']}")
        for q in cl: print(f"       sites {q['sites']}  d_max {q['d_max']:.2f} A  NN orders {q['nn']}  sublattices {q['sublattices']}")
if not out: raise SystemExit(0)
os.makedirs(f"{RESULTS}/figure_data", exist_ok=True); json.dump(out, open(f"{RESULTS}/figure_data/channel_table.json", "w"), indent=1)
print("wrote", f"{RESULTS}/figure_data/channel_table.json")
