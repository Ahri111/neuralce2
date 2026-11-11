"""Score FeNiCr MC runs against the two experiments Su et al. 2024 compare to -- each at ITS OWN composition.

Fig. 5  Fe56Cr21Ni23, Cenedese, Bley & Lefebvre, Acta Cryst. A40, 228 (1984): 1273 K anneal, neutron diffuse scattering,
        per-pair alpha1/alpha2 (Fe-Ni, Fe-Cr, Cr-Ni).  Runs: tags WITHOUT the _F6 suffix, replica nearest 1273 K.
Fig. 6  Fe42.5Cr7.5Ni50, Menshikov, Dimitrov & Teplykh, J. Physique III 7, 1899 (1997): (Fe,Cr)-Ni average SRO after a
        1323 K quench, an 873 K anneal and 583 K 2.5 MeV electron irradiation.  Runs: tags WITH the _F6 suffix.
        alpha_(Fe,Cr)-Ni = (c_Fe alpha_Fe-Ni + c_Cr alpha_Cr-Ni) / (c_Fe + c_Cr) with the run's own composition.
Score = mean |alpha_model - alpha_expt| over the 6 numbers of each figure; the reference is the paper's spin CE (CLAMM)
run through the same MC: clamm_orig_F5 (1273 K in its grid) and clamm_orig_F6.

The module-level constants (F5, F6_T, F6_A1, F6_A2, SI9_*, PAPER6_*, PAPER6) are the experimental / published values
used by the figure scripts; importing this module has no side effects.

Inputs: <run root>/<tag>/sro.json (scripts/mc/fenicr_sro_analyze.py); run root = env FENICR_SRO_ROOT, default
$RUNS/fenicr_select (the tracked copies of the paper's runs live in $RESULTS/fenicr_sro/<tag>/sro.json, which is used
when the run root holds no sro.json).  Writes <run root>/fig56_scores_<set>.json.
Run: python scripts/mc/fenicr_sro_score.py [--set sa|plain|all] [--png6 out.png] [--run6 TAG]
The STFO / Fe-Ni-Cr inputs are not shipped; the script runs when they are provided at the paths above.
"""
import glob, json, os, re, numpy as np
from neuralce.paths import RUNS, RESULTS

R = os.environ.get("FENICR_SRO_ROOT", f"{RUNS}/fenicr_select")
F5 = {"Fe-Ni": (0.017, -0.002), "Fe-Cr": (-0.009, 0.043), "Cr-Ni": (-0.113, 0.148)}
# Fig. 6 of Su et al. = Fe42.5Cr7.5Ni50 (paper text: "Figure 6: ... Fe42.5Cr7.5Ni50"; Menshikov 1997 values as tabulated in
# Wrobel 2015 Table VII: 1300 K quench -0.049/+0.015, 900 K anneal -0.093/+0.134, 600 K irradiated -0.121/+0.148).
# Fe34Cr20Ni46 (-0.152/-0.088/-0.042, +0.098/+0.086/+0.017) is the paper's SI Fig. 9, kept below as SI9.
F6_T = [583, 873, 1323]; F6_A1 = [-0.121, -0.093, -0.049]; F6_A2 = [0.148, 0.134, 0.015]
SI9_A1 = [-0.152, -0.088, -0.042]; SI9_A2 = [0.098, 0.086, 0.017]
# CLAMM reference for Fig. 6 = the paper's own spin-CE curve (Su et al. 2024 Fig. 6, solid lines), read off the figure at
# 500..1500 K and interpolated to the experimental temperatures (reading uncertainty ~0.01)
PAPER6_T = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
PAPER6_A1 = [-0.180, -0.155, -0.130, -0.100, -0.088, -0.082, -0.078, -0.072, -0.068, -0.062, -0.058]
PAPER6_A2 = [0.650, 0.395, 0.288, 0.168, 0.140, 0.118, 0.105, 0.092, 0.082, 0.075, 0.070]
PAPER6 = ([float(np.interp(t, PAPER6_T, PAPER6_A1)) for t in F6_T], [float(np.interp(t, PAPER6_T, PAPER6_A2)) for t in F6_T])


def near(T, t, tol):
    i = int(np.argmin(np.abs(T - t))); return (i if abs(T[i] - t) <= tol else None)


def fig5(d):
    T = np.asarray(d["temps"], float); i = near(T, 1273, 5)
    if i is None: return None
    v = {k: (d["alpha1"][k][i], d["alpha2"][k][i]) for k in F5}
    return float(np.mean([abs(v[k][0] - F5[k][0]) + abs(v[k][1] - F5[k][1]) for k in F5]) / 2), v


def fig6(d):
    T = np.asarray(d["temps"], float); c = d.get("composition")
    if not c: return None
    w = c["Fe"] + c["Cr"]; a1, a2 = [], []
    # targets by the run's own alloy: Ni ~ 0.50 -> Fig. 6 (Fe42.5Cr7.5Ni50); Ni ~ 0.46 -> SI Fig. 9 (Fe34Cr20Ni46)
    fig6_alloy = abs(c["Ni"] - 0.50) < 0.02 and abs(c["Cr"] - 0.075) < 0.02
    T1, T2 = (F6_A1, F6_A2) if fig6_alloy else (SI9_A1, SI9_A2)
    for t in F6_T:
        i = near(T, t, 5)
        if i is None: return None
        a1.append((c["Fe"] * d["alpha1"]["Fe-Ni"][i] + c["Cr"] * d["alpha1"]["Cr-Ni"][i]) / w)
        a2.append((c["Fe"] * d["alpha2"]["Fe-Ni"][i] + c["Cr"] * d["alpha2"]["Cr-Ni"][i]) / w)
    e1 = [abs(x - y) for x, y in zip(a1, T1)]; e2 = [abs(x - y) for x, y in zip(a2, T2)]
    return float(np.mean(e1 + e2)), a1, a2, sum(e1), sum(e2), dict(c, alloy=("Fig6 Fe42.5Cr7.5Ni50" if fig6_alloy else "SI9 Fe34Cr20Ni46"))


def main():
    import argparse
    global R
    ap = argparse.ArgumentParser(); ap.add_argument("--set", default="sa",
        help="spin-rule key: sa/st/fm/... = tags containing _<key>_ (reference clamm_orig_<key>_F5/F6); plain = tags without a "
             "rule key; all = every run")
    ap.add_argument("--png6", default="", help="draw the best Fig. 6 run vs the CLAMM reference vs experiment")
    ap.add_argument("--run6", default="", help="run drawn in --png6 (default: the best-scoring one)")
    a = ap.parse_args(); KEY = "" if a.set in ("plain", "all") else f"_{a.set}_"; REF5 = f"clamm_orig{KEY or '_'}F5"; REF6 = f"clamm_orig{KEY or '_'}F6"
    if not glob.glob(f"{R}/*/sro.json") and glob.glob(f"{RESULTS}/fenicr_sro/*/sro.json"):
        R = f"{RESULTS}/fenicr_sro"; print(f"no sro.json under the run root; scoring the tracked copies in {R}")
    if not glob.glob(f"{R}/*/sro.json"):                                                   # the Fe-Ni-Cr runs are not shipped: skip cleanly
        print(f"[skip] {os.path.basename(__file__)}: {RESULTS}/fenicr_sro/<tag>/sro.json is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under FENICR_SRO_ROOT to run", flush=True); return
    r5, r6 = [], []
    for p in sorted(glob.glob(f"{R}/*/sro.json")):
        tag = os.path.basename(os.path.dirname(p))
        if "smoke" in tag: continue
        if KEY and KEY not in tag + "_": continue
        if a.set == "plain" and re.search(r"_(sa|st|fm|fn|fp|ms\d+)_F[56]$", tag): continue
        if a.set == "plain" and not (tag.endswith("_F6") or tag.endswith("_F6r") or tag.endswith("_F5") or tag.startswith("fenicr_")): continue
        d = json.load(open(p))
        if "Fe-Ni" not in d.get("alpha1", {}): continue
        if tag.endswith("_F6") or tag.endswith("_F6r"):
            s = fig6(d)
            if s: r6.append((s[0], tag, s))
        else:
            s = fig5(d)
            if s: r5.append((s[0], tag, s))
    r5.sort(); r6.sort()
    ref5 = next((x for x in r5 if x[1] == REF5), None)
    if ref5 is None and os.path.exists(f"{R}/clamm_orig_F5/sro.json"):      # the CLAMM Fig. 5 row is shown in every set
        s5 = fig5(json.load(open(f"{R}/clamm_orig_F5/sro.json")))
        if s5:
            REF5 = "clamm_orig_F5"; ref5 = (s5[0], REF5, s5); r5 = sorted([x for x in r5 if x[1] != REF5] + [ref5])
    e1 = [abs(x - y) for x, y in zip(PAPER6[0], F6_A1)]; e2 = [abs(x - y) for x, y in zip(PAPER6[1], F6_A2)]
    ref6 = (float(np.mean(e1 + e2)), "PAPER_CLAMM_Fig6", (float(np.mean(e1 + e2)), PAPER6[0], PAPER6[1], sum(e1), sum(e2), None))
    # Fig. 6 table = runs at the Fig. 6 alloy (Fe42.5Cr7.5Ni50) only; runs at Fe34Cr20Ni46 are the paper's SI Fig. 9 and get
    # their own table (their own experimental targets, no digitised CLAMM curve -> no WIN marks).  Never mix the two.
    is6 = lambda x: bool(x[2][5]) and x[2][5]["alloy"].startswith("Fig6")
    r9 = sorted([x for x in r6 if not is6(x)])
    r6 = sorted([x for x in r6 if is6(x) and (not x[1].startswith("clamm_orig") or x[1].endswith("_F6r"))] + [ref6]); REF6 = "PAPER_CLAMM_Fig6"
    print("== Fig. 5  Fe56Cr21Ni23, 1273 K, per pair (Cenedese 1984) ==")
    print(f"{'':3s} {'run':40s} {'mean|d|':>7s}   Fe-Ni a1/a2     Fe-Cr a1/a2     Cr-Ni a1/a2")
    print(f"{'':3s} {'experiment':40s} {'':7s}   " + "   ".join(f"{F5[k][0]:+.3f}/{F5[k][1]:+.3f}" for k in F5))
    for i, (m, tag, s) in enumerate(r5[:15], 1):
        mark = "  <-- CLAMM" if tag == REF5 else ("  *WIN*" if ref5 and m < ref5[0] else "")
        print(f"{i:3d} {tag:40s} {m:7.3f}   " + "   ".join(f"{s[1][k][0]:+.3f}/{s[1][k][1]:+.3f}" for k in F5) + mark)
    hdr = f"{'':3s} {'run':40s} {'mean|d|':>7s} {'1NNsum':>6s} {'2NNsum':>6s}   a1 583/873/1323        a2 583/873/1323"
    print("\n== Fig. 6  Fe42.5Cr7.5Ni50, (Fe,Cr)-Ni (Menshikov 1997 / Wrobel 2015 Table VII) -- only runs at this alloy ==")
    print(hdr)
    print(f"{'':3s} {'experiment':40s} {'':7s} {'':6s} {'':6s}   " + " ".join(f"{x:+.3f}" for x in F6_A1) + "   " + " ".join(f"{x:+.3f}" for x in F6_A2))
    for i, (m, tag, s) in enumerate(r6, 1):
        allw = ref6 and m < ref6[0] and s[3] < ref6[2][3] and s[4] < ref6[2][4]      # mean, 1NN sum and 2NN sum all below CLAMM
        mark = ("  <-- CLAMM (paper curve)" if tag == REF6 else "  <-- CLAMM (its CE, our MC)" if tag.startswith("clamm_orig") else
                ("  *WIN-ALL*" if allw else ("  *WIN*" if ref6 and m < ref6[0] else "")))
        print(f"{i:3d} {tag:40s} {m:7.3f} {s[3]:6.3f} {s[4]:6.3f}   " + " ".join(f"{x:+.3f}" for x in s[1]) + "   " + " ".join(f"{x:+.3f}" for x in s[2]) + mark)
    print("   Fig. 6 CLAMM = the paper's published spin-CE curve (read off the figure); Fig. 5 CLAMM = its CE through our MC (reproduces the paper's Fig. 5)")
    # every score (not just the printed top rows) -> json, for the combined Fig.5+Fig.6 ranking
    os.makedirs(R, exist_ok=True)
    json.dump(dict(fig5={t: m for m, t, _ in r5}, fig6={t: m for m, t, _ in r6}, si9={t: m for m, t, _ in r9}),
              open(f"{R}/fig56_scores_{a.set}.json", "w"), indent=1)
    if r9:
        print("\n== SI Fig. 9  Fe34Cr20Ni46, (Fe,Cr)-Ni (Menshikov 1997) -- NOT Fig. 6; scored against the SI9 targets ==")
        print(hdr)
        print(f"{'':3s} {'experiment':40s} {'':7s} {'':6s} {'':6s}   " + " ".join(f"{x:+.3f}" for x in SI9_A1) + "   " + " ".join(f"{x:+.3f}" for x in SI9_A2))
        for i, (m, tag, s) in enumerate(r9[:15], 1):
            print(f"{i:3d} {tag:40s} {m:7.3f} {s[3]:6.3f} {s[4]:6.3f}   " + " ".join(f"{x:+.3f}" for x in s[1]) + "   " + " ".join(f"{x:+.3f}" for x in s[2]))
    if a.png6 and r6:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        best = next((x for x in r6 if x[1] == a.run6), None) if a.run6 else next((x for x in r6 if x[1] != REF6), None)
        fig, ax = plt.subplots(figsize=(8.6, 6.4)); RED, BLUE = "#d62728", "#1f4fd8"
        for tag, ls, fill, lab in ((best[1] if best else None, "-", True, "neural CE (ours)"), (REF6, ":", False, "spin CE (Su et al.)")):
            if tag == REF6:
                ax.plot(PAPER6_T, PAPER6_A1, ":", marker="s", color=RED, mfc="none", ms=10, lw=2.3, mew=1.8, label=f"1NN {lab}")
                ax.plot(PAPER6_T, PAPER6_A2, ":", marker="v", color=BLUE, mfc="none", ms=11, lw=2.3, mew=1.8, label=f"2NN {lab}"); continue
            if tag is None or not os.path.exists(f"{R}/{tag}/sro.json"): continue
            d = json.load(open(f"{R}/{tag}/sro.json")); T = np.asarray(d["temps"], float); c = d["composition"]; w = c["Fe"] + c["Cr"]; o = np.argsort(T)
            p1 = ((c["Fe"] * np.asarray(d["alpha1"]["Fe-Ni"]) + c["Cr"] * np.asarray(d["alpha1"]["Cr-Ni"])) / w)[o]
            p2 = ((c["Fe"] * np.asarray(d["alpha2"]["Fe-Ni"]) + c["Cr"] * np.asarray(d["alpha2"]["Cr-Ni"])) / w)[o]
            ax.plot(T[o], p1, ls, marker="s", color=RED, mfc=RED if fill else "none", ms=10, lw=2.3, mew=1.8, label=f"1NN {lab}")
            ax.plot(T[o], p2, ls, marker="v", color=BLUE, mfc=BLUE if fill else "none", ms=11, lw=2.3, mew=1.8, label=f"2NN {lab}")
        ax.plot(F6_T, F6_A1, "s", color="k", ms=12, ls="none", label="1NN (Expt)"); ax.plot(F6_T, F6_A2, "v", color="k", ms=13, ls="none", label="2NN (Expt)")
        ax.set_xlabel("T (K)", fontsize=22); ax.set_ylabel("(Fe,Cr)-Ni SRO", fontsize=22); ax.set_xlim(480, 1520); ax.set_ylim(-0.25, 0.85)
        ax.set_xticks([600, 900, 1200, 1500]); ax.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8]); ax.tick_params(labelsize=19, width=1.6, length=7)
        for sp_ in ax.spines.values(): sp_.set_linewidth(1.8)
        ax.legend(fontsize=12, frameon=False, ncol=2, loc="upper right"); fig.tight_layout(); fig.savefig(a.png6, dpi=200)
        print(f"wrote {a.png6}  (best {best[1] if best else '-'} {best[0] if best else float('nan'):.3f} vs CLAMM {ref6[0] if ref6 else float('nan'):.3f})")


if __name__ == "__main__":
    main()
