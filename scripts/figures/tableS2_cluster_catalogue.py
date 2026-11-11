"""Supplementary Table S2: every cluster of every channel of the cluster-convolution models (the channels of Fig. 3c, Supplementary Figs. S1 and S3
and Table S1).  Read from results/figure_data/channel_table.json (scripts/analysis/channel_table.py); channel names from scripts/figures/channel_labels.py.
One row per cluster: number of sites, species allowed on each site, every site-site distance and, for the fcc systems, the neighbour order of every
site pair.  Spin channels (FeNiCr) are spin-pair features, one row per channel.  Fig. S3 draws one representative per channel; this table is the full list.
Inputs:  results/figure_data/channel_table.json.
Outputs: figures/TableS2_cluster_catalogue.{xlsx,tex} and, when pdflatex is available, TableS2_cluster_catalogue.pdf (skipped with a message otherwise).
Run:     python scripts/figures/tableS2_cluster_catalogue.py"""
import os, sys, ast, json, shutil, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import channel_labels as CL
from neuralce.paths import REPO, RESULTS
import pandas as pd

D = f"{REPO}/figures"; os.makedirs(D, exist_ok=True); CT = f"{RESULTS}/figure_data/channel_table.json"
if not os.path.exists(CT):
    print(f"[skip] tableS2_cluster_catalogue.py: {CT} is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under $NCE_RESULTS/figure_data/ (scripts/analysis/channel_table.py writes it) to run", flush=True); sys.exit(0)
T = json.load(open(CT))
SYS = [("ptni", "PtNi", "PtNi"), ("stfo", "SrTi1−xFexO3−x/2", r"SrTi$_{1-x}$Fe$_x$O$_{3-x/2}$"), ("fenicr", "FeNiCr", "FeNiCr")]
for sysk, _, _ in SYS:                                                                  # systems missing from the table (model not shipped) are skipped
    if sysk not in T: print(f"[skip] tableS2_cluster_catalogue.py: {CT} has no '{sysk}' entry, that model is not shipped in this repository (STFO / Fe-Ni-Cr data are not public); provide it under $NCE_MODELS/nce/ and rerun scripts/analysis/channel_table.py to run", flush=True)
SYS = [t for t in SYS if t[0] in T]
VAC = ("X", "Xe", "Vac", "V")                                                            # vacancy placeholders (the SrTi1-xFexO3-x/2 model uses Xe)


def site_species(sysk, letters, tex=False):
    """Species allowed on each site of the cluster, e.g. Fe/Ti – O/V_O.  Single-sublattice (fcc) systems: '—' (every site is the same)."""
    m = T[sysk]["info"]["sublattice_species"]
    if len(m) == 1: return "—"
    vo = r"V$_\mathrm{O}$" if tex else "V_O"
    return " – ".join("/".join(vo if s in VAC else s for s in m[l]) for l in letters)


rows = []
for sysk, sys_txt, sys_tex in SYS:
    ch_all = T[sysk]["channels"]; first_sys = True
    for key in CL.LABELS[sysk]:                                                         # channel order of channel_labels.py
        if key not in ch_all: continue
        ch = ch_all[key]; lab = CL.label(sysk, key)
        if not ch["clusters"]:                                                           # spin channel: features, not geometric clusters
            sp = T[sysk]["info"].get("spin_pairs", []); pairs = ", ".join("–".join(p) for p in sp)
            desc = f"{ch['n_spin_features']} spin-pair features" + (f" ({pairs})" if ch["n_spin_features"] == len(sp) else "")
            rows.append(dict(sys=sys_txt if first_sys else "", sys_tex=sys_tex if first_sys else "", ch=lab if True else "", n="", sites="2", species="—",
                             d="", nn="", note=desc)); first_sys = False
            continue
        cl = sorted(ch["clusters"], key=lambda c: (c["sites"], c["d_max"], c["d_pairs"]))
        for i, c in enumerate(cl):
            letters = ast.literal_eval(c["sublattices"]) if c["sublattices"] else []
            rows.append(dict(sys=sys_txt if first_sys else "", sys_tex=sys_tex if first_sys else "", ch=lab if i == 0 else "", n=str(i + 1), sites=str(c["sites"]),
                             species=site_species(sysk, letters), species_tex=site_species(sysk, letters, tex=True),
                             d=", ".join(f"{x:.2f}" for x in c["d_pairs"]), nn=", ".join(str(x) for x in c["nn"]) if c["nn"] else "—", note=""))
            first_sys = False
        print(f"{sysk:7s} {CL.plain(lab):60s} {len(cl) if ch['clusters'] else ch['n_spin_features']:4d}", flush=True)

# ---- spreadsheet ----------------------------------------------------------------------------------------------------------------------------------
X = pd.DataFrame([{"System": r["sys"], "Channel": CL.plain(r["ch"]), "Cluster": r["n"], "Sites": r["sites"], "Species on each site": r["species"],
                   "Site–site distances (Å)": r["d"], "Neighbour orders (fcc)": r["nn"], "Note": r["note"]} for r in rows])
NOTE = ("One row per cluster of each channel of the cluster-convolution models (orbits kept by the cluster selection on the full data). "
        "Distances are those of the parent lattice of each model (the primitive cell of its cluster space). Neighbour orders are given for the fcc systems (1 = nearest neighbour). "
        "FeNiCr spin channels are spin-pair features on a neighbour shell, not separate clusters. Point clusters are part of every model and are not listed.")
with pd.ExcelWriter(f"{D}/TableS2_cluster_catalogue.xlsx", engine="openpyxl") as xw:
    X.to_excel(xw, sheet_name="Table S2", index=False); pd.DataFrame({"Note": [NOTE]}).to_excel(xw, sheet_name="Table S2", index=False, startrow=len(X) + 2)

# ---- LaTeX (longtable) ------------------------------------------------------------------------------------------------------------------------------
def tx(s): return CL.tex(s) if s else ""
body = []
for r in rows:
    if r["sys_tex"] and body: body.append(r"\midrule")
    d = r["d"] if r["d"] else r["note"]
    body.append(f"{r['sys_tex']} & {tx(r['ch'])} & {r['n']} & {r['sites']} & {r.get('species_tex', r['species'])} & {d} & {r['nn']} \\\\")
TEX = r"""\begin{longtable}{p{2.1cm}p{3.6cm}rrp{3.4cm}p{4.6cm}p{1.8cm}}
\caption{}\label{tab:cluster_catalogue}\\
\toprule
System & Channel & \# & Sites & Species on each site & Site--site distances (\AA) & Neighbour orders \\
\midrule
\endfirsthead
\toprule
System & Channel & \# & Sites & Species on each site & Site--site distances (\AA) & Neighbour orders \\
\midrule
\endhead
\bottomrule
\endfoot
""" + "\n".join(body) + "\n" + r"""\end{longtable}
{\footnotesize """ + NOTE.replace("Å", r"\AA{}") + "}\n"
open(f"{D}/TableS2_cluster_catalogue.tex", "w").write(TEX)

# optional: render to PDF (needs pdflatex); skipped with a message when it is missing
if shutil.which("pdflatex") is None:
    print("wrote TableS2_cluster_catalogue.{xlsx,tex}; PDF skipped: pdflatex not found", flush=True)
else:
    work = tempfile.mkdtemp(prefix="tableS2_render_")
    open(f"{work}/render.tex", "w").write(r"""\documentclass{article}
\usepackage[a4paper,landscape,margin=1cm]{geometry}\usepackage[T1]{fontenc}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}
\usepackage{booktabs,longtable}\usepackage{sansmath}\sansmath\pagestyle{empty}\makeatletter\renewcommand{\caption}[1]{}\makeatother
\begin{document}\footnotesize
\input{""" + f"{D}/TableS2_cluster_catalogue.tex" + r"""}
\end{document}
""")
    for _ in range(2):                                                                         # longtable needs two passes for column widths
        rc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "render.tex"], cwd=work, capture_output=True, text=True)
    if rc.returncode == 0:
        shutil.copy(f"{work}/render.pdf", f"{D}/TableS2_cluster_catalogue.pdf"); print("wrote TableS2_cluster_catalogue.{xlsx,tex,pdf}", flush=True)
    else:
        print("pdflatex failed:\n", rc.stdout[-3000:], flush=True); sys.exit(1)
print("rows:", len(rows), "| clusters per system:", {s: sum(len(c["clusters"]) for c in T[s]["channels"].values()) for s, _, _ in SYS})
