"""Cluster-channel names used in Fig. 3c, Supplementary Figs. S1 (row labels) and S3 (cell names), Tables S1 and S2 (channel column).
One definition for all of them so the strings are identical.  ONE naming rule for every channel of every system:
    "<n>-body[, <shell or sublattices>] (<distance>)"
  - n-body as in Fig. 1a (1-body ... 5-body); a channel holding two orders says both ("4- and 5-body")
  - after the comma: the neighbour shell (fcc) or the two sublattices (perovskite, B = B site), only when the channel has one kind
  - in parentheses: always a distance in Å - the pair distance for 2-body, "≤ largest distance" for 3-body and up, "d1–d2" for a range
  - spin channels start with "Spin"; nothing else (counts, geometry words, species lists) goes in the name - that is Table S2's job
Distances are those of the model's parent lattice (results/figure_data/channel_table.json from scripts/analysis/channel_table.py).
label(system, key): mathtext string for figures;  plain(): text for spreadsheets;  tex(): LaTeX for the table;  wrapped(): two lines at " (" for figures."""
VO = "V$_{\\mathrm{O}}$"
LABELS = {
    "ptni": {"pairs 1NN": "2-body, 1NN (2.77 Å)", "pairs 2NN": "2-body, 2NN (3.92 Å)", "pairs 3NN+": "2-body, 5NN (6.20 Å)",
             "triplets": "3-body (≤ 4.81 Å)", "quadruplets": "4-body (≤ 3.92 Å)"},
    "stfo": {"B-O pair 1.98 A  (Fe/Ti - V_O)": "2-body, B–O (2.00 Å)", "O-O pair 2.8 A  (V_O-V_O, 90 deg)": "2-body, O–O (2.82 Å)",
             "B-B 1NN and O-O 180 deg pairs (3.96 A)": "2-body, O–O (3.45–4.03 Å)",       # all 5 members are O–O (Table S2); the old name "B–B 1NN" was wrong
             "longer pairs": "2-body (4.20–9.93 Å)",                                     # B–O and O–O mixed, so no sublattice tag
             "triplets": "3-body (≤ 4.68 Å)", "quadruplets": "4- and 5-body (≤ 4.47 Å)"},
    "fenicr": {"chem. pairs 1NN": "2-body, 1NN (2.55 Å)", "chem. pairs 2NN": "2-body, 2NN (3.60 Å)", "chem. pairs 3NN+": "2-body, 3NN–5NN (4.41–5.69 Å)",
               "triplets": "3-body (≤ 4.41 Å)", "spin pairs 1NN": "Spin 2-body, 1NN (2.55 Å)",
               "spin pairs 2NN": "Spin 2-body, 2NN (3.60 Å)", "other spin terms": "Spin 1- and 2-body, species-blind (≤ 3.60 Å)"},
}


def label(system, key): return LABELS[system][key]
def plain(s): return s.replace("$_{\\mathrm{O}}$", "_O")
def wrapped(s):                                                                         # figures: ALWAYS name on line 1, distance on line 2 - same break for every channel
    return s.replace(" (", "\n(", 1) if " (" in s else s
def wrap_to(s, width, head_width=None):                                                 # lines of <= width characters (V_O counts as 3), first break at " ("
    import textwrap
    t = s.replace(VO, "\x01").replace(" Å", "\x02Å").replace("≤ ", "≤\x02"); head, _, tail = t.partition(" (")   # never break inside "≤ 4.68 Å"
    hw = head_width or width; heads = [h + ("," if k < head.count(", ") else "") for k, h in enumerate(head.split(", "))] if len(head) > hw else [head]   # long head: break after its commas first
    parts = [(h, hw) for h in heads] + ([("(" + tail, width)] if tail else [])
    lines = [l for p_, w_ in parts for l in textwrap.wrap(p_.replace("\x01", "V_O"), w_, break_on_hyphens=False, break_long_words=False)]
    return "\n".join(lines).replace("V_O", VO).replace("\x02", " ")
def tex(s):
    t = s.replace("$_{\\mathrm{O}}$", "@VO@").replace("–", "--").replace("Å", r"\,\AA{}").replace("°", r"$^\circ$").replace("≤", r"$\le$").replace("²", r"$^2$")
    return t.replace("@VO@", r"$_\mathrm{O}$").replace(r" \,\AA", r"\,\AA")
