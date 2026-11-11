"""Drop-in artwork for Fig. 1 (the schematic figure), so the drawing side only swaps pictures instead of redrawing from a text order.
Colours are sampled from the current Fig. 1 raster (FIG1_REF, default results/figure_data/fig1_ref.png) so the pieces match it; when that image is
not available the sampled colours fall back to the fixed values given below (the k-means starting colours), with a message.
  panel_a_clusters.svg  right half of panel a: 2-body x3 (1, sqrt2, 2 cells), 3-body x2 (right angle, line), 4-body x2 (square, T),
                        5-body x1 (cross), faint lattice points inside each icon so the distances read, "⋮" under 2-4-body, labels 6 pt Arial.
                        Drawn at final size (130 x 27 mm): insert at 100 %, do not scale (the labels are 6 pt only at 100 %).
  row1_2body.svg ... row4_5body.svg  mini planes for the four rows of the orthonormal-cluster-expansion box (occupation/counting picture:
                        faint lattice + two occurrences of that row's decorated cluster).  No text; may be scaled to the slot.
Inputs:  FIG1_REF (optional reference raster).
Outputs: figures/fig1_dropins/ (SVGs, tile_alloys.txt and, when PyMuPDF is installed, PNG rasters + preview.png).
Run:     python scripts/figures/fig1_dropins.py"""
import os, numpy as np
from PIL import Image
from neuralce.paths import REPO, RESULTS

OUT = f"{REPO}/figures/fig1_dropins"; os.makedirs(OUT, exist_ok=True)
REF = os.environ.get("FIG1_REF", f"{RESULTS}/figure_data/fig1_ref.png")
INIT_T, INIT_O, INIT_N = (93, 193, 176), (245, 178, 107), (100, 122, 170)            # k-means starting colours = the fallback species colours
LINE_FB, BG_FB = np.array([31, 45, 77]), np.array([235, 244, 241])                     # fallback bond-line colour and plane background


def kmeans(px, init, it=30):
    c = np.array(init, float)
    for _ in range(it):
        lab = np.argmin(((px[:, None, :] - c[None]) ** 2).sum(-1), 1); c = np.array([px[lab == k].mean(0) if (lab == k).any() else c[k] for k in range(len(c))])
    return c


hexc = lambda c: "#%02X%02X%02X" % tuple(int(round(v)) for v in c)
if os.path.exists(REF):
    img = np.asarray(Image.open(REF).convert("RGB")).astype(float); Hp, Wp, _ = img.shape
    def region(x0, x1, y0, y1): return img[int(y0 * Hp):int(y1 * Hp), int(x0 * Wp):int(x1 * Wp)].reshape(-1, 3)
    A = region(0.03, 0.17, 0.03, 0.17)                                                    # panel-a lattice
    sat = A[(A.max(1) - A.min(1) > 50) & (A.max(1) > 110)]
    T_, O_, N_ = kmeans(sat, [INIT_T, INIT_O, INIT_N])
    B = region(0.05, 0.16, 0.07, 0.16)                                                    # inside the lattice, below the "Alloy" label
    dk = B[(B.max(1) < 150) & (B[:, 2] - B[:, 0] > 15)]                                   # dark AND bluish: the navy bond lines, not black text
    line = np.median(dk, 0) if len(dk) else LINE_FB
    P = region(0.05, 0.19, 0.72, 0.82); light = P[(P.min(1) > 225) & (P.max(1) < 252)]     # front output plane background (light mint)
    bg = np.median(light, 0) if len(light) else BG_FB
else:
    print(f"reference image {REF} not found: using the fixed fallback colours", flush=True)
    T_, O_, N_, line, bg = np.array(INIT_T, float), np.array(INIT_O, float), np.array(INIT_N, float), LINE_FB, BG_FB
COL = dict(T=hexc(T_), O=hexc(O_), N=hexc(N_)); LINE, BG = hexc(line), hexc(bg)
print("colours:", COL, "line", LINE, "plane background", BG, flush=True)


def darker(h, f=0.72): return hexc([int(h[i:i + 2], 16) * f for i in (1, 3, 5)])
def tint(h, f=0.35): return hexc([255 - (255 - int(h[i:i + 2], 16)) * f for i in (1, 3, 5)])


# cluster = (sites [(x, y, species)], bonds [(i, j)])  -- grid units, y downwards; colours as in the current icons
C1 = [([(0, 0, sp)], []) for sp in "TON"]                                               # 1-body: one kind per species, the complete list
C2 = [([(0, 0, "O"), (1, 0, "N")], [(0, 1)]), ([(0, 0, "T"), (1, 1, "O")], [(0, 1)]), ([(0, 0, "N"), (2, 0, "T")], [(0, 1)])]
C3 = [([(0, 0, "N"), (1, 0, "O"), (1, 1, "T")], [(0, 1), (1, 2)]), ([(0, 0, "T"), (1, 0, "N"), (2, 0, "O")], [(0, 1), (1, 2)])]
C4 = [([(0, 0, "T"), (1, 0, "N"), (0, 1, "O"), (1, 1, "T")], [(0, 1), (1, 3), (3, 2), (2, 0)]), ([(0, 0, "O"), (1, 0, "T"), (2, 0, "N"), (1, 1, "O")], [(0, 1), (1, 2), (1, 3)])]
C5 = ([(1, 0, "O"), (0, 1, "N"), (1, 1, "T"), (2, 1, "N"), (1, 2, "O")], [(2, 0), (2, 1), (2, 3), (2, 4)])
U, R, LW, FAINT = 3.6, 1.0, 0.21, "#D9D9D9"                                            # mm: 1 cell, atom radius, 0.6 pt line


def icon(cl, cx, top, u=U, r=R):
    """SVG of one cluster, horizontally centred at cx, top edge at `top`; faint lattice points fill its bounding box."""
    sites, bonds = cl; xs = [s[0] for s in sites]; ys = [s[1] for s in sites]; w = (max(xs) - min(xs)) * u; h = (max(ys) - min(ys)) * u
    ox, oy = cx - w / 2 - min(xs) * u, top + r - min(ys) * u; pos = lambda s: (ox + s[0] * u, oy + s[1] * u); g = []
    occ = {(s[0], s[1]) for s in sites}
    for gx in range(min(xs), max(xs) + 1):
        for gy in range(min(ys), max(ys) + 1):
            if (gx, gy) not in occ: g.append(f'<circle cx="{ox + gx * u:.3f}" cy="{oy + gy * u:.3f}" r="{0.45 * r:.3f}" fill="{FAINT}"/>')
    for i, j in bonds:
        (x1, y1), (x2, y2) = pos(sites[i]), pos(sites[j]); g.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="{LINE}" stroke-width="{LW}"/>')
    for s in sites:
        x, y = pos(s); g.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{r:.3f}" fill="{COL[s[2]]}" stroke="{darker(COL[s[2]])}" stroke-width="0.12"/>')
    return "\n".join(g), h + 2 * r


def vdots(cx, top, r=0.25, gap=0.9): return "\n".join(f'<circle cx="{cx:.3f}" cy="{top + r + k * gap:.3f}" r="{r}" fill="{LINE}"/>' for k in range(3))


# ---- panel a strip ------------------------------------------------------------------------------------------------------------------------
Wm, Hm, AREA = 130.0, 27.0, (1.0, 22.0)                                               # icons live in y 1-22 mm, labels on baseline 26 mm
columns = [("2-body", C2, True), ("3-body", C3, True), ("4-body", C4, True), ("5-body", [C5], True)]   # no 1-body column; "⋮" = more kinds exist
parts = []
for k, (lab, cls, more) in enumerate(columns):
    cx = Wm * (0.125 + 0.25 * k); hs = [(max(s[1] for s in c[0]) - min(s[1] for s in c[0])) * U + 2 * R for c in cls]
    GAPV = 2.2; tot = sum(hs) + GAPV * (len(hs) - 1) + (1.5 + 2.3 if more else 0.0); y = AREA[0] + (AREA[1] - AREA[0] - tot) / 2
    for c, h in zip(cls, hs):
        svg, _ = icon(c, cx, y); parts.append(svg); y += h + GAPV
    if more: parts.append(vdots(cx, y - GAPV + 1.5))
    parts.append(f'<text x="{cx:.3f}" y="26.0" font-family="Arial" font-size="2.117" text-anchor="middle" fill="#000000">{lab}</text>')   # 2.117 mm = 6 pt
open(f"{OUT}/panel_a_clusters.svg", "w").write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{Wm}mm" height="{Hm}mm" viewBox="0 0 {Wm} {Hm}">\n'
                                                 f'<rect width="{Wm}" height="{Hm}" fill="white"/>\n' + "\n".join(parts) + "\n</svg>\n")

# ---- mini planes (rows of the orthonormal-cluster-expansion box) --------------------------------------------------------------------------
# Each tile: a 6 x 6 alloy with two separated occurrences of its decorated cluster, and the background is repaired until those two are the ONLY
# occurrences (any rotation / reflection counts), so "the plane marks every site where the cluster occurs" is exact and no occurrences overlap.
OPS = [np.array(m) for m in ([[1, 0], [0, 1]], [[0, -1], [1, 0]], [[-1, 0], [0, -1]], [[0, 1], [-1, 0]], [[1, 0], [0, -1]], [[-1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1], [-1, 0]])]


def variants(cl):                                                                                  # decorated orientations, site order kept (bonds reuse it)
    out, seen = [], set()
    for R_ in OPS:
        p_ = [(int(R_[0, 0] * x + R_[0, 1] * y), int(R_[1, 0] * x + R_[1, 1] * y), sp) for x, y, sp in cl[0]]
        mx, my = min(q[0] for q in p_), min(q[1] for q in p_); p_ = [(x - mx, y - my, sp) for x, y, sp in p_]
        k = frozenset(p_)
        if k not in seen: seen.add(k); out.append(p_)
    return out


def find(grid, cl):
    n_ = grid.shape[0]; out = []
    for v in variants(cl):
        wx, wy = max(q[0] for q in v), max(q[1] for q in v)
        for ay_ in range(n_ - wy):
            for ax_ in range(n_ - wx):
                if all(grid[ay_ + y, ax_ + x] == sp for x, y, sp in v): out.append([(ax_ + x, ay_ + y, sp) for x, y, sp in v])
    return out


n = 6; S = 10.0; du = S / n
ROWS = {"row1_2body": (C2[0], [(0, 1), (3, 4)]), "row2_3body": (C3[0], [(0, 0), (3, 3)]), "row3_4body": (C4[0], [(0, 0), (3, 3)]), "row4_5body": (C5, [(0, 0), (3, 3)])}
rng = np.random.default_rng(7); alloys = []
for name, (cl, anchors) in ROWS.items():
    stamped = [[(ax_ + x, ay_ + y, sp) for x, y, sp in cl[0]] for ax_, ay_ in anchors]; fixed = {(x, y) for m in stamped for x, y, _ in m}
    grid = rng.choice(list("TON"), size=(n, n))
    for m in stamped:
        for x, y, sp in m: grid[y, x] = sp
    want = {frozenset((x, y) for x, y, _ in m) for m in stamped}
    for _ in range(5000):                                                                          # repair the background until only the stamps match
        extra = [m for m in find(grid, cl) if frozenset((x, y) for x, y, _ in m) not in want]
        if not extra: break
        free = [(x, y) for x, y, _ in extra[rng.integers(len(extra))] if (x, y) not in fixed]
        x, y = free[rng.integers(len(free))]; grid[y, x] = rng.choice([c for c in "TON" if c != grid[y, x]])
    matches = find(grid, cl)
    assert {frozenset((x, y) for x, y, _ in m) for m in matches} == want, name                      # exactly the two drawn occurrences, nothing else
    lit = {(x, y) for m in matches for x, y, _ in m}
    g = [f'<rect x="0.1" y="0.1" width="{S - 0.2}" height="{S - 0.2}" fill="{BG}" stroke="{LINE}" stroke-width="{LW}"/>']
    for gy in range(n):
        for gx in range(n):
            if (gx, gy) not in lit: g.append(f'<circle cx="{(gx + 0.5) * du:.3f}" cy="{(gy + 0.5) * du:.3f}" r="{0.22 * du:.3f}" fill="{tint(COL[grid[gy, gx]])}"/>')
    P_ = lambda q: ((q[0] + 0.5) * du, (q[1] + 0.5) * du)
    for m in matches:
        for i, j in cl[1]:
            (x1, y1), (x2, y2) = P_(m[i]), P_(m[j]); g.append(f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" stroke="{LINE}" stroke-width="{LW}"/>')
    for x, y in sorted(lit):
        cx_, cy_ = P_((x, y)); g.append(f'<circle cx="{cx_:.3f}" cy="{cy_:.3f}" r="{0.30 * du:.3f}" fill="{COL[grid[y, x]]}" stroke="{darker(COL[grid[y, x]])}" stroke-width="0.08"/>')
    open(f"{OUT}/{name}.svg", "w").write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{S}mm" height="{S}mm" viewBox="0 0 {S} {S}">\n' + "\n".join(g) + "\n</svg>\n")
    alloys.append(f"# {name}\n" + "\n".join(" ".join(r) for r in grid)); print(f"{name}: occurrences = {len(matches)} (both drawn, no others, no overlap)", flush=True)
open(f"{OUT}/tile_alloys.txt", "w").write("\n\n".join(alloys) + "\n")

# ---- preview (rasterised with PyMuPDF; skipped when it is not installed) -------------------------------------------------------------------
try:
    import pymupdf
except ImportError:
    pymupdf = None
if pymupdf is None:
    print("preview skipped: PyMuPDF (pymupdf) not installed; wrote the SVGs to", OUT, flush=True)
else:
    ims = []
    for f in ["panel_a_clusters"] + list(ROWS):
        d = pymupdf.open(f"{OUT}/{f}.svg"); pix = d[0].get_pixmap(dpi=600); pix.save(f"{OUT}/{f}.png"); ims.append(Image.open(f"{OUT}/{f}.png").convert("RGB"))
    Wt = max(ims[0].width, sum(i.width for i in ims[1:]) + 20 * 3); Ht = ims[0].height + 20 + max(i.height for i in ims[1:])
    sheet = Image.new("RGB", (Wt, Ht), "white"); sheet.paste(ims[0], (0, 0)); x = 0
    for im in ims[1:]: sheet.paste(im, (x, ims[0].height + 20)); x += im.width + 20
    sheet.save(f"{OUT}/preview.png"); print("wrote", OUT, flush=True)
