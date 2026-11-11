"""Format check of the v2 PDFs in figures/v2 (or the directory given as the first argument): page size, fonts, text sizes, and that the SVG and PNG exist.
Needs PyMuPDF (pip install pymupdf).
Run:  python scripts/figures/v2/check_pdf.py [dir]"""
import sys, os, glob, collections
import pymupdf
from neuralce.paths import REPO
D = sys.argv[1] if len(sys.argv) > 1 else f"{REPO}/figures/v2"
for f in sorted(glob.glob(f"{D}/*.pdf")):
    p = pymupdf.open(f)[0]; fonts = collections.Counter(); sizes = collections.Counter()
    for b in p.get_text("dict")["blocks"]:
        for l in b.get("lines", []):
            for s in l["spans"]:
                if s["text"].strip(): fonts[s["font"]] += len(s["text"]); sizes[round(s["size"], 1)] += len(s["text"])
    stem = f[:-4]; ok = all(os.path.exists(stem + e) for e in (".svg", ".png"))
    print(f"{os.path.basename(f):28s} {p.rect.width / 72 * 25.4:6.1f} x {p.rect.height / 72 * 25.4:6.1f} mm | fonts {dict(fonts)} | sizes {dict(sorted(sizes.items()))} | svg+png {ok} | images embedded {len(p.get_images())}")
