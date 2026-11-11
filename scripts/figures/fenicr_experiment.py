"""Experimental short-range-order data of Fe-Ni-Cr used by Fig. 5d (scripts/figures/fig5_stf_fenicr.py and its v2 version).
The same constants drive the run scoring in scripts/mc/fenicr_sro_score.py; they are copied here so that the figure scripts need no import from scripts/mc.

Fig. 5 of Su et al. 2024: Fe56Cr21Ni23, Cenedese, Bley & Lefebvre, Acta Cryst. A40, 228 (1984): 1273 K anneal, neutron diffuse scattering, per-pair
    alpha1 / alpha2 (Fe-Ni, Fe-Cr, Cr-Ni).
Fig. 6 of Su et al. 2024: Fe42.5Cr7.5Ni50, Menshikov, Dimitrov & Teplykh, J. Physique III 7, 1899 (1997), values as tabulated in Wrobel 2015 Table VII:
    (Fe,Cr)-Ni average SRO after a 1323 K quench, an 873 K anneal and 583 K 2.5 MeV electron irradiation
    (1300 K quench -0.049/+0.015, 900 K anneal -0.093/+0.134, 600 K irradiated -0.121/+0.148).
    Fe34Cr20Ni46 (-0.152/-0.088/-0.042, +0.098/+0.086/+0.017) is the paper's SI Fig. 9, kept below as SI9.
PAPER6_*: the paper's own spin-CE curve (Su et al. 2024 Fig. 6, solid lines), read off the figure at 500..1500 K (reading uncertainty ~0.01);
    PAPER6 = that curve interpolated to the experimental temperatures."""
import numpy as np

F5 = {"Fe-Ni": (0.017, -0.002), "Fe-Cr": (-0.009, 0.043), "Cr-Ni": (-0.113, 0.148)}
F6_T = [583, 873, 1323]; F6_A1 = [-0.121, -0.093, -0.049]; F6_A2 = [0.148, 0.134, 0.015]
SI9_A1 = [-0.152, -0.088, -0.042]; SI9_A2 = [0.098, 0.086, 0.017]
PAPER6_T = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
PAPER6_A1 = [-0.180, -0.155, -0.130, -0.100, -0.088, -0.082, -0.078, -0.072, -0.068, -0.062, -0.058]
PAPER6_A2 = [0.650, 0.395, 0.288, 0.168, 0.140, 0.118, 0.105, 0.092, 0.082, 0.075, 0.070]
PAPER6 = ([float(np.interp(t, PAPER6_T, PAPER6_A1)) for t in F6_T], [float(np.interp(t, PAPER6_T, PAPER6_A2)) for t in F6_T])
