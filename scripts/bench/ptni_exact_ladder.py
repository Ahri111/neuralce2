"""Temperature ladders and starting cells for the PtNi (50/50) reference runs of the cluster-convolution model.

Ladder rule (one replica per temperature, neighbours exchange configurations):
  * within a phase the exchange acceptance is ~ erfc(db sigma_E / 2): spacing db = 1 / sigma_E(T; N) (acceptance ~ 0.5),
    sigma_E taken from the 16,384-atom two-phase run (ordered branch below T_c, disordered above) and scaled by sqrt(N / 16384);
  * across the first-order transition the acceptance between an ordered and a disordered replica is exp(-db N L),
    L = latent heat per atom (12.3 meV/atom at 16,384 atoms): inside the window T_c +- W the spacing is also <= k T^2 / (N L);
  * low temperatures: dT capped at 60 K (sigma_E -> 0 in the frozen L1_0 phase); above 1400 K: db = 1 / sigma_E(infinity)
    down to beta = 0 (last point 1e8 K = exact random-mixing entropy anchor).
Input: the 16,384-atom two-phase run $RUNS/ptni_two_phase/conv/x50/M16/{energies.npz, branch.json}.
Writes <out>/Tgrid_N<N>.txt and the starting cells: random 50/50 and ideal L1_0 on the same ideal fcc lattice (a = 3.924 A).
usage: python scripts/bench/ptni_exact_ladder.py [--out DIR (default $RESULTS/ptni_exact/ladder)] [--sizes 48 256 864]"""
import os, argparse, json, numpy as np
from neuralce.paths import RUNS, RESULTS
KB = 8.617333262e-5; A_FCC = 3.924; L_LAT = 0.0123; N_REF = 16384
M16 = f"{RUNS}/ptni_two_phase/conv/x50/M16"
CELLS = {48: (2, 2, 3), 108: (3, 3, 3), 256: (4, 4, 4), 864: (6, 6, 6), 2048: (8, 8, 8)}
WINDOW = {48: 0.0, 108: 120.0, 256: 100.0, 864: 60.0, 2048: 40.0}          # T_c +- W where the 1/(N L) spacing applies
TC = 845.0


def sigma_ref():
    """sigma_E(T) (total eV, N_REF atoms) of each branch from the 16,384-atom two-phase run (second half of the samples)."""
    z = np.load(f"{M16}/energies.npz"); tr = z['traj']; T = z['temps']; br = json.load(open(f"{M16}/branch.json"))['branch']
    s = np.array([tr[r, tr.shape[1] // 2:].std() * N_REF for r in range(len(T))])
    o = np.array([b == "ordered" for b in br])
    To, so = T[o & (T <= 850)], s[o & (T <= 850)]; Td, sd = T[~o & (T >= 840)], s[~o & (T >= 840)]
    return (To, so), (Td, sd)


def ladder(N, W, T_lo=300.0, T_mid=1400.0):
    (To, so), (Td, sd) = sigma_ref(); scale = np.sqrt(N / N_REF)
    def sig(T):                                                                # the larger of the two branches near T_c (safe side)
        s_o = np.interp(T, To, so) if T <= 870 else 0.0
        s_d = np.interp(T, Td, sd) if T >= 820 else 0.0
        return max(s_o, s_d, 0.05 * np.sqrt(N_REF)) * scale                   # floor: 0.05 eV/sqrt(atom) (frozen phase has ~0 fluctuation)
    Ts = [T_lo]
    while Ts[-1] < T_mid:
        T = Ts[-1]; dT = KB * T * T / sig(T)
        if abs(T - TC) <= W or abs(T + dT - TC) <= W: dT = min(dT, KB * T * T / (N * L_LAT))
        dT = min(dT, 60.0); Ts.append(min(T + dT, T_mid))
    s_inf = sig(1e8); b = 1.0 / (KB * T_mid); tail = []
    while b > 1.0 / s_inf:
        b -= 1.0 / s_inf; tail.append(1.0 / (KB * b))
    tail = [t for t in tail if t < 5e4] + [1e8]
    return np.array(Ts + tail)


def cells(N, out):
    from ase.build import bulk
    from ase.io import write
    at = bulk("Ni", "fcc", a=A_FCC, cubic=True).repeat(CELLS[N]); assert len(at) == N
    rng = np.random.default_rng(12345 + N); Z = np.full(N, 28); Z[rng.choice(N, N // 2, replace=False)] = 78
    rnd = at.copy(); rnd.numbers = Z; write(f"{out}/init_random_N{N}.cif", rnd)
    z = at.get_scaled_positions(wrap=True)[:, 2] * CELLS[N][2] * 2; layer = np.round(z).astype(int) % 2
    ordd = at.copy(); ordd.numbers = np.where(layer == 1, 78, 28); assert (ordd.numbers == 78).sum() == N // 2
    write(f"{out}/init_L10_N{N}.cif", ordd)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=f"{RESULTS}/ptni_exact/ladder"); ap.add_argument("--sizes", nargs="+", type=int, default=[48, 256, 864])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    for N in a.sizes:
        T = ladder(N, WINDOW[N]); w = (np.abs(T - TC) <= WINDOW[N]).sum()
        np.savetxt(f"{a.out}/Tgrid_N{N}.txt", T, fmt="%.4f", header=f"N={N} exact-reference ladder: {len(T)} temperatures, {w} inside T_c +- {WINDOW[N]:.0f} K")
        cells(N, a.out)
        print(f"N={N}: {len(T)} temperatures ({w} in the window), min spacing {np.diff(T[T < 2000]).min():.2f} K, "
              f"{np.sum(T <= 1400)} up to 1400 K, tail {np.sum(T > 1400)}", flush=True)
