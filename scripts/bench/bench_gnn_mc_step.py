"""Cost of ONE Monte Carlo step for the GNN energy models of the benchmark table (CGCNN, ALIGNN, Disorder_GNN).

None of these models has a local delta-energy: a swap trial needs a FULL forward pass over the whole
structure.  This script measures that forward pass as a function of the number of atoms, on GPU.

Models = exactly the implementations / hyper-parameters of the PtNi benchmark rows
  cgcnn, alignn : scripts/bench/gnn_baselines/neuralce_gnn_jax.py (JAX/flax ports):
                  atom_fea_len 64, n_conv 3, h_fea_len 128, edge_fea_len 8 (class default), fixed sum pool;
                  graph = gnn_baselines/graph_lite.py build_graph_lite (species map from gnn_baselines/nipt_gnn_config.yaml),
                  cutoff 4.5 A, 2 shell one-hot edge features (edges 0/3.3/4.5), max_num_nbr 42 (padded neighbour
                  table, 18 real on fcc).
  disorder_gnn  : external/Configurational-Disorder model_embedding.Transformer (PyTorch-Geometric):
                  2 layers, 64 channels, 4 heads, Conv_list -> ModuleList fix;
                  graph = repo utilities.structureToData, r_cutoff 10 A, max_neighbors 20, dr 0.1 (100 edge feats).
Weights are randomly initialised (timing only).  fp32, TF32 off (JAX_DEFAULT_MATMUL_PRECISION=highest).

Requirements: the optional `gnn` extra (pip install -e ".[gnn]": torch, torch-geometric, flax) and the git submodule
external/Configurational-Disorder (git submodule update --init; https://github.com/qmatyanlab/Configurational-Disorder.git,
commit c5d05ef), whose Disorder_GNN/GNN/{utilities.py, model/model_embedding.py} are loaded by file path.

Structures: ase bulk('Ni','fcc',a=3.924,cubic=True).repeat(M), random 50/50 Ni/Pt; M = 4, 8, 16, 32
            -> 256, 2048, 16384, 131072 atoms.

Every (model, size, batch) is timed in its own subprocess (clean peak-memory counter, OOM isolation):
graph already on the GPU, warm-up passes (incl. jit compile), then N synchronised forward passes.
batch = 1 is one MC replica; batch = 32 is 32 replicas in one pass (ms_per_step = ms per batched pass,
steps_per_s = replica-steps per second = 32 / pass time).
graph_build_ms = wall time of the project's own (python) graph builder for that cell, measured once.
On a fixed lattice only the node species change in a swap, so the graph does NOT have to be rebuilt; the
column is reported because the benchmark pipelines build graphs with exactly this code.

Usage (GPU node, see slurm/bench_gnn_mc_step.sh):  python scripts/bench/bench_gnn_mc_step.py [--sizes 4,8,16,32]
Output: $RESULTS/bench_stage/gnn_mc_step.json (env OUT_JSON overrides, e.g. a separate file per GPU type; env BENCH_OUT_DIR
overrides the directory).  Graph cache: $RUNS/bench_stage/gnn_mc_step_cache (env BENCH_CACHE_DIR).
"""
import os, sys, json, time, argparse, subprocess, importlib.util

from neuralce.paths import REPO, RUNS, RESULTS

EXT = f"{REPO}/external/Configurational-Disorder/Disorder_GNN/GNN"
OUT_DIR = os.environ.get("BENCH_OUT_DIR", f"{RESULTS}/bench_stage")
OUT_JSON = os.environ.get("OUT_JSON", f"{OUT_DIR}/gnn_mc_step.json")                     # env OUT_JSON: e.g. a separate file per GPU type
CACHE = os.environ.get("BENCH_CACHE_DIR", f"{RUNS}/bench_stage/gnn_mc_step_cache")


def ext_module(name, relpath):
    """Import a module of the Disorder-GNN submodule by file path (the submodule is not a package)."""
    path = os.path.join(EXT, relpath)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: initialise the submodule first (git submodule update --init external/Configurational-Disorder)")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod
A_FCC = 3.924
SIZES_M = [4, 8, 16, 32]                      # 256, 2048, 16384, 131072 atoms
BATCHES = [1, 32]
JAX_HP = dict(atom_fea_len=64, n_conv=3, h_fea_len=128)                  # hyper-parameters of the CGCNN / ALIGNN benchmark rows
JAX_GRAPH = dict(cutoff=4.5, n_shells=2, max_num_nbr=42, shell_edges=[0.0, 3.3, 4.5])
DG_HP = dict(num_hidden_layers=2, num_hidden_channels=64, num_heads=4)   # hyper-parameters of the Disorder_GNN benchmark row
DG_GRAPH = dict(r_cutoff=10.0, max_neighbors=20, dr=0.1)

ap = argparse.ArgumentParser()
ap.add_argument("--worker", default=None, choices=[None, "build", "time"])
ap.add_argument("--fw", default=None, choices=[None, "jax", "torch"])
ap.add_argument("--model", default=None)
ap.add_argument("--M", type=int, default=4)
ap.add_argument("--batch", type=int, default=1)
ap.add_argument("--tiled", action="store_true", help="torch build: tiled fcc fallback instead of the repo builder")
ap.add_argument("--sizes", default=None, help="driver: comma list of M (default 4,8,16,32)")
args = ap.parse_args()


def make_atoms(M, seed=0):
    import numpy as np
    from ase.build import bulk
    at = bulk("Ni", "fcc", a=A_FCC, cubic=True).repeat(M)
    rng = np.random.default_rng(seed)
    z = np.full(len(at), 28)
    z[rng.permutation(len(at))[: len(at) // 2]] = 78
    at.set_atomic_numbers(z)
    return at


def emit(d):
    print("RESULT_JSON:" + json.dumps(d), flush=True)


# ════════════════════════════════════════════════════════════════════════════════════════════════
# build workers (CPU; timed once per size, cached)
# ════════════════════════════════════════════════════════════════════════════════════════════════
def build_jax(M):
    import numpy as np
    os.environ["JAX_PLATFORMS"] = "cpu"
    from pymatgen.io.ase import AseAtomsAdaptor
    from gnn_baselines.graph_lite import build_graph_lite                  # species_map Ni/Pt from gnn_baselines/nipt_gnn_config.yaml
    at = make_atoms(M)
    st = AseAtomsAdaptor.get_structure(at)
    t0 = time.perf_counter()
    af, nf, nfi, _ = build_graph_lite(dict(crystal=st, spin_states=np.zeros(len(at))), JAX_GRAPH["cutoff"],
                                      JAX_GRAPH["n_shells"], JAX_GRAPH["max_num_nbr"], False,
                                      shell_edges=JAX_GRAPH["shell_edges"], edge_basis="shell_onehot")
    ms = (time.perf_counter() - t0) * 1e3
    n_real = int((nf.sum(-1) > 0).sum(1).mean())
    np.savez(f"{CACHE}/jax_M{M}.npz", atom_fea=af, nbr_fea=nf, nbr_fea_idx=nfi)
    emit(dict(graph_build_ms=ms, n_atoms=len(at), builder="graph_lite.build_graph_lite",
              nbr_table=list(nfi.shape), real_nbrs_per_atom=n_real))


def _dg_tiled(M):
    """fcc-only vectorised equivalent of utilities.structureToGraph (same neighbour rule: the max_neighbors
    nearest within r_cutoff, same Gaussian edge expansion), by translating the 4 basis-atom patterns of a
    small reference cell.  Used only when the repo's python builder cannot handle the cell."""
    import numpy as np, torch
    from pymatgen.io.ase import AseAtomsAdaptor
    from torch_geometric.data import Data
    M0 = 4
    ref = AseAtomsAdaptor.get_structure(make_atoms(M0))
    nb = ref.get_all_neighbors(DG_GRAPH["r_cutoff"], include_index=True)
    K = DG_GRAPH["max_neighbors"]
    dcell = np.zeros((4, K, 3), dtype=np.int64); bj = np.zeros((4, K), dtype=np.int64); dist = np.zeros((4, K))
    for b in range(4):                                   # atoms 0..3 = basis of cell (0,0,0) (ase repeat order)
        srt = sorted(nb[b], key=lambda x: x.nn_distance)[:K]
        for k, n in enumerate(srt):
            j = n.index
            cj = np.array([j // 4 // M0 // M0, j // 4 // M0 % M0, j // 4 % M0]) + np.asarray(n.image, dtype=int) * M0
            dcell[b, k] = cj; bj[b, k] = j % 4; dist[b, k] = n.nn_distance
    N = 4 * M ** 3
    ii = np.arange(N)
    c = np.stack([ii // 4 // M // M, ii // 4 // M % M, ii // 4 % M], 1); b = ii % 4
    cj = (c[:, None, :] + dcell[b]) % M                                         # (N, K, 3)
    jj = ((cj[..., 0] * M + cj[..., 1]) * M + cj[..., 2]) * 4 + bj[b]           # (N, K)
    d = dist[b]                                                                 # (N, K)
    rgrid = np.arange(0, DG_GRAPH["r_cutoff"], DG_GRAPH["dr"]); sigma = DG_GRAPH["r_cutoff"] / 3
    ea = (np.exp(-0.5 * (rgrid[None, :] - d.reshape(-1, 1)) ** 2 / sigma ** 2) / np.sqrt(2 * np.pi) / sigma)
    ei = np.stack([np.repeat(ii, K), jj.reshape(-1)])
    return Data(edge_index=torch.tensor(ei, dtype=torch.long), edge_attr=torch.tensor(ea, dtype=torch.float)), d, jj


def build_torch(M, tiled):
    import numpy as np, torch
    from pymatgen.io.ase import AseAtomsAdaptor
    at = make_atoms(M)
    info = dict(n_atoms=len(at))
    if not tiled:
        dg_util = ext_module("utilities", "utilities.py")
        st = AseAtomsAdaptor.get_structure(at)
        t0 = time.perf_counter()
        d = dg_util.structureToData(st, E=0.0, normalize_E=False, **DG_GRAPH)
        info.update(graph_build_ms=(time.perf_counter() - t0) * 1e3, builder="utilities.structureToData")
        if M <= 8:                                        # gate for the tiled fallback (same graph statistics)
            dt, dd, jj = _dg_tiled(M)
            pos = at.get_positions(); L = M * A_FCC
            dv = pos[jj.reshape(-1)] - np.repeat(pos, jj.shape[1], 0); dv -= L * np.round(dv / L)
            assert np.allclose(np.linalg.norm(dv, axis=1), dd.reshape(-1), atol=1e-6), "tiled: wrong neighbours"
            assert dt.edge_index.shape == d.edge_index.shape and dt.edge_attr.shape == d.edge_attr.shape
            a = d.edge_attr.view(len(at), -1, d.edge_attr.shape[1]); b = dt.edge_attr.view_as(a)
            assert torch.allclose(a.sum(1), b.sum(1), atol=1e-4), "tiled: edge features differ from repo builder"
            info["tiled_gate"] = "PASS"
    else:
        t0 = time.perf_counter()
        d, _, _ = _dg_tiled(M)
        info.update(graph_build_ms=None, tiled_build_ms=(time.perf_counter() - t0) * 1e3,
                    builder="tiled fcc fallback (vectorised; repo builder not usable at this size)")
    d.x = torch.tensor(at.get_atomic_numbers(), dtype=torch.float)      # embedding index = Z (as in the Disorder_GNN CV benchmark)
    torch.save(d, f"{CACHE}/torch_M{M}.pt")
    info.update(n_edges=int(d.edge_index.shape[1]), num_edge_features=int(d.edge_attr.shape[1]))
    emit(info)


# ════════════════════════════════════════════════════════════════════════════════════════════════
# timing workers (GPU)
# ════════════════════════════════════════════════════════════════════════════════════════════════
def n_timed(t1):
    return int(min(200, max(10, 5.0 / max(t1, 1e-6))))


def time_jax(model_name, M, B):
    import numpy as np
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax, jax.numpy as jnp
    from gnn_baselines.neuralce_gnn_jax import create_neuralce
    g = np.load(f"{CACHE}/jax_M{M}.npz")
    N = g["atom_fea"].shape[0]
    res = dict(n_atoms=int(N), batch=B, framework=f"jax {jax.__version__}")
    model = create_neuralce(model_type=model_name, pool_mode="fixed", readout_type="sum",
                            nbr_fea_len=JAX_GRAPH["n_shells"], **JAX_HP)
    g0 = np.load(f"{CACHE}/jax_M{SIZES_M[0]}.npz"); N0 = g0["atom_fea"].shape[0]
    params = model.init(jax.random.PRNGKey(0), atom_fea=jnp.asarray(g0["atom_fea"]), nbr_fea=jnp.asarray(g0["nbr_fea"]),
                        nbr_fea_idx=jnp.asarray(g0["nbr_fea_idx"]), batch_size=1, n_atoms_per_crystal=N0)
    res["n_params"] = int(sum(p.size for p in jax.tree.leaves(params)))
    try:
        rng = np.random.default_rng(1)
        af = np.zeros((B, N, 2), np.float32)                          # B independent random 50/50 occupations
        for r in range(B):
            occ = np.zeros(N, int); occ[rng.permutation(N)[: N // 2]] = 1
            af[r, np.arange(N), occ] = 1.0
        nfi = (g["nbr_fea_idx"][None] + (np.arange(B) * N)[:, None, None]).astype(np.int32)   # as _flatten_batch
        af = jnp.asarray(af.reshape(B * N, 2))
        nf = jnp.asarray(np.broadcast_to(g["nbr_fea"][None], (B,) + g["nbr_fea"].shape).reshape(B * N, -1, 2))
        nfi = jnp.asarray(nfi.reshape(B * N, -1))

        @jax.jit
        def fwd(p, a, e, i):
            return model.apply(p, atom_fea=a, nbr_fea=e, nbr_fea_idx=i, batch_size=B, n_atoms_per_crystal=N)

        t0 = time.perf_counter(); fwd(params, af, nf, nfi).block_until_ready()
        res["compile_plus_first_ms"] = (time.perf_counter() - t0) * 1e3
        for _ in range(3):
            fwd(params, af, nf, nfi).block_until_ready()
        t0 = time.perf_counter(); fwd(params, af, nf, nfi).block_until_ready(); t1 = time.perf_counter() - t0
        n = n_timed(t1); ts = []
        T0 = time.perf_counter()
        for _ in range(n):
            t0 = time.perf_counter()
            out = fwd(params, af, nf, nfi); out.block_until_ready()
            ts.append(time.perf_counter() - t0)
        total = time.perf_counter() - T0
        assert out.shape == (B, 1) and bool(jnp.all(jnp.isfinite(out)))
        ms = total / n * 1e3
        st = jax.local_devices()[0].memory_stats()
        res.update(ms_per_step=ms, ms_median=float(np.median(ts) * 1e3), steps_per_s=B * 1e3 / ms, n_timed=n,
                   gpu_mem_MB=st["peak_bytes_in_use"] / 2 ** 20, status="OK")
    except Exception as e:                                             # noqa
        msg = str(e)
        oom = "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower()
        res.update(status="OOM" if oom else "ERROR: " + msg[:300])
    emit(res)


def time_torch(M, B):
    import numpy as np, torch
    from torch_geometric.data import Batch
    dg_models = ext_module("model_embedding", os.path.join("model", "model_embedding.py"))
    dev = torch.device("cuda")
    d = torch.load(f"{CACHE}/torch_M{M}.pt", weights_only=False)
    N = d.num_nodes
    res = dict(n_atoms=int(N), batch=B, framework=f"torch {torch.__version__} / pyg", n_edges=int(d.edge_index.shape[1]))
    torch.manual_seed(0)
    m = dg_models.Transformer(num_edge_features=int(d.edge_attr.shape[1]), device=dev, **DG_HP)
    m.Conv_list = torch.nn.ModuleList(m.Conv_list)                     # the repo keeps Conv_list as a plain list (not registered); same fix as in the CV benchmark
    m = m.to(dev).eval()
    res["n_params"] = int(sum(p.numel() for p in m.parameters()))
    try:
        torch.cuda.reset_peak_memory_stats()
        d = d.to(dev)
        if B > 1:
            d = Batch.from_data_list([d] * B)                          # built on the GPU -> CUDA OOM is catchable
            z = torch.full((B * N,), 28.0, device=dev)
            for r in range(B):
                z[r * N + torch.randperm(N, device=dev)[: N // 2]] = 78.0
            d.x = z
        with torch.no_grad():
            for _ in range(5):
                out = m(d)
            torch.cuda.synchronize()
            t0 = time.perf_counter(); m(d); torch.cuda.synchronize(); t1 = time.perf_counter() - t0
            n = n_timed(t1); ts = []
            torch.cuda.synchronize(); T0 = time.perf_counter()
            for _ in range(n):
                t0 = time.perf_counter()
                out = m(d); torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            total = time.perf_counter() - T0
        assert out.numel() == B and bool(torch.isfinite(out).all())
        ms = total / n * 1e3
        res.update(ms_per_step=ms, ms_median=float(np.median(ts) * 1e3), steps_per_s=B * 1e3 / ms, n_timed=n,
                   gpu_mem_MB=torch.cuda.max_memory_allocated() / 2 ** 20, status="OK")
    except torch.cuda.OutOfMemoryError:
        res.update(status="OOM")
    except Exception as e:                                             # noqa
        res.update(status=("OOM" if "out of memory" in str(e).lower() else "ERROR: " + str(e)[:300]))
    emit(res)


# ════════════════════════════════════════════════════════════════════════════════════════════════
# driver
# ════════════════════════════════════════════════════════════════════════════════════════════════
def run_worker(wargs, timeout):
    cmd = [sys.executable, "-u", os.path.abspath(__file__)] + wargs
    t0 = time.time()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err, rc = f"TIMEOUT after {timeout}s", -999
    res = None
    for line in out.splitlines():
        if line.startswith("RESULT_JSON:"):
            res = json.loads(line[len("RESULT_JSON:"):])
    print(f"  [{' '.join(wargs)}] rc={rc} {time.time()-t0:.0f}s -> {res}", flush=True)
    if res is None:
        tail = " | ".join(err.strip().splitlines()[-4:])
        oom = "RESOURCE_EXHAUSTED" in err or "out of memory" in err.lower()
        killed = rc in (-9, 137)
        res = dict(status="OOM" if oom else (f"HOST_KILLED(rc={rc}, host memory?)" if killed else f"FAILED rc={rc}: {tail[-400:]}"))
        print("    stderr tail:", tail[-1500:], flush=True)
    return res


def driver():
    os.makedirs(CACHE, exist_ok=True); os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip().splitlines()[0]
    print("GPU:", gpu, flush=True)
    sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else SIZES_M
    results, builds = [], {}

    def save():
        json.dump(results, open(OUT_JSON, "w"), indent=1)

    def build(fw, M):
        if (fw, M) in builds:
            return builds[(fw, M)]
        b = run_worker(["--worker", "build", "--fw", fw, "--M", str(M)], timeout=3000)
        if "status" in b and fw == "torch":                           # repo builder failed -> tiled fcc fallback
            why = b["status"]
            b = run_worker(["--worker", "build", "--fw", fw, "--M", str(M), "--tiled"], timeout=1800)
            b["repo_builder_failure"] = why
        builds[(fw, M)] = b
        return b

    def bench(model, fw, M):
        b = build(fw, M)
        for B in BATCHES:
            row = dict(model=model, n_atoms=4 * M ** 3, batch=B, ms_per_step=None, steps_per_s=None, gpu_mem_MB=None,
                       graph_build_ms=b.get("graph_build_ms"), n_params=None, gpu_name=gpu, status=None,
                       graph_builder=b.get("builder"))
            if "status" in b:
                row["status"] = "GRAPH_BUILD_" + b["status"]
            else:
                r = run_worker(["--worker", "time", "--fw", fw, "--model", model, "--M", str(M), "--batch", str(B)],
                               timeout=1800)
                row.update(r)
                if row.get("ms_per_step") is not None and row["graph_build_ms"] is not None:
                    row["ms_per_step_with_graph_rebuild"] = row["ms_per_step"] + row["graph_build_ms"]
            for k in ("repo_builder_failure", "tiled_build_ms"):
                if k in b:
                    row[k] = b[k]
            results.append(row); save()

    MODELS = [("cgcnn", "jax"), ("alignn", "jax"), ("disorder_gnn", "torch")]
    for M in [m for m in sizes if m < 32]:                            # small sizes first, 131072 atoms last
        for model, fw in MODELS:
            bench(model, fw, M)
    for M in [m for m in sizes if m >= 32]:
        for model, fw in MODELS:
            bench(model, fw, M)
    save()
    print(f"saved {OUT_JSON}")
    print(f"{'model':13s} {'atoms':>7s} {'B':>3s} {'ms/step':>10s} {'steps/s':>10s} {'mem MB':>9s} {'build ms':>10s} {'params':>8s}  status")
    f = lambda v, s: (s % v) if v is not None else "-"
    for r in results:
        print(f"{r['model']:13s} {r['n_atoms']:7d} {r['batch']:3d} {f(r['ms_per_step'], '%10.3f'):>10s} "
              f"{f(r['steps_per_s'], '%10.2f'):>10s} {f(r['gpu_mem_MB'], '%9.1f'):>9s} "
              f"{f(r['graph_build_ms'], '%10.0f'):>10s} {f(r['n_params'], '%8d'):>8s}  {r['status']}")


if __name__ == "__main__":
    if args.worker == "build":
        (build_jax if args.fw == "jax" else build_torch)(args.M, *([] if args.fw == "jax" else [args.tiled]))
    elif args.worker == "time":
        if args.fw == "jax":
            time_jax(args.model, args.M, args.batch)
        else:
            time_torch(args.M, args.batch)
    else:
        driver()
