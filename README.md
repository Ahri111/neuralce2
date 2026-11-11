# neuralce — cluster expansions on the GPU and the neural cluster expansion

Code, data and trained models for *the cluster expansion as the sliding window of a convolutional network*,
demonstrated on the Pt–Ni alloy:

* **Cluster expansion on the GPU.** Every cluster is a kernel that slides over the lattice. The windows
  that cover a site are found by integer arithmetic on a primitive-cell template, so nothing proportional
  to the number of atoms is stored except the state itself. The same `.ce` file fitted with
  [icet](https://icet.materialsmodeling.org) runs 10⁴–10⁶ times faster per Monte Carlo step and
  temperature than icet's own CPU sampler, in 40–50 times less memory, and reaches cells of 10⁷ atoms
  (29.9 GB on one H100) with 256 temperatures in one compiled kernel.
* **Neural cluster expansion (NeuralCE).** The sliding window returns, for every cluster, a map of where
  it occurs. Averaging the maps over sites gives the cluster expansion; keeping them site-resolved and
  applying graph convolutions on the lattice gives the neural cluster expansion. Its Monte Carlo step
  updates only the receptive field of the swapped sites, so the cost per step does not grow with the
  cell (scaling exponent 0.06 against 0.92–0.98 for CGCNN, ALIGNN and Disorder-GNN).
* **PtNi order–disorder transition** from 256 to 10⁷ atoms: the specific-heat peaks stop moving above
  about 10⁴ atoms. The DFT dataset (121 structures), the trained models and the processed results behind
  the figures are included.

Everything here was verified in a fresh virtual environment built from `requirements.txt`: the test
suite passes on CPU and GPU, the figure scripts re-create the submitted numbers exactly from the shipped
processed results, the orthonormal-CE fit reproduces the shipped `.ce` file, and training runs end to end.

**What is shipped.** The code covers all three systems of the paper (PtNi, SrTi₁₋ₓFeₓO₃₋ₓ/₂, Fe–Ni–Cr),
including training, Monte Carlo, Shapley analysis and every figure script. The DFT datasets, trained models
and processed results are shipped for **PtNi only**; the other two datasets are not public. The trained models of all three systems are included. Scripts that
need the datasets print a `[skip]` line naming the missing input and the environment variable or argument that
points at it, and run unchanged once those files are provided.

---

## Contents

1. [Install](#install)
2. [Quick start](#quick-start)
3. [Repository layout](#repository-layout)
4. [Data and trained models](#data-and-trained-models)
5. [Reproducing the paper](#reproducing-the-paper)
6. [Verification](#verification)
7. [Numerics you must know about](#numerics-you-must-know-about)
8. [Citation and license](#citation-and-license)

---

## Install

Python 3.10–3.12, Linux. `icet` is compiled from source during the install (a C++14 compiler is needed;
about three minutes). The pinned versions in `requirements.txt` are the ones the results were produced with.

**Virtual environment (recommended)**

```bash
git clone --recurse-submodules https://github.com/Ahri111/neuralce2.git
cd neuralce2
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt            # CPU
pip install -r requirements-cuda.txt       # optional: NVIDIA GPU (CUDA 12 wheels for JAX)
pip install -e .                           # the package + the `neuralce-mc` command
source env.sh                              # sets JAX_DEFAULT_MATMUL_PRECISION=highest and NCE_* paths
```

`make venv` does the same. **conda**: `conda env create -f environment.yml && conda activate neuralce && pip install -e .`

**Docker / Apptainer**

```bash
docker build -t neuralce .                                   # CUDA 12.6 runtime image, everything inside
docker run --gpus all -it -v $PWD/runs:/work/runs neuralce bash
docker run --gpus all neuralce pytest -q                     # the test suite

apptainer build --fakeroot neuralce.sif docker/neuralce.def  # same image for HPC systems without Docker
apptainer exec --nv -B $PWD/runs:/work/runs neuralce.sif neuralce-mc --help
```

The GNN baselines of the Monte Carlo cost benchmark need `pip install -r requirements-gnn.txt` and the
`external/Configurational-Disorder` submodule; nothing else does.

---

## Quick start

Parallel tempering of the two energy models on a 2 048-atom PtNi cell (about ten minutes on one GPU;
runs on the CPU as well with `--platform cpu` and fewer steps):

```bash
source env.sh
# neural cluster expansion, receptive-field local update
neuralce-mc --framework nce --ckpt models/nce/nipt_v3_nomax_sel/model.pkl \
    --init_cif models/structures/ptni/ref_L10_NiPt.cif --supercell 2 2 2 \
    --pt --T_min 300 --T_max 1400 --n_replicas 32 --n_steps 2000000 --chunk 5000 \
    --exchange_every 5000 --sample_interval 500 --energy_mode local --out_dir runs/demo/nce

# orthonormal cluster expansion (icet basis), template-only GPU tables
neuralce-mc --framework icet --ce models/ce/ptni/icet_c8_6_5.ce \
    --init_cif models/structures/ptni/ref_L10_NiPt.cif --supercell 2 2 2 \
    --pt --T_min 300 --T_max 1400 --n_replicas 32 --n_steps 2000000 --chunk 5000 \
    --exchange_every 5000 --sample_interval 500 --energy_mode delta --out_dir runs/demo/icet
```

Each run writes the energy trajectory (`energies.npz` / `mcmc_result.npz`), the final structure of every
replica (`final_<T>.cif`) and `mcmc_summary.json`. `neuralce-mc --help` lists all options;
`slurm/mc_demo.sh` is the same demo as a SLURM job.

Using the models from Python:

```python
from ase.build import bulk
from neuralce.nce.mc_model import load_model, setup

mdl = load_model("models/nce/nipt_v3_nomax_sel/model.pkl")
atoms = bulk("Ni", "fcc", a=3.924, cubic=True).repeat((4, 4, 4))   # any supercell of the model's lattice
atoms.numbers[::2] = 78                                               # decorate it (Ni/Pt)
mc = setup(mdl, atoms)                                                # template-only tables, canonical site order
print(mc["energy"](mc["sig0"]), "eV/atom")                            # full evaluation (float64 path)
cache = mc["full_cache"](mc["sig0"])                                  # per-site cache for local updates
```

---

## Repository layout

```
neuralce/              the package (pip install -e .)
  ce/                  orthonormal cluster expansion on the GPU
    classical_ce_gpu   GPUClusterExpansion: icet .ce -> GPU tables, energy and swap kernels
    ce_gpu_tiled       O(N) translation-template tables
    ce_gpu_implicit    template-only kernels (Smith normal form of the supercell; memory = state only)
    pt_mcmc_ce         parallel-tempering engine (jitted lax.scan, vmapped replicas, checkpoints)
    icet_mc, gpu_mc    single-T / multi-T / PT drivers, supercell tiling
    train_ce           Lasso / ARDR fit of an icet cluster expansion
  nce/                 neural cluster expansion
    general            layer 0 = icet's local cluster vector (site mean == cluster vector, gated to 1e-9),
                       neighbour shells, forward model, training utilities
    gpu_layer0*        layer-0 tables on the GPU (legacy / tiled / template-only)
    local_update*      receptive-field local energy update
    mc_model           load model.pkl, build the MC machinery for any supercell
    mcmc               single-T / multi-T / PT drivers
    conv, kernels, lattice, features, deep, spin_features
  mc/                  the shared command line (neuralce-mc) and per-replica start structures
scripts/
  train/               train_nce.py (nested cross-validation and final models, all systems), fit_ce_*.py
                       (orthonormal CE baselines), map_structures.py, stfo_label_corrections.py
  mc/                  PtNi size ladders, two-phase free-energy runs, free energy of mixing, Fe-Ni-Cr SRO
  bench/               per-stage GPU/CPU timing (Fig. 2), GNN Monte Carlo cost (Fig. 3a,b), exact-enumeration checks
  analysis/            cross-validation tables, Shapley decomposition, Fe-Ni-Cr order energy, STFO analyses
  baselines/           CLAMM, EPI, CGCNN/ALIGNN, Disorder-GNN cross-validation (accuracy tables)
  figures/             the paper figures (fig2_*.py ... figS4_*.py, Table S1/S2)
gates/                 correctness gates (template-only == tiled tables, layer 0 == icet, MC == training path)
tests/                 pytest suite (CPU or GPU, about one minute)
slurm/                 SLURM launchers for every script above (site-specific partition names inside)
data/                  PtNi DFT dataset (121 cifs + labels), cross-validation folds
models/                trained NeuralCE model (nce/), fitted cluster expansion (ce/), reference structures  (PtNi)
results/               processed inputs of the PtNi figures (timings, C_V curves, two-phase runs, CV predictions)
figures/paper/         Fig. 2, Fig. 4 and Fig. S2 as submitted (pdf, svg, png)
```

Large Monte Carlo output goes to `runs/` (git-ignored); set `NCE_RUNS` to put it on a scratch file system.

---

## Data and trained models

| what | file |
|---|---|
| PtNi DFT labels, 121 structures (108-atom fcc cells, duplicates merged) | `data/nipt/icet_csv_FINAL_dedup.csv`, `data/nipt/cifs/` |
| cross-validation folds | `data/benchmark_folds.json` |
| NeuralCE PtNi (depth 1, 12 channels, 3 seeds) | `models/nce/nipt_v3_nomax_sel/model.pkl` |
| orthonormal CE PtNi (`icet_c8_6_5.ce`, scaled to the 256-atom total energy) | `models/ce/ptni/` |
| ordered reference cells (L1₂ Pt₃Ni, L1₀ NiPt, L1₂ Ni₃Pt) and random start cells | `models/structures/ptni/` |
| NeuralCE SrTi₁₋ₓFeₓO₃₋ₓ/₂ (387-structure CV model; 475- and 520-structure final models) | `models/nce/stfo_dd_p10_w10_wd5*/model.pkl` |
| orthonormal CE SrTi₁₋ₓFeₓO₃₋ₓ/₂ (`icet_q56.ce` cluster space, `icet_aug467_*.ce`) | `models/ce/stfo/` |
| NeuralCE Fe–Ni–Cr with spin channels; orthonormal CE + Heisenberg pairs | `models/nce/fenicr_*/model.pkl`, `models/ce/fenicr/` |
| Fe–Ni–Cr Monte Carlo start cells, temperature grids, allowed spin values | `models/structures/fenicr/` |

The SrTi₁₋ₓFeₓO₃₋ₓ/₂ and Fe–Ni–Cr models are the ones used for Monte Carlo in the paper and run with
`neuralce-mc`; their training labels are not included in the pickles (the DFT data are not public), so the
gates that compare the MC path with the training set run for PtNi only.

Accuracy (mean absolute error of out-of-fold predictions on 119 structures, meV/atom;
`python scripts/analysis/cv_ptni.py`):

| NeuralCE | orthonormal CE | CLAMM | EPI | Disorder-GNN |
|---|---|---|---|---|
| **0.84** | 1.35 | 2.02 | 3.02 | 5.57 |

---

## Reproducing the paper

Every figure script reads processed inputs from `results/` and writes to `figures/`; the raw Monte Carlo
runs (tens of GB) are not shipped, but every producer script and its SLURM launcher are, so each number can
be regenerated from the model and data in this repository.

| figure | what it shows | render | produce the inputs |
|---|---|---|---|
| Fig. 2 | same `.ce` on GPU vs icet CPU: time per step per temperature and memory up to 10⁷ atoms | `scripts/figures/fig2_speed_memory.py` | `scripts/bench/bench_stage_gpu.py` (GPU, `FW=icet R=256`), `scripts/bench/bench_stage_mchammer.py` (CPU) |
| Fig. 3, S1, Table S1 | NeuralCE MC cost stays flat while GNN costs grow with the cell; Shapley contribution of every cluster channel | `scripts/figures/fig3_gnn.py` (renders the systems whose inputs are present: PtNi) | `scripts/bench/bench_gnn_mc_step.py`, `scripts/bench/bench_stage_gpu.py FW=nce`, `scripts/analysis/shap_cluster_conv.py` |
| Fig. 4, S2 | PtNi order–disorder transition from 256 to 10⁷ atoms; C_V peaks converge above ~10⁴ atoms; ΔG_mix | `scripts/figures/fig4_ptni.py` | `slurm/mc_ptni_ladder*.sh` (parallel tempering), `slurm/mc_ptni_two_phase.sh` + `scripts/mc/ptni_two_phase_analyze.py`, `slurm/mc_ptni_gmix.sh` + `scripts/mc/gmix_ptni.py` |
| Fig. 5, S4 | Fe–Ni–Cr second-neighbour order and STFO extrapolation | `scripts/figures/fig5_stf_fenicr.py`, `figS4_fenicr_order.py` (inputs not shipped) | `slurm/mc_fenicr*.sh`, `scripts/analysis/fenicr_order_energy.py`, `scripts/analysis/stfo_extrapolation.py`, `slurm/mc_stfo_gmix.sh` |
| Fig. S3, Table S2 | cluster definitions of every channel | `scripts/figures/figS3_cluster_defs.py`, `tableS2_cluster_catalogue.py` (models present: PtNi) | `scripts/analysis/channel_table.py` |
| accuracy tables | cross-validation of all models | `scripts/analysis/cv_ptni.py` (`cv_stfo_table.py`, `cv_fenicr.py`: inputs not shipped) | `scripts/train/train_nce.py`, `scripts/train/fit_ce_*.py`, `scripts/baselines/*.py` |

Training the neural cluster expansion (nested 10-fold cross-validation, or `FINAL=1` for the model used
in Monte Carlo; all options are environment variables documented at the top of the script):

```bash
SYSTEM=nipt SELECT=1 AGG=mean python scripts/train/train_nce.py             # cross-validation, 3 seeds
SYSTEM=nipt SELECT=1 AGG=mean FINAL=1 python scripts/train/train_nce.py     # final model -> models/nce/<tag>/
ECI_SCALE=256 python scripts/train/fit_ce_ptni.py                           # orthonormal CE -> models/ce/ptni/
```

Finite-size ladder of one composition (N = 4M³ atoms; `M=136` is the 10⁷-atom cell):

```bash
FW=icet X=x50 M=16 INIT=ordered sbatch slurm/mc_ptni_ladder.sh     # orthonormal CE
FW=nce  X=x50 M=16 INIT=ordered sbatch slurm/mc_ptni_ladder.sh     # neural cluster expansion
python scripts/mc/ptni_ladder_analyze.py                          # C_V peak per cell size
```

Exact enumeration of small cells (`scripts/bench/ptni_exact_*.py`) checks the parallel-tempering averages
against the exact partition function.

---

## Verification

```bash
make test          # pytest: imports, layer 0 == icet cluster vector (1e-9), template-only == tiled tables
                   # (energy and local dE, 1e-9), icet GPU tables == icet, CLI end to end for both models
make gates-cpu     # the correctness gates used during development (gates/)
make gates-gpu     # speed / memory sections and the GPU central-entry checks
```

Re-creating the numbers from the shipped processed results (CPU, under a minute):

```bash
python scripts/figures/fig2_speed_memory.py      # figures/fig2_speed_memory.{pdf,svg,png,json} + SourceData_Fig2.xlsx
python scripts/figures/fig4_ptni.py              # figures/fig4_ptni.*, figS2_ptni_tc.*, SourceData_Fig4/FigS2.xlsx
python scripts/analysis/cv_ptni.py               # figures/cv/ptni_parity.{png,md}
```

The json written next to each figure equals the submitted `results/figure_data/*.json` value for value.

| gate | statement |
|---|---|
| `gates/gate_nce_layer0.py` | site mean of the sliding-window cluster maps == icet cluster vector; layer-0 features == icet local cluster vectors; local ΔE == E_after − E_before at depth 0, 1, 2 |
| `gates/gate_nce_implicit.py` | template-only kernels == full tables for the energy, the local ΔE and the cache after a chain of committed moves (float64 and float32); speed and memory up to 10⁷ atoms; `neuralce-mc --framework nce` reproduces the table energy |
| `gates/gate_ce_implicit.py` | orthonormal CE: template-only == tiled tables; energy extensive under 2×2×2 tiling; `neuralce-mc --framework icet` final replicas have identical tiled and template-only energies |
| `gates/gate_nce_final_models.py` | the MC path reproduces the training-time prediction of the shipped model on its training structures |
| `gates/gate_nce_ensemble.py` | the seed-ensemble energy and ΔE equal the mean over the seeds |

---

## Numerics you must know about

* **`JAX_DEFAULT_MATMUL_PRECISION=highest`** (set by `env.sh` and in the container). On Ampere/Hopper
  GPUs JAX runs float32 matmuls in TF32 by default, which shifts the neural-network energies by
  0.2–0.3 eV per cell in a configuration-dependent way.
* The `.ce` file is scaled to the **total energy of the cell it was fitted on** (256 atoms).
  `GPUClusterExpansion(..., energy_scale=n_cells)` and `neuralce-mc --supercell` take care of this.
* `--energy_mode local` (NeuralCE) and `delta` (orthonormal CE) carry the energy in float32 and re-anchor
  it with a full recompute at every chunk boundary; the drift is logged.
* Sites are stored in a canonical order (`cell * n_basis + basis`); the `perm`/`inv_perm` arrays of
  `setup()` map to the input atom order. The output cifs are written in the input order.
* The template-only tables use the Smith normal form of the supercell matrix, so non-diagonal supercells
  work; only the state (a few tens of bytes per site and replica) grows with the cell.

---

## Citation and license

If you use this code or data, please cite the paper (reference to be added on publication). The code is
released under the MIT license (`LICENSE`). Third-party code: icet (MPL-2.0, dependency), the
Disorder-GNN baseline (git submodule, Fang & Yan, npj Comput. Mater. 10, 91 (2024)).
