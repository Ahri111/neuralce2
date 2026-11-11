#!/bin/bash
#SBATCH --job-name=mc_demo
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=16G --time=00:10:00
#SBATCH --output=logs/%x_%j.out --error=logs/%x_%j.err
# (the GPU partition is site specific: add --partition=... here or on the sbatch command line)
#
# Ten-minute GPU demo of the central MC entry with BOTH energy models on PtNi: the ordered L1_0 NiPt reference cell tiled
# 2x2x2 (2048 atoms, species re-shuffled), parallel tempering 300-1400 K with 8 replicas, 20000 steps per replica,
#   nce  = neural cluster expansion, receptive-field local update (--energy_mode local)
#   icet = orthonormal cluster expansion, cluster-instance delta update (--energy_mode delta)
# Output: $NCE_RUNS/demo/{nce,icet}/ (trajectory, final structures, mcmc_summary.json); the summaries are printed at the end.
# env: OUT (default $NCE_RUNS/demo), N_STEPS (20000), NREP (8).
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"; source env.sh
PY=${NCE_PY:-python}; OUT=${OUT:-$NCE_RUNS/demo}; mkdir -p $OUT
COMMON="--init_cif models/structures/ptni/ref_L10_NiPt.cif --supercell 2 2 2 --pt --T_min 300 --T_max 1400 --n_replicas ${NREP:-8} \
        --n_steps ${N_STEPS:-20000} --chunk 1000 --exchange_every 1000 --sample_interval 100 --log_every 5000 --seed 0"
echo "== demo: nce (neural cluster expansion, --energy_mode local)"
$PY -u -m neuralce.mc.run --framework nce --ckpt models/nce/nipt_v3_nomax_sel/model.pkl --energy_mode local $COMMON --out_dir $OUT/nce || exit 1
echo "== demo: icet (orthonormal cluster expansion, --energy_mode delta)"
$PY -u -m neuralce.mc.run --framework icet --ce models/ce/ptni/icet_c8_6_5.ce --energy_mode delta $COMMON --out_dir $OUT/icet || exit 1
for fw in nce icet; do echo; echo "== $OUT/$fw/mcmc_summary.json"; cat $OUT/$fw/mcmc_summary.json; echo; done
echo "DEMO_DONE"
