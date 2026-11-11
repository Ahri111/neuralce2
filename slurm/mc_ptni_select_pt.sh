#!/bin/bash
#SBATCH --job-name=mc_ptni_select_pt
#SBATCH --partition=eertekin-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out --error=logs/%x_%j.err
# (the partition above is site specific: override it with sbatch --partition=...)
#
# PtNi model selection: parallel tempering of the 256-atom random cell for every saved parameter seed of the neural CE at one
# composition (scripts/mc/ptni_select_prep.py makes the init cells + reference energies; ptni_select_analyze.py reads the runs).
# usage: sbatch slurm/mc_ptni_select_pt.sh x50 "0 1 2" [n_steps (1200000)] [tag]
# env: OUTDIR (default $NCE_RUNS/ptni_select)  CKPT (models/nce/nipt_v3_nomax_sel/model.pkl)  INIT (start cif, default
#      models/structures/ptni/init_<x>.cif)  TMIN (300)  TGRID (optional T-grid file)
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"; source env.sh
export OMP_NUM_THREADS=1
PY=${NCE_PY:-python}
X=${1:-x50}; SEEDS=${2:-"0 1 2"}; NSTEPS=${3:-1200000}; TAG=${4:-}
D=${OUTDIR:-$NCE_RUNS/ptni_select}; mkdir -p $D
INIT=${INIT:-models/structures/ptni/init_$X.cif}
[ -f $INIT ] || { INIT=$D/init_$X.cif; [ -f $INIT ] || OUTDIR=$D $PY -u scripts/mc/ptni_select_prep.py || exit 1; }
for s in $SEEDS; do
  $PY -u -m neuralce.mc.run --framework nce --ckpt ${CKPT:-models/nce/nipt_v3_nomax_sel/model.pkl} \
    --init_cif $INIT --model_seed $s --pt --T_min ${TMIN:-300} --T_max 1400 --n_replicas 32 ${TGRID:+--T_grid_file $TGRID} \
    --n_steps $NSTEPS --chunk 4000 --exchange_every 400 --sample_interval 200 --log_every 200000 \
    --energy_mode local --out_dir $D/pt_${X}_seed$s$TAG --seed $((100 + s)) || exit 1
done
echo "PTNI_SEL_DONE $X"
