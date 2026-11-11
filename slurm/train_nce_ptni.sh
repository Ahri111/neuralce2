#!/bin/bash
#SBATCH --job-name=train_nce_ptni
#SBATCH --partition=eertekin-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# usage: sbatch slurm/train_nce_ptni.sh <tag> <CE_CUT e.g. 8,6,5> <SELECT 0|1> [FINAL_DEPTH]
# FINAL training of one PtNi neural-CE variant -> models/nce/<tag>/model.pkl (skipped when the file exists), then
# parallel-tempering runs at x25/x50/x75 (fixed seed) with the MC driver.  The initial cells
# models/structures/ptni/init_x{25,50,75}.cif are produced by scripts/mc/ptni_select_prep.py.
# The partition name is site specific; override with sbatch --partition=... .
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"
source env.sh
export OMP_NUM_THREADS=1
PY=${NCE_PY:-python}
TAG=$1; CUT=$2; SEL=$3; DEPTH=${4:-}
CKPT=models/nce/$TAG/model.pkl
if [ ! -f $CKPT ]; then
  FINAL=1 SYSTEM=nipt SPIN=0 SELECT=$SEL AGG=mean CE_CUT=$CUT FINAL_PATIENCE=3000 OUT_TAG=$TAG FINAL_DEPTH=$DEPTH $PY -u scripts/train/train_nce.py || exit 1
fi
D=${NCE_RUNS:-runs}/ptni_select/$TAG; INITDIR=${INITDIR:-models/structures/ptni}; NSTEPS=${NSTEPS:-2000000}
for x in x25 x50 x75; do
  $PY -m neuralce.mc.run --framework nce --ckpt $CKPT --init_cif $INITDIR/init_$x.cif --model_seed 0 --pt --T_min ${TMIN:-300} --T_max 1400 --n_replicas 32 ${TGRID:+--T_grid_file $TGRID} \
    --n_steps $NSTEPS --chunk 4000 --exchange_every 400 --sample_interval 200 --log_every 200000 \
    --energy_mode local --out_dir $D/pt_${x}_seed0 --seed 100 || exit 1
done
echo "PTNI_VARIANT_DONE $TAG"
