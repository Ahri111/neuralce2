#!/bin/bash
#SBATCH -J bench_stage_mch --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G -t 06:00:00
#SBATCH --output=logs/%x_%j.out --error=logs/%x_%j.err
# Per-stage timing of the ORIGINAL icet/mchammer canonical swap MC (CPU, single threaded); one process per cell size.
# usage: sbatch slurm/bench_stage_mchammer.sh "4 6 8 10"   (sizes M, run sequentially; env STEPS, WARMUP_SWEEPS, OUT)
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"; source env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
SIZES=${1:-"4 6 8 10"}
STEPS=${STEPS:-2000}
WARMUP_SWEEPS=${WARMUP_SWEEPS:-0}
OUT=${OUT:-$NCE_REPO/results/bench_stage/mchammer.json}
mkdir -p "$(dirname "$OUT")"
hostname; lscpu | grep "Model name"
for M in $SIZES; do
  echo "===== M=$M  $(date) ====="
  ${NCE_PY:-python} -u scripts/bench/bench_stage_mchammer.py \
      --M $M --steps $STEPS --warmup_sweeps $WARMUP_SWEEPS --out "$OUT" || echo "M=$M FAILED (exit $?)"
done
echo "done $(date)"
