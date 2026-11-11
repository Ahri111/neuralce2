#!/bin/bash
#SBATCH -J ptni_exact_enum --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=24G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Exact enumeration of the PtNi nce model on a small fcc cell (scripts/bench/ptni_exact_enum.py).
# env: CELL="2 2 3"  WORKER  NW  [GATE=1]  [MAXN = necklaces per worker, debug]  OUT (default $NCE_RUNS/ptni_exact/enum/N<N>)
# usage: sbatch --export=ALL,CELL="2 2 3",WORKER=0,NW=24 slurm/ptni_exact_enum.sh   (add -p <gpu partition>, site specific)
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"; source env.sh
export OMP_NUM_THREADS=4
PY=${NCE_PY:-python}
ARGS="--cell ${CELL:-2 2 3} --worker ${WORKER:-0} --n_workers ${NW:-1}"
[ "${GATE:-0}" = 1 ] && ARGS="$ARGS --gate"
[ -n "$MAXN" ] && ARGS="$ARGS --max_necklaces $MAXN"
[ -n "$OUT" ] && ARGS="$ARGS --out $OUT"
echo "EXACT_ENUM $ARGS"; nvidia-smi --query-gpu=name --format=csv,noheader
$PY -u scripts/bench/ptni_exact_enum.py $ARGS || exit 1
echo "EXACT_ENUM_DONE"
