#!/bin/bash
#SBATCH -J ptni_exact_analyze --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# CPU analysis of the PT reference runs (scripts/bench/ptni_exact_pt_analyze.py).
# env: ARGS (e.g. "--N 256 --interim" or "--selftest").   usage: sbatch --export=ALL,ARGS="--N 48" slurm/ptni_exact_analyze.sh
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"; source env.sh
export OMP_NUM_THREADS=8 JAX_PLATFORMS=cpu
${NCE_PY:-python} -u scripts/bench/ptni_exact_pt_analyze.py $ARGS || exit 1
echo ANALYZE_DONE
