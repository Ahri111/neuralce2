#!/bin/bash
#SBATCH -J bench_stage --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=100G -t 01:30:00
#SBATCH --output=logs/%x_%j.out --error=logs/%x_%j.err
# Per-stage GPU timing of one MC swap step (scripts/bench/bench_stage_gpu.py), icet-GPU or nce kernel.
# usage: sbatch --export=ALL,FW=icet slurm/bench_stage_gpu.sh      (env FW=icet|nce, R, CH, SIZES; add -p <gpu partition>, site specific)
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"; source env.sh
export OMP_NUM_THREADS=8 CE_SUPERCELL_PERIODIC=1 CE_CNN_SUPERCELL_PERIODIC=1
nvidia-smi --query-gpu=name --format=csv,noheader
${NCE_PY:-python} -u scripts/bench/bench_stage_gpu.py "$@"
