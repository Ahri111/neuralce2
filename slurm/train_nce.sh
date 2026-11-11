#!/bin/bash
#SBATCH --job-name=train_nce
#SBATCH --partition=eertekin-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Nested-CV evaluation of the neural cluster expansion (scripts/train/train_nce.py).
# usage: sbatch slurm/train_nce.sh nipt     (args: SYSTEM [SPIN] [SELECT] [AGG]; only the PtNi data set is shipped)
# The partition name is site specific; override with sbatch --partition=... .
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"
source env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SYSTEM=${1:-nipt} SPIN=${2:-0} SELECT=${3:-0} AGG=${4:-mean+max} ${NCE_PY:-python} -u scripts/train/train_nce.py
