#!/bin/bash
#SBATCH --job-name=fit_ce_stfo
#SBATCH --partition=eertekin-amd
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Orthonormal-CE heavy fit with nested cutoff x regressor selection (scripts/train/fit_ce_stfo.py); CPU only.
# usage: sbatch slurm/fit_ce_stfo.sh stfo|fenicr
# The STFO / Fe-Ni-Cr inputs of this job are not shipped in this repository (only the PtNi data are public); it runs when they are provided.
# The partition name is site specific; override with sbatch --partition=... .
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"
source env.sh
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SYSTEM=${1:-stfo} NJOBS=16 N_REP=3 ${NCE_PY:-python} -u scripts/train/fit_ce_stfo.py
