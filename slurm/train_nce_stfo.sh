#!/bin/bash
#SBATCH --job-name=train_nce_stfo
#SBATCH --partition=eertekin-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# FINAL training of one STFO neural-CE variant -> models/nce/<tag>/model.pkl
# The STFO / Fe-Ni-Cr inputs of this job are not shipped in this repository (only the PtNi data are public); it runs when they are provided.
# usage: sbatch slurm/train_nce_stfo.sh <tag> [ENV=VAL ENV2=VAL ...]
#   e.g. sbatch slurm/train_nce_stfo.sh stfo_dd_p10_w10_wd5 STFO_SHELLS=2.41,3.135,3.705,4.08,10 FINAL_DEPTH=2 FINAL_WD=1e-5 WEIGHT_T=10
# The partition name is site specific; override with sbatch --partition=... .
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"
source env.sh
export OMP_NUM_THREADS=1
TAG=$1; shift
env FINAL=1 SYSTEM=stfo SELECT=1 AGG=mean FINAL_PATIENCE=3000 OUT_TAG=$TAG "$@" ${NCE_PY:-python} -u scripts/train/train_nce.py || exit 1
echo "STFO_VARIANT_DONE $TAG"
