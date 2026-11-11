#!/bin/bash
#SBATCH --job-name=map_structures
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=32G --time=03:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Map the augmented STFO set onto the icet primitive -> data/new_bench_aug467/mapped.pkl (scripts/train/map_structures.py).
# usage: sbatch slurm/map_structures.sh [set]    (default new_bench_aug467)
# The STFO / Fe-Ni-Cr inputs of this job are not shipped in this repository (only the PtNi data are public); it runs when they are provided.
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"
source env.sh
export OMP_NUM_THREADS=1
SET=${1:-new_bench_aug467}
${NCE_PY:-python} -u scripts/train/map_structures.py --csv data/$SET/detailed_info.csv --cif_dir data/$SET --out data/$SET --njobs 8
