#!/bin/bash
#SBATCH --job-name=cv_tables
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Accuracy table and parity figure from results/cv/ -> figures/cv/ (PtNi).
# The STFO / Fe-Ni-Cr tables (scripts/analysis/cv_stfo_table.py, cv_fenicr.py) need inputs that are not shipped in this repository
# (only the PtNi data are public); they are not run here and skip cleanly without their inputs.
# usage: sbatch slurm/cv_tables.sh        (also fine on a workstation: bash slurm/cv_tables.sh)
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"
source env.sh
export JAX_PLATFORMS=cpu
PY=${NCE_PY:-python}
$PY -u scripts/analysis/cv_ptni.py || exit 1
echo "CV_TABLES_DONE"
