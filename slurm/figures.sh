#!/bin/bash
# Draw one paper figure (or run one analysis script) on a CPU node:  SCRIPT=scripts/figures/fig4_ptni.py sbatch slurm/figures.sh
# No partition is fixed here: pass --partition=<your CPU partition> to sbatch (site specific).
#SBATCH -J figures --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G -t 00:20:00
#SBATCH --output=logs/%x_%j.out --error=logs/%x_%j.err
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"; source env.sh
: "${SCRIPT:?set SCRIPT=scripts/figures/<name>.py}"
${NCE_PY:-python} -u "${SCRIPT}"
