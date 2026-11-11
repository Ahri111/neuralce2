#!/bin/bash
# Exact Shapley values of the cluster channels, one system per job (CPU, JAX on the CPU backend):
#   SYSTEM=ptni sbatch slurm/shap.sh          (SYSTEM = ptni | stfo | fenicr; FILES / OUTNAME / TAG are passed through, see the script)
# No partition is fixed here: pass --partition=<your CPU partition> to sbatch (site specific).
# SYSTEM=stfo | fenicr need the STFO / Fe-Ni-Cr models and Monte Carlo runs, which are not shipped in this repository (only the PtNi data are
# public); the script prints a [skip] line and exits 0 without them, and runs when they are provided.
#SBATCH -J shap --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G -t 04:00:00
#SBATCH --output=logs/%x_%j.out --error=logs/%x_%j.err
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"; source env.sh
: "${SYSTEM:?set SYSTEM=ptni|stfo|fenicr}"
export JAX_PLATFORMS=cpu
${NCE_PY:-python} -u scripts/analysis/shap_cluster_conv.py
