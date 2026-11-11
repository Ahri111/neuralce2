#!/bin/bash
#SBATCH -J bench_gnn_mc
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=48G -t 02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# One-MC-step (= full forward pass) cost of CGCNN / ALIGNN / Disorder_GNN vs number of atoms (scripts/bench/bench_gnn_mc_step.py).
# Needs the `gnn` extra and the submodule external/Configurational-Disorder.  Add -p <gpu partition> (site specific).
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"; source env.sh
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$NCE_REPO/results/bench_stage"
${NCE_PY:-python} -u scripts/bench/bench_gnn_mc_step.py "$@"
