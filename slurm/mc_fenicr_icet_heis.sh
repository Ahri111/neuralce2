#!/bin/bash
#SBATCH --job-name=mc_fenicr_icet_heis
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --time=03:59:00 --mem=64G
#SBATCH --output=logs/%x_%j.out --error=logs/%x_%j.err
# (the GPU partition is site specific: add --partition=... here or on the sbatch command line)
# The STFO / Fe-Ni-Cr inputs of this job are not shipped in this repository (only the PtNi data are public); it runs when they are provided.
#
# FeNiCr Fe42.5Cr7.5Ni50 MC with the benchmark icet + Heisenberg-pair model in its depth-0 neural-CE form
# (models/nce/fenicr_icet_heis_bench), SAME protocol as the neural-CE run fenicr_s2b_mm_c65_d1_dense_F6: 4000 atoms,
# 64-temperature PT (Tgrid_dense_F6.txt), DFT spin rule, 1000 + 4000 sweeps, snapshots every 50 sweeps, then the SRO analysis.
# Chained submissions resume from the checkpoint; a finished run is never restarted (sro.json present -> exit; MC done but no
# sro.json -> analysis only).  Everything else is slurm/mc_fenicr.sh.
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"; source env.sh
PY=${NCE_PY:-python}
export TAG=fenicr_icet_heis_bench COMP=F6 MODEL_SEED=0 SW_EQ=1000 SW_PROD=4000 MCTAG=_dense_F6
D=$NCE_RUNS/fenicr_select/${TAG}${MCTAG}
[ -f $D/sro.json ] && { echo "done already: $D"; exit 0; }
if [ -f $D/multiT/energies.npz ]; then $PY -u scripts/mc/fenicr_sro_analyze.py $D; exit $?; fi
exec bash slurm/mc_fenicr.sh
