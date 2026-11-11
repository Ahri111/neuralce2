#!/bin/bash
#SBATCH --job-name=mc_stfo
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=24G --time=02:00:00
#SBATCH --output=logs/%x_%j.out --error=logs/%x_%j.err
# (the GPU partition is site specific: add --partition=... here or on the sbatch command line)
# The STFO / Fe-Ni-Cr inputs of this job are not shipped in this repository (only the PtNi data are public); it runs when they are provided.
#
# STFO parallel tempering of one composition with either energy model.  Start = data/new_bench/A00_<COMP>.cif (lowest DFT structure
# of that composition), K = supercell repeats of the 160-site cell.  Snapshots are kept for the structure analysis (nce).
# env: FW=nce|icet  COMP=125|500|...  SWEEPS (4000)  T_MIN (300)  T_MAX (2000)  TGRID (optional T-grid file)  NREP (32)
#   nce : TAG (model tag, default stfo_dd_p10_w10_wd5)  K (2)  CKE (checkpoint every, 50)  SUFFIX  SKIP_DONE=1
#         -> $NCE_RUNS/stfo_mc/$TAG/c${COMP}_K$K$SUFFIX
#   icet: CE (.ce in the framework convention, default models/ce/stfo/icet_aug467_al2b_c6_5_4.5_4.ce)  NAME (output folder,
#         default icet_aug467_al2b)  K (1)        -> $NCE_RUNS/stfo_mc/$NAME/c${COMP}_K$K
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"; source env.sh
PY=${NCE_PY:-python}; FW=${FW:?nce|icet}; COMP=${COMP:?composition, e.g. 125 or 500}; SWEEPS=${SWEEPS:-4000}
[ "$FW" = conv ] && FW=nce
INIT=data/new_bench/A00_${COMP}.cif
if [ "$FW" = nce ]; then
  TAG=${TAG:-stfo_dd_p10_w10_wd5}; K=${K:-2}; N=$((160*K*K*K)); CH=$((N*2)); NS=$((SWEEPS*N)); NS=$(( (NS+CH-1)/CH*CH ))
  export CE_CNN_SNAPSHOT_EVERY=$(( NS/CH/40 > 1 ? NS/CH/40 : 1 )) CE_CNN_SUPERCELL_PERIODIC=1
  OUT=$NCE_RUNS/stfo_mc/${TAG}/c${COMP}_K${K}${SUFFIX:-}; mkdir -p $OUT
  [ "${SKIP_DONE:-0}" = 1 ] && [ -f $OUT/energies.npz ] && { echo "already finished: $OUT"; exit 0; }
  $PY -u -m neuralce.mc.run --framework nce --ckpt models/nce/${TAG}/model.pkl --init_cif $INIT \
    --supercell $K $K $K --model_seed 0 --pt --T_min ${T_MIN:-300} --T_max ${T_MAX:-2000} ${TGRID:+--T_grid_file $TGRID} --n_replicas ${NREP:-32} --n_steps $NS --chunk $CH --exchange_every $CH --sample_interval $((N/4)) \
    --energy_mode local --log_every $((NS/20)) --out_dir $OUT --seed 7 --checkpoint_every ${CKE:-50} || exit 1
  echo "STFO_MC_DONE $TAG c$COMP K=$K"
else
  CE=${CE:-models/ce/stfo/icet_aug467_al2b_c6_5_4.5_4.ce}; NAME=${NAME:-icet_aug467_al2b}
  export CE_SUPERCELL_PERIODIC=1
  K=${K:-1}; N=$((160*K*K*K)); NS=$((SWEEPS*N)); SI=$(( N/4 )); CH=$(( SI*8 )); NS=$(( ((NS+CH-1)/CH)*CH ))
  OUT=$NCE_RUNS/stfo_mc/${NAME}/c${COMP}_K${K}; mkdir -p $OUT
  $PY -u -m neuralce.mc.run --framework icet --ce $CE --init_cif $INIT --supercell $K $K $K --pt \
    --T_min ${T_MIN:-300} --T_max ${T_MAX:-2000} ${TGRID:+--T_grid_file $TGRID} --n_replicas ${NREP:-32} --n_steps $NS --chunk $CH --sample_interval $SI --exchange_every $CH \
    --energy_mode delta --log_every $((NS/10)) --out_dir $OUT --seed 7 --checkpoint_every 50 || exit 1
  echo "STFO_MC_ICET_DONE $NAME c$COMP K=$K"
fi
