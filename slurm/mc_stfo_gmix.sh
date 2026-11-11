#!/bin/bash
#SBATCH --job-name=mc_stfo_gmix
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=24G --time=04:00:00
#SBATCH --output=logs/%x_%j.out --error=logs/%x_%j.err
# (the GPU partition is site specific: add --partition=... here or on the sbatch command line)
# The STFO / Fe-Ni-Cr inputs of this job are not shipped in this repository (only the PtNi data are public); it runs when they are provided.
#
# Free energy of mixing of SrTi(1-x)Fe(x)O(3-x/2): parallel tempering of the K^3 x 160-site cell at the compositions COMPS (colon list
# of 125:250:375:500:750), temperature ladder 300 K ... infinity (models/structures/stfo/Tgrid_gmix.txt); MBAR + the exact
# infinite-temperature entropy ln[C(32K^3, n_Fe) C(96K^3, n_V)] give G(T) (scripts/mc/gmix_stfo.py).
# env: FW=icet (CE = .ce in the framework convention, default models/ce/stfo/icet_aug467_al2b_c6_5_4.5_4.ce; NAME = output folder,
#      default icet_aug467_al2b) | conv (TAG = nce model tag, default stfo_dd_p10_w10_wd5_aug467_al1a_w0; NAME defaults to TAG)
#      COMPS, K (1), SWEEPS (4000), TGRID.  Start = data/new_bench/A00_<COMP>.cif (ordered ground state; PT melts it).
# -> $NCE_RUNS/stfo_gmix/$FW/$NAME/c${COMP}_K$K
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"; source env.sh
export CE_SUPERCELL_PERIODIC=1 CE_CNN_SUPERCELL_PERIODIC=1
PY=${NCE_PY:-python}; FW=${FW:?icet|conv}; COMPS=${COMPS:?colon-separated compositions}
[ "$FW" = nce ] && FW=conv
G=$NCE_RUNS/stfo_gmix; TG=${TGRID:-models/structures/stfo/Tgrid_gmix.txt}; NR=$(grep -vc "^#" $TG)
K=${K:-1}; N=$((160*K*K*K)); SWEEPS=${SWEEPS:-4000}
if [ "$FW" = icet ]; then CE=${CE:-models/ce/stfo/icet_aug467_al2b_c6_5_4.5_4.ce}; NAME=${NAME:-icet_aug467_al2b}
else TAG=${TAG:-stfo_dd_p10_w10_wd5_aug467_al1a_w0}; NAME=${NAME:-$TAG}; fi
for COMP in ${COMPS//:/ }; do
  OUT=$G/$FW/$NAME/c${COMP}_K$K; mkdir -p $OUT
  [ -f $OUT/DONE_$SWEEPS ] && { echo "done already: $OUT ($SWEEPS sweeps)"; continue; }
  echo "GMIX $FW $NAME x=$COMP K=$K N=$N sweeps=$SWEEPS replicas=$NR"
  if [ "$FW" = icet ]; then
    NS=$((SWEEPS*N)); SI=$((N/4)); CH=$((SI*8)); NS=$(( ((NS+CH-1)/CH)*CH ))
    $PY -u -m neuralce.mc.run --framework icet --ce $CE --init_cif data/new_bench/A00_${COMP}.cif --supercell $K $K $K --pt --T_grid_file $TG --n_replicas $NR \
      --n_steps $NS --chunk $CH --sample_interval $SI --exchange_every $CH --energy_mode delta --log_every $((NS/10)) --out_dir $OUT --seed 7 --checkpoint_every 50 || exit 1
  else
    CH=$((N*2)); NS=$((SWEEPS*N)); NS=$(( (NS+CH-1)/CH*CH )); export CE_CNN_SNAPSHOT_EVERY=$(( NS/CH/40 > 1 ? NS/CH/40 : 1 ))
    $PY -u -m neuralce.mc.run --framework nce --ckpt models/nce/${TAG}/model.pkl --init_cif data/new_bench/A00_${COMP}.cif --supercell $K $K $K --model_seed 0 --pt --T_grid_file $TG --n_replicas $NR \
      --n_steps $NS --chunk $CH --exchange_every $CH --sample_interval $((N/4)) --energy_mode local --log_every $((NS/20)) --out_dir $OUT --seed 7 --checkpoint_every 50 || exit 1
  fi
  touch $OUT/DONE_$SWEEPS
done
echo "GMIX_DONE $FW $NAME $COMPS"
