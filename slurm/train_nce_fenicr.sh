#!/bin/bash
#SBATCH --job-name=train_nce_fenicr
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:H100:1 --partition=secondary --time=03:59:00 --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# The STFO / Fe-Ni-Cr inputs of this job are not shipped in this repository (only the PtNi data are public); it runs when they are provided.
# usage: STAGE=train|mc sbatch slurm/train_nce_fenicr.sh <tag> <SPIN 2|3> <AGG mean|mean+max> <CE_CUT e.g. 6,5> [FINAL_DEPTH]
# train: FINAL model (cluster selection on full data) -> models/nce/<tag>/model.pkl
# mc   : Fe56Cr21Ni23, 4000 atoms (500-atom cif x 2x2x2), 10 temperatures 600..1500 K as replicas (multi-T, no
#        exchange), spin moves p=0.5, EQ+PROD sweeps, snapshots every 50 sweeps for the Warren-Cowley SRO parameters.
#        Output $NCE_RUNS/fenicr_select/<tag><MCTAG>/multiT.  Optional env: INIT_CIF, SC ("2_2_2"), TEMPS, SW_EQ, SW_PROD,
#        MODEL_SEED, SEED, SMP, SPIN_ALLOWED_FILE, PT_ARGS, CKE, MCTAG.
# The partition / GPU type are site specific; override with sbatch --partition=... --gres=... .
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"
source env.sh
export OMP_NUM_THREADS=8
PY=${NCE_PY:-python}
TAG=$1; SPIN=$2; AGG=$3; CUT=$4; DEPTH=${5:-}; CKPT=models/nce/$TAG/model.pkl
if [ "${STAGE:-train}" = train ]; then
  [ -f $CKPT ] && { echo "model exists $CKPT"; exit 0; }
  FINAL=1 SYSTEM=fenicr SPIN=$SPIN SELECT=1 AGG=$AGG CE_CUT=$CUT FINAL_PATIENCE=3000 OUT_TAG=$TAG FINAL_DEPTH=$DEPTH $PY -u scripts/train/train_nce.py || exit 1
  echo "FENICR_TRAIN_DONE $TAG"; exit 0
fi
# screening packs (job name pack_*, 3-4 MCs sharing one GPU) default to 600+1400 sweeps so the slowest model fits the 4 h wall
if [[ ${SLURM_JOB_NAME:-} == pack_* ]]; then SW_EQ=${SW_EQ:-600}; SW_PROD=${SW_PROD:-1400}; fi
SC=${SC//_/ }; export SW_EQ; D=${NCE_RUNS:-runs}/fenicr_select; N=4000; SW_EQ=${SW_EQ:-1000}; SW_PROD=${SW_PROD:-4000}; NS=$(( (SW_EQ+SW_PROD)*N )); MCTAG=${MCTAG:-}
CH=$(( 10*N )); SI=$(( N/4 )); SNAP=5    # chunk = 10 sweeps, snapshot every 50 sweeps
export CE_CNN_SNAPSHOT_EVERY=$SNAP
# initial 500-atom cell: models/structures/fenicr/init_Fe56Cr21Ni23_500.cif (or INIT_CIF)
INIT=${INIT_CIF:-models/structures/fenicr/init_Fe56Cr21Ni23_500.cif}
echo "FENICR_MC $TAG steps/replica=$NS (eq $SW_EQ + prod $SW_PROD sweeps) chunk=$CH"
$PY -m neuralce.mc.run --framework nce --ckpt $CKPT --init_cif $INIT --supercell ${SC:-2 2 2} --model_seed ${MODEL_SEED:-0} \
  --temps ${TEMPS:-600 700 800 900 1000 1100 1200 1300 1400 1500} --n_steps $NS --chunk $CH --sample_interval $SI --log_every $((NS/10)) \
  --energy_mode local --spin_init random --spin_move_prob ${SMP:-0.5} ${SPIN_ALLOWED_FILE:+--spin_allowed_file $SPIN_ALLOWED_FILE} ${PT_ARGS:-} --checkpoint_every ${CKE:-50} --out_dir $D/$TAG$MCTAG/multiT --seed ${SEED:-100} || exit 1
echo "FENICR_MC_DONE $TAG"
