#!/bin/bash
#SBATCH --job-name=mc_fenicr
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=64G --time=10:00:00
#SBATCH --output=logs/%x_%j.out --error=logs/%x_%j.err
# (the GPU partition is site specific: add --partition=... here or on the sbatch command line)
# The STFO / Fe-Ni-Cr inputs of this job are not shipped in this repository (only the PtNi data are public); it runs when they are provided.
#
# FeNiCr Monte Carlo of the neural CE with spins: 4000 atoms, parallel tempering over the 64 temperatures of
# models/structures/fenicr/Tgrid_dense_<COMP>.txt, DFT spin rule (spin_allowed_dft.json), spin moves p = 0.5, SW_EQ + SW_PROD sweeps,
# occupation + spin snapshots every 50 sweeps, then the Warren-Cowley SRO analysis (scripts/mc/fenicr_sro_analyze.py).
# env: COMP=F5 (Fe56Cr21Ni23: 500-atom cell x 2x2x2, species re-shuffled) | F6 (Fe42.5Cr7.5Ni50: 4000-atom cell)   default F5
#      TAG (model tag, default fenicr_s2b_mm_c65_d1)  MODEL_SEED (2)  SW_EQ (1000)  SW_PROD (4000)  MCTAG (output suffix, _dense_$COMP)
#      SEED (100)  SMP (spin move probability, 0.5)  CKE (checkpoint every, 50)  NREP (64)  EXCH (exchange every, 40000)
#      TGRID, INIT_CIF, SC ("2 2 2" / "1 1 1"), SPIN_ALLOWED_FILE, TEMPS (ignored under --pt; kept for a multi-T run without PT_ARGS)
# -> $NCE_RUNS/fenicr_select/$TAG$MCTAG/multiT  and  $NCE_RUNS/fenicr_select/$TAG$MCTAG/sro.json
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"; source env.sh
PY=${NCE_PY:-python}; COMP=${COMP:-F5}; TAG=${TAG:-fenicr_s2b_mm_c65_d1}; MODEL_SEED=${MODEL_SEED:-2}
D=$NCE_RUNS/fenicr_select; S=models/structures/fenicr; N=4000
if [ "$COMP" = F5 ]; then INIT_CIF=${INIT_CIF:-$S/init_Fe56Cr21Ni23_500.cif}; SC=${SC:-2 2 2}; else INIT_CIF=${INIT_CIF:-$S/init_Fe42.5Cr7.5Ni50_4000.cif}; SC=${SC:-1 1 1}; fi
[ -f $INIT_CIF ] || { [ "$COMP" = F5 ] && OUTDIR=$(dirname $INIT_CIF) $PY -u scripts/mc/fenicr_select_prep.py; } || exit 1
SPIN_ALLOWED_FILE=${SPIN_ALLOWED_FILE:-$S/spin_allowed_dft.json}; TGRID=${TGRID:-$S/Tgrid_dense_${COMP}.txt}
PT_ARGS=${PT_ARGS-"--pt --T_grid_file $TGRID --n_replicas ${NREP:-64} --exchange_every ${EXCH:-40000}"}
export SW_EQ=${SW_EQ:-1000}; SW_PROD=${SW_PROD:-4000}; MCTAG=${MCTAG:-_dense_$COMP}; NS=$(( (SW_EQ+SW_PROD)*N ))
CH=$(( 10*N )); SI=$(( N/4 )); export CE_CNN_SNAPSHOT_EVERY=5     # chunk = 10 sweeps, snapshot every 50 sweeps
echo "FENICR_MC $TAG$MCTAG steps/replica=$NS (eq $SW_EQ + prod $SW_PROD sweeps) chunk=$CH init=$INIT_CIF supercell=$SC"
$PY -u -m neuralce.mc.run --framework nce --ckpt models/nce/$TAG/model.pkl --init_cif $INIT_CIF --supercell $SC --model_seed $MODEL_SEED \
  --temps ${TEMPS:-600 700 800 900 1000 1100 1200 1300 1400 1500} --n_steps $NS --chunk $CH --sample_interval $SI --log_every $((NS/10)) \
  --energy_mode local --spin_init random --spin_move_prob ${SMP:-0.5} ${SPIN_ALLOWED_FILE:+--spin_allowed_file $SPIN_ALLOWED_FILE} $PT_ARGS --checkpoint_every ${CKE:-50} --out_dir $D/$TAG$MCTAG/multiT --seed ${SEED:-100} || exit 1
$PY -u scripts/mc/fenicr_sro_analyze.py $D/$TAG$MCTAG || exit 1
echo "FENICR_MC_DONE $TAG$MCTAG"
