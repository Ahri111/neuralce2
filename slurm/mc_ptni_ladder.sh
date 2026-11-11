#!/bin/bash
#SBATCH --job-name=mc_ptni_ladder
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1
#SBATCH --output=logs/%x_%j.out --error=logs/%x_%j.err
# (the GPU partition is site specific: add --partition=... here or on the sbatch command line)
#
# One PtNi size-ladder case.  env: FW=icet|conv  M=4|8|16|32|64|100|136 (conventional cells per side, N = 4 M^3)  X=x25|x50|x75
# -> $NCE_RUNS/ptni_ladder/$FW/$X/M$M[$TAG].  The 256-atom init cif is tiled in memory (--supercell) and re-shuffled; implicit
# (template-only) tables: GPU memory = state only.  SWEEPS (default 200) sets steps/replica = SWEEPS * N.
# INIT=random (default): the 256-atom random cell (scripts/mc/ptni_ladder_init.py, M=4; generated if missing) tiled + re-shuffled.
# INIT=ordered: the ideal ordered reference models/structures/ptni/ref_*.cif (L1_2 Pt3Ni / L1_0 NiPt / L1_2 Ni3Pt) tiled with its
# periodic images kept -> cold start (large cells never order from a random start within the run; exchange acceptance is ~0 there).
# Other env: TAG (output suffix), T_MIN (300), T_MAX (1400), NREP (32), SEED (100), CE, CKPT, INIT_NPZ, SKIP_DONE=1.
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"; source env.sh
PY=${NCE_PY:-python}; FW=${FW:?icet|conv}; M=${M:?conventional cells per side}; X=${X:?x25|x50|x75}
[ "$FW" = nce ] && FW=conv
L=$NCE_RUNS/ptni_ladder; OUT=$L/$FW/$X/M$M${TAG:-}
CE=${CE:-models/ce/ptni/icet_c8_6_5.ce}; CKPT=${CKPT:-models/nce/nipt_v3_nomax_sel/model.pkl}
declare -A REF=([x25]=ref_L12_Pt3Ni [x50]=ref_L10_NiPt [x75]=ref_L12_Ni3Pt)
if [ "${INIT:-random}" = ordered ]; then
  INITCIF=models/structures/ptni/${REF[$X]}.cif; export CE_SUPERCELL_PERIODIC=1 CE_CNN_SUPERCELL_PERIODIC=1
else
  INITCIF=$L/init/init_M4_${X}.cif; [ -f $INITCIF ] || M=4 OUTDIR=$L/init $PY -u scripts/mc/ptni_ladder_init.py || exit 1
fi
INIT=$INITCIF
N=$((4*M*M*M)); K=$((M/4)); SWEEPS=${SWEEPS:-200}; NS=$((SWEEPS*N))
SI=$(( NS/20000 > 200 ? NS/20000 : 200 )); SI=$(( (SI/100)*100 )); CH=$(( SI*10 )); NS=$(( ((NS+CH-1)/CH)*CH ))   # ~20k samples; chunk = 10 samples; steps a multiple of chunk
EX=$(( (3*N/2 + CH - 1) / CH * CH ))     # exchange every ~1.5 sweeps, rounded up to whole chunks (the driver needs chunk | exchange)
[ "${SKIP_DONE:-0}" = 1 ] && { [ -f $OUT/mcmc_result.npz ] || [ -f $OUT/energies.npz ]; } && { echo "already finished: $OUT"; exit 0; }
echo "LADDER $FW $X M=$M N=$N init=${INIT} supercell=$K steps/replica=$NS ($SWEEPS sweeps) sample=$SI chunk=$CH exchange=$EX"
if [ "$FW" = icet ]; then
  $PY -u -m neuralce.mc.run --framework icet --ce $CE --init_cif $INIT --supercell $K $K $K --pt \
    --T_min ${T_MIN:-300} --T_max ${T_MAX:-1400} --n_replicas ${NREP:-32} --n_steps $NS --chunk $CH --sample_interval $SI --exchange_every $EX \
    --energy_mode delta --log_every $((NS/20)) --out_dir $OUT --seed ${SEED:-100} --checkpoint_every $(( NS/CH/40 > 1 ? NS/CH/40 : 1 )) ${INIT_NPZ:+--init_npz $INIT_NPZ} || exit 1
else
  $PY -u -m neuralce.mc.run --framework nce --ckpt $CKPT --init_cif $INIT --supercell $K $K $K --model_seed 0 --pt \
    --T_min ${T_MIN:-300} --T_max ${T_MAX:-1400} --n_replicas ${NREP:-32} --n_steps $NS --chunk $CH --exchange_every $EX --sample_interval $SI \
    --log_every $((NS/20)) --energy_mode local --out_dir $OUT --seed ${SEED:-100} --checkpoint_every $(( NS/CH/40 > 1 ? NS/CH/40 : 1 )) || exit 1
fi
echo "LADDER_DONE $FW $X M=$M"
