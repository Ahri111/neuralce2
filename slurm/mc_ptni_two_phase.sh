#!/bin/bash
#SBATCH --job-name=mc_ptni_two_phase
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1
#SBATCH --output=logs/%x_%j.out --error=logs/%x_%j.err
# (the GPU partition is site specific: add --partition=... here or on the sbatch command line)
#
# Two-phase C_v run of one PtNi composition (scripts/mc/ptni_two_phase_init.py): ordered-start and random-start replicas on their own
# temperatures, NO replica exchange (icet: interval longer than the run; nce: 0), checkpointed.
# env: FW=icet|conv (default icet)  X=x25|x50|x75  M (conventional cells per side, N = 4 M^3)  TC0 (K, transition estimate from PT)
#      SWEEPS (1000)  [WINDOW_ONLY=1] [TAG] [SEED (100)] [CK_DIV (40, nce checkpoint divisor)] [SKIP_DONE=1]
#      plus the grid knobs of the init script (W_UP, W_DN, DENSE, DT, DT_COARSE, N_HI).   -> $NCE_RUNS/ptni_two_phase/$FW/$X/M$M$TAG
# Analysis afterwards: RUN=<that dir> python scripts/mc/ptni_two_phase_analyze.py (or slurm/mc_ptni_two_phase_reanalyze.sh).
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"; source env.sh
export CE_SUPERCELL_PERIODIC=1 CE_CNN_SUPERCELL_PERIODIC=1
PY=${NCE_PY:-python}; FW=${FW:-icet}; X=${X:?x25|x50|x75}; M=${M:?}; TC0=${TC0:?transition estimate in K}
[ "$FW" = nce ] && FW=conv
OUT=$NCE_RUNS/ptni_two_phase/$FW/$X/M$M${TAG:-}; mkdir -p $OUT
CE=${CE:-models/ce/ptni/icet_c8_6_5.ce}; CKPT=${CKPT:-models/nce/nipt_v3_nomax_sel/model.pkl}
declare -A REF=([x25]=ref_L12_Pt3Ni [x50]=ref_L10_NiPt [x75]=ref_L12_Ni3Pt)
INIT=models/structures/ptni/${REF[$X]}.cif
N=$((4*M*M*M)); K=$((M/4)); SWEEPS=${SWEEPS:-1000}; NS=$((SWEEPS*N))
SI=$(( N/4 > 100 ? N/4 : 100 )); SI=$(( (SI/100)*100 )); CH=$(( SI*10 )); NS=$(( ((NS+CH-1)/CH)*CH )); EX=$(( NS + CH ))   # 4 samples per sweep; EX > NS: never exchange
[ "${SKIP_DONE:-0}" = 1 ] && [ -f $OUT/TWO_PHASE_DONE ] && { echo "already finished: $OUT"; exit 0; }
[ -f $OUT/init.npz ] || FW=$FW $PY -u scripts/mc/ptni_two_phase_init.py $X $K $TC0 $OUT || exit 1
NR=$(grep -vc "^#" $OUT/Tgrid.txt)
echo "TWO_PHASE $FW $X M=$M N=$N replicas=$NR steps/replica=$NS ($SWEEPS sweeps) sample=$SI chunk=$CH exchange=$EX (none)"
if [ "$FW" = icet ]; then
  $PY -u -m neuralce.mc.run --framework icet --ce $CE --init_cif $INIT --supercell $K $K $K --pt \
    --T_grid_file $OUT/Tgrid.txt --init_npz $OUT/init.npz --n_steps $NS --chunk $CH --sample_interval $SI --exchange_every $EX \
    --energy_mode delta --log_every $((NS/20)) --out_dir $OUT --seed ${SEED:-100} --checkpoint_every $(( NS/CH/40 > 1 ? NS/CH/40 : 1 )) || exit 1
else
  $PY -u -m neuralce.mc.run --framework nce --ckpt $CKPT --init_cif $INIT --supercell $K $K $K --model_seed 0 --pt \
    --T_grid_file $OUT/Tgrid.txt --init_npz $OUT/init.npz --n_steps $NS --chunk $CH --sample_interval $SI --exchange_every 0 \
    --log_every $((NS/20)) --energy_mode local --out_dir $OUT --seed ${SEED:-100} --checkpoint_every $(( NS/CH/${CK_DIV:-40} > 1 ? NS/CH/${CK_DIV:-40} : 1 )) || exit 1
fi
touch $OUT/TWO_PHASE_DONE; echo "TWO_PHASE_DONE $FW $X M=$M"
