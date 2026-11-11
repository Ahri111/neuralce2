#!/bin/bash
#SBATCH --job-name=mc_ptni_ladder_1m
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=64G
#SBATCH --output=logs/%x_%j.out --error=logs/%x_%j.err
# (the GPU partition is site specific: add --partition=... here or on the sbatch command line)
#
# One equilibrated large-cell PtNi ladder run.  Every replica starts from the SRC run's final structure at the nearest
# temperature, tiled up to this cell (per-replica --init_npz, scripts/mc/ptni_ladder_tile_init.py + gate_ladder_start.py),
# so it starts near equilibrium for its own temperature instead of from one ordered or random configuration.
# Checkpointed: a preempted or time-limited job resumes from $OUT/checkpoint.npz with the same settings.
#   env: FW=icet|conv  X=x25|x50|x75  M (conventional cells per side)  SRC_M  SRC_TAG (default _ord)  SWEEPS (default 1000)
#        TAG (default _eq)  CKPT_EVERY (default: 1/40 of the run)  TGRID (default models/structures/ptni/Tgrid_ladder_1m_$X.txt)
#        CE, CKPT (model files)  RESUME=1: follow-up submission -- exits at once if the run already finished, otherwise
#        re-runs with the same settings (resumes from the checkpoint; never touches a finished run).
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"; source env.sh
export CE_SUPERCELL_PERIODIC=1 CE_CNN_SUPERCELL_PERIODIC=1
PY=${NCE_PY:-python}; FW=${FW:?icet|conv}; X=${X:?x25|x50|x75}; M=${M:?}; SRC_M=${SRC_M:?source M}
[ "$FW" = nce ] && FW=conv
L=$NCE_RUNS/ptni_ladder; SRC_TAG=${SRC_TAG:-_ord}; SWEEPS=${SWEEPS:-1000}; TAG=${TAG:-_eq}
declare -A REF=([x25]=ref_L12_Pt3Ni [x50]=ref_L10_NiPt [x75]=ref_L12_Ni3Pt)
CE=${CE:-models/ce/ptni/icet_c8_6_5.ce}; CKPT=${CKPT:-models/nce/nipt_v3_nomax_sel/model.pkl}
INIT=models/structures/ptni/${REF[$X]}.cif; TG=${TGRID:-models/structures/ptni/Tgrid_ladder_1m_$X.txt}
SRC=$L/$FW/$X/M${SRC_M}${SRC_TAG}; OUT=$L/$FW/$X/M${M}${TAG}; NPZ=$L/init/start_${FW}_${X}_M${SRC_M}${SRC_TAG}_to_M${M}.npz
if [ "${RESUME:-0}" = 1 ]; then
  RES=$([ "$FW" = icet ] && echo mcmc_result.npz || echo energies.npz)
  if [ -f "$OUT/$RES" ]; then echo "RESUME_SKIP $OUT already finished"; exit 0; fi
  [ -f "$OUT/checkpoint.npz" ] || echo "RESUME_NOCKPT $OUT (no checkpoint yet) -> starting over"
fi
mkdir -p $L/init
N=$((4*M*M*M)); K=$((M/4)); SRC_K=$((SRC_M/4)); TILE=$((M/SRC_M))
NS=$((SWEEPS*N)); SI=$(( NS/20000 > 200 ? NS/20000 : 200 )); SI=$(( (SI/100)*100 )); CH=$(( SI*10 )); NS=$(( ((NS+CH-1)/CH)*CH ))
NCH=$(( NS/CH )); CKPT_EVERY=${CKPT_EVERY:-$(( NCH/40 > 1 ? NCH/40 : 1 ))}; EX=$(( 3*N/2 ))
echo "LADDER1M $FW $X M=$M N=$N from M${SRC_M}${SRC_TAG} (tile ${TILE}^3) sweeps=$SWEEPS steps/replica=$NS chunk=$CH sample=$SI chunks=$NCH ckpt_every=$CKPT_EVERY temps=$(grep -vc '^#' $TG)"
if [ ! -f $NPZ ]; then
  if [ "$FW" = icet ]; then
    $PY -u scripts/mc/ptni_ladder_tile_init.py --fw icet --src_dir $SRC --ce $CE --src_init_cif $INIT --src_supercell $SRC_K \
       --tile $TILE $TILE $TILE --T_grid_file $TG --out $NPZ || exit 1
  else
    $PY -u scripts/mc/ptni_ladder_tile_init.py --fw conv --src_dir $SRC --ckpt $CKPT --src_init_cif $INIT --src_supercell $SRC_K \
       --tile $TILE $TILE $TILE --T_grid_file $TG --out $NPZ || exit 1
  fi
  # gate the freshly built starts at the size this run will actually use: every replica's starting energy per atom
  # must equal its source structure's (a wrong site mapping between the two cells shows up as ~1e-3 eV/atom)
  if [ "$FW" = icet ]; then GA="--fw icet --ce $CE"; else GA="--fw conv --ckpt $CKPT"; fi
  $PY -u scripts/mc/gate_ladder_start.py $GA --npz $NPZ --init_cif $INIT --supercell $K || { rm -f $NPZ; exit 1; }
fi
if [ "$FW" = icet ]; then
  $PY -u -m neuralce.mc.run --framework icet --ce $CE --init_cif $INIT --supercell $K $K $K --pt \
    --T_grid_file $TG --init_npz $NPZ --n_steps $NS --chunk $CH --sample_interval $SI --exchange_every $EX \
    --energy_mode delta --checkpoint_every $CKPT_EVERY --log_every $((NS/20)) --out_dir $OUT --seed 100 || exit 1
else
  $PY -u -m neuralce.mc.run --framework nce --ckpt $CKPT --init_cif $INIT --supercell $K $K $K --model_seed 0 --pt \
    --T_grid_file $TG --init_npz $NPZ --n_steps $NS --chunk $CH --sample_interval $SI --exchange_every $EX \
    --energy_mode local --checkpoint_every $CKPT_EVERY --log_every $((NS/20)) --out_dir $OUT --seed 100 || exit 1
fi
# convergence readout of the finished ladder (split-half Tc, dE/dT Tc, drift): a "NOT EQUILIBRATED" line here means
# the run needs more sweeps, not that the job failed -- so it never changes the job's exit status.
$PY -u scripts/mc/ptni_ladder_converge.py $OUT --json $OUT/converge.json || true
echo "LADDER1M_DONE $FW $X M=$M"
