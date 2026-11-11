#!/bin/bash
#SBATCH -J ptni_exact_pt --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=16G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# PT reference run of the final PtNi nce model on a small cell with an explicit temperature grid (python -m neuralce.mc.run).
# env: N (48|256|864)  INIT (random|L10)  SWEEPS  SEED  [TAG]  [SI = sample interval in steps, default N/4]  [SNAP = snapshot every SNAP chunks]
# grid + cells from scripts/bench/ptni_exact_ladder.py in $LAD (default $NCE_REPO/results/ptni_exact/ladder).
# Output $NCE_RUNS/ptni_exact/pt/N$N/${INIT}_s${SEED}$TAG.  A finished run has energies.npz; a cut run resumes from checkpoint.npz
# when resubmitted with the same settings.   usage: sbatch --export=ALL,N=48,INIT=random,SWEEPS=100,SEED=1 slurm/ptni_exact_pt.sh
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"; source env.sh
export OMP_NUM_THREADS=4
PY=${NCE_PY:-python}
LAD=${LAD:-$NCE_REPO/results/ptni_exact/ladder}
OUT=$NCE_RUNS/ptni_exact/pt/N$N/${INIT}_s${SEED}${TAG:-}; mkdir -p $OUT
[ -f $OUT/energies.npz ] && { echo "already finished: $OUT"; exit 0; }
SWEEPS=${SWEEPS:-100}; NS=$((SWEEPS*N))
SI=${SI:-$(( N/4 > 12 ? N/4 : 12 ))}; CH=$(( SI*${CHUNK_MULT:-200} )); NS=$(( ((NS+CH-1)/CH)*CH )); EX=$N   # one exchange round per sweep (driver does chunk/EX rounds per chunk)
[ -n "$SNAP" ] && export CE_CNN_SNAPSHOT_EVERY=$SNAP           # configuration snapshots every SNAP chunks (order parameter / SRO time series)
NREP=$(grep -vc "^#" $LAD/Tgrid_N$N.txt)
CK=$(( NS/CH/400 > 1 ? NS/CH/400 : 1 ))
echo "EXACT_PT N=$N init=$INIT seed=$SEED replicas=$NREP steps/replica=$NS ($SWEEPS sweeps) sample=$SI chunk=$CH exchange=$EX checkpoint_every=$CK chunks"
$PY -u -m neuralce.mc.run --framework nce --ckpt models/nce/nipt_v3_nomax_sel/model.pkl --init_cif $LAD/init_${INIT}_N$N.cif \
  --model_seed 0 --pt --T_grid_file $LAD/Tgrid_N$N.txt --n_steps $NS --chunk $CH --exchange_every $EX --sample_interval $SI \
  --log_every $(( NS/40 > CH ? NS/40 : CH )) --energy_mode local --out_dir $OUT --seed $SEED --checkpoint_every $CK || exit 1
echo "EXACT_PT_DONE N=$N $INIT s$SEED"
