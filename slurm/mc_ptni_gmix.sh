#!/bin/bash
#SBATCH --job-name=mc_ptni_gmix
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --mem=24G --time=03:00:00
#SBATCH --output=logs/%x_%j.out --error=logs/%x_%j.err
# (the GPU partition is site specific: add --partition=... here or on the sbatch command line)
#
# Free energy of mixing of Pt(1-x)Ni(x): parallel tempering of a 4 M^3-atom cell at the compositions NNI_LIST (number of Ni atoms,
# colon-separated, e.g. 16:64:128:192:240), temperature ladder 300 K ... infinity (models/structures/ptni/Tgrid_gmix.txt);
# MBAR + the exact infinite-temperature entropy ln C(N, n_Ni) give G(T) (scripts/mc/gmix_ptni.py).
# env: FW=icet|conv  NNI_LIST  M (4: the 256-atom cell; 3: the 108-atom DFT cell)  SWEEPS (20000)  TGRID  CE  CKPT
# -> $NCE_RUNS/ptni_gmix/$FW[/M$M]/n$NNI
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"; source env.sh
PY=${NCE_PY:-python}; FW=${FW:?icet|conv}; NNI_LIST=${NNI_LIST:?colon-separated list of n_Ni}
[ "$FW" = nce ] && FW=conv
G=$NCE_RUNS/ptni_gmix; L=$NCE_RUNS/ptni_ladder; TG=${TGRID:-models/structures/ptni/Tgrid_gmix.txt}; NR=$(grep -vc "^#" $TG); mkdir -p $G
M=${M:-4}; N=$((4*M*M*M)); SWEEPS=${SWEEPS:-20000}
CE0=${CE:-models/ce/ptni/icet_c8_6_5.ce}; CE=$CE0; CKPT=${CKPT:-models/nce/nipt_v3_nomax_sel/model.pkl}
if [ "$M" != 4 ] && [ "$FW" = icet ]; then                  # the .ce returns the TOTAL energy of a 256-atom cell: rescale its parameters to this cell
  CE=$G/icet_c8_6_5_M$M.ce
  [ -f $CE ] || $PY - <<PYEOF
from icet import ClusterExpansion
ce = ClusterExpansion.read("$CE0")
ClusterExpansion(ce.get_cluster_space_copy(), ce.parameters * ($N / 256.0)).write("$CE"); print("rescaled CE x $N/256 ->", "$CE")
PYEOF
fi
BASE=$L/init/init_M4_x50.cif                                 # lattice constant of the start cells (the ladder's 256-atom random cell)
[ -f $BASE ] || M=4 OUTDIR=$L/init $PY -u scripts/mc/ptni_ladder_init.py || exit 1
NS=$((SWEEPS*N)); SI=$(( NS/20000 > 200 ? NS/20000 : 200 )); SI=$(( (SI/100)*100 )); CH=$(( SI*10 )); NS=$(( ((NS+CH-1)/CH)*CH )); EX=$CH
for NNI in ${NNI_LIST//:/ }; do     # colon-separated list (commas would split sbatch --export)
  if [ "$M" = 4 ]; then INIT=$G/init_n$NNI.cif; OUT=$G/$FW/n$NNI; else INIT=$G/init_M${M}_n$NNI.cif; OUT=$G/$FW/M$M/n$NNI; fi; mkdir -p $OUT
  [ -f $OUT/DONE_$SWEEPS ] && { echo "done already: $OUT ($SWEEPS sweeps)"; continue; }; rm -f $OUT/checkpoint.npz
  $PY - <<PYEOF
import numpy as np
from ase.io import read, write
a = read("$BASE")
if $M != 4:                                                                           # same lattice constant, M x M x M conventional cells
    from ase.build import bulk; a = bulk("Pt", "fcc", a=a.cell.lengths()[0] / 4, cubic=True).repeat(($M, $M, $M))
z = np.full(len(a), 78); z[np.random.default_rng($NNI).choice(len(a), $NNI, replace=False)] = 28; a.numbers = z; write("$INIT", a)
if "$FW" == "icet":                                                                   # energy convention check: CE total for this cell
    from icet import ClusterExpansion
    print("check: CE(256-atom convention) x N/256 for the start cell =", ClusterExpansion.read("$CE0").predict(a) * len(a) / 256.0, "eV")
PYEOF
  echo "GMIX $FW n_Ni=$NNI steps/replica=$NS sample=$SI chunk=$CH replicas=$NR"
  if [ "$FW" = icet ]; then
    $PY -u -m neuralce.mc.run --framework icet --ce $CE --init_cif $INIT --supercell 1 1 1 --pt --T_grid_file $TG --n_replicas $NR \
      --n_steps $NS --chunk $CH --sample_interval $SI --exchange_every $EX --energy_mode delta --log_every $((NS/5)) --out_dir $OUT --seed 100 || exit 1
  else
    $PY -u -m neuralce.mc.run --framework nce --ckpt $CKPT --init_cif $INIT --supercell 1 1 1 --model_seed 0 --pt --T_grid_file $TG --n_replicas $NR \
      --n_steps $NS --chunk $CH --exchange_every $EX --sample_interval $SI --log_every $((NS/5)) --energy_mode local --out_dir $OUT --seed 100 || exit 1
  fi
  touch $OUT/DONE_$SWEEPS
done
echo "GMIX_DONE $FW $NNI_LIST"
