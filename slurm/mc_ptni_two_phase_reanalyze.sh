#!/bin/bash
#SBATCH --job-name=mc_ptni_two_phase_reanalyze
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2 --mem=16G --time=00:30:00
#SBATCH --output=logs/%x_%j.out --error=logs/%x_%j.err
# (CPU job; the partition is site specific: add --partition=... here or on the sbatch command line)
#
# Re-analysis of a finished two-phase run (scripts/mc/ptni_two_phase_analyze.py) on the CPU.
# env: FW (icet)  X (x50)  M (64)  or RUN (the run dir; default $NCE_RUNS/ptni_two_phase/$FW/$X/M$M)
#      BURN (0.3)  ANCHOR_T (919.8 K: T_c of the 131k-atom cell, ANCHOR_N 131072, from which T_c(inf) is taken; set ANCHOR_T="" to
#      integrate this run alone)  CV_LAM (100)  N_CURVE (10061824: also draw the C_v curve of this larger cell from the same branch
#      data, written under M$CURVE_M; CURVE_M 136; N_CURVE=0 skips it).
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"; source env.sh
export JAX_PLATFORMS=cpu
PY=${NCE_PY:-python}; FW=${FW:-icet}; X=${X:-x50}; M=${M:-64}
[ "$FW" = nce ] && FW=conv
RUN=${RUN:-$NCE_RUNS/ptni_two_phase/$FW/$X/M$M}
export BURN=${BURN:-0.3} CV_LAM=${CV_LAM:-100} ANCHOR_T=${ANCHOR_T-919.8} ANCHOR_N=${ANCHOR_N-131072}
FILTER='anchor|T_c|branch C_v|curve drawn|Error|wrote'
echo "== $RUN"; RUN=$RUN OUT=$RUN/two_phase $PY -u scripts/mc/ptni_two_phase_analyze.py 2>&1 | grep -v Warning | grep -E "$FILTER"
N_CURVE=${N_CURVE-10061824}; CURVE_M=${CURVE_M:-136}
if [ "$N_CURVE" != 0 ] && [ -n "$N_CURVE" ]; then
  OUT2=$(dirname $RUN)/M$CURVE_M; mkdir -p $OUT2
  echo "== M$CURVE_M (N = $N_CURVE from the $(basename $RUN) branches)"
  RUN=$RUN N_CURVE=$N_CURVE OUT=$OUT2/two_phase $PY -u scripts/mc/ptni_two_phase_analyze.py 2>&1 | grep -v Warning | grep -E "$FILTER"
fi
