#!/bin/bash
#SBATCH --job-name=nce_gates_gpu
#SBATCH --partition=eertekin-gpu
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# GPU correctness gates (values + speed/memory sections + central-entry MC smokes).
#   gate_nce_implicit    : C1 values, C2 speed/memory up to 10M atoms (MS, TILED_MAX), C3 central PT
#   gate_ce_implicit     : G1/G1b values, G3 speed/memory (MS, TILED_MAX), G4 central PT
#   gate_nce_ensemble    : --model_seed -1 ensemble energies / dE / commit on the 256-atom L1_0 NiPt cell
#   gate_nce_spin_mc     : optional, FeNiCr spin MC path; runs only when models/nce/fenicr_s2b_mm_c65_d1/model.pkl exists
#                          (the Fe-Ni-Cr model and data are not shipped in this repository)
# Prints "GATE <name> PASS|FAIL" per gate and "GATES_DONE FAIL=<n>" at the end.
# The partition name is site specific; override with sbatch --partition=... .
# Run:  sbatch gates/run_gates_gpu.sh
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"
source env.sh
PY="${NCE_PY:-python}"
FAIL=0
run_gate() {                     # run_gate <gate name> [ENV=value ...]
    local name=$1; shift
    echo "==== $name $* ===="
    if env "$@" "$PY" -u "gates/$name.py"; then
        echo "GATE $name PASS"
    else
        echo "GATE $name FAIL"; FAIL=$((FAIL + 1))
    fi
}
run_gate gate_nce_implicit
run_gate gate_ce_implicit
run_gate gate_nce_ensemble
if [ -f models/nce/fenicr_s2b_mm_c65_d1/model.pkl ]; then
    run_gate gate_nce_spin_mc
else
    echo "==== gate_nce_spin_mc skipped: models/nce/fenicr_s2b_mm_c65_d1/model.pkl is not shipped in this repository (Fe-Ni-Cr data are not public) ===="
fi
echo "GATES_DONE FAIL=$FAIL"
exit "$FAIL"
