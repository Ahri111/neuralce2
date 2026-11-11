#!/bin/bash
#SBATCH --job-name=nce_gates_cpu
#SBATCH --partition=eertekin-amd
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# CPU-capable correctness gates (value sections only; JAX on the CPU backend).
#   gate_nce_implicit   SECTION=values : implicit == tiled neural-CE tables (energy, local dE, caches)
#   gate_ce_implicit    SKIP_G3=1      : implicit == tiled icet tables + central-entry PT smoke
#   gate_nce_layer0                    : layer 0 == icet cluster vector / local features, local dE == full dE
#   gate_nce_final_models              : MC path == training path for the shipped PtNi model + central-entry MC smoke
# Prints "GATE <name> PASS|FAIL" per gate and "GATES_DONE FAIL=<n>" at the end.
# The partition name is site specific; override with sbatch --partition=... .
# Run:  sbatch gates/run_gates_cpu.sh        (or  bash gates/run_gates_cpu.sh  from the repository root)
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"
source env.sh
export JAX_PLATFORMS=cpu
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
run_gate gate_nce_implicit SECTION=values
run_gate gate_ce_implicit SKIP_G3=1
run_gate gate_nce_layer0
run_gate gate_nce_final_models
echo "GATES_DONE FAIL=$FAIL"
exit "$FAIL"
