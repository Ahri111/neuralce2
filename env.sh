# source this file from the repository root before running anything:  source env.sh
# Every setting here is optional except JAX_DEFAULT_MATMUL_PRECISION (see README, "Numerics").
export NCE_REPO="${NCE_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export NCE_RUNS="${NCE_RUNS:-$NCE_REPO/runs}"          # large Monte Carlo output (point this at scratch)
export NCE_PY="${NCE_PY:-python}"                       # interpreter used by the SLURM launchers
export JAX_DEFAULT_MATMUL_PRECISION=highest             # TF32 matmuls on Ampere/Hopper corrupt energies by 0.2-0.3 eV
export XLA_PYTHON_CLIENT_PREALLOCATE=false              # share the GPU with the CPU-side tables
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "$NCE_RUNS" "$NCE_REPO/logs"
