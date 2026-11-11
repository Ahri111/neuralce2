#!/bin/bash
#SBATCH -J bench_h100 --nodes=1 --ntasks=1 --cpus-per-task=8 --gres=gpu:H100:1 --mem=120G -t 03:59:00
#SBATCH --output=logs/%x_%j.out --error=logs/%x_%j.err
# Speed / memory matrix through the central entry point (production PT, python -m neuralce.mc.run), PtNi models, one H100.
# steps/s = (S_long - S_short) / (t_long - t_short): two otherwise identical runs, so setup + compile cancel out.
# Memory = max nvidia-smi memory.used during the long run (no preallocation).
# Temperature grids: R replicas on linspace(300, 1400, R) K (R = 1: 900 K).  Output: $NCE_RUNS/bench_h100/results_<job>.csv
cd "${NCE_REPO:-$SLURM_SUBMIT_DIR}"; source env.sh
export OMP_NUM_THREADS=8 CE_SUPERCELL_PERIODIC=1 CE_CNN_SUPERCELL_PERIODIC=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=${NCE_PY:-python}; B=$NCE_RUNS/bench_h100; mkdir -p $B/runs $B/tgrid
CE=models/ce/ptni/icet_c8_6_5.ce; CKPT=models/nce/nipt_v3_nomax_sel/model.pkl; INIT=models/structures/ptni/ref_L10_NiPt.cif
for R in 1 8 32 128; do
  [ -f $B/tgrid/T$R.txt ] || $PY -c "import numpy as np,sys; R=int(sys.argv[1]); T=[900.0] if R==1 else np.linspace(300,1400,R); np.savetxt(sys.argv[2], T, fmt='%.1f')" $R $B/tgrid/T$R.txt
done
OUT=$B/results_${SLURM_JOB_ID}.csv; echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)" | tee $B/gpu_${SLURM_JOB_ID}.txt
echo "fw,K,N,R,chunk,steps_short,steps_long,t_short_s,t_long_s,steps_per_s,mem_MB,ok" > $OUT
run_one () {  # fw K R chunk nchunks tag -> prints wall seconds; memory to $B/runs/$tag.mem
  local fw=$1 K=$2 R=$3 CH=$4 NC=$5 tag=$6; local NS=$((CH*NC)) SI=$((CH/10)) EX=$((4*CH)) d=$B/runs/$tag; rm -rf $d
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > $B/runs/$tag.mem 2>/dev/null & local smi=$!
  local t0=$(date +%s.%N)
  if [ $fw = icet ]; then
    $PY -u -m neuralce.mc.run --framework icet --ce $CE --init_cif $INIT --supercell $K $K $K --pt --T_grid_file $B/tgrid/T$R.txt \
      --n_steps $NS --chunk $CH --sample_interval $SI --exchange_every $EX --energy_mode delta --log_every $NS --out_dir $d --seed 1 > $B/runs/$tag.log 2>&1
  else
    $PY -u -m neuralce.mc.run --framework nce --ckpt $CKPT --init_cif $INIT --supercell $K $K $K --model_seed 0 --pt --T_grid_file $B/tgrid/T$R.txt \
      --n_steps $NS --chunk $CH --sample_interval $SI --exchange_every $EX --energy_mode local --log_every $NS --out_dir $d --seed 1 > $B/runs/$tag.log 2>&1
  fi
  local rc=$?; local t1=$(date +%s.%N); kill $smi 2>/dev/null; wait $smi 2>/dev/null
  echo "$(echo "$t1 - $t0" | bc) $rc"
}
bench () {  # fw K R chunk
  local fw=$1 K=$2 R=$3 CH=$4; local N=$((256*K*K*K)); local NC1=4 NC2=14
  read t1 rc1 <<< "$(run_one $fw $K $R $CH $NC1 ${fw}_K${K}_R${R}_short)"
  read t2 rc2 <<< "$(run_one $fw $K $R $CH $NC2 ${fw}_K${K}_R${R}_long)"
  local S1=$((R*CH*NC1)) S2=$((R*CH*NC2)); local sps=$(echo "($S2 - $S1) / ($t2 - $t1)" | bc -l 2>/dev/null)
  local mem=$(sort -n $B/runs/${fw}_K${K}_R${R}_long.mem 2>/dev/null | tail -1); local ok=$([ "$rc1" = 0 ] && [ "$rc2" = 0 ] && echo 1 || echo 0)
  printf "%s,%d,%d,%d,%d,%d,%d,%.1f,%.1f,%.0f,%s,%s\n" $fw $K $N $R $CH $S1 $S2 $t1 $t2 ${sps:-0} "${mem:-NA}" $ok | tee -a $OUT
}
# size scans at 32 replicas (init cif = 256 atoms, so K=2 -> 2,048 ... K=16 -> 1,048,576, K=34 -> 10.06M, K=64 -> 67M)
for K in 2 4 8 16 34; do bench icet $K 32 20000; done
bench icet 64 32 20000                         # 67M atoms (memory headroom demo)
for K in 2 4 8; do bench nce $K 32 20000; done
bench nce 16 32 4000                           # 1M atoms
bench nce 34 8 1000                            # 10M atoms, 8 replicas (32 replicas exceed 80 GB)
# replica (temperature) scaling at 131k atoms (K=8)
for R in 1 8 128; do bench icet 8 $R 20000; done
for R in 1 8 128; do bench nce 8 $R 10000; done
echo "BENCH_H100_DONE $OUT"
