#!/bin/bash
#SBATCH --job-name=t2-op-r1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --partition=morrill
#SBATCH --time=20:00:00
#SBATCH --account 038814-364640
#SBATCH --mem=128G
#SBATCH --output=morrill-t2-omp-1.out

module purge
module load openmpi

REP=rep1
OUT=../results/$REP
mkdir -p $OUT

export OMP_NUM_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores

{
  echo "== module list =="
  module list 2>&1
  echo "== srun --version =="
  srun --version
  echo "== OMP_NUM_THREADS=$OMP_NUM_THREADS OMP_PROC_BIND=$OMP_PROC_BIND OMP_PLACES=$OMP_PLACES =="
  echo ""
  echo "== KNOWN LIMITATION =="
  echo "Weak-scaling n=64 (256^3 = 16.7M elements) point excluded from this study."
  echo "Same global mesh as the serial n=512 point that OOM'd on rank 0's serial"
  echo "build at 128 GB node memory (see §5.1 environment.txt for the confirming"
  echo "memcheck). Mesh size, not rank/thread count, drives rank-0 memory use, so"
  echo "the same 256^3 build is expected to fail here regardless of only 8 ranks"
  echo "sharing the node. Weak-scaling trend is reported using n=2..32 (5 of 6 points)."
} > ../results/environment.txt

RANKS_STRONG="2 4 8 16 32 64"
RANKS_WEAK="2 4 8 16 32"   # 64 dropped — see environment.txt

# ---- Strong scaling: Mesh A (small, 100^3 fixed) ----
for n in $RANKS_STRONG; do
    srun -n $n --cpu-bind=cores --distribution=block:block ./main 100 100 100 --comms 20 --smooth 3 \
        > $OUT/t2_omp_strongA_n${n}.log 2>&1
    mv test2_results_mesh100x100x100_np${n}_comms20_smooth3.csv $OUT/
done

# ---- Strong scaling: Mesh B (large, 200^3 fixed) ----
for n in $RANKS_STRONG; do
    srun -n $n --cpu-bind=cores --distribution=block:block ./main 200 200 200 --comms 20 --smooth 3 \
        > $OUT/t2_omp_strongB_n${n}.log 2>&1
    mv test2_results_mesh200x200x200_np${n}_comms20_smooth3.csv $OUT/
done

# ---- Weak scaling: 262,144 elems/rank (32,768/core, matches §5.1) ----
# n=64 (256^3) intentionally excluded — see environment.txt
declare -A dims=(
    [2]="128 64 64"
    [4]="128 128 64"
    [8]="128 128 128"
    [16]="256 128 128"
    [32]="256 256 128"
)

for n in $RANKS_WEAK; do
    read -r nx ny nz <<< "${dims[$n]}"
    srun -n $n --cpu-bind=cores --distribution=block:block ./main $nx $ny $nz --comms 20 --smooth 3 \
        > $OUT/t2_omp_weak_n${n}.log 2>&1
    mv test2_results_mesh${nx}x${ny}x${nz}_np${n}_comms20_smooth3.csv $OUT/
done

echo "Run complete. n=64 weak-scaling point intentionally skipped (see environment.txt)." \
    > $OUT/t2_run_summary.log
