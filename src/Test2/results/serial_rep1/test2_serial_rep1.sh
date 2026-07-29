#!/bin/bash
#SBATCH --job-name=t2-sr-r1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --partition=morrill
#SBATCH --time=20:00:00
#SBATCH --account 038814-364640
#SBATCH --mem=128G
#SBATCH --output=morrill-t2-serial-1.out

module purge
module load openmpi

REP=rep1
OUT=../results/$REP
mkdir -p $OUT

{
  echo "== module list =="
  module list 2>&1
  echo "== srun --version =="
  srun --version
  echo ""
  echo "== KNOWN LIMITATION =="
  echo "Weak-scaling n=512 (256^3 = 16.7M elements) point excluded from this study."
  echo "Rank-0 serial mesh build exceeds available node memory (128 GB) when"
  echo "sharing a node with 63 co-located ranks. Confirmed via isolated memcheck:"
  echo "  - n=512, 100^3 mesh: succeeded"
  echo "  - n=512, 256^3 mesh: failed (OOM)"
  echo "Weak-scaling trend is reported using n=2..256 (8 of 9 points)."
} > ../results/environment.txt

RANKS_ALL="2 4 8 16 32 64 128 256 512"
RANKS_STRONG="$RANKS_ALL"
RANKS_WEAK="2 4 8 16 32 64 128 256"   # 512 dropped — see environment.txt

# ---- Strong scaling: Mesh A (small, 100^3 fixed) ----
for n in $RANKS_STRONG; do
    srun -n $n --cpu-bind=cores ./main 100 100 100 --comms 20 --smooth 3 \
        > $OUT/t2_ser_strongA_n${n}.log 2>&1
    mv test2_results_mesh100x100x100_np${n}_comms20_smooth3.csv $OUT/
done

# ---- Strong scaling: Mesh B (large, 200^3 fixed) ----
for n in $RANKS_STRONG; do
    srun -n $n --cpu-bind=cores ./main 200 200 200 --comms 20 --smooth 3 \
        > $OUT/t2_ser_strongB_n${n}.log 2>&1
    mv test2_results_mesh200x200x200_np${n}_comms20_smooth3.csv $OUT/
done

# ---- Weak scaling: 32,768 elems/rank (32^3/core) ----
# n=512 (256^3) intentionally excluded — exceeds 128GB node memory
# during rank-0's serial mesh build. See environment.txt for details.
declare -A dims=(
    [2]="64 32 32"
    [4]="64 64 32"
    [8]="64 64 64"
    [16]="128 64 64"
    [32]="128 128 64"
    [64]="128 128 128"
    [128]="256 128 128"
    [256]="256 256 128"
)

for n in $RANKS_WEAK; do
    read -r nx ny nz <<< "${dims[$n]}"
    srun -n $n --cpu-bind=cores ./main $nx $ny $nz --comms 20 --smooth 3 \
        > $OUT/t2_ser_weak_n${n}.log 2>&1
    mv test2_results_mesh${nx}x${ny}x${nz}_np${n}_comms20_smooth3.csv $OUT/
done

echo "Run complete. n=512 weak-scaling point intentionally skipped (see environment.txt)." \
    >> $OUT/../t2_run_summary_${REP}.log 2>/dev/null || \
    echo "Run complete. n=512 weak-scaling point intentionally skipped (see environment.txt)." \
    > $OUT/t2_run_summary.log
