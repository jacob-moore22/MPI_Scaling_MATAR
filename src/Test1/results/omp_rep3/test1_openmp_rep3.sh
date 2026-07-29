#!/bin/bash
#SBATCH --job-name=t1-op-r3
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --partition=morrill
#SBATCH --time=00:15:00
#SBATCH --account 038814-364640
#SBATCH --mem=128G
#SBATCH --output=morrill-omp-3.out

module purge
module load openmpi

REP=rep3
OUT=../results/omp_$REP
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
} > ../results/environment.txt

RANKS="1 2 4 8 16 32 64"

# Weak: 8M elems/rank fixed (=1M/core), global = 8M * n
for n in $RANKS; do
    srun -n $n --cpu-bind=cores --distribution=block:block ./main 8000000 50 \
        > $OUT/t1_omp_weak_n${n}.log 2>&1
done

# Strong: 512M global fixed
G=512000000
for n in $RANKS; do
    LOCAL=$(( G / n ))
    srun -n $n --cpu-bind=cores --distribution=block:block ./main $LOCAL 50 \
        > $OUT/t1_omp_strong_n${n}.log 2>&1
done
