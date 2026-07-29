#!/bin/bash
#SBATCH --job-name=t1-ser-r1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --partition=morrill
#SBATCH --time=00:10:00
#SBATCH --account 038814-364640
#SBATCH --mem=128G

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
} > ../results/environment.txt

RANKS="1 2 4 8 16 32 64 128 256 512"

# Weak: 1M elems/rank fixed, global = 1M * n
for n in $RANKS; do
    srun -n $n --cpu-bind=cores ./main 1000000 50 \
        > $OUT/t1_ser_weak_n${n}.log 2>&1
done

# Strong: 512M global fixed
G=512000000
for n in $RANKS; do
    LOCAL=$(( G / n ))
    srun -n $n --cpu-bind=cores ./main $LOCAL 50 \
        > $OUT/t1_ser_strong_n${n}.log 2>&1
done
