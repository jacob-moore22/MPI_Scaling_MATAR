# Morrill Scaling Study Plan — MPI_Scaling_MATAR

Publication-quality strong and weak scaling test matrix for **Test1** (`MPICArrayKokkos::all_reduce()` vs. bare `MPI_Allreduce`) and **Test2** (ELEMENTS `decomp_example` mesh ghost-communication benchmark) on **Morrill** (MSU HPCC).

---

## 1. Target hardware (Morrill)

| Resource | Spec | Relevant limits for this study |
|---|---|---|
| CPU compute nodes | 100× Dell C6525: 2× AMD EPYC 7543 (Zen 3), **64 cores/node**, 256 GB RAM, 8 NUMA domains/node (8 cores each) | CPU sweeps use up to **8 nodes = 512 cores** |
| GPU nodes | 4× HPE Apollo 6500: 2× EPYC 7713 (128 cores), 1 TB RAM, **8× A100-SXM4-80GB per node** | CUDA sweeps limited to **2 nodes = 16 GPUs** |
| Interconnect | Mellanox HDR100 InfiniBand, 100 Gb/s node-to-node | All multi-node comm timings ride on this |
| Compiler flags | GCC: `-march=znver3 -O3` (Release default already gives `-O3` — **do not override build type**); NVCC: `-arch=sm_80` | A100 = compute capability 8.0 |

**Design rationale for the sweep endpoints:**
- 8 CPU nodes (512 cores) spans three regimes: single-NUMA (≤8 ranks), single-node (≤64 ranks), and multi-node over IB (128–512 ranks). That's the full story a reviewer wants; going wider on a condo/shared cluster mostly adds queue time, not insight. If allocation permits, an optional 16-node (1024-rank) extension point is noted where relevant.
- GPU sweeps go 1→16 GPUs (Test2: 2→16), crossing the node boundary at 8→16, which isolates NVLink-within-node vs. IB-across-node behavior.
- The **processing element (PE)** differs per backend: Serial PE = 1 core, OpenMP PE = 1 rank = 8 cores (one NUMA domain), CUDA PE = 1 GPU. Weak-scaling workloads below are sized so *work per core* (CPU) or *work per GPU* is what's held constant, so backends remain cross-comparable per-core / per-node.

---

## 2. Build matrix (6 builds total)

One build directory per (test × backend), per the repo guide — never reconfigure a directory between backends.

| Build dir | Configure |
|---|---|
| `src/Test1/build-serial` | `cmake ..` |
| `src/Test1/build-openmp` | `cmake  -DTest1_ENABLE_OPENMP=ON ..` |
| `src/Test1/build-cuda` | `cmake  -DTest1_ENABLE_CUDA=ON ..` (on a GPU node / with `module load cuda`) |
| `src/Test2/build-serial` | `cmake ..` |
| `src/Test2/build-openmp` | `cmake  -DTest2_ENABLE_OPENMP=ON ..` |
| `src/Test2/build-cuda` | `cmake  -DTest2_ENABLE_CUDA=ON ..` |

Notes:
- First configure of each needs internet (FetchContent: Kokkos, MATAR, ELEMENTS, PT-Scotch). Compute nodes may not have outbound internet — do first configures on a **login/dev/DTN node**, then build on the target node type.
- Build the CUDA binaries on (or targeting) the GPU nodes with `-arch=sm_80`.

---

## 3. Common run rules (every run, every sweep)

1. **Exclusive nodes**: every job uses `#SBATCH --exclusive`. No shared-node timing runs.
2. **Core pinning**: `mpirun --bind-to core` (serial/CUDA) or `--bind-to core --map-by ppr:8:node:pe=8` (OpenMP). Never `--oversubscribe`.
3. **Repetitions**: **3 independent job submissions per configuration** (separately scheduled, so node assignment/system noise varies). Report the **min** across reps of the per-run min (`min_ms` for Test1, `min_s` for Test2) as the scaling number; use avg/max spread across reps as the variability bar.
4. **Fixed measurement parameters within a sweep**: Test1 `timed_iters = 50` everywhere; Test2 `--comms 20 --smooth 3` everywhere. Never mix values within a sweep.
5. **Validity checks before recording a data point**: every Test1 `RESULT` line must be `PASS`/`SKIPPED` (never `FAIL`); every Test2 log must echo the intended `Communication steps: 20, smoothing passes/step: 3` and must not have crashed/hung. Discard and rerun bad points.
6. **Raw output archived**: one `.log` per run, named by config; collect Test2's auto-named CSVs (`test2_results_mesh*_np*_comms*_smooth*.csv`) into `results/` immediately after each run (identical configs across reps produce the *same filename* — move each rep into its own subdirectory, e.g. `results/rep1/`, `results/rep2/`, `results/rep3/`, or they will overwrite).
7. Same module environment (compiler, MPI, CUDA versions) for every run in the study; record `module list`, `mpirun --version`, `nvidia-smi` output once into `results/environment.txt`.
8. Test2 requires ≥2 ranks for any communication to occur — no 1-rank Test2 points.

---

## 4. Test1 — `all_reduce` scaling matrix

Invocation: `mpirun -n <ranks> ./main <local_size> 50`

### 4.1 MPI + Serial (CPU nodes, 1 rank/core, 64 ranks/node)

| Sweep | Ranks (nodes) | `local_size` per rank | Global size |
|---|---|---|---|
| **Weak** | 1, 2, 4, 8, 16, 32, 64 (1), 128 (2), 256 (4), 512 (8) | 1,000,000 (fixed) | 1M × ranks |
| **Strong** | 1, 2, 4, 8, 16, 32, 64 (1), 128 (2), 256 (4), 512 (8) | 512,000,000 / ranks | 512,000,000 (fixed) |

- Strong-scaling endpoints: n=1 → 512M/rank (~4 GB array, fits 256 GB node); n=512 → 1M/rank (matches the weak-scaling per-rank load, a clean crossover point).
- 10 configs × 2 sweeps × 3 reps = **60 runs**.
- *Optional extension*: 1024 ranks (16 nodes) at both sweeps if allocation allows.

### 4.2 MPI + OpenMP (CPU nodes, 8 ranks/node × 8 threads/rank = 1 rank per NUMA domain)

`export OMP_NUM_THREADS=8; export OMP_PROC_BIND=close; export OMP_PLACES=cores`

| Sweep | Ranks (nodes) | Total cores | `local_size` per rank | Global size |
|---|---|---|---|---|
| **Weak** | 1, 2, 4, 8 (1), 16 (2), 32 (4), 64 (8) | 8–512 | 8,000,000 (fixed = 1M/core) | 8M × ranks |
| **Strong** | 1, 2, 4, 8 (1), 16 (2), 32 (4), 64 (8) | 8–512 | 512,000,000 / ranks | 512,000,000 (fixed) |

- Same global sizes and same core counts as the Serial sweep → direct serial-vs-OpenMP comparison at equal hardware, isolating the threading/rank-count tradeoff (fewer, larger MPI messages vs. more, smaller ones).
- 7 configs × 2 sweeps × 3 reps = **42 runs**.

### 4.3 MPI + CUDA (GPU nodes, 1 rank/GPU, 8 ranks/node, max 16 GPUs)

| Sweep | Ranks = GPUs (nodes) | `local_size` per rank | Global size |
|---|---|---|---|
| **Weak** | 1, 2, 4, 8 (1), 16 (2) | 100,000,000 (fixed) | 100M × GPUs |
| **Strong** | 1, 2, 4, 8 (1), 16 (2) | 1,600,000,000 / ranks | 1,600,000,000 (fixed) |

- 100M doubles/GPU (roughly 0.8 GB) is large enough to saturate an A100's 2 TB/s HBM on the reduce kernel while leaving the timing sensitive to comm cost; strong-scaling n=1 point is 1.6B doubles (~12.8 GB) — comfortably within 80 GB.
- The 8→16 GPU step is the key data point: it crosses from all-intra-node to inter-node IB.
- GPU-to-rank mapping: pin one A100 per rank via local rank (see §7 wrapper). Verify with `nvidia-smi` in the job log that all 8 GPUs/node are busy.
- 5 configs × 2 sweeps × 3 reps = **30 runs**.

---

## 5. Test2 — mesh ghost-communication scaling matrix

Invocation: `mpirun -n <ranks> ./main <nx> <ny> <nz> --comms 20 --smooth 3`

All meshes below are chosen so element counts double exactly with rank count in the weak sweeps (per-PE load exactly constant), and so strong-scaling meshes divide sensibly at every rank count.

### 5.1 MPI + Serial (CPU nodes, 64 ranks/node)

**Strong scaling — two fixed problem sizes** (a "small" comm-bound case and a "large" compute-heavier case; publication plots benefit from showing where each rolls off):

| Ranks (nodes) | Mesh A (small): `100 100 100` = 1.0M elems | Mesh B (large): `200 200 200` = 8.0M elems |
|---|---|---|
| 2, 4, 8, 16, 32, 64 (1), 128 (2), 256 (4), 512 (8) | elems/rank: 500K → 1.95K | elems/rank: 4M → 15.6K |

**Weak scaling — 32,768 elems/rank (32³ per core):**

| Ranks (nodes) | Mesh (`nx ny nz`) | Total elems |
|---|---|---|
| 2 | `64 32 32` | 65,536 |
| 4 | `64 64 32` | 131,072 |
| 8 | `64 64 64` | 262,144 |
| 16 | `128 64 64` | 524,288 |
| 32 | `128 128 64` | 1,048,576 |
| 64 (1 node) | `128 128 128` | 2,097,152 |
| 128 (2) | `256 128 128` | 4,194,304 |
| 256 (4) | `256 256 128` | 8,388,608 |
| 512 (8) | `256 256 256` | 16,777,216 |

- (9 strong × 2 meshes + 9 weak) × 3 reps = **81 runs**.
- Note: the mesh is built serially on rank 0 before partitioning — the 256³ case is the largest serial-build; if rank-0 memory becomes an issue, place rank 0 alone on its own NUMA domain (it fits in 256 GB, but partition time will visibly grow — that's a reportable result, not a bug: plot `Mesh partitioning time` separately from `communicate()` time).

### 5.2 MPI + OpenMP (CPU nodes, 8 ranks/node × 8 threads, PE = NUMA domain)

**Strong scaling** — same Mesh A (`100³`) and Mesh B (`200³`) as §5.1, ranks = 2, 4, 8 (1 node), 16 (2), 32 (4), 64 (8). Same global problem on the same node counts as serial → direct backend comparison per node.

**Weak scaling — 262,144 elems/rank (= 32,768 per core, matching §5.1 per-core load):**

| Ranks (nodes) | Mesh (`nx ny nz`) | Total elems |
|---|---|---|
| 2 | `128 64 64` | 524,288 |
| 4 | `128 128 64` | 1,048,576 |
| 8 (1 node) | `128 128 128` | 2,097,152 |
| 16 (2) | `256 128 128` | 4,194,304 |
| 32 (4) | `256 256 128` | 8,388,608 |
| 64 (8) | `256 256 256` | 16,777,216 |

- (6 strong × 2 meshes + 6 weak) × 3 reps = **54 runs**.

### 5.3 MPI + CUDA (GPU nodes, 1 rank/GPU, 2–16 GPUs)

**Strong scaling** — fixed `128 128 128` (2,097,152 elems):

| Ranks = GPUs (nodes) | elems/GPU |
|---|---|
| 2, 4, 8 (1), 16 (2) | 1,048,576 → 131,072 |

**Weak scaling — 262,144 elems/GPU (64³ per GPU):**

| GPUs (nodes) | Mesh (`nx ny nz`) | Total elems |
|---|---|---|
| 2 | `128 64 64` | 524,288 |
| 4 | `128 128 64` | 1,048,576 |
| 8 (1 node) | `128 128 128` | 2,097,152 |
| 16 (2) | `256 128 128` | 4,194,304 |

- Weak meshes are *identical* to the OpenMP weak table at matching rank counts → a clean per-PE GPU-vs-NUMA-domain comparison on the exact same global problems.
- (4 strong + 4 weak) × 3 reps = **24 runs**.
- The 8→16 step again isolates intra-node vs. HDR100 inter-node ghost exchange — pair it with the per-neighbor `communication volume` report to separate volume effects from latency/bandwidth effects.

---

## 6. Total run count

| | Configs | Runs (×3 reps) |
|---|---|---|
| Test1 Serial | 20 | 60 |
| Test1 OpenMP | 14 | 42 |
| Test1 CUDA | 10 | 30 |
| Test2 Serial | 27 | 81 |
| Test2 OpenMP | 18 | 54 |
| Test2 CUDA | 8 | 24 |
| **Total** | **97** | **291** |

Individual runs are short (seconds to a few minutes); nearly all wall-clock cost is queue wait. Batch each (backend × test × sweep × rep) as one Slurm job that loops over its rank counts, sized to the sweep's max node count.

---

## 7. Slurm templates

### 7.1 CPU sweep job (example: Test2 serial weak, rep 1)

```bash
#!/bin/bash
#SBATCH --job-name=t2-ser-weak-r1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=64
#SBATCH --exclusive
#SBATCH --time=04:00:00
# (set --partition per current Morrill partition list: docs.hpc.msstate.edu/cluster/operations/partitions.html)

module purge
module load gcc openmpi cmake        # record exact versions in results/environment.txt

cd $SLURM_SUBMIT_DIR/src/Test2/build-serial
REP=rep1; OUT=../results/$REP; mkdir -p $OUT

declare -A dims=( [2]="64 32 32" [4]="64 64 32" [8]="64 64 64" [16]="128 64 64" \
                  [32]="128 128 64" [64]="128 128 128" [128]="256 128 128" \
                  [256]="256 256 128" [512]="256 256 256" )
for n in 2 4 8 16 32 64 128 256 512; do
    mpirun -n $n --bind-to core ./main ${dims[$n]} --comms 20 --smooth 3 \
        > $OUT/weak_serial_n${n}.log 2>&1
    mv test2_results_*.csv $OUT/ 2>/dev/null
done
```

For **OpenMP** jobs, change to `--ntasks-per-node=8 --cpus-per-task=8`, add:

```bash
export OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores
mpirun -n $n --map-by ppr:8:node:pe=8 --bind-to core ./main ...
```

### 7.2 GPU sweep job (example: Test1 CUDA, both sweeps, rep 1)

```bash
#!/bin/bash
#SBATCH --job-name=t1-cuda-r1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:8
#SBATCH --exclusive
#SBATCH --partition=gpu-a100
#SBATCH --time=02:00:00

module purge
module load gcc openmpi cuda cmake

cd $SLURM_SUBMIT_DIR/src/Test1/build-cuda
REP=rep1; OUT=../results/$REP; mkdir -p $OUT

# one-GPU-per-rank wrapper
cat > gpu_wrap.sh <<'EOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=${OMPI_COMM_WORLD_LOCAL_RANK:-$SLURM_LOCALID}
exec "$@"
EOF
chmod +x gpu_wrap.sh

# Weak: 100M elems/GPU
for n in 1 2 4 8 16; do
    mpirun -n $n --map-by ppr:8:node --bind-to core ./gpu_wrap.sh ./main 100000000 50 \
        > $OUT/t1_cuda_weak_n${n}.log 2>&1
done

# Strong: 1.6B global
G=1600000000
for n in 1 2 4 8 16; do
    mpirun -n $n --map-by ppr:8:node --bind-to core ./gpu_wrap.sh ./main $(( G / n )) 50 \
        > $OUT/t1_cuda_strong_n${n}.log 2>&1
done
```

(If the site's MPI is CUDA-aware and Kokkos is configured to pass device pointers, note that in `environment.txt` — it materially affects interpretation of the CUDA comm timings.)

---

## 8. Metrics and publication plots

From **Test1** (`RESULT` CSV lines; use `min_ms`):
1. **Strong scaling speedup + parallel efficiency** vs. ranks/GPUs, per backend, per operator (`sum` primary; `max`/`min` as supplementary). Efficiency: `E(n) = T(base)·base / (T(n)·n)`.
2. **Weak scaling efficiency**: `E(n) = T(base)/T(n)` at fixed per-PE size — flat is ideal.
3. **MATAR overhead ratio**: `matar min_ms / bare_mpi min_ms` at every point of every sweep — this is the paper's headline claim (MATAR abstraction cost ≈ 0 at `-O3`), so plot it explicitly across the full rank range for all three backends.

From **Test2** (auto-generated CSVs; use `min_s`, and `overall_communication` as the primary field, with the four individual fields as supplementary):
4. Strong/weak communication-time scaling per backend (both meshes for strong).
5. **Comm time vs. comm volume**: pair `overall_communication min_s` with the per-run `Total data volume exchanged`; weak-scaling surface-to-volume growth should track volume until latency dominates.
6. **Partition time** (`Mesh partitioning time`, from the `.log`, not the CSV) vs. ranks — reported separately since it's a serial-rank-0 + PT-Scotch cost, not a comm-scaling result.
7. **Cross-backend node-for-node comparison**: at equal node counts (1, 2 nodes … where all three backends have data), serial vs. OpenMP vs. CUDA on the same global problem (§5 weak tables are aligned to make this exact for OpenMP-vs-CUDA).

Variability reporting: for each point, show min across the 3 reps as the marker with a whisker to the max — reviewers on shared clusters expect this, and the guide's own noise findings justify min-based reporting.

---

## 9. Pre-flight checklist (before burning allocation)

- [ ] All 6 builds compile and a 2-rank smoke test passes on the target node type (Test1: all `PASS`/`SKIPPED`; Test2: comm volumes print, CSV written).
- [ ] Test1 `FAIL` never appears at the largest rank count (512 serial / 64 OpenMP / 16 CUDA) in a smoke run.
- [ ] CUDA smoke run: `nvidia-smi` during the run shows 8 distinct busy GPUs per node (device mapping wrapper works).
- [ ] Confirm actual Morrill partition names/QoS limits for ≥8-node CPU jobs and 2-node GPU jobs (condo pre-emption rights may affect which QoS is safe for timing runs — a preempted-and-requeued job is a discarded data point).
- [ ] `results/environment.txt` written (modules, MPI version, CUDA version, commit hash of the repo).
- [ ] `vtk/` dirs deleted between Test2 runs (cosmetic, but keeps scratch clean).