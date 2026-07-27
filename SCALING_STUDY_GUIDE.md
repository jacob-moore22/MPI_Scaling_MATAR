# Running the MPI Scaling Studies (Test1 & Test2)

This guide walks through building and running the two benchmarks in this repo:

- **Test1** — `MPICArrayKokkos::all_reduce()` timing (Benchmark 1 in the paper: basic global reductions).
- **Test2** — the ELEMENTS `decomp_example` mesh ghost-communication benchmark (Benchmark 2 in the paper).

Both are standalone CMake projects under `src/`. Each pulls its own copies of Kokkos, MATAR, (and for Test2, ELEMENTS + PT-Scotch) automatically via CMake `FetchContent` — you don't need to install these yourself, but the **first** configure of each needs network access and takes a while (PT-Scotch in particular compiles from source).

## 0. One-time setup

You'll need: a C/C++ compiler (GCC recommended), CMake ≥ 3.20, an MPI implementation (OpenMPI, MPICH, or Intel MPI — the machine needs `mpirun`/`mpicxx` on the `PATH` or discoverable via `find_package(MPI)`), and for Test2 specifically, `bison` and `flex` (PT-Scotch's parser generator needs them).

Check core-count before running anything:
```bash
nproc
```
Never pass `--oversubscribe` to `mpirun` for real timing runs — that oversubscribes ranks onto fewer physical cores than you asked for and produces meaningless numbers. Only ever run as many ranks as you have physical cores for.

**Build type matters a lot here.** Both `CMakeLists.txt` files default to `CMAKE_BUILD_TYPE=Release` (`-O3`) if you don't override it — leave it alone. We found by direct measurement that an `-O2` build (e.g. `-DCMAKE_BUILD_TYPE=RelWithDebInfo`) makes MATAR look ~4x slower than raw MPI, purely because GCC doesn't fully inline/collapse Kokkos's templated reduction machinery at `-O2`; that gap disappears entirely at `-O3`. So: don't pass `-DCMAKE_BUILD_TYPE=...` at all — the default is already correct, and overriding it to anything else will silently produce misleading timing numbers.

## 1. Test1 — `all_reduce` timing benchmark

### Build

```bash
cd src/Test1
mkdir -p build && cd build
cmake ..
make -j"$(nproc)"
```

This builds one executable, `main`, linked against the Serial Kokkos backend by default.

To build for a different backend instead, pass the matching option at configure time (only enable one backend at a time):
```bash
cmake -DTest1_ENABLE_OPENMP=ON ..
cmake -DTest1_ENABLE_CUDA=ON ..
```
Use a **separate build directory per backend** (e.g. `build-serial`, `build-openmp`, `build-cuda`) rather than reconfiguring the same one back and forth — Kokkos bakes the backend choice deep into its own build, and switching in place is unreliable.

### Run

```bash
mpirun -n <ranks> ./main [local_size] [timed_iters]
```
- `local_size` (default **1,000,000**): number of array elements *per rank*.
- `timed_iters` (default **20**): number of timed repetitions per operator (plus 2 fixed warm-up iterations that aren't counted).

Example:
```bash
mpirun -n 4 ./main 1000000 20
```

### What it does and what it prints

Each rank fills a local array with the constant value `(rank+1)/local_size`, so the array's local sum is exactly `rank+1` — this gives closed-form expected values for `sum`/`max`/`min`/`product` at any rank count or array size, so correctness is self-checked on every run (no separate reference needed).

It then times, back to back, at the **same** `local_size`/rank count:
- **`matar`** — `MPICArrayKokkos<double>::all_reduce()` for each of `sum`, `max`, `min`, `product`.
- **`bare_mpi`** — a plain C++ loop + a single raw `MPI_Allreduce()` doing the identical computation, with zero Kokkos/MATAR involved. This isolates MATAR's overhead from the underlying MPI library's cost.

Output has one human-readable line and one machine-parsable `RESULT,...` CSV line per operator/variant:
```
RESULT,<matar|bare_mpi>,<sum|max|min|product>,world_size,local_size,global_size,min_ms,avg_ms,value,expected,status
```
`status` is `PASS`, `FAIL`, or `SKIPPED` (the `product` reduction is expected to underflow to exactly `0.0` for most problem sizes — that's correct numerical behavior, not a bug, and is reported as `SKIPPED` rather than `FAIL`).

**Always check `status` on every line before trusting a run's timings** — a `FAIL` means something is wrong with that run (e.g. an oversubscribed launch, a build issue) and its timing numbers shouldn't be used.

### Running the scaling sweeps

Redirect output to a log file per configuration so you have a record to parse later — for example:

**Weak scaling** (fixed work per rank, ranks increasing — array grows with rank count):
```bash
mkdir -p ../results
for n in 1 2 4 8; do
    mpirun -n $n ./main 1000000 20 > ../results/weak_n${n}.log 2>&1
done
```

**Strong scaling** (fixed total problem size, split across more ranks — `local_size` shrinks as ranks grow):
```bash
GLOBAL_SIZE=8000000
mkdir -p ../results
for n in 1 2 4 8; do
    local_size=$(( GLOBAL_SIZE / n ))
    mpirun -n $n ./main $local_size 20 > ../results/strong_n${n}.log 2>&1
done
```

To pull just the machine-parsable lines out of a batch of logs for plotting/analysis:
```bash
grep "^RESULT" ../results/*.log > ../results/all_results.csv
```

## 2. Test2 — mesh ghost-communication benchmark (`decomp_example`)

### Build

```bash
cd src/Test2
mkdir -p build && cd build
cmake ..
make -j"$(nproc)"
```

The first configure fetches Kokkos, MATAR, and ELEMENTS (which in turn fetches and **builds PT-Scotch from source** — this step alone can take several minutes). Subsequent builds in the same directory are fast.

Same backend-selection pattern as Test1, using the `Test2_ENABLE_*` options (again, one separate build directory per backend):
```bash
cmake -DTest2_ENABLE_SERIAL=OFF -DTest2_ENABLE_OPENMP=ON ..
```

### Run

```bash
mpirun -n <ranks> ./main [N | nx ny nz] [--comms C] [--smooth S]
```
- No arguments: builds a **20×20×20** element box mesh, and runs **1** communication step with **3** smoothing passes (the defaults).
- One integer `N`: builds an `N×N×N` box mesh.
- Three integers `nx ny nz`: builds an `nx × ny × nz` box mesh.
- `--comms C` (or `-c C`): number of repeated communicate() steps to time (default **1**).
- `--smooth S` (or `-s S`): number of local smoothing passes run between each communicate() step (default **3**).

The `--comms`/`--smooth` flags always come *after* the mesh-size argument(s), in either order, e.g. `./main 30 30 30 --comms 10 --smooth 2` or `./main 30 --smooth 2 --comms 10`.

At **1 rank**, the mesh is built and used locally with no partitioning/communication step (`communicate()` becomes a no-op, so comm-step timings will read ~0). **Partitioning, ghost exchange, and the communication-volume report only happen at 2+ ranks** — for a real scaling study you need ≥2 ranks on every run.

Example:
```bash
mpirun -n 4 ./main 30 30 30 --comms 20 --smooth 3
```

### What it does and what it prints

Per run: builds the mesh on rank 0, partitions it across all ranks with PT-Scotch (building a `CommunicationPlan` for elements and a separate one for nodes), fills Gauss-point and nodal fields with a rank-ID sentinel pattern, runs one **untimed warm-up round** (absorbs first-touch page faults on the MPI buffers so they don't pollute the first timed sample), then repeats `--comms` times: run `--smooth` local self+neighbor averaging passes (this changes the *owned* values each pass, so it's genuinely new data every round, not the same bytes resent) followed by a separately-timed, barrier-isolated `.communicate()` call **for each of the four fields individually** (`fields`, `fields_vec`, `scalar_field`, `vector_field`) that refreshes ghost data from neighbors. Finally it writes VTU visualization output and a CSV log (see below).

Key timing/volume lines to look for in the terminal output:
- `Communication steps: <C>, smoothing passes/step: <S>` (confirms what a run was actually configured with — always check this against what you intended to launch)
- `Mesh partitioning time: ... seconds`
- `Rank <r> gauss point fields communication time over <C> step(s): min=... avg=... max=... seconds` (printed once per rank, in rank order; statistics are over the `C` repeated communicate() calls) — and the same for `fields_vec`, `nodal scalar_field`, `nodal vector_field`
- `Rank <r> communication volume (sent ... bytes, received ... bytes):` plus a per-neighbor breakdown (bytes per communicate() call — constant across all `C` steps, since only the field *values* change round to round, not the mesh partition/topology)
- `Total data volume exchanged across all ranks: ... bytes (... MB)`
- `Total execution time: ... seconds`
- `Wrote scaling-study CSV log: <filename>` (see below)

**CSV log file:** after everything else finishes, rank 0 writes a single combined CSV — `test2_results_mesh<nx>x<ny>x<nz>_np<world_size>_comms<C>_smooth<S>.csv` — with one row per `(rank, field)` for all 6 metrics (`gauss_fields`, `gauss_fields_vec`, `node_scalar_field`, `node_vector_field`, `overall_communication` [the true per-step sum of all four fields, not a sum of separately-computed mins/maxes], `total_execution_time`), each with `min_s`/`avg_s`/`max_s`/`total_s` over the `--comms` steps. The filename already encodes mesh size, rank count, comm steps, and smoothing steps, so sweep runs never collide/overwrite each other — this is the file to actually feed into your scaling-study plots, rather than scraping the terminal log.

**On noise:** communication timings on a shared/laptop machine can be noisy run to run — we confirmed by direct experiment that rerunning the identical configuration back-to-back can show one run with a 10-20x max/min spike on some field/rank and the next run completely clean, with no code change. This is transient OS/system noise (scheduler jitter, background load), not a bug — **use each field's `min_s`** as your primary scaling-study number (least corrupted by noise); `avg_s`/`max_s` are useful for characterizing variability but will stay noisy on a shared machine regardless. Core-pinning (`mpirun --bind-to core -n ...`) and running on dedicated (non-shared) nodes will tighten all three.

Unlike a single Test1 invocation (which times many iterations of one array size), Test2 times `--comms` repeated communication rounds *within* a fixed mesh partition — use a larger `--comms` value (e.g. 20-50) rather than re-launching the binary repeatedly, to get a stable min/avg/max read on repeated-communication cost at a given mesh size and rank count.

**VTU output note:** every run overwrites the same files under `./vtk/` (`vtk/Fierro.00000_rank<r>.vtu`) regardless of mesh size or rank count — these are for visual sanity-checking a single run in ParaView/VisIt, not for archiving across a sweep. They are not needed for the timing/volume data (use the CSV log for that). It's fine (and expected) to ignore or delete `vtk/` between sweep runs.

### Running the scaling sweeps

**Strong scaling** (fixed global mesh size, increasing rank count):
```bash
mkdir -p ../results
for n in 2 4 8; do
    mpirun -n $n ./main 60 60 60 --comms 20 --smooth 3 > ../results/strong_n${n}.log 2>&1
    mv test2_results_mesh60x60x60_np${n}_comms20_smooth3.csv ../results/
done
```

**Weak scaling** (elements-per-rank held roughly fixed — grow the mesh with rank count). Since the mesh is a cube partitioned in 3D, keep `num_elems_dim^3 / world_size` approximately constant, e.g.:
```bash
mkdir -p ../results
declare -A dims=( [2]="38 30 30" [4]="38 38 30" [8]="38 38 38" )
for n in 2 4 8; do
    mpirun -n $n ./main ${dims[$n]} --comms 20 --smooth 3 > ../results/weak_n${n}.log 2>&1
done
mv test2_results_*.csv ../results/
```
(These particular numbers are illustrative — pick `nx ny nz` combinations that keep total elements roughly proportional to rank count for your actual study; there's no built-in helper for this yet, so compute it by hand or with a small script.)

Use the same `--comms`/`--smooth` values across every run in a given sweep — changing them changes what's being measured (a different number of repeated communication rounds), which would make the runs in that sweep incomparable.

The CSV files land in the directory you ran `./main` from (same place as `vtk/`) — move or collect them (as above) rather than leaving them in `build/`, since a later run with the exact same mesh/ranks/comms/smooth would silently overwrite one. To combine every CSV from a sweep into one file for plotting (keeping a single header row):
```bash
head -1 ../results/test2_results_*.csv | grep -m1 mesh_nx > ../results/all_results.csv
tail -n +2 -q ../results/test2_results_*.csv >> ../results/all_results.csv
```
To pull just the timing/volume summary lines out of the terminal logs instead (useful for a quick look, or for the mesh-build/partitioning-time lines that aren't in the CSV):
```bash
grep -E "communication time|communication volume|Total data volume|Total execution time|partitioning time" ../results/*.log
```

## 3. General tips

- **Save raw output, always.** Every number that matters is printed to stdout — redirect every run to a log file (as shown above) rather than reading it off the terminal and typing it in by hand. `src/Test2/results.txt` in this repo is an example of exactly the kind of hand-transcribed log you should instead be keeping as raw `.log`/`.csv` files per run.
- **One build directory per Kokkos backend**, for both Test1 and Test2 — don't reconfigure the same directory between Serial/OpenMP/CUDA.
- **Match `-n <ranks>` to physical cores** (check with `nproc`) — never `--oversubscribe` for a timing run.
- **Don't override the build type.** It defaults to `Release` — see the warning in Section 0 for why that matters.
- If a `RESULT` line (Test1) ever shows `FAIL`, or a Test2 run crashes/hangs at ≥2 ranks, don't record that data point — something is wrong with that specific run (bad launch, wrong core count, etc.), not a real scaling result.
- Test2 needs its `ELEMENTS` dependency's `main` branch to include the `partition_mesh` dimension-broadcast fix (merged upstream as of this writing, PR #58, `lanl/ELEMENTS`). If a *fresh* Test2 build ever crashes at ≥2 ranks with `"Error: mesh.num_dims is not set"`, that means `FetchContent` picked up a stale/pre-fix copy of ELEMENTS — delete `src/Test2/build/_deps/elements-*` and reconfigure to re-fetch.
