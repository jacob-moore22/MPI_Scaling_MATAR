#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <vector>
#include <mpi.h>
#include "matar.h"
#include "timing_common.h"

// Required for MATAR data structures (CArrayDevice, MPICArrayKokkos, operation, FOR_ALL, ...)
using namespace mtr;

// Default problem size / iteration counts; both are overridable from the
// command line so the same binary can drive weak- and strong-scaling sweeps:
//   ./main <local_size> [timed_iters]
//
//   Weak scaling:   keep <local_size> fixed and increase the rank count.
//   Strong scaling: shrink <local_size> as the rank count grows so that
//                   local_size * world_size (the global problem size) stays
//                   constant; the launching script computes that division.
//
// Every run times two variants back to back, at identical local_size and
// rank count, so their RESULT lines (schema in include/timing_common.h,
// "matar" vs "bare_mpi" variant column) are directly comparable:
//   "matar"    -- MPICArrayKokkos<double>::all_reduce()
//   "bare_mpi" -- a plain C++ local reduction + a single MPI_Allreduce(),
//                 with no Kokkos/MATAR involved, reproducing the exact same
//                 access pattern (local reduction over local_size elements,
//                 then a scalar MPI_Allreduce) that all_reduce() takes on
//                 its no-CommunicationPlan path. The gap between the two
//                 isolates MATAR/Kokkos's own array + reduction +
//                 host-device-sync overhead from the underlying MPI
//                 library's collective cost.
#define DEFAULT_LOCAL_SIZE   1000000
#define DEFAULT_TIMED_ITERS  20
#define NUM_WARMUP_ITERS     2

// Times `timed_iters` calls to field.all_reduce(op), with `warmup_iters`
// untimed calls first. A barrier precedes each timed call so that rank skew
// arriving at the collective is not counted against the operation itself.
ReduceResult time_all_reduce(MPICArrayKokkos<double>& field, operation op,
                              int warmup_iters, int timed_iters, MPI_Comm comm)
{
    ReduceResult result;

    for (int iter = 0; iter < warmup_iters; iter++) {
        result.value = field.all_reduce(op);
    }

    Timer timer;
    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::max();

    for (int iter = 0; iter < timed_iters; iter++) {
        MPI_Barrier(comm);
        timer.start();
        result.value = field.all_reduce(op);
        double elapsed_ms = timer.stop();

        total_ms += elapsed_ms;
        min_ms = std::min(min_ms, elapsed_ms);
    }

    result.time_ms_min = min_ms;
    result.time_ms_avg  = total_ms / static_cast<double>(timed_iters);
    return result;
}

// ------------------------------------------------------------------------
// Bare-MPI baseline: no Kokkos, no MATAR types anywhere below this point.
// ------------------------------------------------------------------------

enum class ReduceOp { sum, product, max, min };

MPI_Op to_mpi_op(ReduceOp op)
{
    switch (op) {
        case ReduceOp::sum:     return MPI_SUM;
        case ReduceOp::product: return MPI_PROD;
        case ReduceOp::max:     return MPI_MAX;
        case ReduceOp::min:     return MPI_MIN;
    }
    return MPI_SUM;
}

// Sequential local reduction over a plain host array, using the same
// reduction identities Kokkos uses (0 for sum, 1 for product, lowest/max
// for max/min) so this is mathematically equivalent to MATAR's on-device
// FOR_REDUCE_*_CLASS reduction, not just numerically close to it.
double local_reduce(const std::vector<double>& data, ReduceOp op)
{
    switch (op) {
        case ReduceOp::sum: {
            double s = 0.0;
            for (double v : data) s += v;
            return s;
        }
        case ReduceOp::product: {
            double p = 1.0;
            for (double v : data) p *= v;
            return p;
        }
        case ReduceOp::max: {
            double m = std::numeric_limits<double>::lowest();
            for (double v : data) if (v > m) m = v;
            return m;
        }
        case ReduceOp::min: {
            double m = std::numeric_limits<double>::max();
            for (double v : data) if (v < m) m = v;
            return m;
        }
    }
    return 0.0;
}

// Same harness as the MATAR-side time_all_reduce() above, but timing
// (local_reduce + MPI_Allreduce) directly with no MATAR/Kokkos call in the
// timed region.
ReduceResult time_all_reduce(const std::vector<double>& data, ReduceOp op,
                              int warmup_iters, int timed_iters, MPI_Comm comm)
{
    ReduceResult result;
    MPI_Op mpi_op = to_mpi_op(op);

    for (int iter = 0; iter < warmup_iters; iter++) {
        double local = local_reduce(data, op);
        MPI_Allreduce(&local, &result.value, 1, MPI_DOUBLE, mpi_op, comm);
    }

    Timer timer;
    double total_ms = 0.0;
    double min_ms = std::numeric_limits<double>::max();

    for (int iter = 0; iter < timed_iters; iter++) {
        MPI_Barrier(comm);
        timer.start();
        double local = local_reduce(data, op);
        double global = 0.0;
        MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, mpi_op, comm);
        double elapsed_ms = timer.stop();

        result.value = global;
        total_ms += elapsed_ms;
        min_ms = std::min(min_ms, elapsed_ms);
    }

    result.time_ms_min = min_ms;
    result.time_ms_avg  = total_ms / static_cast<double>(timed_iters);
    return result;
}

// main
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MATAR_INITIALIZE(argc, argv);
    { // MATAR scope

    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t local_size  = DEFAULT_LOCAL_SIZE;
    int timed_iters    = DEFAULT_TIMED_ITERS;
    if (argc > 1) {
        local_size = static_cast<size_t>(std::atoll(argv[1]));
    }
    if (argc > 2) {
        timed_iters = std::atoi(argv[2]);
    }
    if (local_size == 0 || timed_iters <= 0) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " [local_size > 0] [timed_iters > 0]" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        std::cout << "MATAR MPICArrayKokkos::all_reduce timing test" << std::endl;
        std::cout << "  world_size  = " << world_size << std::endl;
        std::cout << "  local_size  = " << local_size << " elements/rank" << std::endl;
        std::cout << "  global_size = " << (local_size * static_cast<size_t>(world_size)) << " elements" << std::endl;
        std::cout << "  iterations  = " << timed_iters << " (+" << NUM_WARMUP_ITERS << " warm-up)" << std::endl;
    }

    // ------------------------------------------------------------------
    // Each rank fills a local_size-length array with a single uniform
    // value, (rank + 1) / local_size, so that the LOCAL sum is exactly
    // (rank + 1) regardless of local_size. That gives closed-form expected
    // values for every all_reduce operator below (see below), so results
    // can be validated on any hardware backend / rank count without a
    // separate reference run.
    //
    // No CommunicationPlan is attached: MPICArrayKokkos::all_reduce()'s
    // ghost-aware branch (taken when a plan is attached) does not perform
    // the local on-device reduction, so it does not apply here. This
    // matches a basic, non-ghosted global reduction, i.e. the intended use
    // case for this benchmark.
    // ------------------------------------------------------------------
    const double rank_value = static_cast<double>(rank + 1) / static_cast<double>(local_size);

    MPICArrayKokkos<double> field(local_size, "field");

    FOR_ALL(i, 0, local_size, {
        field(i) = rank_value;
    });
    MATAR_FENCE();

    // Closed-form expected values across all ranks (all_reduce here has no
    // ghost cells, so it reduces the entire local array on every rank):
    //   sum = 1 + 2 + ... + world_size = world_size * (world_size + 1) / 2
    //   max = value held by the highest rank = world_size / local_size
    //   min = value held by rank 0            = 1 / local_size
    //   product = prod_r ((r + 1) / local_size) ^ local_size
    const double expected_sum = 0.5 * static_cast<double>(world_size) * static_cast<double>(world_size + 1);
    const double expected_max = static_cast<double>(world_size) / static_cast<double>(local_size);
    const double expected_min = 1.0 / static_cast<double>(local_size);

    // Compute the expected product in log-space to avoid overflow while
    // computing it; the product itself underflows to exactly 0.0 in double
    // precision for all but tiny (local_size * world_size) problems, since
    // it multiplies together (local_size * world_size) fractional terms.
    // That is expected numerical behavior, not a bug, so it is reported
    // without a strict pass/fail once the underflow threshold is crossed.
    double expected_log_product = 0.0;
    for (int r = 0; r < world_size; r++) {
        expected_log_product += static_cast<double>(local_size)
            * std::log(static_cast<double>(r + 1) / static_cast<double>(local_size));
    }
    const bool product_underflows = expected_log_product < -700.0; // ~ log(DBL_MIN)
    const double expected_product = product_underflows ? 0.0 : std::exp(expected_log_product);

    if (rank == 0) {
        std::cout << "Timing all_reduce operators..." << std::endl;
    }

    ReduceResult sum_r  = time_all_reduce(field, operation::sum,     NUM_WARMUP_ITERS, timed_iters, MPI_COMM_WORLD);
    ReduceResult max_r  = time_all_reduce(field, operation::max,     NUM_WARMUP_ITERS, timed_iters, MPI_COMM_WORLD);
    ReduceResult min_r  = time_all_reduce(field, operation::min,     NUM_WARMUP_ITERS, timed_iters, MPI_COMM_WORLD);
    ReduceResult prod_r = time_all_reduce(field, operation::product, NUM_WARMUP_ITERS, timed_iters, MPI_COMM_WORLD);

    if (rank == 0) {
        report("matar", "sum",     world_size, local_size, sum_r,  expected_sum, true);
        report("matar", "max",     world_size, local_size, max_r,  expected_max, true);
        report("matar", "min",     world_size, local_size, min_r,  expected_min, true);
        report("matar", "product", world_size, local_size, prod_r, expected_product, !product_underflows);
    }

    // ------------------------------------------------------------------
    // Bare-MPI baseline, run in the same process/rank layout immediately
    // after the MATAR timings above, same local_size/timed_iters, same
    // fill pattern -> same expected_* values computed above still apply.
    // ------------------------------------------------------------------
    std::vector<double> bare_data(local_size, rank_value);

    if (rank == 0) {
        std::cout << "Timing bare-MPI all_reduce baseline..." << std::endl;
    }

    ReduceResult bare_sum_r  = time_all_reduce(bare_data, ReduceOp::sum,     NUM_WARMUP_ITERS, timed_iters, MPI_COMM_WORLD);
    ReduceResult bare_max_r  = time_all_reduce(bare_data, ReduceOp::max,     NUM_WARMUP_ITERS, timed_iters, MPI_COMM_WORLD);
    ReduceResult bare_min_r  = time_all_reduce(bare_data, ReduceOp::min,     NUM_WARMUP_ITERS, timed_iters, MPI_COMM_WORLD);
    ReduceResult bare_prod_r = time_all_reduce(bare_data, ReduceOp::product, NUM_WARMUP_ITERS, timed_iters, MPI_COMM_WORLD);

    if (rank == 0) {
        report("bare_mpi", "sum",     world_size, local_size, bare_sum_r,  expected_sum, true);
        report("bare_mpi", "max",     world_size, local_size, bare_max_r,  expected_max, true);
        report("bare_mpi", "min",     world_size, local_size, bare_min_r,  expected_min, true);
        report("bare_mpi", "product", world_size, local_size, bare_prod_r, expected_product, !product_underflows);
    }

    } // MATAR scope
    MATAR_FINALIZE();
    MPI_Finalize();

    return 0;
}
