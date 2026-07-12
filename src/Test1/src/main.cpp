#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <chrono>
#include <limits>
#include <algorithm>
#include <mpi.h>
#include "matar.h"

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
#define DEFAULT_LOCAL_SIZE   1000000
#define DEFAULT_TIMED_ITERS  20
#define NUM_WARMUP_ITERS     2

// Timer class for timing the execution of the all_reduce calls
class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    bool is_running;

public:
    Timer() : is_running(false) {}

    void start() {
        start_time = std::chrono::high_resolution_clock::now();
        is_running = true;
    }

    double stop() {
        if (!is_running) {
            std::cerr << "Timer was not running!" << std::endl;
            return 0.0;
        }
        end_time = std::chrono::high_resolution_clock::now();
        is_running = false;

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // Convert to milliseconds
    }
};

// Timing statistics + the reduced value from the last iteration, for one operator.
struct ReduceResult {
    double time_ms_min = 0.0;
    double time_ms_avg  = 0.0;
    double value        = 0.0;
};

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

bool nearly_equal(double a, double b, double rel_tol = 1.0e-9, double abs_tol = 1.0e-9)
{
    return std::fabs(a - b) <= std::max(abs_tol, rel_tol * std::max(std::fabs(a), std::fabs(b)));
}

// Prints one machine-parsable line per operator (prefixed "RESULT,") so
// scaling-sweep driver scripts can grep/parse timings across many runs,
// plus a human-readable line for interactive use. Rank-0 only.
void report(const char* op_name, int world_size, size_t local_size,
            const ReduceResult& r, double expected, bool validated)
{
    const char* status = "SKIPPED";
    if (validated) {
        status = nearly_equal(r.value, expected) ? "PASS" : "FAIL";
    }

    std::cout << "  " << op_name
              << ": value=" << r.value
              << " expected=" << (validated ? expected : 0.0)
              << " [" << status << "]"
              << " min_ms=" << r.time_ms_min
              << " avg_ms=" << r.time_ms_avg
              << std::endl;

    std::cout << "RESULT," << op_name << "," << world_size << "," << local_size << ","
              << (local_size * static_cast<size_t>(world_size)) << ","
              << r.time_ms_min << "," << r.time_ms_avg << ","
              << r.value << "," << expected << "," << status
              << std::endl;
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
        report("sum",     world_size, local_size, sum_r,  expected_sum, true);
        report("max",     world_size, local_size, max_r,  expected_max, true);
        report("min",     world_size, local_size, min_r,  expected_min, true);
        report("product", world_size, local_size, prod_r, expected_product, !product_underflows);
    }

    } // MATAR scope
    MATAR_FINALIZE();
    MPI_Finalize();

    return 0;
}
