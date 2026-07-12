#ifndef TIMING_COMMON_H
#define TIMING_COMMON_H

#include <chrono>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <mpi.h>

// Shared by every benchmark variant in this directory (MATAR-based and bare-
// MPI alike) so their timings/output are directly comparable. Deliberately
// depends on nothing but the standard library and MPI.

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

inline bool nearly_equal(double a, double b, double rel_tol = 1.0e-9, double abs_tol = 1.0e-9)
{
    return std::fabs(a - b) <= std::max(abs_tol, rel_tol * std::max(std::fabs(a), std::fabs(b)));
}

// Prints one machine-parsable "RESULT," line (plus a human-readable line) so
// driver scripts can grep/parse timings across many runs *and* line up
// different benchmark variants (e.g. "matar" vs "bare_mpi") at matching
// world_size/local_size for a direct overhead comparison. Rank-0 only.
//
// CSV schema: RESULT,variant,op,world_size,local_size,global_size,min_ms,avg_ms,value,expected,status
inline void report(const char* variant, const char* op_name, int world_size, size_t local_size,
                    const ReduceResult& r, double expected, bool validated)
{
    const char* status = "SKIPPED";
    if (validated) {
        status = nearly_equal(r.value, expected) ? "PASS" : "FAIL";
    }

    std::cout << "  [" << variant << "] " << op_name
              << ": value=" << r.value
              << " expected=" << (validated ? expected : 0.0)
              << " [" << status << "]"
              << " min_ms=" << r.time_ms_min
              << " avg_ms=" << r.time_ms_avg
              << std::endl;

    std::cout << "RESULT," << variant << "," << op_name << "," << world_size << "," << local_size << ","
              << (local_size * static_cast<size_t>(world_size)) << ","
              << r.time_ms_min << "," << r.time_ms_avg << ","
              << r.value << "," << expected << "," << status
              << std::endl;
}

#endif // TIMING_COMMON_H
