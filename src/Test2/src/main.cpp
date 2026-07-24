
//
// Mesh Decomposition Example
//
// This example demonstrates how to:
//   1. Create initial meshes (3D box or 2D polar) on MPI rank 0.
//   2. Partition the mesh across multiple MPI ranks using parallel mesh decomposition (e.g., PT-Scotch).
//   3. Load and distribute node coordinates and mesh connectivity to each rank.
//   4. (Optionally) visualize or post-process partitioned mesh data.
//
// The goal is to show how to use Swage and associated mesh utilities for distributed memory mesh setup,
// which is a typical workflow in scalable finite element or particle-based simulations.
//
// Usage: mpirun -n <num_ranks> ./mesh_decomp_example
//
// Dependencies:
//   - MPI
//   - MATAR (container library)
//   - Swage mesh + decomposition utilities
//   - Optionally: PT-Scotch for mesh partitioning
//


#include <cmath> // for sin
#include <cstdio> // for snprintf
#include <cstdlib> // for strtol, atoi
#include <cstring> // for strcmp
#include <vector>
#include <limits>
#include <algorithm> // for min_element, max_element
#include <fstream>


#include "ELEMENTS.h"
#include "state.h"
#include "mesh_io.h"


// Fills mn/avg/mx/total with the min, average, max, and sum of the values in v.
static void time_series_stats(const std::vector<double>& v, double& mn, double& avg, double& mx, double& total)
{
    mn = *std::min_element(v.begin(), v.end());
    mx = *std::max_element(v.begin(), v.end());
    total = 0.0;
    for (double t : v) total += t;
    avg = total / static_cast<double>(v.size());
}


int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    MATAR_INITIALIZE(argc, argv);
    { // MATAR scope

    int world_size;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int num_dims = 2;
    int Pn_order = 1;

    double t_main_start = MPI_Wtime();

    // Mesh size for 3D box
    double origin[3] = {0.0, 0.0, 0.0};
    double length[3] = {1.0, 1.0, 1.0};
    // Mesh resolution (elements per dimension) and repeated-communication controls,
    // user-settable via command line:
    //   ./main                              -> 20 20 20, 1 comm step, 3 smoothing passes/step (defaults)
    //   ./main N                            -> N N N
    //   ./main nx ny nz                     -> nx ny nz
    //   ./main [mesh args] --comms C        -> C repeated communicate() steps (default 1)
    //   ./main [mesh args] --smooth S       -> S local smoothing passes between each comm step (default 3)
    // --comms/--smooth (or -c/-s) may appear in either order, after any mesh-size arguments.
    int num_elems_dim[3] = {20, 20, 20};
    int num_comms = 1;           // number of repeated communicate() steps to time
    int num_smooth_per_comm = 3; // local smoothing passes (new data) between each comm step
    {
        int provided[3];
        int n_provided = 0;
        int a = 1;
        // Consume leading positive integers as mesh dimensions; stop at the first
        // argument that isn't one (e.g. "--comms") so flag parsing below is unambiguous.
        for (; a < argc && n_provided < 3; ++a) {
            char* end = nullptr;
            long v = std::strtol(argv[a], &end, 10);
            if (end != argv[a] && *end == '\0' && v > 0) {
                provided[n_provided++] = static_cast<int>(v);
            } else {
                break;
            }
        }
        if (n_provided == 1) {
            num_elems_dim[0] = num_elems_dim[1] = num_elems_dim[2] = provided[0];
        } else {
            for (int d = 0; d < n_provided; ++d) num_elems_dim[d] = provided[d];
        }

        for (; a < argc; ++a) {
            if ((std::strcmp(argv[a], "--comms") == 0 || std::strcmp(argv[a], "-c") == 0) && a + 1 < argc) {
                num_comms = std::atoi(argv[++a]);
            } else if ((std::strcmp(argv[a], "--smooth") == 0 || std::strcmp(argv[a], "-s") == 0) && a + 1 < argc) {
                num_smooth_per_comm = std::atoi(argv[++a]);
            }
        }
        if (num_comms < 1) num_comms = 1;
        if (num_smooth_per_comm < 1) num_smooth_per_comm = 1;
    }


    // Initial mesh built on rank zero
    swage::Mesh initial_mesh;
    MPICArrayKokkos<double> initial_node_coords;

    // Mesh partitioned by pt-scotch, including ghost
    swage::Mesh final_mesh;
    node_t final_node;
    MPICArrayKokkos<double> final_node_coords;

    GaussPoint_t gauss_point;

// ********************************************************
//              Build the initial mesh
// ********************************************************

    double t_init_mesh_start = MPI_Wtime();
    if (rank == 0) {
        std::cout<<"World size: "<<world_size<<std::endl;
        std::cout<<"Rank "<<rank<<" Building initial mesh"<<std::endl;
        std::cout<<"Mesh resolution: "<<num_elems_dim[0]<<" x "<<num_elems_dim[1]<<" x "<<num_elems_dim[2]<<std::endl;
        std::cout<<"Communication steps: "<<num_comms<<", smoothing passes/step: "<<num_smooth_per_comm<<std::endl;

        std::cout<<"Initializing mesh"<<std::endl;
        initial_mesh.num_dims = num_dims;
        initial_mesh.Pn = Pn_order;

        build_3d_box(initial_mesh,  initial_node_coords, origin, length, num_elems_dim, Pn_order);

        // Morph the inital mesh to show curvature
        bool morph_mesh = false;
        if(morph_mesh && Pn_order > 1) {

            FOR_ALL(i, 0, initial_mesh.num_nodes, {
                initial_node_coords(i, 0) += 0.0; //0.02 * std::sin(10.0 * initial_node_coords(i, 0));
                initial_node_coords(i, 1) += 0.03 * ( std::sin(10* initial_node_coords(i, 0)) + std::sin(16 * initial_node_coords(i, 0)));
                initial_node_coords(i, 2) += 0.05 * std::sin(12 * std::sqrt(initial_node_coords(i, 0)*initial_node_coords(i, 0) + initial_node_coords(i, 1)*initial_node_coords(i, 1)));
            });
            initial_node_coords.update_device();
        }

        double t_init_mesh_end = MPI_Wtime();
        std::cout << "Initial mesh build time: " << (t_init_mesh_end - t_init_mesh_start) << " seconds" << std::endl;
        std::cout << "Initial mesh has " << initial_mesh.num_elems << " elements and " << initial_mesh.num_nodes << " nodes" << std::endl;
        std::cout <<" Num_nodes_in_elem: " << initial_mesh.num_nodes_in_elem << std::endl;
        std::cout <<" Num_dims: " << initial_mesh.num_dims << std::endl;
        std::cout <<" Num_elems: " << initial_mesh.num_elems << std::endl;
        std::cout <<" Num_nodes: " << initial_mesh.num_nodes << std::endl;
        std::cout <<" Num_nodes_in_elem: " << initial_mesh.num_nodes_in_elem << std::endl;
        std::cout <<" Num_nodes_in_elem: " << initial_mesh.num_nodes_in_elem << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);


// ********************************************************
//             Partition and balance the mesh
// ********************************************************

    double t_partition_start = MPI_Wtime();
    // Create communicaion plans
    CommunicationPlan element_communication_plan;
    element_communication_plan.initialize(MPI_COMM_WORLD);
    CommunicationPlan node_communication_plan;
    node_communication_plan.initialize(MPI_COMM_WORLD);

    if(world_size != 1) { // pass through the partitioning function if not a single rank
        elements::partition_mesh(initial_mesh, final_mesh, initial_node_coords, final_node_coords, element_communication_plan, node_communication_plan, world_size, rank);

        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0) printf("After partitioning\n");

    } else {
        final_mesh = initial_mesh;
        final_mesh.num_owned_elems = initial_mesh.num_elems;
        final_mesh.num_owned_nodes = initial_mesh.num_nodes;
        final_node_coords = initial_node_coords;
        final_mesh.num_dims = num_dims;
        final_mesh.Pn = Pn_order;
    }
    double t_partition_end = MPI_Wtime();




    // Verify communicaiton plans
    // if(world_size != 1) element_communication_plan.verify_graph_communicator();
    // if(world_size != 1) node_communication_plan.verify_graph_communicator();

    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 0) {
        printf("Mesh partitioning time: %.2f seconds\n", t_partition_end - t_partition_start);
    }


    MPI_Barrier(MPI_COMM_WORLD);

    for (int r = 0; r < world_size; ++r) {
        if (rank == r) {
            std::cout << "-----------------------------" << std::endl;
            std::cout << "Rank " << rank << " mesh info:" << std::endl;
            std::cout << "Final mesh has " << final_mesh.num_elems << " elements and " << final_mesh.num_nodes << " nodes" << std::endl;
            std::cout << "Final mesh has " << final_mesh.num_owned_elems << " owned elements and " << final_mesh.num_owned_nodes << " owned nodes" << std::endl;
            std::cout << "Final mesh num_dims = " << final_mesh.num_dims << std::endl;
            std::cout << "-----------------------------" << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }




// ******************************************************************************************
//     Test element communication using MPI_Neighbor_alltoallv
// ******************************************************************************************
    // Gauss points share the same communication plan as elements.
    // This test initializes gauss point fields on owned elements and exchanges them with ghost elements.

    if(world_size != 1) {
        // Test that the shared_tally_owned_nodes mask works correctly by counting all nodes across all ranks and verifying that the number of unique nodes is equal to the number of owned nodes.
        int total_num_nodes = 0;
        int total_local_nodes = 0;
        int total_global_nodes = 0;


        FOR_REDUCE_SUM(node_gid, 0, final_mesh.num_owned_nodes, total_local_nodes, {

            if(final_mesh.shared_tally_owned_nodes(node_gid)){
                total_local_nodes++;
            }

        }, total_num_nodes);
        MATAR_FENCE();

        MPI_Allreduce(&total_num_nodes, &total_global_nodes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        if(rank == 0){
            std::cout<<"Total number of nodes: "<<total_global_nodes<<std::endl;
            std::cout<<"Error in node count = "<<total_global_nodes - initial_mesh.num_nodes<<std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout<<"Generating phony data for gauss points to test MPI communications for the gauss point field communication"<<std::endl;

    std::vector<gauss_pt_state> gauss_pt_states = {gauss_pt_state::fields, gauss_pt_state::fields_vec};
    gauss_point.initialize(final_mesh.num_elems, final_mesh.num_dims, gauss_pt_states, element_communication_plan); // , &element_communication_plan

    // Initialize the gauss point fields on each rank
    // Set owned elements to rank number, ghost elements to -1 (to verify communication)
    for (int i = 0; i < final_mesh.num_owned_elems; i++) {
        gauss_point.fields.host(i) = static_cast<double>(rank);
        for(int dim = 0; dim < final_mesh.num_dims; dim++){
            gauss_point.fields_vec.host(i, dim) = static_cast<double>(rank);
        }
    }
    for (int i = final_mesh.num_owned_elems; i < final_mesh.num_elems; i++) {
        gauss_point.fields.host(i) = -1.0;  // Ghost elements should be updated
        for(int dim = 0; dim < final_mesh.num_dims; dim++){
            gauss_point.fields_vec.host(i, dim) = -100.0;
        }
    }
    gauss_point.fields.update_device();
    gauss_point.fields_vec.update_device();

    MPI_Barrier(MPI_COMM_WORLD);

    // Repeat num_comms times: run num_smooth_per_comm local self+neighbor averaging
    // passes (new owned-element data each pass, using whatever ghost data is
    // currently on hand), then communicate() *each field separately* -- with its
    // own MPI_Barrier immediately beforehand -- to refresh ghost data from
    // neighbors before the next round. Barrier-isolating each field's
    // communicate() call means one field's measurement can never be skewed by
    // rank skew left over from the other field's call. Every communicate() call
    // after the very first is exchanging genuinely new data, not re-sending the
    // same bytes.
    CArrayKokkos <double> tmp(final_mesh.num_elems);

    auto gauss_smooth_pass = [&]() {
        // Self + neighbor average over the element adjacency list
        FOR_ALL(i, 0, final_mesh.num_elems, {
            double value = gauss_point.fields(i);
            for (int j = 0; j < final_mesh.num_elems_in_elem(i); j++) {
                value += gauss_point.fields(final_mesh.elems_in_elem(i, j));
            }
            value /= final_mesh.num_elems_in_elem(i) + 1;
            tmp(i) = value;
        });
        MATAR_FENCE();

        FOR_ALL(i, 0, final_mesh.num_elems, {
            gauss_point.fields(i) = tmp(i);
            for(int dim = 0; dim < final_mesh.num_dims; dim++){
                gauss_point.fields_vec(i, dim) = tmp(i);
            }
        });
        MATAR_FENCE();
    };

    // Untimed warm-up round: absorbs first-touch page faults on the MPI send/recv
    // buffers and any cache/TLB disruption left over from the mesh build + the
    // verbose rank-0-only I/O just before this, so that one-time cost isn't
    // misattributed to the first *timed* communicate() call below (same
    // rationale as Test1's warm-up iterations).
    for (int smooth = 0; smooth < num_smooth_per_comm; smooth++) gauss_smooth_pass();
    MPI_Barrier(MPI_COMM_WORLD);
    gauss_point.fields.communicate();
    MPI_Barrier(MPI_COMM_WORLD);
    gauss_point.fields_vec.communicate();

    std::vector<double> gauss_fields_times(num_comms, 0.0);
    std::vector<double> gauss_fields_vec_times(num_comms, 0.0);

    for (int c = 0; c < num_comms; c++) {

        for (int smooth = 0; smooth < num_smooth_per_comm; smooth++) gauss_smooth_pass();

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        gauss_point.fields.communicate();
        double t1 = MPI_Wtime();
        gauss_fields_times[c] = t1 - t0;

        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        gauss_point.fields_vec.communicate();
        t1 = MPI_Wtime();
        gauss_fields_vec_times[c] = t1 - t0;
    }

    gauss_point.fields.update_host();
    gauss_point.fields_vec.update_host();

    // Print per-rank gauss point communication timing statistics over the num_comms steps
    {
        double gf_min, gf_avg, gf_max, gf_total;
        double gfv_min, gfv_avg, gfv_max, gfv_total;
        time_series_stats(gauss_fields_times, gf_min, gf_avg, gf_max, gf_total);
        time_series_stats(gauss_fields_vec_times, gfv_min, gfv_avg, gfv_max, gfv_total);

        for (int r = 0; r < world_size; ++r) {
            if (rank == r) {
                std::cout << "Rank " << rank << " gauss point fields communication time over " << num_comms
                          << " step(s): min=" << gf_min << " avg=" << gf_avg << " max=" << gf_max << " seconds" << std::endl;
                std::cout << "Rank " << rank << " gauss point fields_vec communication time over " << num_comms
                          << " step(s): min=" << gfv_min << " avg=" << gfv_avg << " max=" << gfv_max << " seconds" << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout<<"Generating phony data for nodes to test MPI communications for the nodal fields"<<std::endl;

    // Test node communication using MPI_Neighbor_alltoallv
    std::vector<node_state> node_states = {node_state::coords, node_state::scalar_field, node_state::vector_field};
    final_node.initialize(final_mesh.num_nodes, final_mesh.num_dims, node_states, node_communication_plan);


    // Copy the final node coordinates to the final node
    final_node.coords = final_node_coords;
    final_node.coords.update_device();

    for (int i = 0; i < final_mesh.num_owned_nodes; i++) {
        final_node.scalar_field.host(i) = static_cast<double>(rank);
        for(int dim = 0; dim < final_mesh.num_dims; dim++){
            final_node.vector_field.host(i, dim) = static_cast<double>(rank);
        }
    }
    for (int i = final_mesh.num_owned_nodes; i < final_mesh.num_nodes; i++) {
        final_node.scalar_field.host(i) = -100.0;
        for(int dim = 0; dim < final_mesh.num_dims; dim++){
            final_node.vector_field.host(i, dim) = -100;
        }
    }

    final_node.coords.update_device();
    final_node.scalar_field.update_device();
    final_node.vector_field.update_device();
    MATAR_FENCE();
    MPI_Barrier(MPI_COMM_WORLD);

    // Same repeated smooth-then-communicate pattern as the gauss point fields
    // above, including the untimed warm-up round and per-field barrier isolation.
    DCArrayKokkos <double> tmp_too(final_mesh.num_nodes);

    auto node_smooth_pass = [&]() {
        FOR_ALL(i, 0, final_mesh.num_nodes, {

            double value = final_node.scalar_field(i);
            for(int j = 0; j < final_mesh.num_nodes_in_node(i); j++){
                value += final_node.scalar_field(final_mesh.nodes_in_node(i, j));
            }
            value /= final_mesh.num_nodes_in_node(i) + 1;
            tmp_too(i) = value;
        });
        MATAR_FENCE();

        FOR_ALL(i, 0, final_mesh.num_nodes, {
            final_node.scalar_field(i) = tmp_too(i);
            for(int dim = 0; dim < final_mesh.num_dims; dim++){
                final_node.vector_field(i, dim) = tmp_too(i);
            }
        });
        MATAR_FENCE();
    };

    for (int smooth = 0; smooth < num_smooth_per_comm; smooth++) node_smooth_pass();
    MPI_Barrier(MPI_COMM_WORLD);
    final_node.scalar_field.communicate();
    MPI_Barrier(MPI_COMM_WORLD);
    final_node.vector_field.communicate();

    std::vector<double> node_scalar_times(num_comms, 0.0);
    std::vector<double> node_vector_times(num_comms, 0.0);

    for (int c = 0; c < num_comms; c++) {

        for (int smooth = 0; smooth < num_smooth_per_comm; smooth++) node_smooth_pass();

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        final_node.scalar_field.communicate();
        double t1 = MPI_Wtime();
        node_scalar_times[c] = t1 - t0;

        MPI_Barrier(MPI_COMM_WORLD);
        t0 = MPI_Wtime();
        final_node.vector_field.communicate();
        t1 = MPI_Wtime();
        node_vector_times[c] = t1 - t0;
    }

    final_node.scalar_field.update_host();
    final_node.vector_field.update_host();

    // Print per-rank nodal field communication timing statistics over the num_comms steps
    {
        double ns_min, ns_avg, ns_max, ns_total;
        double nv_min, nv_avg, nv_max, nv_total;
        time_series_stats(node_scalar_times, ns_min, ns_avg, ns_max, ns_total);
        time_series_stats(node_vector_times, nv_min, nv_avg, nv_max, nv_total);

        for (int r = 0; r < world_size; ++r) {
            if (rank == r) {
                std::cout << "Rank " << rank << " nodal scalar_field communication time over " << num_comms
                          << " step(s): min=" << ns_min << " avg=" << ns_avg << " max=" << ns_max << " seconds" << std::endl;
                std::cout << "Rank " << rank << " nodal vector_field communication time over " << num_comms
                          << " step(s): min=" << nv_min << " avg=" << nv_avg << " max=" << nv_max << " seconds" << std::endl;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }

    MATAR_FENCE();
    MPI_Barrier(MPI_COMM_WORLD);

// ******************************************************************************************
//     Report the volume of data transferred between ranks
// ******************************************************************************************
    // Bytes per exchanged item = (components per item) x sizeof(double), with the component count
    // taken from MATAR metadata as size()/dims(0). Per-neighbor item counts come from the shared
    // CommunicationPlan. Gauss fields share the element plan; node fields share the node plan.

    if(world_size != 1) {

        auto item_bytes = [](MPICArrayKokkos<double>& f){ return (f.size() / f.dims(0)) * sizeof(double); };

        CommunicationPlan& elem_plan = *gauss_point.fields.comm_plan_;
        CommunicationPlan& node_plan = *final_node.scalar_field.comm_plan_;
        size_t elem_item_bytes = item_bytes(gauss_point.fields) + item_bytes(gauss_point.fields_vec);
        size_t node_item_bytes = item_bytes(final_node.scalar_field) + item_bytes(final_node.vector_field);

        // Print the per-neighbor send/recv volume for one plan on the current rank.
        auto print_plan = [](const char* name, CommunicationPlan& p, size_t ib){
            std::cout << "  " << name << ":" << std::endl;
            for (int i = 0; i < p.num_send_ranks; ++i)
                std::cout << "    -> rank " << p.send_rank_ids.host(i) << ": " << p.send_counts_.host(i) * ib << " bytes" << std::endl;
            for (int i = 0; i < p.num_recv_ranks; ++i)
                std::cout << "    <- rank " << p.recv_rank_ids.host(i) << ": " << p.recv_counts_.host(i) * ib << " bytes" << std::endl;
        };

        unsigned long long send_bytes = (unsigned long long)elem_plan.total_send_count * elem_item_bytes
                                      + (unsigned long long)node_plan.total_send_count * node_item_bytes;
        unsigned long long recv_bytes = (unsigned long long)elem_plan.total_recv_count * elem_item_bytes
                                      + (unsigned long long)node_plan.total_recv_count * node_item_bytes;

        for (int r = 0; r < world_size; ++r) {
            if (rank == r) {
                std::cout << "Rank " << rank << " communication volume (sent " << send_bytes
                          << " bytes, received " << recv_bytes << " bytes):" << std::endl;
                print_plan("element (gauss) fields", elem_plan, elem_item_bytes);
                print_plan("node fields", node_plan, node_item_bytes);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

        // Total volume moved = sum of bytes sent by every rank.
        unsigned long long global_bytes = 0;
        MPI_Reduce(&send_bytes, &global_bytes, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::cout << "Total data volume exchanged across all ranks: " << global_bytes
                      << " bytes (" << (static_cast<double>(global_bytes) / (1024.0 * 1024.0)) << " MB)" << std::endl;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0) std::cout<<"Writing VTU file for rank "<<rank<<std::endl;
    write_vtu(final_mesh, final_node, gauss_point, rank, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // Stop timer and get execution time
    double t_main_end = MPI_Wtime();

    if(rank == 0) {
        printf("Total execution time: %.2f seconds\n", t_main_end - t_main_start);
    }

// ******************************************************************************************
//     Write a combined per-rank, per-field CSV log for the scaling study
// ******************************************************************************************
    // One row per (rank, field): gauss_fields, gauss_fields_vec, node_scalar_field,
    // node_vector_field, overall_communication (elementwise sum of the four per-step
    // series above, so its own min/max reflect true worst/best combined steps rather
    // than a sum of separately-computed mins/maxes), and total_execution_time.
    // Written once by rank 0 after MPI_Gather-ing every rank's stats, into a file
    // whose name encodes mesh size, rank count, comm steps, and smoothing steps so
    // sweeping any of those parameters never overwrites another run's log.
    {
        constexpr int NUM_METRICS = 6; // gauss_fields, gauss_fields_vec, node_scalar_field, node_vector_field, overall, total_execution_time
        constexpr int STATS_PER_METRIC = 4; // min, avg, max, total
        constexpr int NUM_LOCAL_VALUES = NUM_METRICS * STATS_PER_METRIC;

        double gf_min, gf_avg, gf_max, gf_total;
        double gfv_min, gfv_avg, gfv_max, gfv_total;
        double ns_min, ns_avg, ns_max, ns_total;
        double nv_min, nv_avg, nv_max, nv_total;
        time_series_stats(gauss_fields_times, gf_min, gf_avg, gf_max, gf_total);
        time_series_stats(gauss_fields_vec_times, gfv_min, gfv_avg, gfv_max, gfv_total);
        time_series_stats(node_scalar_times, ns_min, ns_avg, ns_max, ns_total);
        time_series_stats(node_vector_times, nv_min, nv_avg, nv_max, nv_total);

        std::vector<double> overall_times(num_comms, 0.0);
        for (int c = 0; c < num_comms; c++) {
            overall_times[c] = gauss_fields_times[c] + gauss_fields_vec_times[c]
                              + node_scalar_times[c] + node_vector_times[c];
        }
        double ov_min, ov_avg, ov_max, ov_total;
        time_series_stats(overall_times, ov_min, ov_avg, ov_max, ov_total);

        double total_execution_time = t_main_end - t_main_start;

        double local_values[NUM_LOCAL_VALUES] = {
            gf_min,  gf_avg,  gf_max,  gf_total,
            gfv_min, gfv_avg, gfv_max, gfv_total,
            ns_min,  ns_avg,  ns_max,  ns_total,
            nv_min,  nv_avg,  nv_max,  nv_total,
            ov_min,  ov_avg,  ov_max,  ov_total,
            total_execution_time, total_execution_time, total_execution_time, total_execution_time
        };

        std::vector<double> gathered;
        if (rank == 0) gathered.resize(static_cast<size_t>(NUM_LOCAL_VALUES) * world_size);
        MPI_Gather(local_values, NUM_LOCAL_VALUES, MPI_DOUBLE,
                   rank == 0 ? gathered.data() : nullptr, NUM_LOCAL_VALUES, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);

        if (rank == 0) {
            char csv_filename[256];
            std::snprintf(csv_filename, sizeof(csv_filename),
                          "test2_results_mesh%dx%dx%d_np%d_comms%d_smooth%d.csv",
                          num_elems_dim[0], num_elems_dim[1], num_elems_dim[2],
                          world_size, num_comms, num_smooth_per_comm);

            std::ofstream csv(csv_filename);
            csv << "mesh_nx,mesh_ny,mesh_nz,world_size,num_comms,num_smooth_per_comm,rank,field,min_s,avg_s,max_s,total_s\n";

            const char* metric_names[NUM_METRICS] = {
                "gauss_fields", "gauss_fields_vec", "node_scalar_field",
                "node_vector_field", "overall_communication", "total_execution_time"
            };

            for (int r = 0; r < world_size; r++) {
                const double* row = &gathered[static_cast<size_t>(r) * NUM_LOCAL_VALUES];
                for (int m = 0; m < NUM_METRICS; m++) {
                    csv << num_elems_dim[0] << "," << num_elems_dim[1] << "," << num_elems_dim[2] << ","
                        << world_size << "," << num_comms << "," << num_smooth_per_comm << ","
                        << r << "," << metric_names[m] << ","
                        << row[m * STATS_PER_METRIC + 0] << "," << row[m * STATS_PER_METRIC + 1] << ","
                        << row[m * STATS_PER_METRIC + 2] << "," << row[m * STATS_PER_METRIC + 3] << "\n";
                }
            }
            csv.close();
            std::cout << "Wrote scaling-study CSV log: " << csv_filename << std::endl;
        }
    }

    } // end MATAR scope
    MATAR_FINALIZE();
    MPI_Finalize();

    return 0;
}
