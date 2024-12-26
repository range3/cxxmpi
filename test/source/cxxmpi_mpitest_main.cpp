#include <mpi.h>

#define CATCH_CONFIG_RUNNER
#include <exception>
#include <iostream>

#include <catch2/catch_session.hpp>
#include <cxxmpi/universe.hpp>

auto main(int argc, char* argv[]) -> int {
  try {
    auto const universe = cxxmpi::universe(argc, argv, MPI_THREAD_MULTIPLE);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    // Run tests
    const int result = Catch::Session().run(argc, argv);

    // Ensure all processes return the same result
    int global_result = 0;
    MPI_Allreduce(&result, &global_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    return global_result;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}
