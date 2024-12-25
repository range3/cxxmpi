#include <mpi.h>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>

auto main(int argc, char* argv[]) -> int {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

  // Run tests
  const int result = Catch::Session().run(argc, argv);

  // Ensure all processes return the same result
  int global_result = 0;
  MPI_Allreduce(&result, &global_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

  MPI_Finalize();

  return global_result;
}
