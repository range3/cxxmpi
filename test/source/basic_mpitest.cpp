#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <mpi.h>

TEST_CASE("Basic MPI Test", "[mpi]") {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  INFO("Rank: " << rank << ", Size: " << size);

  SECTION("Point-to-Point Communication") {
    if (size < 2) {
      SKIP("This test requires at least 2 processes");
    }

    if (rank == 0) {
      int send_data = 42;
      MPI_Send(&send_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
      int recv_data = 0;
      MPI_Status status;
      MPI_Recv(&recv_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
      REQUIRE(recv_data == 42);
    }
  }
}
