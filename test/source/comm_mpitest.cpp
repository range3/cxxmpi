#include <array>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cxxmpi/comm.hpp>
#include <cxxmpi/error.hpp>
#include <cxxmpi/request.hpp>
#include <mpi.h>

#include "cxxmpi/dtype.hpp"

TEST_CASE("Basic comm constructor operations", "[mpi][comm][constructor]") {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  SECTION("Default constructor behavior") {
    auto weak = cxxmpi::basic_comm<cxxmpi::weak_comm_handle>{};
    CHECK(weak.rank() == 0);
    CHECK(weak.size() == 0);
    CHECK(weak.native() == MPI_COMM_NULL);

    auto managed = cxxmpi::basic_comm<cxxmpi::comm_handle>{};
    CHECK(managed.rank() == 0);
    CHECK(managed.size() == 0);
    CHECK(managed.native() == MPI_COMM_NULL);
  }

  SECTION("Construction from MPI_COMM_WORLD") {
    const auto& world = cxxmpi::comm_world();
    CHECK(world.rank() == rank);
    CHECK(world.size() == static_cast<size_t>(size));
    CHECK(world.native() == MPI_COMM_WORLD);
  }

  SECTION("Copy construction of weak_comm") {
    const auto& world = cxxmpi::comm_world();
    auto copied = world;  // Copy constructor

    // Verify copied properties match original
    CHECK(copied.rank() == world.rank());
    CHECK(copied.size() == world.size());
    CHECK(copied.native() == world.native());
  }

  SECTION("Move construction of managed_comm") {
    // Create a managed comm via split
    const auto& world = cxxmpi::comm_world();
    auto original = cxxmpi::basic_comm<cxxmpi::comm_handle>{world, 0};
    auto original_rank = original.rank();
    auto original_size = original.size();
    MPI_Comm original_comm = original.native();

    // Move construct
    auto moved = std::move(original);

    // Verify moved properties
    CHECK(moved.rank() == original_rank);
    CHECK(moved.size() == original_size);
    CHECK(moved.native() == original_comm);
  }

  SECTION("Copy assignment of weak_comm") {
    const auto& world = cxxmpi::comm_world();
    auto copied = cxxmpi::basic_comm<cxxmpi::weak_comm_handle>{};
    copied = world;  // Copy assignment

    // Verify assigned properties
    CHECK(copied.rank() == world.rank());
    CHECK(copied.size() == world.size());
    CHECK(copied.native() == world.native());
  }

  SECTION("Move assignment of managed_comm") {
    const auto& world = cxxmpi::comm_world();
    auto original = cxxmpi::basic_comm<cxxmpi::comm_handle>{world, 0};
    auto target = cxxmpi::basic_comm<cxxmpi::comm_handle>{};

    auto original_rank = original.rank();
    auto original_size = original.size();
    MPI_Comm original_comm = original.native();

    // Move assign
    target = std::move(original);

    // Verify moved properties
    CHECK(target.rank() == original_rank);
    CHECK(target.size() == original_size);
    CHECK(target.native() == original_comm);
  }

  SECTION("Verify RAII behavior") {
    MPI_Comm tracked_comm = MPI_COMM_NULL;
    {
      const auto& world = cxxmpi::comm_world();
      auto managed = cxxmpi::basic_comm<cxxmpi::comm_handle>{world, 0};
      tracked_comm = managed.native();
    }
    // After destruction, the communicator should be freed
    auto const result = MPI_Comm_compare(tracked_comm, MPI_COMM_NULL, nullptr);
    CHECK(result == MPI_ERR_COMM);
  }
}

TEST_CASE("Comm constructor error handling",
          "[mpi][comm][constructor][error]") {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  SECTION("Construction with invalid comm handle") {
    auto invalid_handle = cxxmpi::weak_comm_handle{MPI_COMM_NULL};
    CHECK_THROWS_AS(
        cxxmpi::basic_comm<cxxmpi::weak_comm_handle>{invalid_handle},
        cxxmpi::mpi_error);
  }
}

// NOLINTNEXTLINE
TEST_CASE("Basic Communication Tests", "[mpi]") {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    SKIP("This test requires at least 2 processes");
  }

  const auto& comm = cxxmpi::comm_world();

  SECTION("Basic rank and size") {
    REQUIRE(comm.rank() == static_cast<size_t>(rank));
    REQUIRE(comm.size() == static_cast<size_t>(size));
  }

  SECTION("Single value send/recv") {
    if (rank == 0) {
      const int send_val = 42;
      comm.send(send_val, 1);
    } else if (rank == 1) {
      int recv_val = 0;
      auto st = comm.recv(recv_val, 0);
      REQUIRE(st.source() == 0);
      REQUIRE(recv_val == 42);
    }
  }

  SECTION("Array send/recv with status") {
    if (rank == 0) {
      std::array<double, 3> send_data = {1.0, 2.0, 3.0};
      comm.send(std::span<const double>{send_data}, 1);
    } else if (rank == 1) {
      std::array<double, 3> recv_data = {};
      auto st = comm.recv(std::span{recv_data}, 0);
      REQUIRE(st.source() == 0);
      REQUIRE(recv_data == std::array{1.0, 2.0, 3.0});
      REQUIRE(st.count<double>() == 3);
    }
  }

  SECTION("Vector send/recv without status") {
    if (rank == 0) {
      std::vector<int> send_data = {1, 2, 3, 4, 5};
      comm.send(std::span<const int>{send_data}, 1);
    } else if (rank == 1) {
      std::vector<int> recv_data(5);
      comm.recv_without_status(std::span{recv_data}, 0);
      REQUIRE(recv_data == std::vector{1, 2, 3, 4, 5});
    }
  }
}

// NOLINTNEXTLINE
TEST_CASE("Custom Datatype Communication Tests", "[mpi]") {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    SKIP("This test requires at least 2 processes");
  }

  const auto& comm = cxxmpi::comm_world();

  SECTION("Vector datatype") {
    // Create a datatype for pairs of integers
    auto vector_type = cxxmpi::dtype{cxxmpi::as_weak_dtype<int>(), 2, 1, 1};
    vector_type.commit();

    if (rank == 0) {
      const std::array<int, 4> send_data = {1, 2, 3, 4};  // 2 pairs
      comm.send(std::span<const int>{send_data},
                cxxmpi::weak_dtype{vector_type}, 2, 1);
    } else if (rank == 1) {
      std::array<int, 4> recv_data = {};
      auto st = comm.recv(std::span<int>{recv_data},
                          cxxmpi::weak_dtype{vector_type}, 2, 0);
      REQUIRE(st.source() == 0);
      REQUIRE(recv_data == std::array{1, 2, 3, 4});
      REQUIRE(st.count(cxxmpi::weak_dtype{vector_type}) == 2);  // 2 pairs
    }
  }
}

// NOLINTNEXTLINE
TEST_CASE("Non-blocking Communication Tests", "[mpi]") {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    SKIP("This test requires at least 2 processes");
  }

  const auto& comm = cxxmpi::comm_world();

  SECTION("Single request send/recv") {
    if (rank == 0) {
      const int send_val = 42;
      MPI_Request request = MPI_REQUEST_NULL;
      comm.isend(send_val, 1, 0, request);

      MPI_Wait(&request, MPI_STATUS_IGNORE);
    } else if (rank == 1) {
      int recv_val = 0;
      MPI_Request request = MPI_REQUEST_NULL;
      comm.irecv(recv_val, 0, 0, request);

      cxxmpi::status st;
      MPI_Wait(&request, &st.native());
      REQUIRE(recv_val == 42);
      REQUIRE(st.source() == 0);
    }
  }

  SECTION("Multiple requests with request_group") {
    if (rank == 0) {
      std::array<int, 3> send_data = {1, 2, 3};
      cxxmpi::request_group requests;

      // Send data to all other processes
      for (int i = 1; i < size; ++i) {
        comm.isend(std::span<const int>{send_data}, i, 0, requests.add());
      }

      requests.wait_all_without_status();
    } else {
      std::array<int, 3> recv_data = {};
      cxxmpi::request_group requests;

      MPI_Request& req = requests.add();
      comm.irecv(std::span{recv_data}, 0, 0, req);

      auto statuses = requests.wait_all();
      REQUIRE(statuses.size() == 1);
      REQUIRE(statuses[0].source() == 0);
      REQUIRE(recv_data == std::array{1, 2, 3});
    }
  }
}

// NOLINTNEXTLINE
TEST_CASE("Error Handling Tests", "[mpi]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto& comm = cxxmpi::comm_world();

  SECTION("Invalid rank send") {
    if (rank == 0) {
      const int val = 42;
      REQUIRE_THROWS_AS(comm.send(val, static_cast<int>(comm.size())),
                        cxxmpi::mpi_error);
    }
  }
}
