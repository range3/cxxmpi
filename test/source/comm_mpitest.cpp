#include <cstddef>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <cxxmpi/comm.hpp>
#include <cxxmpi/error.hpp>
#include <mpi.h>

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
    CHECK(world.rank() == static_cast<size_t>(rank));
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
