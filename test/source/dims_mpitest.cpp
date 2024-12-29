#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <cxxmpi/dims.hpp>
#include <mpi.h>

// NOLINTNEXTLINE
TEST_CASE("create_dims basic functionality", "[mpi][dims]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {  // Only test on rank 0
    SECTION("create with size") {
      auto dims = cxxmpi::create_dims(6, 2);  // 6 processes in 2D
      REQUIRE(dims.size() == 2);

      // MPI_Dims_create should divide 6 processes into 2x3 or 3x2
      const std::vector<int> expected1{2, 3};
      const std::vector<int> expected2{3, 2};
      CHECK_THAT(dims, Catch::Matchers::Equals(expected1)
                           || Catch::Matchers::Equals(expected2));
    }

    SECTION("create with existing dimensions") {
      std::vector<int> init{0, 2, 0};  // Fix middle dimension to 2
      auto dims = cxxmpi::create_dims(12, std::span{init});
      REQUIRE(dims.size() == 3);

      // Middle dimension should remain 2
      REQUIRE(dims[1] == 2);

      // Total product should equal number of processes
      int product = 1;
      for (auto const dim : dims) {
        product *= dim;
      }
      REQUIRE(product == 12);
    }

    SECTION("create with initializer list") {
      auto dims = cxxmpi::create_dims(8, {2, 2, 2});  // 8 processes in 3D
      REQUIRE(dims.size() == 3);

      // MPI_Dims_create should divide 8 processes into 2x2x2
      const std::vector<int> expected{2, 2, 2};
      CHECK_THAT(dims, Catch::Matchers::Equals(expected));
    }

    SECTION("zero dimensions throws") {
      REQUIRE_THROWS_AS(cxxmpi::create_dims(4, 0), std::invalid_argument);
    }

    SECTION("empty dimensions array throws") {
      std::vector<int> empty;
      REQUIRE_THROWS_AS(cxxmpi::create_dims(4, std::span{empty}),
                        std::invalid_argument);
    }

    SECTION("prime number of processes") {
      // For 7 processes in 2D, one dimension must be 7
      auto dims = cxxmpi::create_dims(7, 2);
      CHECK_THAT(dims, Catch::Matchers::Contains(std::vector{7}));
    }

    SECTION("perfect square processes") {
      // For 9 processes in 2D, should be 3x3
      auto dims = cxxmpi::create_dims(9, 2);
      auto const expected = std::vector<int>{3, 3};
      CHECK_THAT(dims, Catch::Matchers::Equals(expected));
    }
  }
}
