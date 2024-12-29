#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <cxxmpi/cart_comm.hpp>
#include <cxxmpi/comm.hpp>
#include <cxxmpi/error.hpp>

// NOLINTNEXTLINE
TEST_CASE("Cartesian Communicator Basic Test", "[cart][mpi]") {
  using namespace cxxmpi;  // NOLINT

  SECTION("2D Grid Creation and Properties") {
    if (comm_world().size() != 4) {
      SKIP("This test requires exactly 4 processes");
    }

    auto cart = cart_comm(comm_world(), {2, 2}, {true, true}, false);

    REQUIRE(cart.size() == 4);
    REQUIRE(cart.ndims() == 2);

    auto rank = cart.rank();
    auto coords = cart.coords();

    switch (rank) {  // NOLINT
      case 0:
        REQUIRE(coords == std::vector{0, 0});
        REQUIRE(cart.rank({0, 0}) == 0);
        break;
      case 1:
        REQUIRE(coords == std::vector{0, 1});
        REQUIRE(cart.rank({0, 1}) == 1);
        break;
      case 2:
        REQUIRE(coords == std::vector{1, 0});
        REQUIRE(cart.rank({1, 0}) == 2);
        break;
      case 3:
        REQUIRE(coords == std::vector{1, 1});
        REQUIRE(cart.rank({1, 1}) == 3);
        break;
    }
  }

  SECTION("Conversion between cart_comm and weak_cart_comm") {
    if (comm_world().size() != 2) {
      SKIP("This test requires exactly 2 processes");
    }
    auto cart = cart_comm(comm_world(), {2, 1}, {false, false}, false);
    auto const weak_cart = weak_cart_comm{cart};
    REQUIRE(weak_cart.native() == cart.native());
  }

  SECTION("2D Grid Neighbors") {
    if (comm_world().size() != 4) {
      SKIP("This test requires exactly 4 processes");
    }

    auto cart = cart_comm(comm_world(), {2, 2}, {true, true}, false);
    auto neighbors = cart.neighbors_2d();
    auto rank = cart.rank();

    switch (rank) {                     // NOLINT
      case 0:                           // (0,0)
        REQUIRE(neighbors.up == 2);     // (1,0)
        REQUIRE(neighbors.down == 2);   // (1,0)
        REQUIRE(neighbors.left == 1);   // (0,1)
        REQUIRE(neighbors.right == 1);  // (0,1)
        break;
      case 1:  // (0,1)
        REQUIRE(neighbors.up == 3);
        REQUIRE(neighbors.down == 3);
        REQUIRE(neighbors.left == 0);
        REQUIRE(neighbors.right == 0);
        break;
      case 2:  // (1,0)
        REQUIRE(neighbors.up == 0);
        REQUIRE(neighbors.down == 0);
        REQUIRE(neighbors.left == 3);
        REQUIRE(neighbors.right == 3);
        break;
      case 3:  // (1,1)
        REQUIRE(neighbors.up == 1);
        REQUIRE(neighbors.down == 1);
        REQUIRE(neighbors.left == 2);
        REQUIRE(neighbors.right == 2);
        break;
    }
  }
}

TEST_CASE("Cartesian Communicator Error Handling", "[cart][mpi][error]") {
  using namespace cxxmpi;  // NOLINT

  SECTION("Mismatched dims and periods size") {
    REQUIRE_THROWS_AS(cart_comm(comm_world(), {2, 2}, {true}, false),
                      std::invalid_argument);
  }

  SECTION("Too many processes requested") {
    auto const world_size = comm_world().size();
    // Request more processes than available
    REQUIRE_THROWS_AS(cart_comm(comm_world(), {world_size + 1}, {true}, false),
                      mpi_error);
  }
}
