#include <stdexcept>

#include <catch2/catch_test_macros.hpp>
#include <cxxmpi/cart_comm.hpp>
#include <cxxmpi/comm.hpp>
#include <cxxmpi/error.hpp>

TEST_CASE("Cartesian Communicator Basic Test", "[cart][mpi]") {
  using namespace cxxmpi;  // NOLINT

  SECTION("2D Grid Creation") {
    if (comm_world().size() != 4) {
      SKIP("This test requires exactly 4 processes");
    }

    // Create 2x2 grid topology
    auto cart = cart_comm(comm_world(), {2, 2}, {true, false}, false);
    REQUIRE(cart.size() == 4);
  }

  SECTION("Conversion between cart_comm and weak_cart_comm") {
    if (comm_world().size() != 2) {
      SKIP("This test requires exactly 2 processes");
    }
    auto cart = cart_comm(comm_world(), {2, 1}, {false, false}, false);
    auto const weak_cart = weak_cart_comm{cart};
    REQUIRE(weak_cart.native() == cart.native());
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
