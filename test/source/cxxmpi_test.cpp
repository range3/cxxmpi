#include "cxxmpi/cxxmpi.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Name is cxxmpi", "[library]")
{
  REQUIRE(name() == "cxxmpi");
}
