#include <system_error>

#include <catch2/catch_test_macros.hpp>
#include <cxxmpi/error.hpp>
#include <mpi.h>

// NOLINTNEXTLINE
TEST_CASE("MPI error handling basic features", "[error]") {
  SECTION("error category basics") {
    const auto& cat = cxxmpi::error_category::instance();
    CHECK(cat.name() == std::string{"cxxmpi"});
    CHECK(&cat == &cxxmpi::error_category::instance());
  }

  SECTION("error code creation") {
    auto ec = cxxmpi::make_error_code(MPI_ERR_BUFFER);
    CHECK(ec.category() == cxxmpi::error_category::instance());
    CHECK(ec.value() == MPI_ERR_BUFFER);

    ec = cxxmpi::make_error_code(MPI_SUCCESS);
    CHECK_FALSE(ec);  // Success codes should return false
  }

  SECTION("mpi_error exception") {
    try {
      throw cxxmpi::mpi_error(MPI_ERR_BUFFER);
      FAIL("Exception was not thrown");
    } catch (const cxxmpi::mpi_error& e) {
      const auto& ec = e.code();
      CHECK(ec.value() == MPI_ERR_BUFFER);
      CHECK(ec.category() == cxxmpi::error_category::instance());

      const std::string what = e.what();
      CHECK(what.find("error_mpitest.cpp") != std::string::npos);
    }
  }

  SECTION("check_mpi_result functionality") {
    CHECK_NOTHROW(cxxmpi::check_mpi_result(MPI_SUCCESS));
    CHECK_THROWS_AS(cxxmpi::check_mpi_result(MPI_ERR_BUFFER),
                    cxxmpi::mpi_error);

    try {
      cxxmpi::check_mpi_result(MPI_ERR_BUFFER);
      FAIL("Exception was not thrown");
    } catch (const cxxmpi::mpi_error& e) {
      CHECK(e.code().value() == MPI_ERR_BUFFER);
      const std::string what = e.what();
      CHECK(what.find("error_mpitest.cpp") != std::string::npos);
    }
  }
}

TEST_CASE("MPI error code to error condition mapping", "[error]") {
  const auto& cat = cxxmpi::error_category::instance();

  SECTION("memory errors") {
    CHECK(cat.default_error_condition(MPI_ERR_NO_MEM)
          == std::errc::not_enough_memory);
  }

  SECTION("buffer errors") {
    CHECK(cat.default_error_condition(MPI_ERR_BUFFER)
          == std::errc::no_buffer_space);
  }

  SECTION("file system errors") {
    CHECK(cat.default_error_condition(MPI_ERR_NO_SPACE)
          == std::errc::no_space_on_device);
    CHECK(cat.default_error_condition(MPI_ERR_FILE_EXISTS)
          == std::errc::file_exists);
    CHECK(cat.default_error_condition(MPI_ERR_NO_SUCH_FILE)
          == std::errc::no_such_file_or_directory);
    CHECK(cat.default_error_condition(MPI_ERR_READ_ONLY)
          == std::errc::read_only_file_system);
  }

  SECTION("I/O errors") {
    CHECK(cat.default_error_condition(MPI_ERR_IO) == std::errc::io_error);
  }

  SECTION("access errors") {
    CHECK(cat.default_error_condition(MPI_ERR_ACCESS)
          == std::errc::permission_denied);
  }
}

// NOLINTNEXTLINE
TEST_CASE("MPI error string functionality", "[error][mpi]") {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {  // Only test on rank 0
    const auto& cat = cxxmpi::error_category::instance();

    SECTION("error message generation") {
      const auto success_msg = cat.message(MPI_SUCCESS);
      CHECK_FALSE(success_msg.empty());

      const auto buffer_msg = cat.message(MPI_ERR_BUFFER);
      CHECK_FALSE(buffer_msg.empty());

      const auto invalid_msg = cat.message(-99999);
      CHECK_FALSE(invalid_msg.empty());
    }
  }
}
