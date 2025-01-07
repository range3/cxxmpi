#include <array>
#include <complex>
#include <cstddef>
#include <utility>

#include <catch2/catch_test_macros.hpp>
#include <cxxmpi/dtype.hpp>
#include <cxxmpi/error.hpp>
#include <mpi.h>

TEST_CASE("MPI Datatype basic operations", "[mpi][dtype]") {
  SECTION("builtin datatypes") {
    CHECK(cxxmpi::as_builtin_datatype<int>() == MPI_INT);
    CHECK(cxxmpi::as_builtin_datatype<double>() == MPI_DOUBLE);
    CHECK(cxxmpi::as_builtin_datatype<char>() == MPI_CHAR);
    CHECK(cxxmpi::as_builtin_datatype<std::complex<double>>()
          == MPI_C_DOUBLE_COMPLEX);
  }

  SECTION("weak_dtype creation from builtin types") {
    auto int_type = cxxmpi::as_weak_dtype<int>();
    CHECK(int_type.native() == MPI_INT);

    auto double_type = cxxmpi::as_weak_dtype<double>();
    CHECK(double_type.native() == MPI_DOUBLE);
  }
}

TEST_CASE("MPI Datatype custom types", "[mpi][dtype]") {
  SECTION("contiguous datatype") {
    auto base_type = cxxmpi::as_weak_dtype<int>();
    auto contiguous_type = cxxmpi::dtype(base_type, 3);
    contiguous_type.commit();

    int size = 0;
    MPI_Type_size(contiguous_type.native(), &size);
    CHECK(size == sizeof(int) * 3);
  }

  SECTION("vector datatype") {
    auto base_type = cxxmpi::as_weak_dtype<double>();
    constexpr int count = 2;
    constexpr int blocklength = 3;
    constexpr int stride = 4;

    auto vector_type = cxxmpi::dtype(base_type, count, blocklength, stride);
    vector_type.commit();

    int size = 0;
    MPI_Type_size(vector_type.native(), &size);
    CHECK(size == sizeof(double) * count * blocklength);
  }

  SECTION("subarray datatype") {
    auto base_type = cxxmpi::as_weak_dtype<float>();
    std::array<int, 2> sizes = {4, 4};
    std::array<int, 2> subsizes = {2, 2};
    std::array<int, 2> starts = {1, 1};

    auto subarray_type = cxxmpi::dtype(base_type, sizes, subsizes, starts);
    subarray_type.commit();

    int size = 0;
    MPI_Type_size(subarray_type.native(), &size);
    CHECK(static_cast<size_t>(size)
          == sizeof(float) * static_cast<size_t>(subsizes[0] * subsizes[1]));
  }
}

TEST_CASE("MPI Datatype move semantics", "[mpi][dtype]") {
  SECTION("move construction") {
    auto base_type = cxxmpi::as_weak_dtype<int>();
    auto original = cxxmpi::dtype(base_type, 3);
    auto moved = std::move(original);

    CHECK(moved.native() != MPI_DATATYPE_NULL);
  }

  SECTION("move assignment") {
    auto base_type = cxxmpi::as_weak_dtype<int>();
    auto original = cxxmpi::dtype(base_type, 3);
    auto other = cxxmpi::dtype(base_type, 2);

    other = std::move(original);
    CHECK(other.native() != MPI_DATATYPE_NULL);
  }
}

struct test_struct {
  int a;
  double b;
  char c[10];  // NOLINT
};

TEST_CASE("MPI Datatype struct creation", "[mpi][dtype]") {
  SECTION("create struct datatype") {
    std::array<int, 3> blocklengths = {1, 1, 10};
    std::array<MPI_Aint, 3> displacements = {offsetof(test_struct, a),
                                             offsetof(test_struct, b),
                                             offsetof(test_struct, c)};
    std::array<MPI_Datatype, 3> types = {MPI_INT, MPI_DOUBLE, MPI_CHAR};

    auto struct_type = cxxmpi::dtype(blocklengths, displacements, types);
    struct_type.commit();

    int size = 0;
    MPI_Type_size(struct_type.native(), &size);
    CHECK(static_cast<size_t>(size)
          == sizeof(test_struct::a) + sizeof(test_struct::b)
                 + sizeof(test_struct::c));
  }
}

TEST_CASE("MPI Datatype error handling", "[mpi][dtype]") {
  SECTION("invalid type creation") {
    auto base_type = cxxmpi::as_weak_dtype<int>();

    CHECK_THROWS_AS(cxxmpi::dtype(base_type, -1), cxxmpi::mpi_error);

    std::array<int, 2> invalid_sizes = {-1, -1};
    std::array<int, 2> subsizes = {1, 1};
    std::array<int, 2> starts = {0, 0};

    CHECK_THROWS_AS(cxxmpi::dtype(base_type, invalid_sizes, subsizes, starts),
                    cxxmpi::mpi_error);
  }

  SECTION("null type handling") {
    cxxmpi::weak_dtype_handle null_handle;
    CHECK_FALSE(null_handle);
    CHECK(null_handle.native() == MPI_DATATYPE_NULL);
  }
}
