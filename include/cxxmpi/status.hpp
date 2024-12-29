#pragma once

#include <mpi.h>

#include "cxxmpi/dtype.hpp"
#include "cxxmpi/error.hpp"

namespace cxxmpi {
class status {
  MPI_Status status_{};

 public:
  constexpr status() noexcept = default;

  [[nodiscard]]
  constexpr auto native() const noexcept -> const MPI_Status& {
    return status_;
  }

  [[nodiscard]]
  constexpr auto native() noexcept -> MPI_Status& {
    return status_;
  }

  [[nodiscard]]
  auto source() const noexcept -> int {
    return status_.MPI_SOURCE;
  }

  [[nodiscard]]
  auto tag() const noexcept -> int {
    return status_.MPI_TAG;
  }

  [[nodiscard]]
  auto error() const noexcept -> int {
    return status_.MPI_ERROR;
  }

  template <typename T>
  [[nodiscard]]
  auto count() const -> int {
    int count = -1;
    check_mpi_result(MPI_Get_count(&status_, as_builtin_datatype<T>(), &count));
    return count;
  }

  [[nodiscard]]
  auto count(const weak_dtype& wdtype) const -> int {
    int count = -1;
    check_mpi_result(MPI_Get_count(&status_, wdtype.native(), &count));
    return count;
  }
};

}  // namespace cxxmpi
