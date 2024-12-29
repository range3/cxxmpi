#pragma once

#include <span>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include "error.hpp"

namespace cxxmpi {

[[nodiscard]]
inline auto create_dims(int nprocs, std::size_t ndims) -> std::vector<int> {
  if (ndims == 0) {
    throw std::invalid_argument("Number of dimensions cannot be zero");
  }

  auto dims = std::vector<int>(ndims, 0);
  check_mpi_result(
      MPI_Dims_create(nprocs, static_cast<int>(ndims), dims.data()));
  return dims;
}

[[nodiscard]]
inline auto create_dims(int nprocs,
                        std::span<const int> init_dims) -> std::vector<int> {
  if (init_dims.empty()) {
    throw std::invalid_argument("Dimensions array cannot be empty");
  }

  auto dims = std::vector<int>(init_dims.begin(), init_dims.end());
  check_mpi_result(
      MPI_Dims_create(nprocs, static_cast<int>(dims.size()), dims.data()));
  return dims;
}

[[nodiscard]]
inline auto create_dims(int nprocs, std::initializer_list<int> init_dims)
    -> std::vector<int> {
  return create_dims(nprocs, std::span{init_dims.begin(), init_dims.size()});
}

}  // namespace cxxmpi
