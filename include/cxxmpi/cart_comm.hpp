#pragma once

#include <algorithm>
#include <initializer_list>
#include <span>
#include <stdexcept>
#include <vector>

#include <mpi.h>

#include "comm.hpp"
#include "error.hpp"

namespace cxxmpi {

template <typename Handle = comm_handle>
class basic_cart_comm : public basic_comm<Handle> {
 public:
  using handle_type = Handle;

  constexpr basic_cart_comm() noexcept = default;
  constexpr basic_cart_comm(const basic_cart_comm& other) = delete;
  constexpr basic_cart_comm(basic_cart_comm&& other) noexcept = default;
  constexpr auto operator=(const basic_cart_comm& other) -> basic_cart_comm& =
                                                                delete;
  auto operator=(basic_cart_comm&& other) noexcept -> basic_cart_comm& =
                                                          default;
  constexpr ~basic_cart_comm() = default;

  // weak_cart_comm can be copyable / assignable
  constexpr basic_cart_comm(const basic_cart_comm& other)
    requires std::is_copy_constructible_v<Handle>
      : basic_comm<Handle>{other} {}
  constexpr auto operator=(const basic_cart_comm& other) -> basic_cart_comm&
    requires std::is_copy_assignable_v<Handle>
  {
    if (this != &other) {
      basic_comm<Handle>::operator=(other);
    }
    return *this;
  }

  // cart_comm to weak_cart_comm conversion constructor
  constexpr explicit basic_cart_comm(const basic_cart_comm<comm_handle>& other)
    requires std::same_as<Handle, weak_comm_handle>
      : basic_comm<Handle>{other} {}

  template <typename BaseHandle>
  basic_cart_comm(const basic_comm<BaseHandle>& base,
                  std::span<const size_t> dims,
                  std::span<const bool> periods,
                  bool reorder)
      : basic_comm<Handle>{create_cart_comm(base, dims, periods, reorder)} {}

  template <typename BaseHandle>
  basic_cart_comm(const basic_comm<BaseHandle>& base,
                  std::initializer_list<size_t> dims,
                  std::initializer_list<bool> periods,
                  bool reorder)
      : basic_cart_comm{base, std::span{dims.begin(), dims.size()},
                        std::span{periods.begin(), periods.size()}, reorder} {}

  [[nodiscard]]
  auto ndims() const -> std::size_t {
    int ndims = 0;
    check_mpi_result(MPI_Cartdim_get(this->native(), &ndims));
    return static_cast<size_t>(ndims);
  }

  [[nodiscard]]
  auto dims() const -> std::vector<int> {
    auto const num_dims = ndims();

    auto dims = std::vector<int>(num_dims);
    auto periods = std::vector<int>(num_dims);
    auto coords = std::vector<int>(num_dims);

    check_mpi_result(MPI_Cart_get(this->native(), static_cast<int>(num_dims),
                                  dims.data(), periods.data(), coords.data()));

    return dims;
  }

  [[nodiscard]]
  auto coords(int rank) const -> std::vector<int> {
    std::vector<int> coords(ndims());
    check_mpi_result(MPI_Cart_coords(
        this->native(), rank, static_cast<int>(coords.size()), coords.data()));
    return coords;
  }

  [[nodiscard]]
  auto coords() const -> std::vector<int> {
    return coords(this->rank());
  }

  using basic_comm<Handle>::rank;

  [[nodiscard]]
  auto rank(std::span<const int> coords) const -> int {
    int rank = MPI_PROC_NULL;
    check_mpi_result(MPI_Cart_rank(this->native(), coords.data(), &rank));
    return rank;
  }

  [[nodiscard]]
  auto rank(std::initializer_list<int> coords) const -> int {
    return rank(std::span{coords.begin(), coords.size()});
  }

  // Get ranks of neighboring processes
  [[nodiscard]]
  auto shift(int direction, int disp) const -> std::pair<int, int> {
    int source, dest;  // NOLINT
    check_mpi_result(
        MPI_Cart_shift(this->native(), direction, disp, &source, &dest));
    return {source, dest};
  }

  struct neighbors_2d {
    int up;
    int down;
    int left;
    int right;
  };

  [[nodiscard]]
  auto neighbors_2d() const -> neighbors_2d {
    auto [up, down] = shift(0, 1);
    auto [left, right] = shift(1, 1);
    return {up, down, left, right};
  }

 private:
  template <typename BaseHandle>
  static auto create_cart_comm(const basic_comm<BaseHandle>& base,
                               std::span<const size_t> dims,
                               std::span<const bool> periods,
                               bool reorder) -> handle_type {
    if (dims.size() != periods.size()) {
      throw std::invalid_argument("dims and periods must have same size");
    }

    std::vector<int> periods_int(dims.size());
    std::ranges::transform(periods, periods_int.begin(),
                           [](bool p) { return p ? 1 : 0; });
    std::vector<int> dims_int(dims.begin(), dims.end());
    std::ranges::transform(dims, dims_int.begin(),
                           [](size_t d) { return static_cast<int>(d); });

    weak_comm_handle new_comm;
    check_mpi_result(MPI_Cart_create(
        base.native(), static_cast<int>(dims.size()), dims_int.data(),
        periods_int.data(), static_cast<int>(reorder), &new_comm.native()));
    return handle_type{new_comm};
  }
};

using cart_comm = basic_cart_comm<comm_handle>;
using weak_cart_comm = basic_cart_comm<weak_comm_handle>;

}  // namespace cxxmpi
