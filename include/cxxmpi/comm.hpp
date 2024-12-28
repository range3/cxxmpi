#pragma once

#include <memory>

#include <mpi.h>

#include "error.hpp"

namespace cxxmpi {

class weak_comm_handle {
  MPI_Comm comm_{MPI_COMM_NULL};

 public:
  using pointer = weak_comm_handle;

  constexpr weak_comm_handle() noexcept = default;
  // NOLINTNEXTLINE
  weak_comm_handle(std::nullptr_t) noexcept {}
  // NOLINTNEXTLINE
  constexpr weak_comm_handle(MPI_Comm comm) noexcept : comm_{comm} {}

  [[nodiscard]]
  explicit operator bool() const noexcept {
    return comm_ != MPI_COMM_NULL;
  }

  // Smart reference pattern
  [[nodiscard]]
  constexpr auto operator->() noexcept -> pointer* {
    return this;
  }

  [[nodiscard]]
  constexpr auto operator->() const noexcept -> const pointer* {
    return this;
  }

  [[nodiscard]]
  constexpr auto native() const noexcept -> MPI_Comm {
    return comm_;
  }

  [[nodiscard]]
  constexpr auto get() const noexcept -> pointer {
    return *this;
  }

  [[nodiscard]]
  auto release() noexcept -> MPI_Comm {
    return std::exchange(comm_, MPI_COMM_NULL);
  }

  constexpr friend auto operator==(const weak_comm_handle& l,
                                   const weak_comm_handle& r) noexcept -> bool {
    return l.comm_ == r.comm_;
  }

  constexpr friend auto operator!=(const weak_comm_handle& l,
                                   const weak_comm_handle& r) noexcept -> bool {
    return !(l == r);
  }
};

namespace detail {
struct comm_deleter {
  using pointer = weak_comm_handle;

  void operator()(weak_comm_handle handle) const noexcept {
    if (handle && handle.native() != MPI_COMM_WORLD
        && handle.native() != MPI_COMM_SELF) {
      MPI_Comm comm = handle.release();
      MPI_Comm_free(&comm);
    }
  }
};
}  // namespace detail

using comm_handle = std::unique_ptr<weak_comm_handle, detail::comm_deleter>;

template <typename Handle = comm_handle>
class basic_comm {
 public:
  using handle_type = Handle;

  constexpr basic_comm() noexcept = default;
  constexpr basic_comm(const basic_comm& other) = delete;
  constexpr basic_comm(const basic_comm& other)
    requires std::is_copy_constructible_v<Handle>
      : handle_{other.handle_}, rank_{other.rank_}, size_{other.size_} {}
  constexpr auto operator=(const basic_comm& other) -> basic_comm& = delete;
  constexpr auto operator=(const basic_comm& other) -> basic_comm&
    requires std::is_copy_assignable_v<Handle>
  {
    if (this != &other) {
      handle_ = other.handle_;
      rank_ = other.rank_;
      size_ = other.size_;
    }
    return *this;
  }
  constexpr basic_comm(basic_comm&& other) noexcept = default;
  constexpr auto operator=(basic_comm&& other) noexcept -> basic_comm& =
                                                               default;
  constexpr ~basic_comm() = default;

  // comm to weak_comm
  constexpr explicit basic_comm(const basic_comm<comm_handle>& other)
    requires std::same_as<Handle, weak_comm_handle>
      : handle_{weak_comm_handle{other.native()}},
        rank_{other.rank()},
        size_{other.size()} {}

  explicit basic_comm(handle_type handle)
      : handle_{std::move(handle)},
        rank_{do_get_rank()},
        size_{do_get_size()} {}

  // split constructor
  template <typename BaseHandle>
  basic_comm(const basic_comm<BaseHandle>& base, int color, int key = 0)
    requires std::same_as<Handle, comm_handle>
      : handle_{create_split_comm(base, color, key)},
        rank_{do_get_rank()},
        size_{do_get_size()} {}

  [[nodiscard]]
  constexpr auto rank() const noexcept -> int {
    return rank_;
  }

  [[nodiscard]]
  constexpr auto size() const noexcept -> int {
    return size_;
  }

  void barrier() const { check_mpi_result(MPI_Barrier(native())); }

  [[nodiscard]]
  constexpr auto native() const noexcept -> MPI_Comm {
    return handle_->native();
  }

 private:
  template <typename BaseHandler>
  auto create_split_comm(const basic_comm<BaseHandler>& base,
                         int color,
                         int key) const -> handle_type {
    MPI_Comm new_comm;  // NOLINT
    check_mpi_result(MPI_Comm_split(base.native(), color, key, &new_comm));
    return comm_handle{weak_comm_handle{new_comm}};
  }

  auto do_get_rank() const -> int {
    int rank{};
    check_mpi_result(MPI_Comm_rank(native(), &rank));
    return rank;
  }

  auto do_get_size() const -> int {
    int size{};
    check_mpi_result(MPI_Comm_size(native(), &size));
    return size;
  }

  handle_type handle_{};
  int rank_{-1};
  int size_{-1};
};

using comm = basic_comm<comm_handle>;
using weak_comm = basic_comm<weak_comm_handle>;

[[nodiscard]]
inline auto comm_world() -> const weak_comm& {
  static const weak_comm instance{weak_comm_handle{MPI_COMM_WORLD}};
  return instance;
}

[[nodiscard]]
inline auto comm_self() -> const weak_comm& {
  static const weak_comm instance{weak_comm_handle{MPI_COMM_SELF}};
  return instance;
}

}  // namespace cxxmpi
