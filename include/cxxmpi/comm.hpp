#pragma once

#include <memory>

#include <mpi.h>

#include "error.hpp"

namespace cxxmpi {

class weak_comm_handle {
  MPI_Comm comm_{MPI_COMM_NULL};

 public:
  using pointer = weak_comm_handle;

  weak_comm_handle() noexcept = default;
  // NOLINTNEXTLINE
  weak_comm_handle(std::nullptr_t) noexcept {}
  // NOLINTNEXTLINE
  weak_comm_handle(MPI_Comm comm) noexcept : comm_{comm} {}

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
class comm {
  Handle handle_{};
  int rank_{-1};
  int size_{-1};

 public:
  using handle_type = Handle;

  constexpr comm() noexcept = default;
  comm(const comm& other) = delete;
  comm(const comm& other)
    requires std::is_copy_constructible_v<Handle>
      : handle_{other.handle_}, rank_{other.rank_}, size_{other.size_} {}
  auto operator=(const comm& other) -> comm& = delete;
  auto operator=(const comm& other) -> comm&
    requires std::is_copy_assignable_v<Handle>
  {
    if (this != &other) {
      handle_ = other.handle_;
      rank_ = other.rank_;
      size_ = other.size_;
    }
    return *this;
  }
  comm(comm&& other) noexcept = default;
  auto operator=(comm&& other) noexcept -> comm& = default;
  ~comm() = default;

  constexpr explicit comm(handle_type handle)
      : handle_{std::move(handle)},
        rank_{do_get_rank()},
        size_{do_get_size()} {}

  // split constructor
  template <typename BaseHandle>
  comm(const comm<BaseHandle>& base, int color, int key = 0)
    requires std::same_as<Handle, comm_handle>
      : handle_{create_split_comm(base, color, key)},
        rank_{do_get_rank()},
        size_{do_get_size()} {}

  [[nodiscard]]
  auto rank() const noexcept -> int {
    return rank_;
  }

  [[nodiscard]]
  auto size() const noexcept -> int {
    return size_;
  }

  void barrier() const { check_mpi_result(MPI_Barrier(native())); }

  [[nodiscard]]
  constexpr auto native() const noexcept -> MPI_Comm {
    return handle_->native();
  }

 private:
  template <typename BaseHandler>
  auto create_split_comm(const comm<BaseHandler>& base,
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
};

using weak_comm = comm<weak_comm_handle>;

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
