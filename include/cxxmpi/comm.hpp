#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include <mpi.h>

#include "cxxmpi/dtype.hpp"
#include "cxxmpi/error.hpp"
#include "cxxmpi/request.hpp"
#include "cxxmpi/status.hpp"

namespace cxxmpi {

namespace detail {

template <typename T>
concept is_std_span = requires {
  typename T::element_type;
  { T::extent } -> std::convertible_to<std::size_t>;
} && std::same_as<T, std::span<typename T::element_type, T::extent>>;

}  // namespace detail

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
  constexpr auto native() noexcept -> MPI_Comm& {
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
  constexpr auto rank() const noexcept -> size_t {
    return rank_;
  }

  [[nodiscard]]
  constexpr auto size() const noexcept -> size_t {
    return size_;
  }

  [[nodiscard]]
  constexpr auto native() const noexcept -> MPI_Comm {
    return handle_->native();
  }

  void barrier() const { check_mpi_result(MPI_Barrier(native())); }

  // Blocking send - custom datatype with count
  template <typename T, size_t Extent>
  void send(std::span<const T, Extent> data,
            const weak_dtype& data_type,
            int count,
            int dest,
            int tag = 0) const {
    check_mpi_result(
        MPI_Send(data.data(), count, data_type.native(), dest, tag, native()));
  }

  // Blocking send - builtin datatype with count
  template <typename T, size_t Extent>
  void send(std::span<const T, Extent> data, int dest, int tag = 0) const {
    send(data, as_weak_dtype<T>(), static_cast<int>(data.size()), dest, tag);
  }

  // Blocking receive - custom datatype with count
  template <typename T, size_t Extent>
  auto recv(std::span<T, Extent> data,
            const weak_dtype& data_type,
            int count,
            int source,
            int tag = 0) const -> status {
    status st;
    check_mpi_result(MPI_Recv(data.data(), count, data_type.native(), source,
                              tag, native(), &st.native()));
    return st;
  }

  template <typename T, size_t Extent>
  void recv_without_status(std::span<T, Extent> data,
                           const weak_dtype& data_type,
                           int count,
                           int source,
                           int tag = 0) const {
    check_mpi_result(MPI_Recv(data.data(), count, data_type.native(), source,
                              tag, native(), MPI_STATUS_IGNORE));
  }

  // Blocking receive - builtin datatype with count
  template <typename T, size_t Extent>
  auto recv(std::span<T, Extent> data,
            int source,
            int tag = 0) const -> status {
    return recv(data, as_weak_dtype<T>(), static_cast<int>(data.size()), source,
                tag);
  }

  template <typename T, size_t Extent>
  void recv_without_status(std::span<T, Extent> data,
                           int source,
                           int tag = 0) const {
    recv_without_status(data, as_weak_dtype<T>(), static_cast<int>(data.size()),
                        source, tag);
  }

  // Non-blocking send - custom datatype with count
  template <typename T, size_t Extent>
  void isend(std::span<const T, Extent> data,
             const weak_dtype& data_type,
             int count,
             int dest,
             int tag,
             MPI_Request& request) const {
    check_mpi_result(MPI_Isend(data.data(), count, data_type.native(), dest,
                               tag, native(), &request));
  }

  // Non-blocking send - builtin datatype with count
  template <typename T, size_t Extent>
  void isend(std::span<const T, Extent> data,
             int dest,
             int tag,
             MPI_Request& request) const {
    isend(data, as_weak_dtype<T>(), static_cast<int>(data.size()), dest, tag,
          request);
  }

  // Non-blocking receive - custom datatype with count
  template <typename T, size_t Extent>
  void irecv(std::span<T, Extent> data,
             const weak_dtype& data_type,
             int count,
             int source,
             int tag,
             MPI_Request& request) const {
    check_mpi_result(MPI_Irecv(data.data(), count, data_type.native(), source,
                               tag, native(), &request));
  }

  // Non-blocking receive - builtin datatype with count
  template <typename T, size_t Extent>
  void irecv(std::span<T, Extent> data,
             int source,
             int tag,
             MPI_Request& request) const {
    irecv(data, as_weak_dtype<T>(), static_cast<int>(data.size()), source, tag,
          request);
  }

  // Single value overloads
  template <typename T>
  void send(const T& value, int dest, int tag = 0) const
    requires(!detail::is_std_span<T>)
  {
    send(std::span<const T, 1>(&value, 1), dest, tag);
  }

  template <typename T>
  auto recv(T& value, int source, int tag = 0) const -> status
    requires(!detail::is_std_span<T>)
  {
    return recv(std::span<T, 1>(&value, 1), source, tag);
  }

  template <typename T>
  void recv_without_status(T& value, int source, int tag = 0) const
    requires(!detail::is_std_span<T>)
  {
    recv_without_status(std::span<T, 1>(&value, 1), source, tag);
  }

  template <typename T>
  void isend(const T& value, int dest, int tag, MPI_Request& request) const
    requires(!detail::is_std_span<T>)
  {
    isend(std::span<const T, 1>(&value, 1), dest, tag, request);
  }

  template <typename T>
  void irecv(T& value, int source, int tag, MPI_Request& request) const
    requires(!detail::is_std_span<T>)
  {
    irecv(std::span<T, 1>(&value, 1), source, tag, request);
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

  auto do_get_rank() const -> size_t {
    int rank{};
    check_mpi_result(MPI_Comm_rank(native(), &rank));
    return static_cast<size_t>(rank);
  }

  auto do_get_size() const -> size_t {
    int size{};
    check_mpi_result(MPI_Comm_size(native(), &size));
    return static_cast<size_t>(size);
  }

  handle_type handle_{};
  size_t rank_{};
  size_t size_{};
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
