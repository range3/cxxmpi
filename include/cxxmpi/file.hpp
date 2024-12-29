#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include "cxxmpi/comm.hpp"
#include "cxxmpi/dtype.hpp"
#include "cxxmpi/error.hpp"

namespace cxxmpi {

class weak_file_handle {
  MPI_File file_{MPI_FILE_NULL};

 public:
  constexpr weak_file_handle() noexcept = default;
  // NOLINTNEXTLINE
  weak_file_handle(std::nullptr_t) noexcept {}
  // NOLINTNEXTLINE
  constexpr weak_file_handle(MPI_File file) noexcept : file_{file} {}

  [[nodiscard]]
  explicit operator bool() const noexcept {
    return file_ != MPI_FILE_NULL;
  }

  // Smart reference pattern
  [[nodiscard]]
  constexpr auto operator->() noexcept -> weak_file_handle* {
    return this;
  }

  [[nodiscard]]
  constexpr auto operator->() const noexcept -> const weak_file_handle* {
    return this;
  }

  [[nodiscard]]
  constexpr auto native() const noexcept -> MPI_File {
    return file_;
  }

  [[nodiscard]]
  constexpr auto native() noexcept -> MPI_File& {
    return file_;
  }

  [[nodiscard]]
  constexpr auto get() const noexcept -> weak_file_handle {
    return *this;
  }

  [[nodiscard]]
  auto release() noexcept -> MPI_File {
    return std::exchange(file_, MPI_FILE_NULL);
  }

  constexpr friend auto operator==(const weak_file_handle& l,
                                   const weak_file_handle& r) noexcept -> bool {
    return l.file_ == r.file_;
  }

  constexpr friend auto operator!=(const weak_file_handle& l,
                                   const weak_file_handle& r) noexcept -> bool {
    return !(l == r);
  }
};

namespace detail {
struct file_deleter {
  using pointer = weak_file_handle;

  void operator()(weak_file_handle handle) const noexcept {
    if (handle) {
      MPI_File file = handle.release();
      MPI_File_close(&file);
    }
  }
};
}  // namespace detail

using file_handle = std::unique_ptr<weak_file_handle, detail::file_deleter>;

template <typename Handle>
class basic_file {
 public:
  using handle_type = Handle;

  constexpr basic_file() noexcept = default;
  constexpr basic_file(const basic_file&) = delete;
  constexpr auto operator=(const basic_file&) -> basic_file& = delete;
  constexpr basic_file(basic_file&&) noexcept = default;
  constexpr auto operator=(basic_file&&) noexcept -> basic_file& = default;
  constexpr ~basic_file() noexcept = default;

  constexpr basic_file(const basic_file&)
    requires std::is_copy_constructible_v<handle_type>
  = default;
  constexpr auto operator=(const basic_file&) -> basic_file&
    requires std::is_copy_assignable_v<handle_type>
  = default;

  explicit basic_file(handle_type handle) : handle_{std::move(handle)} {}

  [[nodiscard]]
  constexpr auto native() const noexcept -> MPI_File {
    return handle_->native();
  }

  // file to weak_file
  constexpr explicit basic_file(const basic_file<file_handle>& other)
    requires std::same_as<handle_type, weak_file_handle>
      : handle_{weak_file_handle{other.native()}} {}

  void set_atomicity(bool flag) {
    check_mpi_result(MPI_File_set_atomicity(native(), flag ? 1 : 0));
  }

  void sync() { check_mpi_result(MPI_File_sync(native())); }

  void set_view(MPI_Offset disp,
                const weak_dtype& etype,
                const weak_dtype& filetype,
                const std::string& datarep = "native",
                MPI_Info info = MPI_INFO_NULL) {
    check_mpi_result(MPI_File_set_view(native(), disp, etype.native(),
                                       filetype.native(), datarep.c_str(),
                                       info));
  }

  template <typename T, std::size_t Extent>
  void write_at(MPI_Offset offset,
                std::span<const T, Extent> data,
                MPI_Status* status = MPI_STATUS_IGNORE) {
    check_mpi_result(MPI_File_write_at(native(), offset, data.data(),
                                       static_cast<int>(data.size()),
                                       as_builtin_datatype<T>(), status));
  }

  template <typename T, std::size_t Extent>
  void write_at(MPI_Offset offset,
                std::span<const T, Extent> data,
                const weak_dtype& dtype,
                MPI_Status* status = MPI_STATUS_IGNORE) {
    check_mpi_result(MPI_File_write_at(native(), offset, data.data(),
                                       static_cast<int>(data.size()),
                                       dtype.native(), status));
  }

  template <typename T, std::size_t Extent>
  void read_at(MPI_Offset offset,
               std::span<T, Extent> data,
               MPI_Status* status = MPI_STATUS_IGNORE) {
    check_mpi_result(MPI_File_read_at(native(), offset, data.data(),
                                      static_cast<int>(data.size()),
                                      as_builtin_datatype<T>(), status));
  }

  template <typename T, std::size_t Extent>
  void read_at(MPI_Offset offset,
               std::span<T, Extent> data,
               const weak_dtype& dtype,
               MPI_Status* status = MPI_STATUS_IGNORE) {
    check_mpi_result(MPI_File_read_at(native(), offset, data.data(),
                                      static_cast<int>(data.size()),
                                      dtype.native(), status));
  }

  template <typename T, std::size_t Extent>
  void write_at_all(MPI_Offset offset,
                    std::span<const T, Extent> data,
                    MPI_Status* status = MPI_STATUS_IGNORE) {
    check_mpi_result(MPI_File_write_at_all(native(), offset, data.data(),
                                           static_cast<int>(data.size()),
                                           as_builtin_datatype<T>(), status));
  }

  template <typename T, std::size_t Extent>
  void write_at_all(MPI_Offset offset,
                    std::span<const T, Extent> data,
                    const weak_dtype& dtype,
                    MPI_Status* status = MPI_STATUS_IGNORE) {
    check_mpi_result(MPI_File_write_at_all(native(), offset, data.data(),
                                           static_cast<int>(data.size()),
                                           dtype.native(), status));
  }

  template <typename T, std::size_t Extent>
  void read_at_all(MPI_Offset offset,
                   std::span<T, Extent> data,
                   MPI_Status* status = MPI_STATUS_IGNORE) {
    check_mpi_result(MPI_File_read_at_all(native(), offset, data.data(),
                                          static_cast<int>(data.size()),
                                          as_builtin_datatype<T>(), status));
  }

  template <typename T, std::size_t Extent>
  void read_at_all(MPI_Offset offset,
                   std::span<T, Extent> data,
                   const weak_dtype& dtype,
                   MPI_Status* status = MPI_STATUS_IGNORE) {
    check_mpi_result(MPI_File_read_at_all(native(), offset, data.data(),
                                          static_cast<int>(data.size()),
                                          dtype.native(), status));
  }

 private:
  handle_type handle_;
};

using file = basic_file<file_handle>;
using weak_file = basic_file<weak_file_handle>;

[[nodiscard]]
inline auto open(const std::string& filename,
                 const weak_comm& comm,
                 int mode,
                 MPI_Info info = MPI_INFO_NULL) -> file {
  weak_file_handle fh;
  check_mpi_result(
      MPI_File_open(comm.native(), filename.c_str(), mode, info, &fh.native()));
  return file{file_handle{fh}};
}

}  // namespace cxxmpi
