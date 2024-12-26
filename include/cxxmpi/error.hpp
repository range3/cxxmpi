#pragma once

#include <array>
#include <format>
#include <source_location>
#include <string>
#include <system_error>

#include <mpi.h>

namespace cxxmpi {

class error_category : public std::error_category {
 public:
  static auto instance() noexcept -> const error_category& {
    static const error_category instance;
    return instance;
  }

  [[nodiscard]]
  auto name() const noexcept -> const char* override {
    return "cxxmpi";
  }

  [[nodiscard]]
  auto message(int ev) const -> std::string override {
    auto error_string = std::array<char, MPI_MAX_ERROR_STRING>{};
    auto length = int{};
    if (auto const ret = MPI_Error_string(ev, error_string.data(), &length);
        ret != MPI_SUCCESS) {
      return std::format("Failed to get MPI error message for error code: {}",
                         ev);
    }
    return {error_string.data(), static_cast<size_t>(length)};
  }

  [[nodiscard]]
  auto default_error_condition(int ev) const noexcept
      -> std::error_condition override {
    switch (ev) {
      case MPI_ERR_NO_MEM:
        return std::errc::not_enough_memory;
      case MPI_ERR_BUFFER:
        return std::errc::no_buffer_space;
      case MPI_ERR_ACCESS:
        return std::errc::permission_denied;
      case MPI_ERR_NO_SPACE:
        return std::errc::no_space_on_device;
      case MPI_ERR_FILE_EXISTS:
        return std::errc::file_exists;
      case MPI_ERR_NO_SUCH_FILE:
        return std::errc::no_such_file_or_directory;
      case MPI_ERR_IO:
        return std::errc::io_error;
      case MPI_ERR_READ_ONLY:
        return std::errc::read_only_file_system;
      default:
        return std::error_condition{ev, *this};
    }
  }

 private:
  error_category() = default;
};

[[nodiscard]]
inline auto make_error_code(int e) noexcept -> std::error_code {
  return {e, error_category::instance()};
}

// Base exception class
class mpi_error : public std::system_error {
 public:
  explicit mpi_error(
      int error_code,
      const std::source_location& loc = std::source_location::current())
      : std::system_error{make_error_code(error_code),
                          create_what_message(error_code, loc)} {}

 private:
  static auto create_what_message(int error_code,
                                  const std::source_location& loc)
      -> std::string {
    return std::format("{}:{} in {}: {}", loc.file_name(), loc.line(),
                       loc.function_name(),
                       make_error_code(error_code).message());
  }
};

constexpr void check_mpi_result(
    int result,
    const std::source_location& loc = std::source_location::current()) {
  if (result != MPI_SUCCESS) {
    throw mpi_error{result, loc};
  }
}

}  // namespace cxxmpi
