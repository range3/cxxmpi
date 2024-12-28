#pragma once

#include <memory>
#include <string>

#include <cxxmpi/error.hpp>
#include <mpi.h>

namespace cxxmpi {

class universe {
  struct deleter {
    void operator()(void* p) const noexcept {
      if (p != nullptr) {
        MPI_Finalize();
      }
    }
  };

  using handle_type = std::unique_ptr<void, deleter>;
  handle_type handle_;

 public:
  universe() {
    if (!is_initialized()) {
      check_mpi_result(MPI_Init(nullptr, nullptr));
      handle_ = handle_type{this};
      set_errhandlers();
    }
  }

  explicit universe(int& argc, char**& argv) {
    if (!is_initialized()) {
      check_mpi_result(MPI_Init(&argc, &argv));
      handle_ = handle_type{this};
      set_errhandlers();
    }
  }

  explicit universe(int& argc, char**& argv, int required) {
    if (!is_initialized()) {
      int provided = 0;
      check_mpi_result(MPI_Init_thread(&argc, &argv, required, &provided));
      handle_ = handle_type{this};
      set_errhandlers();
    }
  }

  universe(const universe&) = delete;
  auto operator=(const universe&) -> universe& = delete;
  universe(universe&&) noexcept = default;
  auto operator=(universe&&) noexcept -> universe& = default;
  ~universe() = default;

  [[nodiscard]]
  explicit operator bool() const noexcept {
    return handle_ != nullptr;
  }

  [[nodiscard]]
  static auto is_initialized() -> bool {
    int flag = 0;
    check_mpi_result(MPI_Initialized(&flag));
    return flag != 0;
  }

  [[nodiscard]]
  static auto is_finalized() -> bool {
    int flag = 0;
    check_mpi_result(MPI_Finalized(&flag));
    return flag != 0;
  }

  [[nodiscard]]
  static auto is_thread_main() -> bool {
    int flag = 0;
    check_mpi_result(MPI_Is_thread_main(&flag));
    return flag != 0;
  }

  [[nodiscard]]
  static auto processor_name() -> std::string {
    auto name = std::string(MPI_MAX_PROCESSOR_NAME, '\0');
    auto len = int{};
    check_mpi_result(MPI_Get_processor_name(name.data(), &len));
    name.resize(static_cast<std::size_t>(len));
    return name;
  }

 private:
  static void set_errhandlers() {
    check_mpi_result(
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    check_mpi_result(MPI_Comm_set_errhandler(MPI_COMM_SELF, MPI_ERRORS_RETURN));
    check_mpi_result(MPI_File_set_errhandler(MPI_FILE_NULL, MPI_ERRORS_RETURN));
  }
};

}  // namespace cxxmpi
