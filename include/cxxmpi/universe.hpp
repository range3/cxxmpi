#include <memory>
#include <stdexcept>

#include <mpi.h>

namespace cxxmpi {

class universe {
  struct deleter {
    void operator()(void* /*unused*/) const noexcept { MPI_Finalize(); }
  };

  using handle_type = std::unique_ptr<void, deleter>;
  handle_type handle_;

 public:
  universe(int& argc, char**& argv, int required = MPI_THREAD_MULTIPLE) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, required, &provided) != MPI_SUCCESS) {
      throw std::runtime_error("Failed to initialize MPI");
    }
    handle_ = handle_type{this};
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
};

}  // namespace cxxmpi
