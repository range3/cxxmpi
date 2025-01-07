#pragma once

#include <optional>
#include <vector>

#include <mpi.h>

#include "cxxmpi/error.hpp"
#include "cxxmpi/status.hpp"

namespace cxxmpi {
class request_group {
  std::vector<MPI_Request> requests_;

 public:
  request_group() = default;

  explicit request_group(size_t reserve_size) {
    requests_.reserve(reserve_size);
  }

  [[nodiscard]]
  auto add() -> MPI_Request& {
    requests_.push_back(MPI_REQUEST_NULL);
    return requests_.back();
  }

  [[nodiscard]]
  auto data() noexcept -> MPI_Request* {
    return requests_.data();
  }

  [[nodiscard]]
  auto data() const noexcept -> const MPI_Request* {
    return requests_.data();
  }

  [[nodiscard]]
  auto operator[](size_t index) noexcept -> MPI_Request& {
    return requests_[index];
  }

  [[nodiscard]]
  auto operator[](size_t index) const noexcept -> const MPI_Request& {
    return requests_[index];
  }

  [[nodiscard]]
  auto size() const noexcept -> size_t {
    return requests_.size();
  }

  [[nodiscard]]
  auto empty() const noexcept -> bool {
    return requests_.empty();
  }

  void wait_all_without_status() {
    if (empty()) {
      return;
    }

    check_mpi_result(MPI_Waitall(static_cast<int>(requests_.size()),
                                 requests_.data(), MPI_STATUSES_IGNORE));
    requests_.clear();
  }

  [[nodiscard]]
  auto wait_all() -> std::vector<status> {
    if (empty()) {
      return {};
    }

    std::vector<status> statuses(requests_.size());
    check_mpi_result(
        MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(),
                    reinterpret_cast<MPI_Status*>(statuses.data())));
    requests_.clear();
    return statuses;
  }

  [[nodiscard]]
  auto wait_any() -> std::pair<size_t, status> {
    if (empty()) {
      throw std::runtime_error("No requests to wait on");
    }

    status st{};
    int index = -1;
    check_mpi_result(MPI_Waitany(static_cast<int>(requests_.size()),
                                 requests_.data(), &index, &st.native()));

    if (index >= 0) {
      requests_[static_cast<size_t>(index)] = MPI_REQUEST_NULL;
    }

    return {static_cast<size_t>(index), st};
  }

  [[nodiscard]]
  auto test_all_without_status() -> bool {
    if (empty()) {
      return true;
    }

    int flag = 0;
    check_mpi_result(MPI_Testall(static_cast<int>(requests_.size()),
                                 requests_.data(), &flag, MPI_STATUSES_IGNORE));

    if (flag != 0) {
      requests_.clear();
      return true;
    }
    return false;
  }

  auto test_all(std::vector<status>& statuses) -> bool {
    if (empty()) {
      statuses.clear();
      return true;
    }

    statuses.resize(requests_.size());

    int flag = 0;
    check_mpi_result(
        MPI_Testall(static_cast<int>(requests_.size()), requests_.data(), &flag,
                    reinterpret_cast<MPI_Status*>(statuses.data())));

    if (flag != 0) {
      requests_.clear();
      return true;
    }
    return false;
  }

  [[nodiscard]]
  auto test_any(status& st) -> std::optional<size_t> {
    if (empty()) {
      return std::nullopt;
    }

    int index, flag;  // NOLINT
    check_mpi_result(MPI_Testany(static_cast<int>(requests_.size()),
                                 requests_.data(), &index, &flag,
                                 &st.native()));

    if ((flag != 0) && (index >= 0)) {
      requests_[static_cast<size_t>(index)] = MPI_REQUEST_NULL;
      return static_cast<size_t>(index);
    }
    return std::nullopt;
  }
};

}  // namespace cxxmpi
