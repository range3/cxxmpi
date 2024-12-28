#pragma once

#include <cassert>
#include <complex>
#include <concepts>
#include <cstddef>
#include <memory>
#include <span>
#include <type_traits>
#include <utility>

#include <mpi.h>

#include "cxxmpi/error.hpp"

namespace cxxmpi {

template <typename T>
[[nodiscard]] constexpr auto as_builtin_datatype() noexcept -> MPI_Datatype {
  if constexpr (std::is_same_v<T, char>) {
    return MPI_CHAR;
  } else if constexpr (std::is_same_v<T, signed char>) {
    return MPI_SIGNED_CHAR;
  } else if constexpr (std::is_same_v<T, unsigned char>) {
    return MPI_UNSIGNED_CHAR;
  } else if constexpr (std::is_same_v<T, wchar_t>) {
    return MPI_WCHAR;
  } else if constexpr (std::is_same_v<T, signed short>) {  // NOLINT
    return MPI_SHORT;
  } else if constexpr (std::is_same_v<T, unsigned short>) {  // NOLINT
    return MPI_UNSIGNED_SHORT;
  } else if constexpr (std::is_same_v<T, signed int>) {
    return MPI_INT;
  } else if constexpr (std::is_same_v<T, unsigned int>) {
    return MPI_UNSIGNED;
  } else if constexpr (std::is_same_v<T, float>) {
    return MPI_FLOAT;
  } else if constexpr (std::is_same_v<T, double>) {
    return MPI_DOUBLE;
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    return MPI_C_COMPLEX;
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    return MPI_C_DOUBLE_COMPLEX;
  } else if constexpr (std::is_same_v<T, long double>) {
    return MPI_LONG_DOUBLE;
  } else if constexpr (std::is_same_v<T, bool>) {
    return MPI_C_BOOL;
  } else if constexpr (std::is_same_v<T, std::byte>) {
    return MPI_BYTE;
  } else {
    static_assert(false, "Unsupported builtin type for MPI communication");
  }
}

template <typename T>
concept has_builtin_datatype = requires {
  { as_builtin_datatype<T>() } -> std::same_as<MPI_Datatype>;
};

class weak_dtype_handle {
  MPI_Datatype dtype_{MPI_DATATYPE_NULL};

 public:
  constexpr weak_dtype_handle() noexcept = default;
  // NOLINTNEXTLINE
  weak_dtype_handle(std::nullptr_t) noexcept {}
  // NOLINTNEXTLINE
  constexpr weak_dtype_handle(MPI_Datatype dtype) noexcept : dtype_{dtype} {}

  [[nodiscard]]
  explicit operator bool() const noexcept {
    return dtype_ != MPI_DATATYPE_NULL;
  }

  // Smart reference pattern
  [[nodiscard]]
  constexpr auto operator->() noexcept -> weak_dtype_handle* {
    return this;
  }
  [[nodiscard]]
  constexpr auto operator->() const noexcept -> const weak_dtype_handle* {
    return this;
  }

  [[nodiscard]]
  constexpr auto native() const noexcept -> MPI_Datatype {
    return dtype_;
  }

  [[nodiscard]]
  constexpr auto get() const noexcept -> weak_dtype_handle {
    return *this;
  }

  [[nodiscard]]
  auto release() noexcept -> MPI_Datatype {
    return std::exchange(dtype_, MPI_DATATYPE_NULL);
  }

  constexpr friend auto operator==(const weak_dtype_handle& l,
                                   const weak_dtype_handle& r) noexcept
      -> bool {
    return l.dtype_ == r.dtype_;
  }

  constexpr friend auto operator!=(const weak_dtype_handle& l,
                                   const weak_dtype_handle& r) noexcept
      -> bool {
    return !(l == r);
  }
};

namespace detail {
struct dtype_deleter {
  using pointer = weak_dtype_handle;

  void operator()(weak_dtype_handle handle) const noexcept {
    if (handle) {
      MPI_Datatype dtype = handle.release();
      MPI_Type_free(&dtype);
    }
  }
};
}  // namespace detail

using dtype_handle = std::unique_ptr<weak_dtype_handle, detail::dtype_deleter>;

template <typename Handle>
class basic_dtype {
 public:
  using handle_type = Handle;

  constexpr basic_dtype() noexcept = default;
  constexpr basic_dtype(const basic_dtype& other) = delete;
  constexpr basic_dtype(basic_dtype&& other) noexcept = default;
  constexpr auto operator=(const basic_dtype& other) -> basic_dtype& = delete;
  constexpr auto operator=(basic_dtype&& other) noexcept -> basic_dtype& =
                                                                default;
  constexpr ~basic_dtype() = default;

  constexpr basic_dtype(const basic_dtype& other)
    requires std::copy_constructible<handle_type>
      : handle_{other.handle_} {}
  constexpr auto operator=(const basic_dtype& other) -> basic_dtype&
    requires std::is_copy_assignable_v<handle_type>
  {
    if (this != &other) {
      handle_ = other.handle_;
    }
    return *this;
  }

  explicit basic_dtype(handle_type handle) : handle_{std::move(handle)} {}

  // dtype to weak_dtype
  constexpr explicit basic_dtype(const basic_dtype<dtype_handle>& other)
    requires std::same_as<handle_type, weak_dtype_handle>
      : handle_{weak_dtype_handle{other.native()}} {}

  // contiguous constructor
  template <typename OtherHandle>
  explicit basic_dtype(const basic_dtype<OtherHandle>& base, const int count)
    requires std::same_as<handle_type, dtype_handle>
      : handle_{create_contiguous_dtype(base, count)} {}

  // vector constructor
  template <typename OtherHandle>
  explicit basic_dtype(const basic_dtype<OtherHandle>& base,
                       const int count,
                       const int blocklength,
                       const int stride)
    requires std::same_as<handle_type, dtype_handle>
      : handle_{create_vector_dtype(base, count, blocklength, stride)} {}

  // subarray constructor
  template <typename OtherHandle>
  explicit basic_dtype(const basic_dtype<OtherHandle>& base,
                       std::span<const int> sizes,
                       std::span<const int> subsizes,
                       std::span<const int> starts,
                       const int order = MPI_ORDER_C)
    requires std::same_as<handle_type, dtype_handle>
      : handle_{create_subarray_dtype(base, sizes, subsizes, starts, order)} {}

  // struct constructor
  explicit basic_dtype(std::span<const int> blocklengths,
                       std::span<const MPI_Aint> displacements,
                       std::span<const MPI_Datatype> types)
    requires std::same_as<handle_type, dtype_handle>
      : handle_{create_struct_dtype(blocklengths, displacements, types)} {}

  [[nodiscard]]
  constexpr auto native() const noexcept -> MPI_Datatype {
    return handle_->native();
  }

  void commit()
    requires std::same_as<handle_type, dtype_handle>
  {
    auto native_dtype = native();
    check_mpi_result(MPI_Type_commit(&native_dtype));
    assert(native_dtype == native());
  }

 private:
  handle_type handle_;

  template <typename OtherHandle>
  [[nodiscard]]
  static auto create_contiguous_dtype(const basic_dtype<OtherHandle>& base,
                                      const int count) -> dtype_handle {
    MPI_Datatype dtype = MPI_DATATYPE_NULL;
    check_mpi_result(MPI_Type_contiguous(count, base.native(), &dtype));
    return dtype_handle{dtype};
  }

  template <typename OtherHandle>
  [[nodiscard]]
  static auto create_vector_dtype(const basic_dtype<OtherHandle>& base,
                                  const int count,
                                  const int blocklength,
                                  const int stride) -> dtype_handle {
    MPI_Datatype dtype = MPI_DATATYPE_NULL;
    check_mpi_result(
        MPI_Type_vector(count, blocklength, stride, base.native(), &dtype));
    return dtype_handle{dtype};
  }

  template <typename OtherHandle>
  [[nodiscard]]
  static auto create_subarray_dtype(const basic_dtype<OtherHandle>& base,
                                    std::span<const int> sizes,
                                    std::span<const int> subsizes,
                                    std::span<const int> starts,
                                    const int order = MPI_ORDER_C)
      -> dtype_handle {
    assert(sizes.size() == subsizes.size());
    assert(sizes.size() == starts.size());
    MPI_Datatype dtype = MPI_DATATYPE_NULL;
    check_mpi_result(MPI_Type_create_subarray(
        static_cast<int>(sizes.size()), sizes.data(), subsizes.data(),
        starts.data(), order, base.native(), &dtype));
    return dtype_handle{dtype};
  }

  [[nodiscard]]
  static auto create_struct_dtype(std::span<const int> blocklengths,
                                  std::span<const MPI_Aint> displacements,
                                  std::span<const MPI_Datatype> types)
      -> dtype_handle {
    assert(blocklengths.size() == displacements.size());
    assert(blocklengths.size() == types.size());
    MPI_Datatype dtype = MPI_DATATYPE_NULL;
    check_mpi_result(MPI_Type_create_struct(
        static_cast<int>(blocklengths.size()), blocklengths.data(),
        displacements.data(), types.data(), &dtype));
    return dtype_handle{dtype};
  }
};

using dtype = basic_dtype<dtype_handle>;
using weak_dtype = basic_dtype<weak_dtype_handle>;

template <has_builtin_datatype T>
[[nodiscard]]
constexpr auto as_weak_dtype() noexcept -> weak_dtype {
  return weak_dtype{weak_dtype_handle{as_builtin_datatype<T>()}};
}

template <typename T>
struct dtype_traits {
  static_assert(false, "No MPI datatype conversion available for this type");
  static auto create(const T&) -> dtype;
};

}  // namespace cxxmpi
