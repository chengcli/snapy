#pragma once

// torch
#include <torch/torch.h>

//! \file cache_lifecycle.hpp
//! \brief Drop grid-derived caches when the owning module is converted.
//!
//! `torch::nn::Module::to` converts parameters and buffers with `set_data`
//! (torch/csrc/api/include/torch/nn/module.h, `Module::to_impl`), which KEEPS
//! the `TensorImpl` identity and does NOT bump the version counter while it
//! replaces the values. A cache keyed on the identity and the version of a
//! grid buffer therefore cannot see the conversion at all, and a cache that
//! additionally compares dtype and device sees a ONE-WAY conversion but not a
//! float64 -> float32 -> float64 ROUND TRIP: the dtype returns to its original
//! value while the buffer now holds float32-rounded numbers, so every term of
//! the key compares equal to the stale entry.
//!
//! Keying on `data_ptr()` cannot be relied on either: after a round trip the
//! allocator may hand the same address back (it does sometimes on the CPU),
//! so a changed pointer is not guaranteed.
//!
//! The reliable signal is the conversion itself. `Module::to_impl` first calls
//! `to()` on every child, through the VIRTUAL `to()` declared in `Module`, so a
//! module that overrides the three `to()` overloads is notified whether the
//! conversion was requested on it or on any ancestor. Invalidation is O(1) and
//! adds no per-step work, no host synchronisation and no allocation; a
//! conversion that happens to be a no-op costs one rebuild on next use.
//!
//! The existing version/identity guards stay: they catch in-place edits of a
//! grid buffer, which a conversion hook cannot see.
//!
//! Usage, inside a module that derives from `torch::nn::Cloneable<T>`:
//! \code
//!   SNAP_INVALIDATE_CACHES_ON_TO(torch::nn::Cloneable<MyImpl>,
//!                                invalidate_my_caches())
//! \endcode
namespace snap {

//! \brief Can this tensor be used as a cache key at all?
/*!
 * A tensor made under `InferenceMode` keeps NO version counter: reading
 * `_version()` on one throws, and mapping it to a constant instead would let
 * an in-place write through unnoticed, which is a silent stale value and worse
 * than the throw. Such a key is uncacheable, and the value it guards has to be
 * rebuilt on every call.
 */
inline bool cacheable_key(torch::Tensor const& t) {
  return t.defined() && !t.is_inference();
}

//! \brief The version of a cache key, or -1 when it has none.
/*!
 * Safe on an inference tensor, which `cacheable_key` rejects anyway, so the
 * returned -1 is never used to decide that a cache is still valid.
 */
inline int64_t key_version(torch::Tensor const& t) {
  return cacheable_key(t) ? t._version() : -1;
}

//! \brief Would a value computed from this tensor carry an autograd graph?
/*!
 * A memo of such a value would hand an earlier call's graph to a later one:
 * once that graph has been used by a backward pass, a second backward through
 * the memo fails, and a graph kept alive pins its saved tensors. Values that
 * need gradients are computed afresh and not memoised.
 */
inline bool needs_grad(torch::Tensor const& t) {
  return t.defined() && t.requires_grad() && torch::GradMode::is_enabled();
}

}  // namespace snap

#define SNAP_INVALIDATE_CACHES_ON_TO(BaseModule, invalidate_expr)              \
  void to(torch::Device device, torch::Dtype dtype, bool non_blocking = false) \
      override {                                                               \
    BaseModule::to(device, dtype, non_blocking);                               \
    invalidate_expr;                                                           \
  }                                                                            \
  void to(torch::Dtype dtype, bool non_blocking = false) override {            \
    BaseModule::to(dtype, non_blocking);                                       \
    invalidate_expr;                                                           \
  }                                                                            \
  void to(torch::Device device, bool non_blocking = false) override {          \
    BaseModule::to(device, non_blocking);                                      \
    invalidate_expr;                                                           \
  }
