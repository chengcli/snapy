#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <thread>
#include <utility>

namespace snap {

//! Bounded single-producer/single-consumer queue.
//!
//! Only the producer may call try_push()/wait_push() and only the consumer may
//! call try_consume()/wait_consume().  The producer owns write_, the consumer
//! owns read_, and release/acquire publication synchronizes access to slots.
template <typename T, std::size_t Capacity>
class SpscQueue {
  static_assert(Capacity > 0, "SpscQueue capacity must be positive");

 public:
  SpscQueue() = default;
  SpscQueue(SpscQueue const&) = delete;
  SpscQueue& operator=(SpscQueue const&) = delete;

  template <typename U>
  bool try_push(U&& value, std::uint64_t* ticket = nullptr) {
    auto write = write_.value.load(std::memory_order_relaxed);
    auto read = read_.value.load(std::memory_order_acquire);
    if (write - read == Capacity) return false;

    slots_[write % Capacity].emplace(std::forward<U>(value));
    write_.value.store(write + 1, std::memory_order_release);
    if (ticket != nullptr) *ticket = write + 1;
    return true;
  }

  template <typename U>
  std::uint64_t wait_push(U&& value) {
    std::uint64_t ticket = 0;
    while (!try_push(std::forward<U>(value), &ticket)) {
      std::this_thread::yield();
    }
    return ticket;
  }

  template <typename Consumer>
  bool try_consume(Consumer&& consumer) {
    auto read = read_.value.load(std::memory_order_relaxed);
    auto write = write_.value.load(std::memory_order_acquire);
    if (read == write) return false;

    auto& slot = slots_[read % Capacity];
    std::forward<Consumer>(consumer)(*slot);
    slot.reset();
    read_.value.store(read + 1, std::memory_order_release);
    return true;
  }

  template <typename Consumer>
  void wait_consume(Consumer&& consumer) {
    while (!try_consume(std::forward<Consumer>(consumer))) {
      std::this_thread::yield();
    }
  }

  //! Consumer-side peek. The returned pointer remains valid until consume.
  T const* front() const noexcept {
    auto read = read_.value.load(std::memory_order_relaxed);
    auto write = write_.value.load(std::memory_order_acquire);
    return read == write ? nullptr : &*slots_[read % Capacity];
  }

  //! Producer-side check that a published ticket has been consumed.
  bool consumed(std::uint64_t ticket) const noexcept {
    return read_.value.load(std::memory_order_acquire) >= ticket;
  }

 private:
  struct alignas(64) Counter {
    std::atomic<std::uint64_t> value{0};
  };

  std::array<std::optional<T>, Capacity> slots_;
  Counter write_;
  Counter read_;
};

}  // namespace snap
