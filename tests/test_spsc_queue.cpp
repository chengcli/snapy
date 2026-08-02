#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <snap/utils/spsc_queue.hpp>
#include <thread>
#include <vector>

using snap::SpscQueue;

TEST(SpscQueue, reports_empty_and_full_and_wraps) {
  SpscQueue<int, 2> queue;
  EXPECT_EQ(queue.front(), nullptr);
  EXPECT_TRUE(queue.try_push(1));
  EXPECT_TRUE(queue.try_push(2));
  EXPECT_FALSE(queue.try_push(3));

  int value = 0;
  EXPECT_TRUE(queue.try_consume([&](int current) { value = current; }));
  EXPECT_EQ(value, 1);
  EXPECT_TRUE(queue.try_push(3));
  EXPECT_TRUE(queue.try_consume([&](int current) { value = current; }));
  EXPECT_EQ(value, 2);
  EXPECT_TRUE(queue.try_consume([&](int current) { value = current; }));
  EXPECT_EQ(value, 3);
  EXPECT_FALSE(queue.try_consume([](int) {}));
}

TEST(SpscQueue, supports_move_only_payloads) {
  SpscQueue<std::unique_ptr<int>, 1> queue;
  queue.wait_push(std::make_unique<int>(42));
  std::unique_ptr<int> value;
  queue.wait_consume(
      [&](std::unique_ptr<int>& current) { value = std::move(current); });
  ASSERT_NE(value, nullptr);
  EXPECT_EQ(*value, 42);
}

TEST(SpscQueue, preserves_fifo_order_under_contention) {
  constexpr std::uint64_t kCount = 200000;
  SpscQueue<std::uint64_t, 64> queue;
  std::atomic<bool> ordered{true};

  std::thread producer([&]() {
    for (std::uint64_t value = 0; value < kCount; ++value) {
      queue.wait_push(value);
    }
  });
  std::thread consumer([&]() {
    for (std::uint64_t expected = 0; expected < kCount; ++expected) {
      queue.wait_consume([&](std::uint64_t value) {
        if (value != expected) ordered.store(false, std::memory_order_relaxed);
      });
    }
  });

  producer.join();
  consumer.join();
  EXPECT_TRUE(ordered.load(std::memory_order_relaxed));
}
