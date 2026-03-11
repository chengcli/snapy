#pragma once

// C/C++
#include <memory>
#include <mutex>
#include <string>

// torch
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>

namespace snap {

struct LayoutOptionsImpl;
using LayoutOptions = std::shared_ptr<LayoutOptionsImpl>;

class ProcessGroupContext {
 public:
  static std::shared_ptr<ProcessGroupContext> create(LayoutOptions const& opts);

  at::intrusive_ptr<c10d::Store> store;
  std::shared_ptr<c10d::Backend> pg;

  bool is_nccl() const { return backend == "nccl"; }
  void group_start() const;
  void group_end() const;
  void sync_device() const;

 private:
  explicit ProcessGroupContext(LayoutOptions const& opts);
  void _init();
  void _init_gloo();
  void _init_nccl();

  LayoutOptions options_;
  std::string backend;

  static std::mutex mutex_;
};

}  // namespace snap
