// torch
#include "distributed.hpp"

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace snap {

namespace {

c10::intrusive_ptr<c10d::ProcessGroup> g_process_group;

}  // namespace

void set_process_group(c10::intrusive_ptr<c10d::ProcessGroup> pg) {
  g_process_group = std::move(pg);
}

c10::intrusive_ptr<c10d::ProcessGroup> get_process_group() {
  return g_process_group;
}

bool is_process_group_initialized() { return g_process_group.defined(); }

}  // namespace snap
