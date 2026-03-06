// torch
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

#include "distributed.hpp"

namespace snap {

namespace {

// Global process group - set by Python after torch.distributed.init_process_group()
c10::intrusive_ptr<c10d::ProcessGroup> g_pg;

}  // anonymous namespace

void set_process_group(c10::intrusive_ptr<c10d::ProcessGroup> pg) {
  g_pg = std::move(pg);
}

c10::intrusive_ptr<c10d::ProcessGroup> get_process_group() { return g_pg; }

bool is_process_group_initialized() { return g_pg.defined(); }

void destroy_process_group() { g_pg.reset(); }

}  // namespace snap
