#pragma once

// torch
#include <c10/util/intrusive_ptr.h>

#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace snap {

// Register a process group that was initialized outside snapy, typically from
// Python via torch.distributed.init_process_group().
void set_process_group(c10::intrusive_ptr<c10d::ProcessGroup> pg);

// Return the externally registered process group, if any.
c10::intrusive_ptr<c10d::ProcessGroup> get_process_group();

// True when an external process group has been registered.
bool is_process_group_initialized();

}  // namespace snap
