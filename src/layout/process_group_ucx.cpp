#include <configure.h>

#ifdef USE_UCX

#include <commux/process_group_ucx.hpp>
#include <cstdlib>

#include "layout.hpp"
#include "process_group.hpp"

namespace snap {
namespace {

void set_default_env(char const* name, char const* value) {
  if (std::getenv(name) == nullptr) {
    setenv(name, value, /*overwrite=*/0);
  }
}

}  // namespace

void ProcessGroupContext::_init_ucx() {
  // commux owns CUDA stream synchronization. In particular, grouped point-to-
  // point operations are flushed after one stream synchronization at
  // endCoalescing(); synchronizing each buffer here would defeat that batching.
  set_default_env("COMMUX_COALESCE", "1");
  set_default_env("COMMUX_GROUP", "1");
  if (options_->device() == "cpu") {
    set_default_env("UCX_TLS", "^cuda_copy,cuda_ipc,gdr_copy");
  }
  ucx_ = c10::make_intrusive<commux::ProcessGroupUCX>(
      store, options_->process_rank(), options_->process_world_size());
  owns_process_group_ = true;
}

}  // namespace snap

#endif
