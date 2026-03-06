#pragma once

// C/C++
#include <memory>

// torch
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>

namespace snap {

//! \brief Set the global process group from an already-initialized backend.
/*!
 * This should be called from Python after torch.distributed.init_process_group()
 * and before creating any Layout objects that require distributed communication.
 *
 * Python usage:
 * \code{.py}
 *   import torch.distributed as dist
 *   import snapy
 *   dist.init_process_group(backend="gloo", init_method="env://")
 *   pg = dist.distributed_c10d._get_default_group()
 *   snapy.distributed.set_process_group(pg)
 * \endcode
 */
void set_process_group(c10::intrusive_ptr<c10d::ProcessGroup> pg);

//! \brief Get the globally set process group
/*!
 * \return intrusive_ptr to the global ProcessGroup, or null if not set
 */
c10::intrusive_ptr<c10d::ProcessGroup> get_process_group();

//! \brief Check whether the process group has been set
bool is_process_group_initialized();

//! \brief Clear the global process group reference
void destroy_process_group();

}  // namespace snap
