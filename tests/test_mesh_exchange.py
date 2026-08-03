from pathlib import Path

import torch
import snapy


options = snapy.MeshOptions.from_yaml(
    str(Path(__file__).with_name("test_mesh_multi_block.yaml"))
)
options.blocks_per_process(4)
options.set_local_horizontal_cells(4, 4)
layout_options = options.block().layout()
layout_options.process_rank(0)
layout_options.process_world_size(1)
layout_options.blocks_per_process(options.blocks_per_process())
layout_options.device_id(0)
assert layout_options.process_rank() == 0
assert layout_options.process_world_size() == 1
assert layout_options.blocks_per_process() == 4
assert layout_options.device_id() == 0

mesh = snapy.Mesh(options)

variables = []
for rank, block in enumerate(mesh.blocks):
    coord = block.options.coord()
    shape = (
        1,
        coord.nx3() + 2 * coord.nghost(),
        coord.nx2() + 2 * coord.nghost(),
        coord.nx1(),
    )
    tensor = torch.full(shape, -1.0)
    tensor[block.part((0, 0, 0), False)] = float(rank)
    variables.append({"field": tensor})

sync_options = snapy.SyncOptions()
sync_options.interpolate(True).type(snapy.kScalar)
for generation in range(16):
    for rank, (block, block_vars) in enumerate(zip(mesh.blocks, variables)):
        block_vars["field"][block.part((0, 0, 0), False)] = float(
            rank + generation
        )
    mesh.exchange(variables, sync_options)

for rank, (block, block_vars) in enumerate(zip(mesh.blocks, variables)):
    layout = block.get_layout()
    location = layout.loc_of(rank)
    neighbor = layout.neighbor_rank(location, (0, 1, 0))
    ghost = block_vars["field"][block.part((0, 1, 0), True)]
    assert torch.all(ghost == float(neighbor + generation)), (
        rank,
        neighbor,
        ghost.unique(),
    )


# A local logical neighbor can also be masked by a physical boundary function.
# The sender must publish an inactive message instead of reading the buffer that
# serialize() deliberately left unprepared for that direction.
physical_options = snapy.MeshOptions.from_yaml(
    str(Path(__file__).with_name("test_mesh_multi_block.yaml"))
)
physical_options.blocks_per_process(4)
physical_options.set_local_horizontal_cells(4, 4)
physical_options.block().set_bfunc(0, -1, 0, lambda var, dim, opts: None)
physical_mesh = snapy.Mesh(physical_options)
physical_variables = []
for rank, block in enumerate(physical_mesh.blocks):
    coord = block.options.coord()
    shape = (
        1,
        coord.nx3() + 2 * coord.nghost(),
        coord.nx2() + 2 * coord.nghost(),
        coord.nx1(),
    )
    tensor = torch.full(shape, -1.0)
    tensor[block.part((0, 0, 0), False)] = float(rank)
    physical_variables.append({"field": tensor})

physical_mesh.exchange(physical_variables, sync_options)
