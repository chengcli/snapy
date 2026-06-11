from pathlib import Path

import torch
import snapy


options = snapy.MeshOptions.from_yaml(
    str(Path(__file__).with_name("test_mesh_multi_block.yaml"))
)
options.blocks_per_process(4)
options.set_local_horizontal_cells(4, 4)
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

mesh.exchange_ghost_zones(variables, snapy.kScalar)

for rank, (block, block_vars) in enumerate(zip(mesh.blocks, variables)):
    layout = block.get_layout()
    location = layout.loc_of(rank)
    neighbor = layout.neighbor_rank(location, (0, 1, 0))
    ghost = block_vars["field"][block.part((0, 1, 0), True)]
    assert torch.all(ghost == float(neighbor)), (
        rank,
        neighbor,
        ghost.unique(),
    )
