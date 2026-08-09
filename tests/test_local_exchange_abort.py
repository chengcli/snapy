from pathlib import Path

import torch
import snapy


def build_mesh():
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
    return snapy.Mesh(options)


def make_variables(mesh, bad_block=None):
    variables = []
    for rank, block in enumerate(mesh.blocks):
        coord = block.options.coord()
        shape = (
            1,
            coord.nx3() + 2 * coord.nghost(),
            coord.nx2() + 2 * coord.nghost(),
            coord.nx1(),
        )
        if rank == bad_block:
            # One block publishes a buffer set that cannot match its
            # neighbours' receive buffers.
            shape = (2,) + shape[1:]
        tensor = torch.full(shape, float(rank))
        variables.append({"field": tensor})
    return variables


sync_options = snapy.SyncOptions()
sync_options.interpolate(True).type(snapy.kScalar)

# Control: a well-formed multi-block exchange completes.
mesh = build_mesh()
mesh.exchange(make_variables(mesh), sync_options)

# Fault: block 0 publishes a mismatched buffer set. The worker that consumes
# it raises, and every peer spinning in the local ghost queues must observe
# the abort and unwind, so the pool completes and exchange() rethrows the
# root cause. Without abort propagation this exchange never returns; the
# ctest TIMEOUT on this file is the regression guard for that deadlock.
mesh = build_mesh()
try:
    mesh.exchange(make_variables(mesh, bad_block=0), sync_options)
except RuntimeError as err:
    print("exchange failed as expected:", str(err).splitlines()[0])
else:
    raise SystemExit("exchange with a mismatched block must raise")
