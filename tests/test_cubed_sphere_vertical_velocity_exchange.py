from pathlib import Path

import torch
import snapy


VERTICAL_VELOCITY = 1.2345
ABS_TOL = 1.0e-12
REL_TOL = 1.0e-12
HORIZONTAL_OFFSETS = (
    (0, -1, 0),
    (0, 1, 0),
    (-1, 0, 0),
    (1, 0, 0),
)


def make_mesh():
    options = snapy.MeshOptions.from_yaml(
        str(Path(__file__).with_name("test_exchange.yaml"))
    )
    options.blocks_per_process(6)
    options.set_local_horizontal_cells(6, 6)
    return snapy.Mesh(options)


def make_uniform_vertical_velocity_state(mesh):
    variables = []

    for block in mesh.blocks:
        coord = block.options.coord()
        eos = block.module("hydro.eos")
        nvar = eos.nvar()
        shape = (
            nvar,
            coord.nx3() + 2 * coord.nghost(),
            coord.nx2() + 2 * coord.nghost(),
            coord.nx1(),
        )
        hydro_w = torch.full(shape, -999.0, dtype=torch.float64)
        interior = block.part((0, 0, 0), False)
        interior_cells = interior[1:]

        hydro_w[snapy.kIDN][interior_cells] = 1.0
        hydro_w[snapy.kIV1][interior_cells] = VERTICAL_VELOCITY
        hydro_w[snapy.kIV2][interior_cells] = 0.0
        hydro_w[snapy.kIV3][interior_cells] = 0.0
        hydro_w[snapy.kIPR][interior_cells] = 1.0e5

        variables.append({"hydro_w": hydro_w})

    return variables


def gather_vertical_velocity_ghost_values(block, block_vars):
    values = []
    for offset in HORIZONTAL_OFFSETS:
        ghost = block.part(offset, True)
        values.append(block_vars["hydro_w"][snapy.kIV1][ghost[1:]].reshape(-1))
    return torch.cat(values)


def main():
    mesh = make_mesh()
    variables = make_uniform_vertical_velocity_state(mesh)

    mesh.exchange_ghost_zones(variables, snapy.kPrimitive)

    global_min = float("inf")
    global_max = float("-inf")
    global_max_abs_error = 0.0

    print(
        "setup: test_exchange.yaml, cubed-sphere, ideal-gas, "
        "blocks_per_process=6, local nx2=nx3=6"
    )
    print(
        f"interior vertical velocity kIV1={snapy.kIV1}: "
        f"{VERTICAL_VELOCITY:.16g}"
    )
    print("block loc ghost_min ghost_max max_abs_error")

    for rank, (block, block_vars) in enumerate(zip(mesh.blocks, variables)):
        values = gather_vertical_velocity_ghost_values(block, block_vars)
        ghost_min = values.min().item()
        ghost_max = values.max().item()
        max_abs_error = torch.max(torch.abs(values - VERTICAL_VELOCITY)).item()

        global_min = min(global_min, ghost_min)
        global_max = max(global_max, ghost_max)
        global_max_abs_error = max(global_max_abs_error, max_abs_error)

        loc = tuple(block.get_layout().loc_of(rank))
        print(
            f"{rank:5d} {str(loc):9s} "
            f"{ghost_min:.16g} {ghost_max:.16g} {max_abs_error:.16e}"
        )

    tolerance = ABS_TOL + REL_TOL * abs(VERTICAL_VELOCITY)
    ideal_match = global_max_abs_error <= tolerance
    print(f"global ghost range: [{global_min:.16g}, {global_max:.16g}]")
    print(f"global max abs error: {global_max_abs_error:.16e}")
    print(f"tolerance: {tolerance:.16e}")
    print(f"ideal_match: {ideal_match}")

    assert ideal_match, (
        "cubed-sphere primitive ghost exchange did not preserve uniform "
        f"vertical velocity: max_abs_error={global_max_abs_error:.16e}, "
        f"tolerance={tolerance:.16e}"
    )


if __name__ == "__main__":
    main()
