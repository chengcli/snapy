"""MeshOptions.set_local_horizontal_cells must leave every axis resolved.

The setter writes the global CELL COUNTS for x2 and x3 but not the global
BOUNDS, so the sentinel (global count 0) can no longer fire for those axes.
It is public API and needs no card, so nothing else in the suite reaches it.
"""

import snapy

NG = 2
# distinct span AND distinct count per axis, so an axis mix-up cannot pass
AXES = (("x1", 4, 0.0, 8.0), ("x2", 4, 10.0, 30.0), ("x3", 6, -1.0, 2.0))

coord = snapy.CoordinateOptions()
coord.nghost(NG)
coord.nx1(AXES[0][1])  # the horizontal counts are the setter's job
for name, _, lo, hi in AXES:
    getattr(coord, name + "min")(lo)
    getattr(coord, name + "max")(hi)

block = snapy.MeshBlockOptions()
block.coord(coord)
block.layout(snapy.LayoutOptions())

options = snapy.MeshOptions()
options.block(block)
options.set_local_horizontal_cells(AXES[1][1], AXES[2][1])
assert (coord.nx2(), coord.nx3()) == (AXES[1][1], AXES[2][1])

# without the fix the x2 offset resolves to 40 into a 9-element face array
cart = snapy.Cartesian(coord)
for name, nx, lo, hi in AXES:
    xf = cart.buffer(name + "f")
    dx = (hi - lo) / nx
    assert xf.shape[0] == nx + 2 * NG + 1, (name, tuple(xf.shape))
    assert abs(float(xf[NG]) - lo) <= 1.0e-12 * abs(dx), (name, "lower")
    assert abs(float(xf[NG + nx]) - hi) <= 1.0e-12 * abs(dx), (name, "upper")

# Multi-block options built in code: a later call must rewrite what an
# earlier one wrote, not check the half-set state it left (whole-domain
# local bounds with per-block counts, made consistent only by repartition).
PX, PY = 2, 3
mcoord = snapy.CoordinateOptions()
mcoord.nghost(NG)
mcoord.nx1(AXES[0][1])
for name, _, lo, hi in AXES:
    getattr(mcoord, name + "min")(lo)
    getattr(mcoord, name + "max")(hi)

layout = snapy.LayoutOptions()
layout.px(PX)
layout.py(PY)
mblock = snapy.MeshBlockOptions()
mblock.coord(mcoord)
mblock.layout(layout)

moptions = snapy.MeshOptions()
moptions.block(mblock)
moptions.set_local_horizontal_cells(2 * AXES[1][1], 1)
moptions.set_local_horizontal_cells(AXES[1][1], AXES[2][1])
assert (mcoord.nx2(), mcoord.nx3()) == (AXES[1][1], AXES[2][1])

# stand in for repartition: take the last block's slice. Its faces fit the
# global grid only if the global counts are nx2 * PX and nx3 * PY.
for name, nb in (("x2", PX), ("x3", PY)):
    lo, hi = [a[2:] for a in AXES if a[0] == name][0]
    getattr(mcoord, name + "min")(lo + (nb - 1) * (hi - lo) / nb)
    getattr(mcoord, name + "max")(hi)

cart = snapy.Cartesian(mcoord)
for (name, nx, lo, hi), nb in zip(AXES, (1, PX, PY)):
    xf = cart.buffer(name + "f")
    dx = (hi - lo) / (nx * nb)
    first = hi - nx * dx
    assert xf.shape[0] == nx + 2 * NG + 1, (name, tuple(xf.shape))
    assert abs(float(xf[NG]) - first) <= 1.0e-12 * abs(hi - lo), (name, "lower")
    assert abs(float(xf[NG + nx]) - hi) <= 1.0e-12 * abs(hi - lo), (name, "upper")
    assert abs(float(xf[NG + 1] - xf[NG]) - dx) <= 1.0e-12 * abs(dx), (name, "dx")

print("test_local_horizontal_cells: OK")
