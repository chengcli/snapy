"""Manual Python boundary fills must reuse the initialized reference state."""
from pathlib import Path

import torch
import snapy

options = snapy.MeshBlockOptions.from_yaml(
    str(Path(__file__).with_name("test_radiating_boundary.yaml"))
)
options.hydro().eos().type("ideal-gas")
options.scalar().nvar(1)
block = snapy.MeshBlock(options)
w = torch.zeros((5, 14, 14, 14), dtype=torch.float64)
w[0] = 1.0
w[4] = 100000.0
variables, _ = block.initialize(
    {"hydro_w": w, "scalar_r": torch.full((1, 14, 14, 14), 0.2)}
)
u = variables["hydro_u"]
u[1, 3:-3, 3:-3, 3:-3] += 0.1
active = u[:, 3:-3, 3:-3, 3:-3].clone()
reference = variables["boundary_reference_w"].clone()

try:
    block.apply_hydro_bc(u)
except RuntimeError as exc:
    assert "missing initial boundary reference" in str(exc)
else:
    raise AssertionError("A manual characteristic fill needs its saved reference")

block.apply_hydro_bc(u, snapy.kConserved, variables)
block.apply_boundaries(variables, u)
block.apply_boundaries(variables, u, variables["scalar_s"])
assert torch.equal(active, u[:, 3:-3, 3:-3, 3:-3])
assert torch.equal(reference, variables["boundary_reference_w"])
assert torch.isfinite(u).all()
print("Python characteristic hydro/tracer boundary API passed")
