"""Covered storage, Kintera adjustment and kinetics on a Cartesian block."""

import math
import tempfile
import unittest
from pathlib import Path

import kintera as kt
import torch
import yaml

from snapy import EmbeddedBoundaryGeometry, MeshBlock, MeshBlockOptions, kICY, kIDN, kIPR
from test_sk4_chemistry_source import config


def check_moist_covered_cells(tmp_path: Path, all_covered: bool):
    cfg = config("float64")
    cfg["dynamics"].update({"anomaly": False, "perturbation-flux": False})
    cfg["integration"]["implicit-scheme"] = 0
    cfg.pop("forcing")
    cfg.pop("sedimentation")
    cfg["scalar"] = {"names": ["tracer"], "reconstruct": {"type": "dc"}}
    path = tmp_path / "chemistry.yaml"
    path.write_text(yaml.safe_dump(cfg))

    block = MeshBlock(MeshBlockOptions.from_yaml(str(path)))
    block.set_chemistry(kt.KineticsOptions.from_yaml(str(path)))
    coord = block.module("coord")
    x, y, z = [coord.buffer(name) for name in ("x1f", "x2f", "x3f")]
    phi = (x - (x[-1] + 1 if all_covered else x[x.numel() // 2]))
    phi = phi.expand(z.numel(), y.numel(), -1)
    geom = EmbeddedBoundaryGeometry.from_level_set(phi, x, y, z)
    block.set_embedded_boundary(geom)

    w = block.module("hydro").buffer("D").clone().zero_()
    thermo = block.module("hydro.eos.thermo")
    w[kIPR] = 1.0e6
    w[kIDN] = w[kIPR] / (8.31446261815324 * thermo.buffer("inv_mu")[0] * 300)
    w[kICY] = 0.01
    w[kICY + 1] = 0.001
    w[kICY + 3] = 1.0e-5
    w[kICY + 5] = 1.0e-6
    solid = geom.flags == 2
    w[:, solid] = math.nan
    v, _ = block.initialize({"hydro_w": w, "scalar_r": torch.full_like(w[:1], 0.1)})
    sl = block.part((0, 0, 0), False)
    owned_solid = solid[sl[1:]]
    nfluid = int((~owned_solid).sum())
    u0 = v["hydro_u"][sl].clone()
    s0 = v["scalar_s"][sl].clone()
    assert torch.isfinite(v["hydro_w"]).all()
    dt = min(0.001, block.max_time_step(v))
    assert math.isfinite(dt) and dt > 0
    for stage in range(len(block.module("intg").stages)):
        block.forward(v, dt, stage)
        assert torch.equal(v["hydro_u"][sl][:, owned_solid], u0[:, owned_solid])
        assert torch.equal(v["scalar_s"][sl][:, owned_solid], s0[:, owned_solid])
    assert thermo.make_active_set_storage()[1].shape == (1, 1, nfluid)
    assert block.check_redo(v) == 0
    if nfluid:
        assert not torch.equal(v["hydro_u"][sl][kICY:, ~owned_solid], u0[kICY:, ~owned_solid])
    # An invalid inactive value must have no vote in the EOS floor or redo.
    v["hydro_u"][:, solid] = math.nan
    assert block.check_redo(v) == 0

class CoveredMoistTest(unittest.TestCase):
    def test_partial_and_fully_covered_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            for all_covered in (False, True):
                with self.subTest(all_covered=all_covered):
                    check_moist_covered_cells(Path(directory), all_covered)


if __name__ == "__main__":
    unittest.main()
