"""Advect a smooth density wave alongside a thin Cartesian cut cell."""
import math
from pathlib import Path
import tempfile

import torch
import yaml
import snapy


def evolve(n, device):
    cfg = yaml.safe_load(Path(__file__).with_name("test_eb_transport.yaml").read_text())
    cfg["geometry"]["cells"].update(nx1=8, nx2=n)
    cfg["boundary-condition"]["external"].update({"x2-inner": "periodic", "x2-outer": "periodic"})
    cfg["dynamics"]["reconstruct"] = {d: {"type": "weno3", "scale": False}
                                      for d in ("vertical", "horizontal")}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "wave.yaml"
        path.write_text(yaml.safe_dump(cfg))
        opt = snapy.MeshBlockOptions.from_yaml(str(path))
    opt.layout().device(device)
    block = snapy.MeshBlock(opt)
    block.to(torch.device(device), torch.float64)
    c = block.module("coord")
    x, y, z = [c.buffer(f"x{d}f") for d in (1, 2, 3)]
    phi = (x - (0.5 - 1.e-6 / 8)).expand(z.numel(), y.numel(), -1)
    eb = snapy.EmbeddedBoundaryGeometry.from_level_set(phi, x, y, z)
    block.set_embedded_boundary(eb)
    w = torch.zeros_like(block.buffer("hydro.D"))
    def density(t):
        return 1. + 0.1 * (2 * math.pi * (eb.cell_centroid[1] - 0.5 * t)).sin()
    w[snapy.kIDN] = density(0.)
    w[snapy.kIV2] = 0.5
    w[snapy.kIPR] = 1.
    v, _ = block.initialize({"hydro_w": w, "scalar_r": torch.full_like(w[:1], 0.23)})
    dt = block.max_time_step(v)
    assert dt > 0.1 / n
    steps = math.ceil(0.12 / dt)
    for _ in range(steps):
        for stage in range(len(block.module("intg").stages)):
            block.forward(v, 0.12 / steps, stage)
        assert block.check_redo(v) == 0
    cut = block.part((0, 0, 0), False)[1:]
    rho = v["hydro_u"][snapy.kIDN]
    error = ((rho-density(0.12)).abs() * eb.volume_fraction)[cut].sum() / eb.volume_fraction[cut].sum()
    mask = eb.volume_fraction[cut] > 0
    torch.testing.assert_close((v["scalar_s"][0][cut] / rho[cut])[mask],
                               torch.full_like(rho[cut][mask], 0.23))
    return float(error), rho[cut].cpu()


if __name__ == "__main__":
    e1, _ = evolve(16, "cpu")
    e2, reference = evolve(32, "cpu")
    print(f"Volume-weighted density error at 16 and 32 cells: {e1:g}, {e2:g}")
    assert 0 < e2 < 0.75 * e1, (e1, e2)
    if torch.cuda.is_available():
        _, gpu = evolve(32, "cuda")
        torch.testing.assert_close(gpu, reference)
