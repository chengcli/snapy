"""Compare one Cartesian block with two Gloo ranks. Run python tests/test_eb_distributed.py."""

import math
import os
import socket
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml

import snapy


def evolve(case):
    mesh = snapy.Mesh(snapy.MeshOptions.from_yaml(str(case)))
    block = mesh.blocks[0]
    xf, yf, zf = [block.buffer(f"coord.x{d}f") for d in (1, 2, 3)]
    _, y, x = torch.meshgrid(zf, yf, xf, indexing="ij")
    # Small fluid fragments open across y=0 (periodic wrap) and y=1/2
    # (interior process boundary). The wall is oblique near each seam.
    phase = 4 * math.pi * (y.remainder(1.0) + 0.1)
    phi = phase.cos() - math.cos(4 * math.pi * 0.11) + 0.03 * (2 * math.pi * x).sin()
    geometry = snapy.EmbeddedBoundaryGeometry.from_level_set(phi, xf, yf, zf)
    block.set_embedded_boundary(geometry)
    w = torch.zeros_like(block.buffer("hydro.D"))
    xc, yc = geometry.cell_centroid[:2]
    w[snapy.kIDN] = 1.
    w[snapy.kIPR] = 1. + 0.04 * (2 * math.pi * yc).sin() * (math.pi * xc).cos()
    variables, _ = mesh.initialize([{"hydro_w": w, "scalar_r": torch.full_like(w[:1], 0.02)}])
    cut = block.part((0, 0, 0), False)
    fraction = geometry.volume_fraction.unsqueeze(0)[cut][0]
    start = variables[0]["hydro_u"][cut].clone()
    mesh.set_cycle(1)
    dt = mesh.max_time_step(variables)
    for stage in range(len(block.module("intg").stages)):
        mesh.forward(variables, dt, stage)
    assert mesh.check_redo(variables) == 0
    final = variables[0]["hydro_u"][cut].clone()
    assert torch.isfinite(final).all()
    assert (final[1:4] - start[1:4]).abs().max() > 1.e-6
    tracer = variables[0]["scalar_s"][cut]
    valid = fraction > 0
    torch.testing.assert_close(tracer[0][valid], 0.02 * final[0][valid], atol=2.e-11, rtol=2.e-11)
    return fraction, torch.cat((final, tracer), dim=0)


def run_rank(rank, directory):
    directory = Path(directory)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank),
                      MASTER_ADDR="127.0.0.1", MASTER_PORT="29500",
                      BACKEND="gloo", DEVICE="cpu")
    for _, interface in socket.if_nameindex():
        if interface in ("lo", "lo0"):
            os.environ.setdefault("GLOO_SOCKET_IFNAME", interface)
            break
    # FileStore rendezvous isolates each test invocation. Gloo's peer links
    # still exercise the same inter-process tensor transfers as production.
    dist.init_process_group("gloo", init_method=(directory / "store").as_uri(),
                            rank=rank, world_size=2)
    snapy.distributed.set_process_group(dist.group.WORLD)
    try:
        vf, final = evolve(directory / "split.yaml")
        torch.save((vf, final), directory / f"rank-{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_distributed():
    case = yaml.safe_load(Path(__file__).with_name("test_eb_transport.yaml").read_text())
    bc = case["boundary-condition"]["external"]
    bc["x2-inner"] = bc["x2-outer"] = "periodic"
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for name, nblocks in (("single", 1), ("split", 2)):
            case["distribute"] = {"layout": "slab", "nb2": nblocks}
            (directory / f"{name}.yaml").write_text(yaml.safe_dump(case))
        vf, expected = evolve(directory / "single.yaml")
        for row in (0, vf.size(1) // 2):
            assert ((vf[:, row] > 0.) & (vf[:, row] < 0.5)).any()
        mp.spawn(run_rank, (str(directory),), nprocs=2)
        chunks = [torch.load(directory / f"rank-{rank}.pt", weights_only=True)
                  for rank in range(2)]
        torch.testing.assert_close(torch.cat([s[0] for s in chunks], dim=1), vf,
                                   atol=2.e-12, rtol=2.e-12)
        torch.testing.assert_close(torch.cat([s[1] for s in chunks], dim=2), expected,
                                   atol=2.e-11, rtol=2.e-11)


if __name__ == "__main__":
    test_distributed()
