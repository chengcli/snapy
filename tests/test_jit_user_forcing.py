from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict

import torch

import snapy


class _NestedScale(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.scale * value


class _StageForcing(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nested = _NestedScale()
        self.register_buffer("offset", torch.tensor(0.5, dtype=torch.float64))

    def forward(
        self,
        variables: Dict[str, torch.Tensor],
        dt: float,
        stage: int,
    ) -> Dict[str, torch.Tensor]:
        hydro_u = variables["hydro_u"]
        x2v = variables["coord.x2v"]
        hydro_du = torch.zeros_like(hydro_u)
        hydro_du[0] = (
            self.nested(torch.ones_like(hydro_u[0]))
            + self.offset
            + 0.0 * torch.sum(x2v)
        )
        return {"hydro_du": hydro_du}


def main() -> None:
    options = snapy.MeshBlockOptions.from_yaml("test_coordinate.yaml")
    block = snapy.MeshBlock(options)
    block.to(torch.device("cpu"))

    with TemporaryDirectory(prefix="snapy-jit-forcing-") as directory:
        forcing_path = Path(directory) / "forcing.pt"
        torch.jit.script(_StageForcing().eval()).save(str(forcing_path))
        block.set_user_stage_forcings([str(forcing_path)])

        coord = block.module("coord")
        eos = block.module("hydro.eos")
        shape = (
            eos.nvar(),
            coord.buffer("x3v").shape[0],
            coord.buffer("x2v").shape[0],
            coord.buffer("x1v").shape[0],
        )
        hydro_w = torch.zeros(shape, dtype=torch.float64)
        hydro_w[snapy.kIDN].fill_(1.0)
        hydro_w[snapy.kIPR].fill_(3.0)

        variables, _ = block.initialize({"hydro_w": hydro_w})
        initial_density = variables["hydro_u"][snapy.kIDN].clone()
        block.advance_local(variables, 0.0, 0)

        expected = initial_density + 0.75
        torch.testing.assert_close(variables["hydro_u"][snapy.kIDN], expected)


if __name__ == "__main__":
    main()
