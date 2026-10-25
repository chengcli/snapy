"""Construct the six-panel diffusion configuration using the selected package.

Use PYTHONPATH and LD_LIBRARY_PATH to select a rebuilt checkout. This is an
API/capability smoke check, not a convection simulation.
"""
from pathlib import Path
import tempfile

import snapy
import yaml


def main():
    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((root / "tests/test_forcing_cubed_sphere.yaml").read_text())
    cfg["forcing"] = {"diffusion": {"nu_iso": 0.01, "kappa_iso": 0.01}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml") as f:
        yaml.safe_dump(cfg, f)
        f.flush()
        options = snapy.MeshOptions.from_yaml(f.name)
        options.blocks_per_process(6)
        options.set_local_horizontal_cells(6, 6)
        mesh = snapy.Mesh(options)
        assert len(mesh.blocks) == 6
        print("PASS: six-panel cubed-sphere diffusion constructed")
    print("Loaded:", snapy.__file__)


if __name__ == "__main__":
    main()
