"""Removed gravity-work keys are rejected even with their former defaults."""
import tempfile
from pathlib import Path

import yaml
from snapy import MeshBlockOptions

from gravity_box import config


def test_removed_gravity_options():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "gravity.yaml"
        deck = config()
        path.write_text(yaml.safe_dump(deck))
        gravity = MeshBlockOptions.from_yaml(str(path)).hydro().grav()
        assert gravity.grav1() == deck["forcing"]["const-gravity"]["grav1"]
        for key, value in (("gravity-work", "face"), ("gravity-work-fixer", False)):
            assert not hasattr(gravity, key.replace("-", "_"))
            deck["forcing"]["const-gravity"][key] = value
            path.write_text(yaml.safe_dump(deck))
            try:
                MeshBlockOptions.from_yaml(str(path))
            except RuntimeError as error:
                assert f"unknown key 'forcing/const-gravity/{key}'" in str(error), error
            else:
                raise AssertionError(f"accepted removed option {key}")
            del deck["forcing"]["const-gravity"][key]


if __name__ == "__main__":
    test_removed_gravity_options()
