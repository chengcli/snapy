"""Capture the built-in forcings together and compare each replay to eager."""
import torch

from cuda_build import build_has_cuda, skip
from test_cubed_sphere_capture import make as make_sphere


def cartesian(precision):
    from test_graph_chemistry_slot import assert_equal, build, config, forward_both, moist_state

    forcing = {
        "coriolis": {"type": "123", "omega1": 0.0003, "omega2": 0.0001},
        "diffusion": {"nu_iso": 0.1, "kappa_iso": 0.2,
                      "nu_scale_x1": {"x1": [0., 10000.], "scale": [0.8, 1.2]}},
        "body-heat": {"dTdt": 0.013, "pmin": 0., "pmax": 200000.},
        "bot-heat": {"flux": 7., "depth": 1},
        "top-cool": {"flux": -5., "thickness": 750.},
        "relax-bot-temp": {"tau": 70., "btemp": 300., "at-face": True},
        "relax-bot-comp": {"tau": 90., "species": ["H2O"], "xfrac": [0.015]},
        "relax-bot-velo": {"tau": 110., "bvy": -0.7},
        "top-sponge-lyr": {"tau": 80., "width": 900.},
        "bot-sponge-lyr": {"tau": 100., "width": 800.},
    }
    pair = []
    for mode in ("cuda-graph", "eager"):
        cfg = config(precision, mode, strict=(mode == "cuda-graph"))
        cfg["forcing"].update(forcing)
        block = build(cfg, torch.device("cuda"))
        pair.append((block, moist_state(block)))
    graph, state = pair[0]
    for dt in (0.173, 0.041, 0.123):
        forward_both(pair, dt, sync_error=True)
        assert_equal(state, pair[1][1], (precision, dt))
    report = graph.execution_report()
    stages = len(graph.intg.stages)
    assert report["eager"] == 0 and report["captures"] == stages, report
    assert report["replays"] == 3 * stages, report


def cubed_sphere(split):
    from test_graph_chemistry_slot import assert_equal

    forcing = {"coriolis": {"omega1": 0.0001},
               "diffusion": {"nu_iso": 0.1, "kappa_iso": 0.2, "implicit_radial": split}}
    graph, blocks, state = make_sphere("cuda-graph", True, False, False, forcing)
    eager, _, expected = make_sphere("eager", False, False, False, forcing)
    # Replay twice at the same dt and once at a new dt, including the two
    # radial half-step flux posts and the cross-panel energy/momentum fluxes.
    for dt in (0.2, 0.2, 0.1):
        for stage in range(3):
            graph.forward(state, dt, stage)
            eager.forward(expected, dt, stage)
        for i, values in enumerate(state):
            assert_equal(values, expected[i], ("cubed sphere", split, i, dt))
    report = graph.execution_report()
    assert report["eager"] == 0 and report["replays"] == 9 * len(blocks), report
    assert report["captures"] == (6 if split else 3) * len(blocks), report


if __name__ == "__main__":
    if not build_has_cuda():
        skip("CUDA unavailable")
    for precision in ("float64", "float32"):
        cartesian(precision)
    for split in (False, True):
        cubed_sphere(split)
