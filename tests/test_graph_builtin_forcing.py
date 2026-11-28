"""Built-in increments, changing stage dt, mixed scripts and radial replay.

CUDA FP32/FP64 on Cartesian; sphere FP64.
Plume's unavailable EOS is covered at registry level in test_source_registry.
"""
import copy
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from cuda_build import build_has_cuda, skip
from test_cubed_sphere_capture import make as make_sphere
from test_graph_chemistry_slot import assert_equal, build, config, moist_state
from test_graph_user_forcing import _Heating

FORCING = {
    "coriolis": {"type": "123", "omega1": 0.0003, "omega2": 0.0001, "omega3": 0.0002},
    "diffusion": {"nu_iso": 10., "kappa_iso": 20.,
                  "nu_scale_x1": {"x1": [0., 10000.], "scale": [0.8, 1.2]}},
    "body-heat": {"dTdt": 0.13, "pmin": 30000., "pmax": 80000.},
    "bot-heat": {"flux": 70., "depth": 2},
    "top-cool": {"flux": -50., "thickness": 750.},
    "relax-bot-temp": {"tau": 70., "btemp": 300., "at-face": True},
    "relax-bot-comp": {"tau": 90., "species": ["H2O"], "xfrac": [0.015]},
    "relax-bot-velo": {"tau": 110., "bvy": -0.7},
    "top-sponge-lyr": {"tau": 80., "width": 900.},
    "bot-sponge-lyr": {"tau": 100., "width": 800.},
}
DTS = (0.173, 0.041, 0.123, 0.173)


def cartesian(precision, forcing=None, implicit=0):
    forcing = FORCING if forcing is None else forcing
    pair = []
    for mode, sources in (("cuda-graph", forcing), ("eager", forcing), ("eager", {})):
        cfg = config(precision, mode, strict=(mode == "cuda-graph"), implicit=implicit)
        cfg["forcing"].update(sources)
        block = build(cfg, torch.device("cuda"))
        pair.append((block, moist_state(block)))
    graph, state = pair[0]
    for dt in DTS:
        for stage in range(3):
            for block, values in pair:
                torch.cuda.set_sync_debug_mode("error" if block is graph else "default")
                try:
                    block.forward(values, dt * (1. + 0.25 * stage), stage)
                finally:
                    torch.cuda.set_sync_debug_mode("default")
            assert_equal(state, pair[1][1], (precision, forcing, dt, stage))
    effect = (state["hydro_u"].double() - pair[2][1]["hydro_u"].double()).abs().max().item()
    assert effect > 0., ("inactive forcing", forcing, precision)
    report = graph.execution_report()
    assert report["eager"] == 0 and report["captures"] == 3, report
    assert report["replays"] == len(DTS) * 3, report
    print("cartesian", precision, list(forcing), "effect", effect, report, flush=True)


def cubed_sphere(split, script=None, extra=None, stretched=False):
    forcing = {"coriolis": {"omega1": 0.0001, "traditional": True},
               "diffusion": {"nu_iso": 1.e5, "kappa_iso": 2.e5, "implicit_radial": split},
               "body-heat": {"dTdt": 0.13, "pmin": 0., "pmax": 2.e5},
               "bot-heat": {"flux": 70., "thickness": 1500.},
               "top-cool": {"flux": -50., "depth": 2},
               "relax-bot-temp": {"tau": 70., "btemp": 300.},
               "relax-bot-velo": {"tau": 110., "bvy": -0.7},
               "top-sponge-lyr": {"tau": 80., "width": 1900.},
               "bot-sponge-lyr": {"tau": 100., "width": 1800.}}
    if stretched:
        forcing["top-cool"] = {"flux": -50., "thickness": 2000.}
    if extra:
        forcing["diffusion"].update(extra)
    graph, blocks, state = make_sphere("cuda-graph", not script, False, False, forcing, stretched=stretched)
    eager, eblocks, expected = make_sphere("eager", False, False, False, forcing, stretched=stretched)
    # Keep the same EOS in the control, while disabling explicit/radial diffusion.
    control_forcing = copy.deepcopy(forcing)
    control_forcing["diffusion"].update(nu_iso=0., kappa_iso=0.)
    control, _, unforced = make_sphere("eager", False, False, False, control_forcing, stretched=stretched)
    if script:
        for block in blocks + eblocks:
            block.set_user_stage_forcings([script])
    for dt in DTS:
        for stage in range(3):
            stage_dt = dt * (1. + 0.25 * stage)
            graph.forward(state, stage_dt, stage)
            eager.forward(expected, stage_dt, stage)
            control.forward(unforced, stage_dt, stage)
            for i, values in enumerate(state):
                assert_equal(values, expected[i], ("sphere", split, i, dt, stage))
    effect = max((v["hydro_u"] - u["hydro_u"]).abs().max().item()
                 for v, u in zip(expected, unforced))
    assert effect > 1., ("weak diffusion", effect)
    report = graph.execution_report()
    segments = 2 if script else 1
    assert report["eager"] == (len(DTS) * 3 * len(blocks) if script else 0), report
    assert report["replays"] == len(DTS) * 3 * len(blocks) * segments, report
    assert report["captures"] == 3 * len(blocks) * segments, report
    print("sphere", split, bool(script), extra, "effect", effect, report, flush=True)


def sphere_composition():
    forcing = {"relax-bot-comp": FORCING["relax-bot-comp"]}
    graph, blocks, state = make_sphere("cuda-graph", True, False, False, forcing, vapor=True)
    eager, _, expected = make_sphere("eager", False, False, False, forcing, vapor=True)
    control, _, unforced = make_sphere("eager", False, False, False, vapor=True)
    for dt in DTS:
        for stage in range(3):
            for mesh, values in ((graph, state), (eager, expected), (control, unforced)):
                mesh.forward(values, dt * (1. + .25 * stage), stage)
            for a, b in zip(state, expected):
                assert_equal(a, b, ("sphere composition", dt, stage))
    effect = max((a["hydro_u"] - b["hydro_u"]).abs().max().item()
                 for a, b in zip(expected, unforced))
    assert effect > 0., effect
    report = graph.execution_report()
    assert report["captures"] == 3 * len(blocks) and report["eager"] == 0, report
    assert report["replays"] == len(DTS) * 3 * len(blocks), report
    print("sphere composition", "effect", effect, report, flush=True)


def parameter_invalidation():
    pair = []
    for mode in ("cuda-graph", "eager", "eager"):
        cfg = config("float64", mode, strict=(mode == "cuda-graph"))
        cfg["forcing"]["coriolis"] = {"type": "123", "omega3": .0001}
        block = build(cfg, torch.device("cuda"))
        pair.append((block, moist_state(block)))
    for block, state in pair:
        for stage in range(3):
            block.forward(state, .1, stage)
    for block, _ in pair[:2]:
        block.options.hydro().coriolis().omega3(.003)
        block.bump_config_generation()
    for block, state in pair:
        for stage in range(3):
            block.forward(state, .07, stage)
    assert_equal(pair[0][1], pair[1][1], "forcing option after explicit invalidation")
    assert not torch.equal(pair[1][1]["hydro_u"], pair[2][1]["hydro_u"])
    assert pair[0][0].execution_report()["captures"] == 6

    # Profiles are immutable even after a configuration-generation bump.
    cfg = config("float64", "cuda-graph", strict=True)
    cfg["forcing"]["diffusion"] = FORCING["diffusion"]
    block = build(cfg, torch.device("cuda"))
    state = moist_state(block)
    for stage in range(3):
        block.forward(state, .1, stage)
    block.options.hydro().diffusion().nu_scale_x1_table().mul_(1.01)
    block.bump_config_generation()
    try:
        block.forward(state, .1, 0)
    except RuntimeError as error:
        assert "not changed afterwards" in str(error), str(error)
    else:
        raise AssertionError("modified diffusion profile was accepted")
    print("forcing parameter invalidation and profile refusal passed", flush=True)


def main():
    if not build_has_cuda():
        skip("CUDA unavailable")
    parameter_invalidation()
    variants = [{name: opts} for name, opts in FORCING.items()]
    variants += [{"coriolis": {"type": "xyz", "omega1": .0003, "omega2": .0001, "omega3": .0002}},
                 {"bot-heat": {"flux": 70., "thickness": 750.}},
                 {"top-cool": {"flux": -50., "depth": 2}},
                 {"relax-bot-temp": {"tau": 70., "btemp": 300.}},
                 {"diffusion": {"nu_iso": 10.}},
                 {"diffusion": {"kappa_iso": 2.e4, "dynamic": True}},
                 {"diffusion": {"kappa_iso": 20., "temperature_inner": 300.,
                                "temperature_outer": 200.,
                                "kappa_scale_x1": {"x1": [0., 10000.], "scale": [.8, 1.2]}}}]
    for precision in ("float64", "float32"):
        cartesian(precision, implicit=9)
        for forcing in variants:
            cartesian(precision, forcing)
    sphere_composition()
    for split in (False, True):
        cubed_sphere(split)
        cubed_sphere(split, extra={"dynamic": True, "kappa_iso": 2.e8})
        cubed_sphere(split, stretched=True, extra={"wall_velocity": "no-slip", "temperature_inner": 280.,
                                 "temperature_outer": 270.,
                                 "kappa_scale_x1": {"x1": [6.371e6, 6.381e6], "scale": [.8, 1.2]}})
    with TemporaryDirectory() as directory:
        path = str(Path(directory) / "heating.pt")
        torch.jit.script(_Heating().eval()).save(path)
        cubed_sphere(True, script=path)


if __name__ == "__main__":
    main()
