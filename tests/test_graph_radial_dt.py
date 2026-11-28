"""Radial split accepts zero and rejects invalid dt in eager and graph modes."""
import torch
from cuda_build import build_has_cuda, skip
from test_cubed_sphere_capture import make
from test_graph_chemistry_slot import assert_equal


def main():
    if not build_has_cuda():
        skip("CUDA unavailable")
    forcing = {"diffusion": {"kappa_iso": 2.e5, "implicit_radial": True}}
    graph, blocks, state = make("cuda-graph", True, False, False, forcing)
    eager, _, expected = make("eager", False, False, False, forcing)
    for dt in (0., .1, 0., .07):
        for stage in range(3):
            graph.forward(state, dt, stage)
            eager.forward(expected, dt, stage)
        for a, b in zip(state, expected):
            assert_equal(a, b, ("radial dt", dt))
    assert graph.execution_report()["captures"] == 3 * len(blocks)
    for mesh, variables in ((graph, state), (eager, expected)):
        for dt in (-.1, float("nan"), float("inf")):
            try:
                mesh.forward(variables, dt, 0)
            except RuntimeError as error:
                assert "Invalid radial step dt" in str(error), str(error)
            else:
                raise AssertionError("invalid radial dt accepted")


if __name__ == "__main__":
    main()
