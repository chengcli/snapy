"""Python print reads live bound options; report_input describes the deck."""

import io
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

import snapy


def printed(value):
    stream = io.StringIO()
    print(value, file=stream)
    return stream.getvalue()


def test_option_print():
    with TemporaryDirectory() as tmp:
        deck = Path(tmp) / "input.yaml"
        deck.write_text("integration: {implicit-scheme: 1}\noutputs: [{type: netcdf}]\n")
        opts = snapy.ImplicitOptions.from_yaml(str(deck))
        assert "scheme=1" in printed(opts)
        opts.scheme(9)
        assert "scheme=9" in printed(opts)
        section = snapy.report_input(str(deck), "outputs")
        output, = yaml.safe_load(printed(section))["outputs"]
        assert output["type"] == "netcdf" and output["combine"] is True

    assert "Hydro(options=None)" in printed(snapy.Hydro())
    opts = snapy.OutputOptions()
    opts.variables(["temp"])
    assert "variables=['temp']" in printed(opts)


if __name__ == "__main__":
    test_option_print()
