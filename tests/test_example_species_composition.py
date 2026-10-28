#!/usr/bin/env python3
"""Every ammonia species in the example decks carries ammonia's formula unit.

`composition` defines the formula unit a species is built from, so it fixes the molar
mass and everything derived from it: mass fractions, the gas constant of the mixture
and the latent heat per kilogram. Giving NH3 the composition of water ({H: 2, O: 1})
silently runs ammonia at 18.015 g/mol instead of 17.031 g/mol, a 5.8 % error that no
run reports, because both compositions parse and both molar masses are plausible.

The decks are parsed, not grepped: the check reads the `composition` mapping of every
species whose name is NH3 or a phase of it (NH3(s), NH3(s,p), ...) and compares both
the mapping and the molar mass it implies.

  python test_example_species_composition.py
"""
import re
import sys
from pathlib import Path

import yaml

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# Decks that carry an ammonia species. jupiter_evap_precip_1d is the one that was
# already right and is kept here as a control: it must stay right.
DECKS = (
    "jupiter_crm.yaml",
    "jupiter_gcm.yaml",
    "uranus.yaml",
    "jupiter_evap_precip_1d.yaml",
)

AMMONIA = {"N": 1, "H": 3}

# CODATA/IUPAC standard atomic weights, g/mol, for the elements these decks use.
WEIGHTS = {"H": 1.008, "He": 4.002602, "N": 14.007, "O": 15.999, "S": 32.06}

# NH3 against the H2O that was written in its place; 5 mg/mol is far below the gap.
NH3_MOLAR_MASS = 17.031
TOL = 5.0e-3


def molar_mass(composition):
    return sum(WEIGHTS[element] * count for element, count in composition.items())


def is_ammonia(name):
    """NH3 itself or one of its phases: NH3(s), NH3(s,p), NH3(l), ... but not NH4SH."""
    return re.fullmatch(r"NH3(\([^)]*\))?", name) is not None


def main():
    problems = []
    checked = 0
    for deck in DECKS:
        path = EXAMPLES / deck
        if not path.exists():
            problems.append(f"{deck}: missing")
            continue
        species = yaml.safe_load(path.read_text()).get("species", [])
        found = [s for s in species if is_ammonia(s["name"])]
        if not found:
            problems.append(f"{deck}: no ammonia species found; the deck or this test moved")
            continue
        for s in found:
            checked += 1
            name, composition = s["name"], s.get("composition")
            if composition != AMMONIA:
                problems.append(
                    f"{deck}: {name} has composition {composition}, expected {AMMONIA}"
                )
                continue
            mass = molar_mass(composition)
            if abs(mass - NH3_MOLAR_MASS) > TOL:
                problems.append(
                    f"{deck}: {name} composition {composition} gives "
                    f"{mass:.3f} g/mol, expected {NH3_MOLAR_MASS:.3f} g/mol"
                )

    for p in problems:
        print(f"FAIL {p}")
    if problems:
        print(f"ammonia composition: {len(problems)} problem(s) in {checked} species")
        return 1
    print(f"ammonia composition: {checked} species in {len(DECKS)} decks, all N=1 H=3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
