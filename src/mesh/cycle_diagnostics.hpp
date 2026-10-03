#pragma once

#include <utility>
#include <vector>

#include "meshblock.hpp"

namespace snap {

// Aggregate local blocks before entering any process collective.
void print_cycle_diagnostics(
    std::vector<std::pair<MeshBlockImpl const*, Variables const*>> const&
        blocks,
    double time, double dt, int precision, char const* energy_label);

}  // namespace snap
