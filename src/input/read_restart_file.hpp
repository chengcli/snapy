#pragma once

// C/C++
#include <cstdio>
#include <cstring>
#include <string>

// snap
#include <snap/mesh/meshblock.hpp>

// kintera
#include <kintera/utils/serialize.hpp>

namespace snap {

void set_hydro_interior(MeshBlockImpl *block, torch::Tensor &hydro_w,
                        Variables &in_vars) {
  auto interior = block->part({0, 0, 0});
  hydro_w.index(interior)[IDN] = in_vars["rho"];
  hydro_w.index(interior)[IVX] = in_vars["vel1"];
  hydro_w.index(interior)[IVY] = in_vars["vel2"];
  hydro_w.index(interior)[IVZ] = in_vars["vel3"];
  hydro_w.index(interior)[IPR] = in_vars["press"];
}

}  // namespace snap
