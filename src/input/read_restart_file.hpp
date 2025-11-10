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

// restart files are named as: <file_basename>.<block_id>.<fileid>.restart
void read_restart_file(MeshBlockImpl *pmb, std::string fileid,
                       Variables &in_vars) {
  // create filename: <file_basename>.<block_id>.<fileid>.restart
  std::string fname;
  char blockid[12];
  snprintf(blockid, sizeof(blockid), "block%d", pmb->options.dist().gid());

  fname.append(pmb->options.basename());
  fname.append(".");
  fname.append(blockid);
  fname.append(".");
  fname.append(fileid);
  fname.append(".restart");

  // load from disk
  kintera::load_tensors(in_vars, fname);
}

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
