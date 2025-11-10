// C/C++
#include <cstdio>
#include <cstring>
#include <string>

// snapy
#include <snap/mesh/meshblock.hpp>

// kintera
#include <kintera/utils/serialize.hpp>

namespace snap {

// restart files are named as: <file_basename>.<block_id>.<file_number>.restart
void read_restart_file(MeshBlock pmb, std::string file_basename,
                       int file_number, Variables &in_vars) {
  // create filename: <file_basename>.<block_id>.<file_number>.restart
  std::string fname;
  char number[6];
  snprintf(number, sizeof(number), "%05d", file_number);
  char blockid[12];
  snprintf(blockid, sizeof(blockid), "block%d", pmb->options.dist().gid());

  fname.append(file_basename);
  fname.append(".");
  fname.append(blockid);
  fname.append(".");
  fname.append(number);
  fname.append(".restart");

  // load from disk
  load_tensors(in_vars, fname);
}

}  // namespace snap
