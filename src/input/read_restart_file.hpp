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

}  // namespace snap
