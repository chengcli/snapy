// C/C++
#include <cstdio>
#include <cstring>
#include <string>

// kintera
#include <kintera/utils/serialize.hpp>

// snap
#include <snap/mesh/meshblock.hpp>

#include "output_formats.hpp"
#include "output_utils.hpp"

namespace snap {

RestartOutput::RestartOutput(OutputOptions const &options_)
    : OutputType(options_) {}

void RestartOutput::write_output_file(MeshBlockImpl *pmb, Variables const &vars,
                                      double current_time, bool final_write) {
  // make a cpu copy of variables
  Variables out_vars;
  for (auto const &[name, var] : vars) {
    if (var.defined()) {
      out_vars[name] = var.to(torch::kCPU);
    }
  }

  // store current time and cycle
  out_vars["current_time"] = torch::tensor({current_time}, torch::kFloat64);
  out_vars["current_cycle"] =
      torch::tensor({(int64_t)pmb->cycle}, torch::kInt64);

  // save file number and next time for each output type
  std::vector<int> output_file_numbers;
  std::vector<double> output_next_times;

  for (auto out : pmb->output_types) {
    output_file_numbers.push_back(out->file_number);
    output_next_times.push_back(out->next_time);
  }

  out_vars["file_number"] = torch::tensor(output_file_numbers, torch::kInt64);
  out_vars["next_time"] = torch::tensor(output_next_times, torch::kFloat64);

  // create filename: <basename>.<blockid>.<file_number>.restart
  std::string fname;
  char number[6];
  snprintf(number, sizeof(number), "%05d", file_number);
  char blockid[12];
  snprintf(blockid, sizeof(blockid), "block%d", pmb->options.dist().gid());

  fname.append(pmb->options.basename());
  fname.append(".");
  fname.append(blockid);
  fname.append(".");
  if (final_write) {
    fname.append("final");
  } else {
    fname.append(number);
  }
  fname.append(".restart");

  // save to disk
  kintera::save_tensors(out_vars, fname);
}

}  // namespace snap
