// torch
#include <torch/script.h>
#include <torch/torch.h>

std::map<std::string, torch::Tensor> load_tensors(const std::string& filename) {
  std::map<std::string, torch::Tensor> tensor_map;

  // Load scripted module
  torch::jit::script::Module module = torch::jit::load(filename);

  // Get all named buffers
  for (const auto& p : module.named_buffers(/*recurse=*/false)) {
    tensor_map[p.name] = p.value;
  }

  // Optionally, also load parameters (if register_parameter was used)
  for (const auto& p : module.named_parameters(/*recurse=*/false)) {
    tensor_map[p.name] = p.value;
  }

  return tensor_map;
}

void test1() {
  // read topo_20m.pt
  auto data =
      load_tensors("topo_20kmx10km_32.7719_32.9881_-106.5098_-106.3802.pt");

  std::cout << "solid10 = " << data["solid_10m"].sizes() << std::endl;
  std::cout << "solid20 = " << data["solid_20m"].sizes() << std::endl;
  std::cout << "solid40 = " << data["solid_40m"].sizes() << std::endl;
  std::cout << "solid80 = " << data["solid_80m"].sizes() << std::endl;
  std::cout << "solid160 = " << data["solid_160m"].sizes() << std::endl;
  // std::cout << data["solid_160m"] << std::endl;
}

int main(int argc, char* argv[]) { test1(); }
