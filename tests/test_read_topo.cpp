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
  auto data = load_tensors("topo_20m_new.pt");

  std::cout << "solid = " << data["solid_20m"].sizes() << std::endl;
}

int main(int argc, char* argv[]) { test1(); }
