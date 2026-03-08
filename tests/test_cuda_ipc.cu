#include <torch/torch.h>
#include <snap/layout/layout.hpp>
#include <snap/utils/cuda_ipc_pg.cuh>

int main() {
  int rank = snap::get_rank();
  std::cout << "I'm rank " << rank << std::endl;

  int world_size = snap::get_world_size();
  std::cout << "World size " << world_size << std::endl;

  auto pg = std::make_shared<snapy::distributed::CudaIpcProcessGroup>(
      rank,
      world_size,          // must be 2 in this version
      0,                   // shared GPU index
      "/tmp/snapy_cuda_ipc.sock",
      64 * 1024 * 1024,    // slot bytes
      8);                  // num slots

  std::vector<c10::intrusive_ptr<c10d::Work>> works;
  auto opts = torch::dtype(torch::kDouble).device(torch::kCUDA, 1);

  std::vector<torch::Tensor> send_data = {(1 + rank) * torch::ones(2, opts)};
  std::vector<torch::Tensor> recv_data = {(1 + rank) * torch::zeros(2, opts)};

  auto send_work = pg->send(send_data, 1 - rank, 0);
  works.push_back(send_work);

  auto recv_work = pg->recv(recv_data, 1 - rank, 0);
  works.push_back(recv_work);

  for (auto& w : works) w->wait();

  cudaDeviceSynchronize();

  std::cout << "rank " << rank << " send = " << send_data[0] << std::endl;
  std::cout << "rank " << rank << " recv = " << recv_data[0] << std::endl;

  // Temporary debugging delay to reduce teardown race
  //std::this_thread::sleep_for(std::chrono::seconds(1));

  return 0;
}
