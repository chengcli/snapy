#include "cuda_ipc_pg.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <torch/csrc/distributed/c10d/Types.hpp>

#include <cerrno>
#include <stdexcept>

namespace snapy {
namespace distributed {

#define TORCH_CHECK_CUDA_TENSOR(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define TORCH_CHECK_CONTIGUOUS_TENSOR(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

CudaIpcWork::CudaIpcWork(
    c10d::OpType op_type,
    std::shared_future<void> fut,
    std::vector<at::Tensor> result)
    : c10d::Work(-1, op_type), fut_(std::move(fut)), result_(std::move(result)) {}

bool CudaIpcWork::isCompleted() {
  return fut_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
}

bool CudaIpcWork::isSuccess() const {
  return fut_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready;
}

bool CudaIpcWork::wait(std::chrono::milliseconds timeout) {
  if (timeout == kUnsetTimeout) {
    fut_.wait();
    return true;
  }
  return fut_.wait_for(timeout) == std::future_status::ready;
}

c10::intrusive_ptr<c10::ivalue::Future> CudaIpcWork::getFuture() {
  auto fut = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
  fut_.wait();
  fut->markCompleted(c10::IValue(result_));
  return fut;
}

/************ UnixSocket ************/

CudaIpcProcessGroup::UnixSocket::UnixSocket(int fd) : fd_(fd) {}

CudaIpcProcessGroup::UnixSocket::~UnixSocket() {
  close();
}

CudaIpcProcessGroup::UnixSocket::UnixSocket(UnixSocket&& other) noexcept : fd_(other.fd_) {
  other.fd_ = -1;
}

auto CudaIpcProcessGroup::UnixSocket::operator=(UnixSocket&& other) noexcept -> UnixSocket& {
  if (this != &other) {
    close();
    fd_ = other.fd_;
    other.fd_ = -1;
  }
  return *this;
}

void CudaIpcProcessGroup::UnixSocket::close() {
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

auto CudaIpcProcessGroup::UnixSocket::server_accept(const std::string& path) -> UnixSocket {
  int listen_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  sys_check(listen_fd >= 0, "socket");

  ::unlink(path.c_str());

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  TORCH_CHECK(path.size() < sizeof(addr.sun_path), "socket path too long");
  std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

  int rc = ::bind(listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
  sys_check(rc == 0, "bind");

  rc = ::listen(listen_fd, 1);
  sys_check(rc == 0, "listen");

  int conn_fd = ::accept(listen_fd, nullptr, nullptr);
  sys_check(conn_fd >= 0, "accept");

  ::close(listen_fd);
  return UnixSocket(conn_fd);
}

auto CudaIpcProcessGroup::UnixSocket::client_connect(const std::string& path) -> UnixSocket {
  int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  sys_check(fd >= 0, "socket");

  sockaddr_un addr{};
  addr.sun_family = AF_UNIX;
  TORCH_CHECK(path.size() < sizeof(addr.sun_path), "socket path too long");
  std::strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

  int rc = -1;
  for (int i = 0; i < 500; ++i) {
    rc = ::connect(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr));
    if (rc == 0) {
      return UnixSocket(fd);
    }
    if (errno != ENOENT && errno != ECONNREFUSED) {
      break;
    }
    ::usleep(10000);
  }
  sys_check(rc == 0, "connect");
  return UnixSocket(fd);
}

void CudaIpcProcessGroup::UnixSocket::send_all(const void* buf, size_t len) {
  const char* p = static_cast<const char*>(buf);
  while (len > 0) {
    ssize_t n = ::send(fd_, p, len, 0);
    sys_check(n >= 0, "send");
    p += n;
    len -= static_cast<size_t>(n);
  }
}

void CudaIpcProcessGroup::UnixSocket::recv_all(void* buf, size_t len) {
  char* p = static_cast<char*>(buf);
  while (len > 0) {
    ssize_t n = ::recv(fd_, p, len, MSG_WAITALL);
    sys_check(n > 0, "recv");
    p += n;
    len -= static_cast<size_t>(n);
  }
}

/************ Helpers ************/

void CudaIpcProcessGroup::cuda_check(cudaError_t err, const char* what) {
  TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err));
}

void CudaIpcProcessGroup::sys_check(bool ok, const char* what) {
  TORCH_CHECK(ok, what, ": ", std::strerror(errno));
}

void CudaIpcProcessGroup::check_tensor_list(const std::vector<at::Tensor>& tensors) {
  for (const auto& t : tensors) {
    TORCH_CHECK_CUDA_TENSOR(t);
    TORCH_CHECK_CONTIGUOUS_TENSOR(t);
  }
}

size_t CudaIpcProcessGroup::tensor_nbytes(const at::Tensor& t) {
  return static_cast<size_t>(t.numel()) * static_cast<size_t>(t.element_size());
}

/************ CudaIpcProcessGroup ************/

CudaIpcProcessGroup::CudaIpcProcessGroup(
    int rank,
    int world_size,
    int device_index,
    std::string socket_path,
    size_t slot_bytes,
    int num_slots)
    : rank_(rank),
      world_size_(world_size),
      device_index_(device_index),
      socket_path_(std::move(socket_path)),
      slot_bytes_(slot_bytes),
      num_slots_(num_slots) {
  TORCH_CHECK(world_size_ == 2, "This implementation supports exactly 2 ranks");
  TORCH_CHECK(rank_ == 0 || rank_ == 1, "rank must be 0 or 1");
  TORCH_CHECK(num_slots_ >= 1, "num_slots must be >= 1");

  cuda_check(cudaSetDevice(device_index_), "cudaSetDevice");
  cuda_check(cudaStreamCreate(&stream_), "cudaStreamCreate");

  local_slots_.resize(num_slots_);
  remote_slots_.resize(num_slots_);

  create_local_slots();
  establish_socket();
  exchange_slot_descriptors();

  running_.store(true);
  progress_thread_ = std::thread(&CudaIpcProcessGroup::progress_loop, this);
}

CudaIpcProcessGroup::~CudaIpcProcessGroup() {
  if (running_.exchange(false)) {
    if (sock_.valid()) {
      CtrlMsg msg{};
      msg.type = CtrlType::kShutdown;
      std::lock_guard<std::mutex> g(sock_mu_);
      if (sock_.valid()) {
        // Best effort
        ::send(sock_.fd(), &msg, sizeof(msg), MSG_NOSIGNAL);
      }
    }
  }

  if (progress_thread_.joinable()) {
    progress_thread_.join();
  }

  cleanup_remote_slots();
  cleanup_local_slots();

  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }

  sock_.close();

  if (rank_ == 0) {
    ::unlink(socket_path_.c_str());
  }
}

size_t CudaIpcProcessGroup::total_nbytes(const std::vector<at::Tensor>& tensors) const {
  size_t total = 0;
  for (const auto& t : tensors) {
    total += tensor_nbytes(t);
  }
  return total;
}

void CudaIpcProcessGroup::pack_batch_cuda(const std::vector<at::Tensor>& tensors, void* dst) {
  char* base = static_cast<char*>(dst);
  size_t offset = 0;
  for (const auto& t : tensors) {
    const size_t nbytes = tensor_nbytes(t);
    cuda_check(
        cudaMemcpyAsync(
            base + offset,
            t.data_ptr(),
            nbytes,
            cudaMemcpyDeviceToDevice,
            stream_),
        "cudaMemcpyAsync pack");
    offset += nbytes;
  }
}

void CudaIpcProcessGroup::unpack_batch_cuda(const void* src, const std::vector<at::Tensor>& tensors) {
  const char* base = static_cast<const char*>(src);
  size_t offset = 0;
  for (const auto& t : tensors) {
    const size_t nbytes = tensor_nbytes(t);
    cuda_check(
        cudaMemcpyAsync(
            t.data_ptr(),
            base + offset,
            nbytes,
            cudaMemcpyDeviceToDevice,
            stream_),
        "cudaMemcpyAsync unpack");
    offset += nbytes;
  }
}

void CudaIpcProcessGroup::create_local_slots() {
  for (int i = 0; i < num_slots_; ++i) {
    auto& s = local_slots_[i];
    cuda_check(cudaMalloc(&s.ptr, slot_bytes_), "cudaMalloc slot");
    s.capacity = slot_bytes_;
    cuda_check(
        cudaEventCreateWithFlags(
            &s.event,
            cudaEventDisableTiming | cudaEventInterprocess),
        "cudaEventCreateWithFlags");
  }
}

void CudaIpcProcessGroup::cleanup_local_slots() {
  for (auto& s : local_slots_) {
    if (s.event) {
      cudaEventDestroy(s.event);
      s.event = nullptr;
    }
    if (s.ptr) {
      cudaFree(s.ptr);
      s.ptr = nullptr;
    }
  }
}

void CudaIpcProcessGroup::cleanup_remote_slots() {
  for (auto& s : remote_slots_) {
    if (s.event) {
      cudaEventDestroy(s.event);
      s.event = nullptr;
    }
    if (s.ptr) {
      cuda_check(cudaIpcCloseMemHandle(s.ptr), "cudaIpcCloseMemHandle");
      s.ptr = nullptr;
    }
  }
}

void CudaIpcProcessGroup::establish_socket() {
  if (rank_ == 0) {
    sock_ = UnixSocket::server_accept(socket_path_);
  } else {
    sock_ = UnixSocket::client_connect(socket_path_);
  }
}

void CudaIpcProcessGroup::exchange_slot_descriptors() {
  std::vector<ExportSlotDesc> mine(num_slots_);
  std::vector<ExportSlotDesc> peer(num_slots_);

  for (int i = 0; i < num_slots_; ++i) {
    mine[i].capacity = local_slots_[i].capacity;
    cuda_check(cudaIpcGetMemHandle(&mine[i].mem_handle, local_slots_[i].ptr), "cudaIpcGetMemHandle");
    cuda_check(cudaIpcGetEventHandle(&mine[i].event_handle, local_slots_[i].event), "cudaIpcGetEventHandle");
  }

  {
    std::lock_guard<std::mutex> g(sock_mu_);
    if (rank_ == 0) {
      sock_.send_all(mine.data(), mine.size() * sizeof(ExportSlotDesc));
      sock_.recv_all(peer.data(), peer.size() * sizeof(ExportSlotDesc));
    } else {
      sock_.recv_all(peer.data(), peer.size() * sizeof(ExportSlotDesc));
      sock_.send_all(mine.data(), mine.size() * sizeof(ExportSlotDesc));
    }
  }

  for (int i = 0; i < num_slots_; ++i) {
    auto& s = remote_slots_[i];
    s.capacity = static_cast<size_t>(peer[i].capacity);
    cuda_check(
        cudaIpcOpenMemHandle(
            &s.ptr,
            peer[i].mem_handle,
            cudaIpcMemLazyEnablePeerAccess),
        "cudaIpcOpenMemHandle");
    cuda_check(cudaIpcOpenEventHandle(&s.event, peer[i].event_handle), "cudaIpcOpenEventHandle");
  }
}

int CudaIpcProcessGroup::other_rank() const {
  return rank_ == 0 ? 1 : 0;
}

int CudaIpcProcessGroup::acquire_send_slot() {
  while (true) {
    for (int i = 0; i < num_slots_; ++i) {
      auto& s = local_slots_[i];
      if (s.acked_seq + 1 == s.next_seq) {
        return i;
      }
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

void CudaIpcProcessGroup::send_ctrl(const CtrlMsg& msg) {
  std::lock_guard<std::mutex> g(sock_mu_);
  sock_.send_all(&msg, sizeof(msg));
}

void CudaIpcProcessGroup::send_ack(int peer, uint32_t slot, uint64_t seq, int tag) {
  CtrlMsg msg{};
  msg.type = CtrlType::kAck;
  msg.slot = slot;
  msg.seq = seq;
  msg.nbytes = 0;
  msg.tag = tag;
  msg.peer = peer;
  send_ctrl(msg);
}

auto CudaIpcProcessGroup::pop_matching_msg(int peer, int tag) -> PendingMsg {
  std::unique_lock<std::mutex> lk(pending_mu_);
  while (true) {
    for (auto it = pending_.begin(); it != pending_.end(); ++it) {
      if (it->peer == peer && it->tag == tag) {
        PendingMsg out = *it;
        pending_.erase(it);
        return out;
      }
    }
    pending_cv_.wait(lk);
  }
}

void CudaIpcProcessGroup::handle_ctrl_msg(const CtrlMsg& msg) {
  if (msg.type == CtrlType::kAck) {
    TORCH_CHECK(msg.slot < static_cast<uint32_t>(num_slots_), "bad ack slot");
    local_slots_[msg.slot].acked_seq = msg.seq;
    return;
  }

  if (msg.type == CtrlType::kData) {
    PendingMsg p{};
    p.slot = msg.slot;
    p.seq = msg.seq;
    p.nbytes = msg.nbytes;
    p.tag = msg.tag;
    p.peer = msg.peer;
    {
      std::lock_guard<std::mutex> g(pending_mu_);
      pending_.push_back(p);
    }
    pending_cv_.notify_all();
    return;
  }

  if (msg.type == CtrlType::kShutdown) {
    running_.store(false);
    pending_cv_.notify_all();
    return;
  }

  TORCH_CHECK(false, "unknown control message");
}

void CudaIpcProcessGroup::progress_loop() {
  while (running_.load()) {
    CtrlMsg msg{};
    {
      std::lock_guard<std::mutex> g(sock_mu_);
      if (!sock_.valid()) {
        break;
      }
      ssize_t n = ::recv(sock_.fd(), &msg, sizeof(msg), MSG_WAITALL);
      if (n == 0) {
        break;
      }
      sys_check(n == static_cast<ssize_t>(sizeof(msg)), "recv control");
    }
    handle_ctrl_msg(msg);
  }
}

c10::intrusive_ptr<c10d::Work> CudaIpcProcessGroup::send(
    std::vector<at::Tensor>& tensors,
    int dst,
    int tag) {
  check_tensor_list(tensors);
  TORCH_CHECK(dst == other_rank(), "only one peer supported");

  auto promise = std::make_shared<std::promise<void>>();
  auto fut = promise->get_future().share();

  std::vector<at::Tensor> result = tensors;

  std::thread([this, tensors, dst, tag, promise]() mutable {
    try {
      const size_t nbytes = total_nbytes(tensors);
      TORCH_CHECK(nbytes <= slot_bytes_, "message exceeds slot capacity");

      int slot = acquire_send_slot();
      auto& s = local_slots_[slot];
      uint64_t seq = s.next_seq++;

      pack_batch_cuda(tensors, s.ptr);
      cuda_check(cudaEventRecord(s.event, stream_), "cudaEventRecord");

      CtrlMsg msg{};
      msg.type = CtrlType::kData;
      msg.slot = static_cast<uint32_t>(slot);
      msg.seq = seq;
      msg.nbytes = nbytes;
      msg.tag = tag;
      msg.peer = rank_;
      send_ctrl(msg);

      promise->set_value();
    } catch (...) {
      promise->set_exception(std::current_exception());
    }
  }).detach();

  return c10::make_intrusive<CudaIpcWork>(c10d::OpType::SEND, fut, std::move(result));
}

c10::intrusive_ptr<c10d::Work> CudaIpcProcessGroup::recv(
    std::vector<at::Tensor>& tensors,
    int src,
    int tag) {
  check_tensor_list(tensors);
  TORCH_CHECK(src == other_rank(), "only one peer supported");

  auto promise = std::make_shared<std::promise<void>>();
  auto fut = promise->get_future().share();

  std::vector<at::Tensor> result = tensors;

  std::thread([this, tensors, src, tag, promise]() mutable {
    try {
      PendingMsg msg = pop_matching_msg(src, tag);
      TORCH_CHECK(msg.slot < static_cast<uint32_t>(num_slots_), "bad recv slot");
      TORCH_CHECK(msg.nbytes <= total_nbytes(tensors), "destination tensors too small");

      auto& s = remote_slots_[msg.slot];
      cuda_check(cudaStreamWaitEvent(stream_, s.event, 0), "cudaStreamWaitEvent");
      unpack_batch_cuda(s.ptr, tensors);
      cuda_check(cudaStreamSynchronize(stream_), "cudaStreamSynchronize recv");

      send_ack(src, msg.slot, msg.seq, tag);
      promise->set_value();
    } catch (...) {
      promise->set_exception(std::current_exception());
    }
  }).detach();

  return c10::make_intrusive<CudaIpcWork>(c10d::OpType::RECV, fut, std::move(result));
}

} // namespace distributed
} // namespace snapy
