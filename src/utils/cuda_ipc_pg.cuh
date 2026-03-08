#pragma once

#include <ATen/ATen.h>
#include <c10/core/Device.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/Work.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>

#include <cuda_runtime.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace snapy {
namespace distributed {

class CudaIpcWork : public c10d::Work {
 public:
  CudaIpcWork(
      c10d::OpType op_type,
      std::shared_future<void> fut,
      std::vector<at::Tensor> result = {});

  bool isCompleted() override;
  bool isSuccess() const override;
  bool wait(std::chrono::milliseconds timeout = c10d::kUnsetTimeout) override;
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

 private:
  std::shared_future<void> fut_;
  std::vector<at::Tensor> result_;
};

class CudaIpcProcessGroup {
 public:
  CudaIpcProcessGroup(
      int rank,
      int world_size,
      int device_index,
      std::string socket_path,
      size_t slot_bytes = 64 * 1024 * 1024,
      int num_slots = 8);

  ~CudaIpcProcessGroup();

  c10::intrusive_ptr<c10d::Work> send(
      std::vector<at::Tensor>& tensors,
      int dst,
      int tag);

  c10::intrusive_ptr<c10d::Work> recv(
      std::vector<at::Tensor>& tensors,
      int src,
      int tag);

  int rank() const { return rank_; }
  int world_size() const { return world_size_; }
  int device_index() const { return device_index_; }

 private:
  struct ExportSlotDesc {
    uint64_t capacity;
    cudaIpcMemHandle_t mem_handle;
    cudaIpcEventHandle_t event_handle;
  };

  enum class CtrlType : uint32_t {
    kData = 1,
    kAck = 2,
    kShutdown = 3,
  };

  struct CtrlMsg {
    CtrlType type;
    uint32_t slot;
    uint64_t seq;
    uint64_t nbytes;
    int32_t tag;
    int32_t peer;
  };

  struct PendingMsg {
    uint32_t slot;
    uint64_t seq;
    uint64_t nbytes;
    int32_t tag;
    int32_t peer;
  };

  struct LocalSlot {
    void* ptr{nullptr};
    size_t capacity{0};
    cudaEvent_t event{nullptr};
    uint64_t next_seq{1};
    uint64_t acked_seq{0};
  };

  struct RemoteSlot {
    void* ptr{nullptr};
    size_t capacity{0};
    cudaEvent_t event{nullptr};
  };

  class UnixSocket {
   public:
    explicit UnixSocket(int fd = -1);
    ~UnixSocket();

    UnixSocket(const UnixSocket&) = delete;
    UnixSocket& operator=(const UnixSocket&) = delete;
    UnixSocket(UnixSocket&& other) noexcept;
    UnixSocket& operator=(UnixSocket&& other) noexcept;

    int fd() const { return fd_; }
    bool valid() const { return fd_ >= 0; }
    void close();

    static UnixSocket server_accept(const std::string& path);
    static UnixSocket client_connect(const std::string& path);

    void send_all(const void* buf, size_t len);
    void recv_all(void* buf, size_t len);

   private:
    int fd_;
  };

  static void cuda_check(cudaError_t err, const char* what);
  static void sys_check(bool ok, const char* what);
  static void check_tensor_list(const std::vector<at::Tensor>& tensors);
  static size_t tensor_nbytes(const at::Tensor& t);

  size_t total_nbytes(const std::vector<at::Tensor>& tensors) const;
  void pack_batch_cuda(const std::vector<at::Tensor>& tensors, void* dst);
  void unpack_batch_cuda(const void* src, const std::vector<at::Tensor>& tensors);

  void create_local_slots();
  void cleanup_local_slots();
  void cleanup_remote_slots();

  void establish_socket();
  void exchange_slot_descriptors();

  int other_rank() const;

  int acquire_send_slot();
  void send_ctrl(const CtrlMsg& msg);
  void send_ack(int peer, uint32_t slot, uint64_t seq, int tag);

  PendingMsg pop_matching_msg(int peer, int tag);

  void progress_loop();
  void handle_ctrl_msg(const CtrlMsg& msg);

 private:
  int rank_;
  int world_size_;
  int device_index_;
  std::string socket_path_;
  size_t slot_bytes_;
  int num_slots_;

  UnixSocket sock_;
  std::mutex sock_mu_;

  cudaStream_t stream_{nullptr};

  std::vector<LocalSlot> local_slots_;
  std::vector<RemoteSlot> remote_slots_;

  std::atomic<bool> running_{false};
  std::thread progress_thread_;

  std::mutex pending_mu_;
  std::condition_variable pending_cv_;
  std::deque<PendingMsg> pending_;

  std::mutex ack_mu_;
};

} // namespace distributed
} // namespace snapy
