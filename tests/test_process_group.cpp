// C/C++
#include <unistd.h>

#include <cstdlib>
#include <optional>
#include <string>

// external
#include <gtest/gtest.h>

// snap
#include <snap/layout/distributed.hpp>
#include <snap/layout/layout.hpp>
#include <snap/layout/process_group.hpp>

using namespace snap;

namespace {

int unique_test_port() { return 29501 + (getpid() % 1000); }

class ScopedEnvVar {
 public:
  explicit ScopedEnvVar(char const* name) : name_(name) {
    if (auto* value = std::getenv(name)) {
      original_ = std::string(value);
    }
  }

  ~ScopedEnvVar() {
    if (original_) {
      setenv(name_.c_str(), original_->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::optional<std::string> original_;
};

LayoutOptions make_test_options() {
  auto opts = LayoutOptionsImpl::create();
  opts->backend("gloo");
  opts->master_addr("127.0.0.1");
  opts->master_port(unique_test_port());
  opts->process_rank(0);
  opts->rank(0);
  opts->root_rank(0);
  opts->local_rank(0);
  opts->process_world_size(1);
  opts->world_size(1);
  opts->blocks_per_process(1);
  return opts;
}

}  // namespace

TEST(LayoutOptions, DefaultsToAvailableCommunicationBackend) {
  auto opts = LayoutOptionsImpl::create();
#ifdef USE_UCX
  EXPECT_EQ(opts->backend(), "ucx");
#else
  EXPECT_EQ(opts->backend(), "gloo");
#endif
}

TEST(LayoutOptions, RandomizesDefaultMasterPortWhenEnvUnset) {
  ScopedEnvVar master_port("MASTER_PORT");
  unsetenv("MASTER_PORT");

  auto opts = LayoutOptionsImpl::create();

  EXPECT_GE(opts->master_port(), 29500);
  EXPECT_LE(opts->master_port(), 29600);
}

TEST(ProcessGroupContext, SkipsSingleProcessCommunication) {
  auto opts = make_test_options();
  auto ctx = ProcessGroupContext::create(opts);

  EXPECT_FALSE(ctx->owns_process_group());
  EXPECT_FALSE(ctx->pg.defined());

  set_process_group(nullptr);
}
