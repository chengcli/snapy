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
  ScopedEnvVar process_world_size("PROCESS_WORLD_SIZE");
  ScopedEnvVar world_size("WORLD_SIZE");
  unsetenv("MASTER_PORT");
  unsetenv("PROCESS_WORLD_SIZE");
  unsetenv("WORLD_SIZE");

  auto opts = LayoutOptionsImpl::create();

  EXPECT_GE(opts->master_port(), 29500);
  EXPECT_LE(opts->master_port(), 29600);
}

TEST(LayoutOptions, RequiresMasterPortForMultiProcessWhenEnvUnset) {
  ScopedEnvVar master_port("MASTER_PORT");
  ScopedEnvVar process_world_size("PROCESS_WORLD_SIZE");
  ScopedEnvVar world_size("WORLD_SIZE");
  unsetenv("MASTER_PORT");
  setenv("PROCESS_WORLD_SIZE", "2", 1);
  unsetenv("WORLD_SIZE");

  EXPECT_THROW((void)LayoutOptionsImpl::create(), c10::Error);
}

TEST(LayoutOptions, RequiresMasterPortWhenWorldSizeImpliesMultiProcess) {
  ScopedEnvVar master_port("MASTER_PORT");
  ScopedEnvVar process_world_size("PROCESS_WORLD_SIZE");
  ScopedEnvVar world_size("WORLD_SIZE");
  unsetenv("MASTER_PORT");
  unsetenv("PROCESS_WORLD_SIZE");
  setenv("WORLD_SIZE", "2", 1);

  EXPECT_THROW((void)LayoutOptionsImpl::create(), c10::Error);
}

TEST(LayoutOptions, UsesProvidedMasterPortForMultiProcess) {
  ScopedEnvVar master_port("MASTER_PORT");
  ScopedEnvVar process_world_size("PROCESS_WORLD_SIZE");
  ScopedEnvVar world_size("WORLD_SIZE");
  setenv("MASTER_PORT", "29577", 1);
  setenv("PROCESS_WORLD_SIZE", "2", 1);
  setenv("WORLD_SIZE", "2", 1);

  auto opts = LayoutOptionsImpl::create();

  EXPECT_EQ(opts->master_port(), 29577);
  EXPECT_EQ(opts->process_world_size(), 2);
  EXPECT_EQ(opts->world_size(), 2);
}

TEST(ProcessGroupContext, SkipsSingleProcessCommunication) {
  auto opts = make_test_options();
  auto ctx = ProcessGroupContext::create(opts);

  EXPECT_FALSE(ctx->owns_process_group());
  EXPECT_FALSE(ctx->pg.defined());

  set_process_group(nullptr);
}
