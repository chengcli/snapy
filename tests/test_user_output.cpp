// external
#include <gtest/gtest.h>

// torch
#include <torch/torch.h>

// snap
#include <snap/mesh/meshblock.hpp>
#include <snap/output/output_type.hpp>

using namespace snap;

namespace {

class TestOutputType : public OutputType {
 public:
  explicit TestOutputType(OutputOptions const& options) : OutputType(options) {}

  void load_user_output(MeshBlockImpl* pmb, Variables const& vars) {
    loadUserOutputData(pmb, vars);
  }
};

}  // namespace

TEST(UserOutput, missing_callback_reports_meaningful_error) {
  auto options = OutputOptionsImpl::create();
  options->variables({"uov"});
  TestOutputType output(options);

  MeshBlockImpl block;
  Variables vars;
  vars["hydro_w"] = torch::ones({1, 1, 1, 1});

  try {
    output.load_user_output(&block, vars);
    FAIL() << "Expected missing user output callback to throw";
  } catch (std::exception const& exc) {
    auto msg = std::string(exc.what());
    EXPECT_NE(msg.find("set_user_output_func"), std::string::npos);
    EXPECT_NE(msg.find("uov"), std::string::npos);
  }
}

TEST(UserOutput, registered_callback_allows_uov_output) {
  auto options = OutputOptionsImpl::create();
  options->variables({"uov"});
  TestOutputType output(options);

  MeshBlockImpl block;
  block.user_output_callback = [](Variables const&) {
    Variables out;
    out["my_uov"] = torch::ones({1, 1, 1});
    return out;
  };

  Variables vars;
  vars["hydro_w"] = torch::ones({1, 1, 1, 1});

  EXPECT_NO_THROW(output.load_user_output(&block, vars));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
