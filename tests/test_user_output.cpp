// external
#include <gtest/gtest.h>

// C/C++
#include <algorithm>
#include <filesystem>
#include <memory>
#include <vector>

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

  void load_hydro(MeshBlockImpl* pmb, Variables const& vars) {
    loadHydroOutputData(pmb, vars);
  }

  void load_scalar(MeshBlockImpl* pmb, Variables const& vars) {
    loadScalarOutputData(pmb, vars);
  }

  void load_user_output(MeshBlockImpl* pmb, Variables const& vars) {
    loadUserOutputData(pmb, vars);
  }

  std::vector<std::string> output_names() const {
    std::vector<std::string> names;
    for (auto* pdata = pfirst_data_; pdata != nullptr; pdata = pdata->pnext) {
      names.push_back(pdata->name);
    }
    return names;
  }
};

std::shared_ptr<MeshBlockImpl> make_block(
    std::string eos_type, std::vector<std::string> scalars = {}) {
  std::vector<std::filesystem::path> candidates = {
      "test_coordinate.yaml",
      std::filesystem::path("tests") / "test_coordinate.yaml",
      std::filesystem::path("build") / "tests" / "test_coordinate.yaml",
  };
  auto it = std::find_if(candidates.begin(), candidates.end(),
                         [](std::filesystem::path const& path) {
                           return std::filesystem::exists(path);
                         });
  if (it == candidates.end()) {
    throw std::runtime_error("could not locate test_coordinate.yaml");
  }

  auto options = MeshBlockOptionsImpl::from_yaml(it->string());
  options->hydro()->eos()->type() = std::move(eos_type);
  options->scalar()->nvar() = scalars.size();
  options->scalar()->names() = std::move(scalars);
  return std::make_shared<MeshBlockImpl>(options);
}

bool contains(std::vector<std::string> const& names, std::string const& name) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

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

TEST(OutputSelection, hydro_fields_depend_on_eos_and_requested_mode) {
  auto shallow_block = make_block("shallow-water");
  auto ideal_block = make_block("ideal-gas");

  Variables shallow_vars;
  shallow_vars["hydro_w"] = torch::ones(
      {shallow_block->phydro->peos->nvar(), 1, 4, 4}, torch::kFloat64);
  shallow_vars["hydro_u"] = torch::ones_like(shallow_vars["hydro_w"]);

  Variables ideal_vars;
  ideal_vars["hydro_w"] = torch::ones(
      {ideal_block->phydro->peos->nvar(), 1, 4, 4}, torch::kFloat64);
  ideal_vars["hydro_u"] = torch::ones_like(ideal_vars["hydro_w"]);

  auto prim_opts = OutputOptionsImpl::create();
  prim_opts->variables({"prim"});
  TestOutputType prim_output(prim_opts);
  prim_output.load_hydro(shallow_block.get(), shallow_vars);
  auto prim_names = prim_output.output_names();
  EXPECT_TRUE(contains(prim_names, "rho"));
  EXPECT_TRUE(contains(prim_names, "vel"));
  EXPECT_FALSE(contains(prim_names, "press"));
  prim_output.ClearOutputData();

  auto cons_opts = OutputOptionsImpl::create();
  cons_opts->variables({"cons"});
  TestOutputType cons_output(cons_opts);
  cons_output.load_hydro(shallow_block.get(), shallow_vars);
  auto shallow_cons_names = cons_output.output_names();
  EXPECT_TRUE(contains(shallow_cons_names, "dens"));
  EXPECT_TRUE(contains(shallow_cons_names, "mom"));
  EXPECT_FALSE(contains(shallow_cons_names, "Etot"));
  cons_output.ClearOutputData();

  cons_output.load_hydro(ideal_block.get(), ideal_vars);
  auto ideal_cons_names = cons_output.output_names();
  EXPECT_TRUE(contains(ideal_cons_names, "dens"));
  EXPECT_TRUE(contains(ideal_cons_names, "mom"));
  EXPECT_TRUE(contains(ideal_cons_names, "Etot"));
}

TEST(OutputSelection, scalar_fields_follow_primitive_and_conserved_requests) {
  auto block = make_block("ideal-gas", {"tracer_a", "tracer_b"});

  Variables vars;
  vars["scalar_r"] = torch::ones({2, 1, 4, 4}, torch::kFloat64);
  vars["scalar_s"] = torch::ones({2, 1, 4, 4}, torch::kFloat64);

  auto scalar_opts = OutputOptionsImpl::create();
  scalar_opts->variables({"scalar"});
  TestOutputType scalar_output(scalar_opts);
  scalar_output.load_scalar(block.get(), vars);

  auto names = scalar_output.output_names();
  EXPECT_TRUE(contains(names, "r_tracer_a"));
  EXPECT_TRUE(contains(names, "r_tracer_b"));
  EXPECT_TRUE(contains(names, "s_tracer_a"));
  EXPECT_TRUE(contains(names, "s_tracer_b"));
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
