// external
#include <gtest/gtest.h>

// C/C++
#include <algorithm>
#include <cmath>
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

  double output_value(std::string const& name) const {
    for (auto* pdata = pfirst_data_; pdata != nullptr; pdata = pdata->pnext) {
      if (pdata->name == name) {
        return pdata->data(0, 0, 0, 0);
      }
    }
    throw std::runtime_error("missing output variable: " + name);
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
  options->hydro()->riemann()->type() = "lmars";
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

TEST(UserForcing, callback_adds_extra_hydro_and_scalar_tendencies) {
  auto block = make_block("ideal-gas", {"tracer_a"});
  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();

  Variables vars;
  vars["hydro_w"] = torch::zeros({block->phydro->peos->nvar(), nc3, nc2, nc1},
                                 torch::dtype(torch::kFloat64));
  vars["hydro_w"][IDN].fill_(1.0);
  vars["hydro_w"][IPR].fill_(3.0);
  vars["hydro_u"] = block->phydro->peos->compute("W->U", {vars["hydro_w"]});
  vars["scalar_s"] =
      torch::ones({1, nc3, nc2, nc1}, torch::dtype(torch::kFloat64));
  vars["scalar_r"] = vars["scalar_s"] / vars["hydro_u"][IDN].unsqueeze(0);

  block->user_forcing_callback = [](Variables const& forcing_vars, double dt,
                                    int stage) {
    EXPECT_DOUBLE_EQ(dt, 0.0);
    EXPECT_EQ(stage, 0);
    EXPECT_TRUE(forcing_vars.count("hydro_u"));
    EXPECT_TRUE(forcing_vars.count("scalar_s"));

    Variables out;
    out["hydro_du"] = torch::zeros_like(forcing_vars.at("hydro_u"));
    out["hydro_du"][IDN].fill_(0.5);
    out["scalar_ds"] = torch::full_like(forcing_vars.at("scalar_s"), 0.25);
    return out;
  };

  block->advance_local(vars, 0.0, 0);

  auto hydro_u = vars.at("hydro_u");
  auto scalar_s = vars.at("scalar_s");

  EXPECT_TRUE(torch::allclose(hydro_u[IDN], torch::full_like(hydro_u[IDN], 1.5),
                              1.e-12, 1.e-12));
  EXPECT_TRUE(torch::allclose(scalar_s, torch::full_like(scalar_s, 1.25),
                              1.e-12, 1.e-12));
}

TEST(UserForcing, callback_rejects_unsupported_keys) {
  auto block = make_block("ideal-gas");
  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();

  Variables vars;
  vars["hydro_w"] = torch::zeros({block->phydro->peos->nvar(), nc3, nc2, nc1},
                                 torch::dtype(torch::kFloat64));
  vars["hydro_w"][IDN].fill_(1.0);
  vars["hydro_w"][IPR].fill_(3.0);
  vars["hydro_u"] = block->phydro->peos->compute("W->U", {vars["hydro_w"]});

  block->user_forcing_callback = [](Variables const& forcing_vars, double,
                                    int) {
    Variables out;
    out["hydro_w"] = torch::zeros_like(forcing_vars.at("hydro_u"));
    return out;
  };

  EXPECT_THROW(block->advance_local(vars, 0.0, 0), c10::Error);
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

TEST(OutputSelection, scalar_statistics_are_explicitly_selected) {
  auto block = make_block("ideal-gas", {"tracer_a", "tracer_b"});

  Variables vars;
  vars["scalar_r"] = torch::ones({2, 1, 4, 4}, torch::kFloat64);
  vars["scalar_s"] = torch::ones({2, 1, 4, 4}, torch::kFloat64);

  auto scalar_opts = OutputOptionsImpl::create();
  scalar_opts->variables({"scalar_prim"});
  TestOutputType scalar_output(scalar_opts);
  scalar_output.load_scalar(block.get(), vars);
  auto scalar_names = scalar_output.output_names();
  EXPECT_TRUE(contains(scalar_names, "r_tracer_a"));
  EXPECT_FALSE(contains(scalar_names, "r_tracer_a_mean"));
  scalar_output.ClearOutputData();

  auto stat_opts = OutputOptionsImpl::create();
  stat_opts->variables({"scalar_stat"});
  TestOutputType stat_output(stat_opts);
  stat_output.load_scalar(block.get(), vars);
  auto stat_names = stat_output.output_names();
  EXPECT_TRUE(contains(stat_names, "r_tracer_a_mean"));
  EXPECT_TRUE(contains(stat_names, "r_tracer_a_std"));
  EXPECT_TRUE(contains(stat_names, "r_tracer_b_mean"));
  EXPECT_TRUE(contains(stat_names, "r_tracer_b_std"));
}

TEST(OutputSelection, primitive_statistics_are_explicitly_selected) {
  auto block = make_block("ideal-gas");

  Variables vars;
  vars["hydro_w"] =
      torch::ones({block->phydro->peos->nvar(), 1, 4, 4}, torch::kFloat64);
  vars["hydro_u"] = torch::ones_like(vars["hydro_w"]);

  auto prim_opts = OutputOptionsImpl::create();
  prim_opts->variables({"prim"});
  TestOutputType prim_output(prim_opts);
  prim_output.load_hydro(block.get(), vars);
  auto prim_names = prim_output.output_names();
  EXPECT_FALSE(contains(prim_names, "rho_mean"));
  EXPECT_FALSE(contains(prim_names, "vel1_mean"));
  prim_output.ClearOutputData();

  auto stat_opts = OutputOptionsImpl::create();
  stat_opts->variables({"prim_stat"});
  TestOutputType stat_output(stat_opts);
  stat_output.load_hydro(block.get(), vars);
  auto stat_names = stat_output.output_names();
  EXPECT_TRUE(contains(stat_names, "rho_mean"));
  EXPECT_TRUE(contains(stat_names, "rho_std"));
  EXPECT_TRUE(contains(stat_names, "press_mean"));
  EXPECT_TRUE(contains(stat_names, "press_std"));
  EXPECT_TRUE(contains(stat_names, "vel1_mean"));
  EXPECT_TRUE(contains(stat_names, "vel2_mean"));
  EXPECT_TRUE(contains(stat_names, "vel3_mean"));
  EXPECT_TRUE(contains(stat_names, "vel1_std"));
  EXPECT_TRUE(contains(stat_names, "vel2_std"));
  EXPECT_TRUE(contains(stat_names, "vel3_std"));
}

TEST(OutputStatistics, primitive_statistics_are_time_weighted_and_reset) {
  auto block = make_block("ideal-gas");

  Variables vars;
  vars["hydro_w"] =
      torch::ones({block->phydro->peos->nvar(), 1, 4, 4}, torch::kFloat64);
  vars["hydro_u"] = torch::ones_like(vars["hydro_w"]);

  auto opts = OutputOptionsImpl::create();
  opts->variables({"prim_stat"});
  TestOutputType output(opts);

  vars["hydro_w"].fill_(1.0);
  output.AccumulatePrimStat(vars, 0.0);
  output.load_hydro(block.get(), vars);
  EXPECT_DOUBLE_EQ(output.output_value("rho_mean"), 1.0);
  EXPECT_DOUBLE_EQ(output.output_value("rho_std"), 0.0);
  output.ClearOutputData();

  vars["hydro_w"].fill_(3.0);
  output.AccumulatePrimStat(vars, 1.0);
  vars["hydro_w"].fill_(5.0);
  output.AccumulatePrimStat(vars, 3.0);

  output.load_hydro(block.get(), vars);
  EXPECT_NEAR(output.output_value("rho_mean"), 13.0 / 3.0, 1.e-12);
  EXPECT_NEAR(output.output_value("rho_std"), std::sqrt(8.0 / 9.0), 1.e-12);
  EXPECT_NEAR(output.output_value("vel1_mean"), 13.0 / 3.0, 1.e-12);
  EXPECT_NEAR(output.output_value("vel1_std"), std::sqrt(8.0 / 9.0), 1.e-12);
  output.ClearOutputData();

  output.ResetPrimStat(3.0);
  vars["hydro_w"].fill_(7.0);
  output.AccumulatePrimStat(vars, 4.0);
  output.load_hydro(block.get(), vars);
  EXPECT_DOUBLE_EQ(output.output_value("rho_mean"), 7.0);
  EXPECT_DOUBLE_EQ(output.output_value("rho_std"), 0.0);
}

TEST(OutputStatistics, scalar_statistics_are_time_weighted_and_reset) {
  auto block = make_block("ideal-gas", {"tracer_a"});

  Variables vars;
  vars["scalar_r"] = torch::ones({1, 1, 4, 4}, torch::kFloat64);

  auto opts = OutputOptionsImpl::create();
  opts->variables({"scalar_stat"});
  TestOutputType output(opts);

  vars["scalar_r"].fill_(1.0);
  output.AccumulatePrimStat(vars, 0.0);
  output.load_scalar(block.get(), vars);
  EXPECT_DOUBLE_EQ(output.output_value("r_tracer_a_mean"), 1.0);
  EXPECT_DOUBLE_EQ(output.output_value("r_tracer_a_std"), 0.0);
  output.ClearOutputData();

  vars["scalar_r"].fill_(3.0);
  output.AccumulatePrimStat(vars, 1.0);
  vars["scalar_r"].fill_(5.0);
  output.AccumulatePrimStat(vars, 3.0);

  output.load_scalar(block.get(), vars);
  EXPECT_NEAR(output.output_value("r_tracer_a_mean"), 13.0 / 3.0, 1.e-12);
  EXPECT_NEAR(output.output_value("r_tracer_a_std"), std::sqrt(8.0 / 9.0),
              1.e-12);
  output.ClearOutputData();

  output.ResetPrimStat(3.0);
  vars["scalar_r"].fill_(7.0);
  output.AccumulatePrimStat(vars, 4.0);
  output.load_scalar(block.get(), vars);
  EXPECT_DOUBLE_EQ(output.output_value("r_tracer_a_mean"), 7.0);
  EXPECT_DOUBLE_EQ(output.output_value("r_tracer_a_std"), 0.0);
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
