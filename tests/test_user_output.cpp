// external
#include <gtest/gtest.h>

// C/C++
#include <netcdf.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <vector>

// torch
#include <torch/torch.h>

// snap
#include <snap/mesh/meshblock.hpp>
#include <snap/output/output_formats.hpp>
#include <snap/output/output_type.hpp>

using namespace snap;

namespace {

class TestOutputType : public OutputType {
 public:
  explicit TestOutputType(OutputOptions const &options) : OutputType(options) {}

  void load_hydro(MeshBlockImpl *pmb, Variables const &vars) {
    loadHydroOutputData(pmb, vars);
  }

  void load_scalar(MeshBlockImpl *pmb, Variables const &vars) {
    loadScalarOutputData(pmb, vars);
  }

  void load_user_output(MeshBlockImpl *pmb, Variables const &vars) {
    loadUserOutputData(pmb, vars);
  }

  void append(std::string name, torch::Tensor const &tensor) {
    appendTensorSliceOutput("SCALARS", std::move(name), tensor, 4, 0,
                            tensor.size(0));
  }

  std::vector<int> output_shape(std::string const &name) const {
    for (auto *pdata = pfirst_data_; pdata != nullptr; pdata = pdata->pnext) {
      if (pdata->name == name) {
        return {pdata->data.GetDim4(), pdata->data.GetDim3(),
                pdata->data.GetDim2(), pdata->data.GetDim1()};
      }
    }
    throw std::runtime_error("missing output variable: " + name);
  }

  double output_value(std::string const &name, int n, int k, int j,
                      int i) const {
    for (auto *pdata = pfirst_data_; pdata != nullptr; pdata = pdata->pnext) {
      if (pdata->name == name) return pdata->data(n, k, j, i);
    }
    throw std::runtime_error("missing output variable: " + name);
  }

  std::vector<std::string> output_names() const {
    std::vector<std::string> names;
    for (auto *pdata = pfirst_data_; pdata != nullptr; pdata = pdata->pnext) {
      names.push_back(pdata->name);
    }
    return names;
  }

  double output_value(std::string const &name) const {
    for (auto *pdata = pfirst_data_; pdata != nullptr; pdata = pdata->pnext) {
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
                         [](std::filesystem::path const &path) {
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

std::shared_ptr<MeshBlockImpl> make_3d_block() {
  auto options = MeshBlockOptionsImpl::from_yaml("test_forcing_3d.yaml");
  return std::make_shared<MeshBlockImpl>(options);
}

bool contains(std::vector<std::string> const &names, std::string const &name) {
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
  } catch (std::exception const &exc) {
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
  block.user_output_callback = [](Variables const &) {
    Variables out;
    out["my_uov"] = torch::ones({1, 1, 1});
    return out;
  };

  Variables vars;
  vars["hydro_w"] = torch::ones({1, 1, 1, 1});

  EXPECT_NO_THROW(output.load_user_output(&block, vars));
}

TEST(OutputSlice, selects_boundary_cell_and_collapses_axis) {
  auto block = make_3d_block();
  auto opts = OutputOptionsImpl::create();
  opts->x1_slice(2.0);
  TestOutputType output(opts);

  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();
  auto values = torch::arange(nc1 * nc2 * nc3, torch::kFloat64)
                    .reshape({1, nc3, nc2, nc1});
  output.append("field", values);
  output.out_is = block->pcoord->il();
  output.out_ie = block->pcoord->iu();
  output.out_js = block->pcoord->jl();
  output.out_je = block->pcoord->ju();
  output.out_ks = block->pcoord->kl();
  output.out_ke = block->pcoord->ku();

  ASSERT_TRUE(output.TransformOutputData(block.get()));
  EXPECT_EQ(output.islice, block->pcoord->il() + 2);
  EXPECT_EQ(output.output_shape("field"), (std::vector<int>{1, nc3, nc2, 1}));
  EXPECT_DOUBLE_EQ(output.output_value("field", 0, block->pcoord->kl(),
                                       block->pcoord->jl(), 0),
                   values
                       .index({0, block->pcoord->kl(), block->pcoord->jl(),
                               block->pcoord->il() + 2})
                       .item<double>());
  output.ClearOutputData();
}

TEST(OutputSlice, supports_multiple_axes_and_rejects_nonintersecting_block) {
  auto block = make_3d_block();
  auto opts = OutputOptionsImpl::create();
  opts->x2_slice(1.5);
  opts->x3_slice(2.5);
  TestOutputType output(opts);

  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();
  output.append("field", torch::ones({1, nc3, nc2, nc1}, torch::kFloat64));
  output.out_is = block->pcoord->il();
  output.out_ie = block->pcoord->iu();
  output.out_js = block->pcoord->jl();
  output.out_je = block->pcoord->ju();
  output.out_ks = block->pcoord->kl();
  output.out_ke = block->pcoord->ku();

  ASSERT_TRUE(output.TransformOutputData(block.get()));
  EXPECT_EQ(output.output_shape("field"), (std::vector<int>{1, 1, 1, nc1}));
  output.ClearOutputData();

  auto outside_opts = OutputOptionsImpl::create();
  outside_opts->x1_slice(block->pcoord->options->x1max());
  TestOutputType outside(outside_opts);
  outside.append("field", torch::ones({1, nc3, nc2, nc1}, torch::kFloat64));
  EXPECT_FALSE(outside.TransformOutputData(block.get()));
  outside.ClearOutputData();
}

TEST(OutputSlice, yaml_coordinate_presence_activates_slice_and_rejects_sum) {
  auto opts = OutputOptionsImpl::from_yaml(
      YAML::Load("{type: netcdf, x1_slice: 1.25}"));
  ASSERT_TRUE(opts->x1_slice().has_value());
  EXPECT_DOUBLE_EQ(*opts->x1_slice(), 1.25);
  EXPECT_FALSE(opts->x2_slice().has_value());

  EXPECT_THROW(
      OutputOptionsImpl::from_yaml(YAML::Load("{type: netcdf, x1_slice: 1.25, "
                                              "output_sumx1: true}")),
      std::invalid_argument);
}

#ifdef NETCDFOUTPUT
TEST(OutputSlice, netcdf_writes_selected_coordinate_and_collapsed_dimension) {
  auto block = make_3d_block();
  auto dir = std::filesystem::temp_directory_path() /
             ("snapy_slice_" +
              std::to_string(reinterpret_cast<std::uintptr_t>(block.get())));
  block->options->output_dir(dir.string());
  block->options->basename("slice");

  auto opts = OutputOptionsImpl::create();
  opts->file_type("netcdf");
  opts->variables({"d"});
  opts->x1_slice(2.0);
  opts->combine(false);
  NetcdfOutput output(opts);

  int nc1 = block->pcoord->options->nc1();
  int nc2 = block->pcoord->options->nc2();
  int nc3 = block->pcoord->options->nc3();
  int nvar = block->phydro->peos->nvar();
  Variables vars;
  vars["hydro_w"] = torch::zeros({nvar, nc3, nc2, nc1}, torch::kFloat64);
  vars["hydro_u"] = torch::zeros_like(vars["hydro_w"]);
  vars["hydro_w"][IDN].copy_(
      torch::arange(nc1 * nc2 * nc3, torch::kFloat64).reshape({nc3, nc2, nc1}));

  output.write_output_file(block.get(), vars, 0.0, false);
  auto file = dir / "slice.block0.out0.00000.nc";

  int ncid;
  ASSERT_EQ(nc_open(file.c_str(), NC_NOWRITE, &ncid), NC_NOERR);
  int dimid;
  ASSERT_EQ(nc_inq_dimid(ncid, "x1", &dimid), NC_NOERR);
  size_t length;
  ASSERT_EQ(nc_inq_dimlen(ncid, dimid, &length), NC_NOERR);
  EXPECT_EQ(length, 1);

  int varid;
  ASSERT_EQ(nc_inq_varid(ncid, "x1", &varid), NC_NOERR);
  float coordinate;
  ASSERT_EQ(nc_get_var_float(ncid, varid, &coordinate), NC_NOERR);
  EXPECT_FLOAT_EQ(coordinate, 2.5F);
  EXPECT_EQ(nc_close(ncid), NC_NOERR);
  std::remove(file.c_str());
  std::remove(dir.c_str());
}
#endif

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

  block->user_forcing_callback = [](Variables const &forcing_vars, double dt,
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

  block->user_forcing_callback = [](Variables const &forcing_vars, double,
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
  output.AccumulateStats(vars, 0.0);
  output.load_hydro(block.get(), vars);
  EXPECT_DOUBLE_EQ(output.output_value("rho_mean"), 1.0);
  EXPECT_DOUBLE_EQ(output.output_value("rho_std"), 0.0);
  output.ClearOutputData();

  vars["hydro_w"].fill_(3.0);
  output.AccumulateStats(vars, 1.0);
  vars["hydro_w"].fill_(5.0);
  output.AccumulateStats(vars, 3.0);

  output.load_hydro(block.get(), vars);
  EXPECT_NEAR(output.output_value("rho_mean"), 13.0 / 3.0, 1.e-12);
  EXPECT_NEAR(output.output_value("rho_std"), std::sqrt(8.0 / 9.0), 1.e-12);
  EXPECT_NEAR(output.output_value("vel1_mean"), 13.0 / 3.0, 1.e-12);
  EXPECT_NEAR(output.output_value("vel1_std"), std::sqrt(8.0 / 9.0), 1.e-12);
  output.ClearOutputData();

  output.ResetStats(3.0);
  vars["hydro_w"].fill_(7.0);
  output.AccumulateStats(vars, 4.0);
  output.load_hydro(block.get(), vars);
  EXPECT_DOUBLE_EQ(output.output_value("rho_mean"), 7.0);
  EXPECT_DOUBLE_EQ(output.output_value("rho_std"), 0.0);
  output.ClearOutputData();

  TestOutputType stable_output(opts);
  vars["hydro_w"].fill_(1.e9 + 1.0);
  stable_output.AccumulateStats(vars, 0.0);
  vars["hydro_w"].fill_(1.e9 + 1.0);
  stable_output.AccumulateStats(vars, 1.0);
  vars["hydro_w"].fill_(1.e9 + 3.0);
  stable_output.AccumulateStats(vars, 2.0);
  stable_output.load_hydro(block.get(), vars);
  EXPECT_NEAR(stable_output.output_value("rho_mean"), 1.e9 + 2.0, 1.e-6);
  EXPECT_NEAR(stable_output.output_value("rho_std"), 1.0, 1.e-12);
}

TEST(OutputStatistics, scalar_statistics_are_time_weighted_and_reset) {
  auto block = make_block("ideal-gas", {"tracer_a"});

  Variables vars;
  vars["scalar_r"] = torch::ones({1, 1, 4, 4}, torch::kFloat64);

  auto opts = OutputOptionsImpl::create();
  opts->variables({"scalar_stat"});
  TestOutputType output(opts);

  vars["scalar_r"].fill_(1.0);
  output.AccumulateStats(vars, 0.0);
  output.load_scalar(block.get(), vars);
  EXPECT_DOUBLE_EQ(output.output_value("r_tracer_a_mean"), 1.0);
  EXPECT_DOUBLE_EQ(output.output_value("r_tracer_a_std"), 0.0);
  output.ClearOutputData();

  vars["scalar_r"].fill_(3.0);
  output.AccumulateStats(vars, 1.0);
  vars["scalar_r"].fill_(5.0);
  output.AccumulateStats(vars, 3.0);

  output.load_scalar(block.get(), vars);
  EXPECT_NEAR(output.output_value("r_tracer_a_mean"), 13.0 / 3.0, 1.e-12);
  EXPECT_NEAR(output.output_value("r_tracer_a_std"), std::sqrt(8.0 / 9.0),
              1.e-12);
  output.ClearOutputData();

  output.ResetStats(3.0);
  vars["scalar_r"].fill_(7.0);
  output.AccumulateStats(vars, 4.0);
  output.load_scalar(block.get(), vars);
  EXPECT_DOUBLE_EQ(output.output_value("r_tracer_a_mean"), 7.0);
  EXPECT_DOUBLE_EQ(output.output_value("r_tracer_a_std"), 0.0);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
