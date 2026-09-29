// #250 T4 robustness driver: a 2-D (x1 vertical, x2 periodic) column with an
// analytic T(z) background, an optional cosine temperature anomaly at fixed
// pressure (cold pool / warm bubble), an optional geometric x1 stretch, and
// study diagnostics:
//   - ref.<rank>.<blk>.bin: the well-balanced x1 references (pref, dref at
//     cells; psf_lo, dsf at lower faces) at t = 0, for the block-seam check
//   - state.<rank>.<blk>.bin: interior primitives at the end, for the
//     bit-reproducibility and decomposition checks
//   - summary.<rank>.json: wall time, cycles, min rho / p, non-finite counts,
//     max |v| / c_s (rest residual), reference cost, fault-injection counts
// Study-only; not meant for main. Build with ../build_driver.sh-style flags
// through build_t4.sh.
// C/C++
#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

// yaml
#include <yaml-cpp/yaml.h>

// kintera
#include <kintera/constants.h>

// snap
#include <snap/snap.h>

#include <snap/hydro/hydro.hpp>
#include <snap/mesh/mesh.hpp>

using namespace snap;

namespace {

using RefTuple =
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>;

// _hydro_ref_x1 is protected; a derived class may name it, and the resulting
// member pointer (of type RefTuple (HydroImpl::*)(...) const) applies to any
// HydroImpl. Study diagnostics only; the production path is untouched.
struct RefPeek : HydroImpl {
  static auto ptr() { return &RefPeek::_hydro_ref_x1; }
};

RefTuple call_ref(Hydro const& h, torch::Tensor const& w) {
  return ((*h).*RefPeek::ptr())(w);
}

struct Profile {
  std::string kind;  // isothermal | polytrope | inversion | tropopause
  double T0, p0, lapse, ztrop, Rd, grav;

  // temperature and pressure at height z (z measured from x1min)
  std::pair<torch::Tensor, torch::Tensor> tp(torch::Tensor z) const {
    if (kind == "isothermal") {
      auto T = torch::full_like(z, T0);
      return {T, p0 * torch::exp(-grav * z / (Rd * T0))};
    }
    auto poly_T = [&](torch::Tensor zz) { return T0 - lapse * zz; };
    auto poly_p = [&](torch::Tensor zz) {
      return p0 * torch::pow(poly_T(zz) / T0, grav / (Rd * lapse));
    };
    if (kind == "polytrope" || kind == "inversion") {
      TORCH_CHECK(std::abs(lapse) > 0, "lapse must be nonzero for ", kind);
      return {poly_T(z), poly_p(z)};
    }
    if (kind == "tropopause") {  // polytrope below ztrop, isothermal above
      double Tt = T0 - lapse * ztrop;
      double pt = p0 * std::pow(Tt / T0, grav / (Rd * lapse));
      auto zc = z.clamp_max(ztrop);
      auto T = torch::where(z < ztrop, poly_T(zc), torch::full_like(z, Tt));
      auto p = torch::where(z < ztrop, poly_p(zc),
                            pt * torch::exp(-grav * (z - ztrop) / (Rd * Tt)));
      return {T, p};
    }
    TORCH_CHECK(false, "unknown profile ", kind);
  }
};

void write_bin(std::string const& fname, std::vector<torch::Tensor> const& ts) {
  std::ofstream f(fname, std::ios::binary);
  for (auto t : ts) {
    t = t.to(torch::kCPU, torch::kFloat64).contiguous();
    int64_t nd = t.dim();
    f.write(reinterpret_cast<char const*>(&nd), sizeof(nd));
    for (auto s : t.sizes()) {
      int64_t v = s;
      f.write(reinterpret_cast<char const*>(&v), sizeof(v));
    }
    f.write(reinterpret_cast<char const*>(t.data_ptr<double>()),
            t.numel() * sizeof(double));
  }
}

// geometric x1 faces over [x1min, x1max], dz_{k+1} = q dz_k, ghosts continue
// the same ratio outward
torch::Tensor stretched_faces(double x1min, double x1max, int nx1, int ng,
                              double q) {
  std::vector<double> dz(nx1 + 2 * ng);
  double dz0 = (q == 1.) ? (x1max - x1min) / nx1
                         : (x1max - x1min) * (q - 1.) / (std::pow(q, nx1) - 1.);
  for (int k = 0; k < nx1 + 2 * ng; ++k) dz[k] = dz0 * std::pow(q, k - ng);
  std::vector<double> f(nx1 + 2 * ng + 1);
  f[ng] = x1min;
  for (int k = ng; k < nx1 + 2 * ng; ++k) f[k + 1] = f[k] + dz[k];
  for (int k = ng - 1; k >= 0; --k) f[k] = f[k + 1] - dz[k];
  f[ng + nx1] = x1max;  // exact top
  return torch::tensor(f, torch::kFloat64);
}

}  // namespace

int main(int argc, char** argv) {
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  TORCH_CHECK(argc >= 2, "usage: t4_driver.release <input.yaml>");
  std::string input = argv[1];
  auto config = YAML::LoadFile(input);
  auto prob = config["problem"];

  auto mesh = Mesh(MeshOptionsImpl::from_yaml(input));
  auto device = torch::Device(mesh->options->device_str());
  mesh->to(device);

  int rank = 0;
  auto layout0 = mesh->blocks.front()->get_layout();
  if (layout0) rank = layout0->options->rank();
  int nb1 = config["distribute"]["nb1"].as<int>(1);

  double q = prob["stretch"].as<double>(1.);
  auto grav = -config["forcing"]["const-gravity"]["grav1"].as<double>();
  auto x1min = config["geometry"]["bounds"]["x1min"].as<double>();
  auto x1max = config["geometry"]["bounds"]["x1max"].as<double>();
  int gnx1 = config["geometry"]["cells"]["nx1"].as<int>();
  int ng = config["geometry"]["cells"]["nghost"].as<int>();

  MeshVariables vars(mesh->blocks.size());
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto block = mesh->blocks[b];
    auto pcoord = block->pcoord;
    auto peos = block->phydro->peos;
    if (q != 1.) {
      TORCH_CHECK(nb1 == 1, "stretch needs nb1 == 1 in this driver");
      pcoord->x1f.copy_(stretched_faces(x1min, x1max, gnx1, ng, q));
      pcoord->reset_coordinates({nullptr, nullptr, nullptr});
    }
    Profile prof{prob["profile"].as<std::string>(),
                 prob["T0"].as<double>(),
                 prob["p0"].as<double>(),
                 prob["lapse"].as<double>(0.),
                 prob["ztrop"].as<double>(0.),
                 kintera::constants::Rgas / peos->species_weight(),
                 grav};
    auto grids =
        torch::meshgrid({pcoord->x3v, pcoord->x2v, pcoord->x1v}, "ij");
    auto z = grids[2] - x1min;
    auto x = grids[1];
    auto [T, p] = prof.tp(z);
    double dT = prob["dT"].as<double>(0.);
    if (dT != 0.) {
      auto L = torch::sqrt(
          ((x - prob["xc"].as<double>()) / prob["xr"].as<double>()).square() +
          ((z - prob["zc"].as<double>()) / prob["zr"].as<double>()).square());
      T = T + torch::where(L <= 1, dT * (torch::cos(L * M_PI) + 1.) / 2., 0.);
    }
    auto w = torch::zeros(
        {peos->nvar(), pcoord->options->nc3(), pcoord->options->nc2(),
         pcoord->options->nc1()},
        torch::TensorOptions().dtype(torch::kFloat64).device(device));
    w[IPR] = p;
    w[IDN] = p / (prof.Rd * T);
    vars[b]["hydro_w"] = w;
  }

  double current_time = mesh->initialize(vars);

  // ---- t = 0 references, all ranks together (the x1 relay is collective)
  std::ostringstream extra;
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto block = mesh->blocks[b];
    auto pcoord = block->pcoord;
    auto w = vars[b].at("hydro_w");
    auto [psf, pref, dsf, dref] = call_ref(block->phydro, w);
    int il = pcoord->il(), iu = pcoord->iu();
    int jl = pcoord->jl(), ju = pcoord->ju();
    int n1 = iu - il + 1, n2 = ju - jl + 1;
    auto cells = [&](torch::Tensor t) {
      return t.narrow(-1, il, n1).narrow(-2, jl, n2);
    };
    auto faces = [&](torch::Tensor t) {  // lower faces il..iu+1
      return t.narrow(-1, il, n1 + 1).narrow(-2, jl, n2);
    };
    write_bin("ref." + std::to_string(rank) + "." + std::to_string(b) + ".bin",
              {pcoord->x1v.narrow(0, il, n1), pcoord->x2v.narrow(0, jl, n2),
               pcoord->x1f.narrow(0, il, n1 + 1), cells(pref), cells(dref),
               faces(psf), faces(dsf), cells(w[IDN]), cells(w[IPR])});
    auto nonfin = [](torch::Tensor t) {
      return (~torch::isfinite(t)).sum().item<int64_t>();
    };
    extra << "\"ref_nonfinite_b" << b << "\": " << nonfin(dref) + nonfin(dsf)
          << ", ";
  }

  // ---- reference cost: repeated calls on an unsplit column only
  int nbench = prob["ref_bench"].as<int>(0);
  double ref_us = -1.;
  if (nbench > 0 && nb1 == 1) {
    auto block = mesh->blocks.front();
    auto w = vars[0].at("hydro_w");
    for (int k = 0; k < 5; ++k) call_ref(block->phydro, w);
    if (device.is_cuda()) torch::cuda::synchronize();
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < nbench; ++k) call_ref(block->phydro, w);
    if (device.is_cuda()) torch::cuda::synchronize();
    auto t1 = std::chrono::steady_clock::now();
    ref_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / nbench;
  }

  // ---- fault injection: corrupt one cell of a copy of w, count non-finite
  // references (the guard is dynamics/wb-rop-guard in the YAML)
  std::string fault = prob["fault"].as<std::string>("");
  int64_t fault_nonfinite = -1;
  if (!fault.empty() && nb1 == 1) {
    auto block = mesh->blocks.front();
    auto w = vars[0].at("hydro_w").clone();
    int il = block->pcoord->il();
    int i = il + block->pcoord->options->nx1() / 2;
    int j = block->pcoord->jl();
    if (fault == "p_zero") {  // one interior cell with p = 0
      w[IPR].select(-1, i).select(-1, j).fill_(0.);
    } else if (fault == "p_negative") {
      w[IPR].select(-1, i).select(-1, j).fill_(-1.);
    } else if (fault == "bottom_p_zero") {  // the isentrope anchor
      w[IPR].select(-1, il).select(-1, j).fill_(0.);
    } else if (fault == "rho_overflow") {  // rho/p finite, weighted sum not
      w[IDN].select(-1, i).select(-1, j).fill_(
          0.2 * std::numeric_limits<double>::max());
      w[IPR].select(-1, i).select(-1, j).fill_(0.1);
    } else {
      TORCH_CHECK(false, "unknown fault ", fault);
    }
    auto [psf, pref, dsf, dref] = call_ref(block->phydro, w);
    fault_nonfinite = (~torch::isfinite(dref)).sum().item<int64_t>() +
                      (~torch::isfinite(dsf)).sum().item<int64_t>();
  }

  // ---- time loop with positivity tracking
  double rho_min = std::numeric_limits<double>::max();
  double p_min = rho_min;
  int64_t nonfinite = 0;
  int bad_cycle = -1;
  auto interior = [&](size_t b, torch::Tensor t) {
    auto pc = mesh->blocks[b]->pcoord;
    return t.narrow(-1, pc->il(), pc->iu() - pc->il() + 1)
        .narrow(-2, pc->jl(), pc->ju() - pc->jl() + 1);
  };
  auto track = [&](int cycle) {
    for (size_t b = 0; b < mesh->blocks.size(); ++b) {
      auto w = vars[b].at("hydro_w");
      auto r = interior(b, w[IDN]), p = interior(b, w[IPR]);
      rho_min = std::min(rho_min, r.min().item<double>());
      p_min = std::min(p_min, p.min().item<double>());
      auto nf = (~torch::isfinite(interior(b, w))).sum().item<int64_t>();
      if (nf > 0 && bad_cycle < 0) bad_cycle = cycle;
      nonfinite += nf;
    }
  };
  track(0);

  int cycle = mesh->blocks.front()->cycle;
  bool abnormal = false;
  auto t0 = std::chrono::steady_clock::now();
  try {
    while (!mesh->blocks.front()->pintg->stop(cycle, current_time)) {
      ++cycle;
      mesh->set_cycle(cycle);
      auto dt = mesh->max_time_step(vars);
      for (int stage = 0; stage < mesh->blocks.front()->pintg->stages.size();
           ++stage) {
        mesh->forward(vars, dt, stage);
      }
      int redo = mesh->check_redo(vars);
      if (redo > 0) {
        cycle = mesh->blocks.front()->cycle;
        continue;
      }
      if (redo < 0) break;
      current_time += dt;
      track(cycle);
      if (bad_cycle >= 0) break;
    }
  } catch (std::exception const& e) {
    abnormal = true;
    std::cerr << "t4_driver: " << e.what() << std::endl;
  }
  if (device.is_cuda()) torch::cuda::synchronize();
  double wall = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0)
                    .count();

  // ---- final state, rest residual max |v| / c_s
  double vmax_cs = 0.;
  for (size_t b = 0; b < mesh->blocks.size(); ++b) {
    auto block = mesh->blocks[b];
    auto pcoord = block->pcoord;
    auto w = vars[b].at("hydro_w");
    int il = pcoord->il(), iu = pcoord->iu();
    int jl = pcoord->jl(), ju = pcoord->ju();
    int n1 = iu - il + 1, n2 = ju - jl + 1;
    auto gamma = block->phydro->peos->compute("W->A", {w});
    auto cs = torch::sqrt(gamma * w[IPR] / w[IDN]);
    auto vm = torch::sqrt(w[IVX].square() + w[IVY].square()) / cs;
    vmax_cs = std::max(vmax_cs, interior(b, vm).max().item<double>());
    write_bin(
        "state." + std::to_string(rank) + "." + std::to_string(b) + ".bin",
        {pcoord->x1v.narrow(0, il, n1), pcoord->x2v.narrow(0, jl, n2),
         interior(b, w[IDN]), interior(b, w[IPR]), interior(b, w[IVX]),
         interior(b, w[IVY])});
  }

  std::ofstream s("summary." + std::to_string(rank) + ".json");
  s.precision(17);
  s << "{" << extra.str() << "\"rank\": " << rank
    << ", \"cycles\": " << cycle << ", \"time\": " << current_time
    << ", \"wall_s\": " << wall << ", \"rho_min\": " << rho_min
    << ", \"p_min\": " << p_min << ", \"nonfinite\": " << nonfinite
    << ", \"bad_cycle\": " << bad_cycle << ", \"abnormal\": "
    << (abnormal ? "true" : "false") << ", \"vmax_over_cs\": " << vmax_cs
    << ", \"ref_us_per_call\": " << ref_us
    << ", \"fault\": \"" << fault << "\", \"fault_nonfinite\": "
    << fault_nonfinite << "}\n";
  if (abnormal) std::cout << "Terminating abnormally" << std::endl;
  mesh->finalize(vars, current_time);
  return abnormal ? 3 : 0;
}
