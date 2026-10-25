// Matched compressible spherical-shell onset; eta=inner/outer radius.
// Units d=T_mid=rho_mid=R=1. Evolution uses the requested device.
#include <kintera/constants.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/hydro/balance_column.hpp>
#include <snap/mesh/mesh.hpp>

using namespace snap;
int main(int argc, char **argv) {
  try {
    TORCH_CHECK(argc >= 14 && argc <= 18,
                "usage: spherical_convection yaml Ra eta ell N nr end "
                "seed cfl a D output.csv device [radial_profile.txt "
                "[z|diagonal [diffusion_step_scale [float64|float32]]]]");
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    torch::NoGradGuard no_grad;
    double ra = std::stod(argv[2]), eta = std::stod(argv[3]),
           end = std::stod(argv[7]), seed = std::stod(argv[8]),
           cfl = std::stod(argv[9]), a = std::stod(argv[10]),
           dissipation = std::stod(argv[11]);
    int ell = std::stoi(argv[4]), n = std::stoi(argv[5]),
        nr = std::stoi(argv[6]);
    std::string orientation = argc >= 16 ? argv[15] : "z";
    double step_scale = argc >= 17 ? std::stod(argv[16]) : 1.;
    std::string precision = argc == 18 ? argv[17] : "float64";
    TORCH_CHECK(precision == "float64" || precision == "float32",
                "invalid precision");
    auto dtype = precision == "float64" ? torch::kFloat64 : torch::kFloat32;
    TORCH_CHECK(orientation == "z" || orientation == "diagonal",
                "invalid mode orientation");
    TORCH_CHECK(step_scale >= 1. && step_scale <= 8.,
                "invalid diffusion step scale");
    TORCH_CHECK(ra > 0. && eta >= .5 && eta < 1. && ell >= 1 && n >= 8 &&
                    n % 2 == 0 && nr >= 8 && nr % 2 == 0 && end > 0. &&
                    seed >= 0. && cfl > 0. && cfl <= 1. && a > dissipation &&
                    dissipation > 0. && a < 2.,
                "invalid shell parameters");
    std::string device_name = argv[13];
    if (device_name == "cuda:local") {
      auto local_rank = std::getenv("LOCAL_RANK");
      device_name = "cuda:" + std::string(local_rank ? local_rank : "0");
    }
    torch::Device device(device_name);
    TORCH_CHECK(!device.is_cuda() || torch::cuda::is_available(),
                "CUDA was requested but is unavailable");
    c10::OptionalDeviceGuard guard(device);
    auto opts = MeshOptionsImpl::from_yaml(argv[1]);
    int processes = get_world_size();
    TORCH_CHECK(processes >= 1 && 6 % processes == 0,
                "the process count must divide six cubed-sphere panels");
    opts->blocks_per_process(6 / processes);
    auto bo = opts->block();
    int process_rank = bo->layout()->process_rank();
    bo->layout()
        ->device(device.is_cuda() ? "cuda" : "cpu")
        .device_id(device.is_cuda() ? device.index() : 0);
    bo->intg()->cfl(cfl);
    double ri = eta / (1. - eta), ro = 1. / (1. - eta), rm = .5 * (ri + ro);
    auto co = bo->coord();
    co->nx1(nr).global_nx1(nr).nx2(n).global_nx2(n).nx3(n).global_nx3(n);
    co->x1min(ri).global_x1min(ri).x1max(ro).global_x1max(ro);
    auto eo = bo->hydro()->eos();
    eo->weight(kintera::constants::Rgas);
    double gamma = eo->gammad(), cp = gamma / (gamma - 1.);
    double gravity = cp * dissipation,
           chi = std::sqrt(gravity * (a - dissipation) / ra);
    double B = a * ri * ro, A = 1. - B / rm;
    double ti = 1. + a / (1. + eta), to = 1. - a * eta / (1. + eta);
    bo->hydro()->grav()->grav1(-gravity);
    bo->hydro()
        ->diffusion()
        ->dynamic(true)
        .nu_iso(chi)
        .kappa_iso(cp * chi)
        .temperature_inner(ti)
        .temperature_outer(to)
        .implicit_radial(false);
    auto mesh = Mesh(opts);
    MeshVariables vars(mesh->blocks.size());
    std::vector<torch::Tensor> modes, targets, initial_rhos;
    double max_balance_error = 0.;
    auto profile = torch::zeros({4, nr}, torch::kFloat64);
    if (argc >= 15) {
      std::ifstream input(argv[14]);
      TORCH_CHECK(input.good(), "cannot open radial eigenmode profile");
      auto values = profile.accessor<double, 2>();
      for (int i = 0; i < nr; ++i) {
        double z;
        TORCH_CHECK(bool(input >> z), "missing eigenmode row");
        TORCH_CHECK(std::abs(z - ((i + .5) / nr - .5)) < 1.e-10,
                    "eigenmode profile must use active cell centers");
        for (int q = 0; q < 4; ++q)
          TORCH_CHECK(
              bool(input >> values[q][i]) && std::isfinite(values[q][i]),
              "invalid eigenmode value");
      }
    }
    for (size_t ib = 0; ib < mesh->blocks.size(); ++ib) {
      auto b = mesh->blocks[ib];
      auto c = b->pcoord;
      int ng = c->options->nghost();
      auto r = c->x1v, zrad = r - rm;
      auto temperature = 1. - B * zrad / (r * rm);
      TORCH_CHECK(temperature.min().item<double>() > 0.,
                  "nonpositive ghost temperature");
      torch::Tensor integral;
      if (std::abs(A) < 1.e-8)
        integral = (r.square() - rm * rm) / (2. * B);
      else
        integral = zrad / A - B / (A * A) * torch::log1p(A * zrad / rm);
      auto pressure = torch::exp(-gravity * integral);
      auto col = torch::zeros({5, 1, 1, nr}, torch::kFloat64);
      col[IDN][0][0].copy_((pressure / temperature).slice(0, ng, ng + nr));
      col[IPR][0][0].copy_(pressure.slice(0, ng, ng + nr));
      auto [balanced, balance_error, sweeps] =
          balance_column(col, c->dx1f.slice(0, ng, ng + nr).contiguous(),
                         gravity, true, 1.e-10, 240, "cartesian");
      max_balance_error = std::max(max_balance_error, balance_error);
      auto rb = balanced[IDN][0][0];
      double rho_mid = (-rb[nr / 2 - 2] + 9 * rb[nr / 2 - 1] + 9 * rb[nr / 2] -
                        rb[nr / 2 + 1])
                           .item<double>() /
                       16.;
      balanced[IDN].div_(rho_mid);
      balanced[IPR].div_(rho_mid);
      auto w = torch::zeros(
          {5, c->options->nc3(), c->options->nc2(), c->options->nc1()},
          torch::kFloat64);
      for (int q : {IDN, IPR}) {
        w[q].narrow(-1, ng, nr).copy_(balanced[q]);
        w[q].narrow(-1, 0, ng).copy_(w[q].narrow(-1, ng, 1));
        w[q].narrow(-1, ng + nr, ng).copy_(w[q].narrow(-1, ng + nr - 1, 1));
      }
      int panel =
          std::get<2>(b->get_layout()->loc_of(b->options->layout()->rank()));
      auto av = c->x2v.view({1, -1, 1}), bv = c->x3v.view({-1, 1, 1});
      // The unit normal and spherical harmonic depend only on angle. Keep
      // their radial extent one, rather than repeating the high-degree
      // Legendre recurrence independently in every radial cell.
      auto normal = torch::zeros({3, c->options->nc3(), c->options->nc2(), 1},
                                 w.options());
      normal[0].fill_(1.);
      cs_contra_to_cart_(normal, av, bv, panel);
      auto axis = torch::zeros({3, 1, 1, 1}, normal.options());
      if (orientation == "diagonal")
        axis.fill_(1. / std::sqrt(3.));
      else
        axis[2].fill_(1.);
      auto z = (normal * axis).sum(0), prev = torch::ones_like(z),
           y = z.clone();
      for (int l = 2; l <= ell; ++l) {
        auto next = ((2. * l - 1.) * z * y - (l - 1.) * prev) / l;
        prev = y;
        y = next;
      }
      auto angular = axis - normal * z;
      angular *= ell * (prev - z * y) / (1. - z.square()).clamp_min(1.e-15);
      double degree = ell * (ell + 1.);
      auto rr = r.view({1, 1, -1});
      auto radial = torch::zeros({4, c->options->nc1()}, torch::kFloat64);
      if (argc >= 15) {
        radial.narrow(1, ng, nr).copy_(profile);
        radial.narrow(1, 0, ng).copy_(profile.narrow(1, 0, 1));
        radial.narrow(1, ng + nr, ng).copy_(profile.narrow(1, nr - 1, 1));
      } else {
        auto W = torch::sin(M_PI * (r - ri));
        auto DW = M_PI * torch::cos(M_PI * (r - ri));
        double Q = M_PI * M_PI + degree / (rm * rm);
        radial[0].copy_(chi * Q * W);
        radial[1].copy_(
            chi * Q *
            (r * DW + (2. - r * (gravity - B / r.square()) / temperature) * W) /
            std::sqrt(degree));
        radial[2].copy_((a - dissipation) * W);
      }
      auto velocity =
          seed * (normal * radial[0].view({1, 1, -1}) * y +
                  angular * radial[1].view({1, 1, -1}) / std::sqrt(degree));
      cs_cart_to_contra_(velocity, av, bv, panel);
      w.narrow(0, IVX, 3).copy_(velocity);
      auto base_temperature = w[IPR] / w[IDN];
      w[IPR].add_(seed * radial[3].view({1, 1, -1}) * y);
      w[IDN].copy_(w[IPR] /
                   (base_temperature + seed * radial[2].view({1, 1, -1}) * y));
      vars[ib]["hydro_w"] = w.to(device, dtype);
      modes.push_back((torch::sin(M_PI * (rr - ri)) * y).to(device, dtype));
      targets.push_back(temperature.view({1, 1, -1}).to(device, dtype));
    }
    mesh->to(device, dtype);
    mesh->initialize(vars);
    for (auto const &state : vars)
      initial_rhos.push_back(state.at("hydro_u")[IDN].clone());
    std::string path(argv[12]);
    std::ofstream csv;
    if (process_rank == 0) {
      csv.open(path);
      TORCH_CHECK(csv.good(), "cannot open output CSV");
      csv << std::setprecision(17)
          << "time,diffusive_time,amplitude,radial_rms,temperature_drift,"
             "density_drift\n";
    }
    auto record = [&](double time) {
      auto sums = torch::zeros({4}, modes[0].options().dtype(torch::kFloat64));
      auto errors = torch::zeros({2}, modes[0].options());
      for (size_t ib = 0; ib < mesh->blocks.size(); ++ib) {
        auto b = mesh->blocks[ib];
        auto sl = b->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
        auto u = vars[ib]["hydro_u"], vr = (u[IVX] / u[IDN]).index(sl);
        auto vol = b->pcoord->cell_volume().index(sl), f = modes[ib].index(sl);
        auto primitive = b->phydro->peos->compute("U->W", {u.clone()});
        auto t = b->phydro->peos->compute("W->T", {primitive});
        sums += torch::stack({(vol * f * vr).sum(torch::kFloat64),
                              (vol * f * f).sum(torch::kFloat64),
                              (vol * vr * vr).sum(torch::kFloat64),
                              vol.sum(torch::kFloat64)});
        errors = torch::maximum(
            errors,
            torch::stack({(t - targets[ib]).index(sl).abs().max(),
                          (u[IDN] - initial_rhos[ib]).index(sl).abs().max()}));
      }
      auto layout = mesh->blocks.front()->get_layout();
      if (layout->has_process_group()) {
        std::vector<torch::Tensor> sum_values{sums}, max_values{errors};
        layout->comm->allreduce(sum_values, c10d::ReduceOp::SUM);
        layout->comm->allreduce(max_values, c10d::ReduceOp::MAX);
      }
      if (process_rank != 0) return;
      auto stats =
          torch::stack({sums[0] / sums[1], torch::sqrt(sums[2] / sums[3]),
                        errors[0], errors[1]})
              .cpu();
      csv << time << ',' << time * chi;
      for (int j = 0; j < 4; ++j) csv << ',' << stats[j].item<double>();
      csv << '\n';
      csv.flush();
    };
    std::cout << std::setprecision(14) << "device=" << device << " Ra=" << ra
              << " eta=" << eta << " ell=" << ell << " N=" << n << " nr=" << nr
              << " chi=" << chi << " balance_residual=" << max_balance_error
              << std::endl;
    record(0.);
    auto start = std::chrono::steady_clock::now();
    double time = 0., physical_end = end / chi;
    int cycle = 0, retries = 0;
    while (time < physical_end) {
      double dt = step_scale * mesh->max_time_step(vars);
      // Optional benchmark-only relaxation of the conservative diffusion
      // guard. Keep the original acoustic bound, and validate by reducing the
      // scale. The production timestep policy is unchanged.
      if (step_scale > 1.) {
        auto acoustic = torch::full({}, 1.e30, modes[0].options());
        for (size_t ib = 0; ib < mesh->blocks.size(); ++ib) {
          auto b = mesh->blocks[ib];
          auto w = vars[ib]["hydro_w"];
          auto sound = torch::sqrt(gamma * w[IPR] / w[IDN]);
          auto sl = b->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
          auto c = b->pcoord;
          acoustic = torch::minimum(
              acoustic,
              torch::stack({(c->center_width1() / (w[IVX].abs() + sound))
                                .index(sl)
                                .min(),
                            (c->center_width2() / (w[IVY].abs() + sound))
                                .index(sl)
                                .min(),
                            (c->center_width3() / (w[IVZ].abs() + sound))
                                .index(sl)
                                .min()})
                  .min());
        }
        auto layout = mesh->blocks.front()->get_layout();
        if (layout->has_process_group()) {
          std::vector<torch::Tensor> values{acoustic};
          layout->comm->allreduce(values, c10d::ReduceOp::MIN);
        }
        dt = std::min(dt, cfl * acoustic.item<double>());
      }
      dt = std::min(dt, physical_end - time);
      TORCH_CHECK(std::isfinite(dt) && dt > 0., "invalid timestep");
      mesh->set_cycle(++cycle);
      for (;;) {
        for (int stage = 0; stage < mesh->blocks[0]->pintg->stages.size();
             ++stage)
          mesh->forward(vars, dt, stage);
        int redo = mesh->check_redo(vars);
        TORCH_CHECK(redo >= 0, "step retry limit");
        if (redo == 0) break;
        ++retries;
        dt *= .5;
      }
      time += dt;
      if (cycle % 2 == 0 || time >= physical_end) record(time);
      if (cycle % 100 == 0 && process_rank == 0)
        std::cout << "cycle=" << cycle << " t_diff=" << time * chi << std::endl;
    }
    double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
            .count();
    if (process_rank != 0) return 0;
    std::ofstream metadata(path + ".json");
    TORCH_CHECK(metadata.good(), "cannot open metadata output");
    metadata << std::setprecision(16) << "{\n\"device\": \"" << device
             << "\", \"Ra\": " << ra << ", \"eta\": " << eta << ",\n"
             << "\"ell\": " << ell << ", \"N\": " << n << ", \"nr\": " << nr
             << ",\n"
             << "\"a\": " << a << ", \"D\": " << dissipation
             << ", \"gamma\": " << gamma << ", \"Pr\": 1, \"chi_mid\": " << chi
             << ",\n"
             << "\"seed\": " << seed << ", \"cfl\": " << cfl
             << ", \"eigenmode_seed\": " << (argc >= 15 ? "true" : "false")
             << ", \"orientation\": \"" << orientation << "\",\n"
             << "\"diffusion_step_scale\": " << step_scale << ",\n"
             << "\"dtype\": \"" << precision << "\",\n"
             << "\"processes\": " << processes << ",\n"
             << "\"end_diffusive_time\": " << time * chi
             << ", \"cycles\": " << cycle << ", \"retries\": " << retries
             << ", \"seconds\": " << elapsed << ",\n"
             << "\"balance_residual\": " << max_balance_error << "\n}\n";
    std::cout << "completed " << cycle << " cycles in " << elapsed << " s"
              << std::endl;
  } catch (std::exception const &error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
