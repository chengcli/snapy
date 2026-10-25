// Dry compressible onset experiment. CSV records a spherical-harmonic radial
// velocity projection; fit its logarithmic slope after acoustic transients.
// Usage: executable yaml Ra end_diffusive_time N nr epsilon output.csv seed
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
    TORCH_CHECK(argc == 9,
                "usage: cubed_sphere_convection yaml Ra end_time N "
                "nr epsilon output.csv seed_amplitude");
    torch::set_num_threads(1);
    double ra = std::stod(argv[2]), end = std::stod(argv[3]),
           eps = std::stod(argv[6]), seed = std::stod(argv[8]);
    int n = std::stoi(argv[4]), nr = std::stoi(argv[5]);
    TORCH_CHECK(ra > 0. && end > 0. && eps > 0. && eps < 1. && n >= 6 &&
                    nr >= 6 && seed >= 0.,
                "invalid experiment parameters");
    auto opts = MeshOptionsImpl::from_yaml(argv[1]);
    auto c = opts->block()->coord();
    c->nx1() = nr;
    c->global_nx1() = nr;
    c->nx2() = n;
    c->global_nx2() = n;
    c->nx3() = n;
    c->global_nx3() = n;
    double ri = 1. / eps - .5, ro = 1. / eps + .5;
    c->x1min() = ri;
    c->global_x1min() = ri;
    c->x1max() = ro;
    c->global_x1max() = ro;
    auto diff = opts->block()->hydro()->diffusion();
    double ti = diff->temperature_inner(), to = diff->temperature_outer();
    auto eos = opts->block()->hydro()->eos();
    double rd = 8.31446261815324 / eos->weight(),
           cv = rd / (eos->gammad() - 1.), cp = cv + rd;
    double gravity = -opts->block()->hydro()->grav()->grav1();
    double tm = 0.5 * (ti + to), sa = ti - to - gravity / cp;
    TORCH_CHECK(sa > 0., "temperature drop must exceed the adiabatic drop");
    double chi =
        std::sqrt(gravity * sa / (tm * ra));  // Pr=1, d=1, reference rho=1
    diff->nu_iso() = chi;
    diff->kappa_iso() = cp * chi;
    double ellstar =
        0.5 * (-1. + std::sqrt(1. + 2. * M_PI * M_PI / (eps * eps)));
    int ell = std::max(1, static_cast<int>(std::round(ellstar)));
    double degree = ell * (ell + 1), a2 = eps * eps * degree,
           Q = M_PI * M_PI + a2;
    auto mesh = Mesh(opts);
    MeshVariables vars(mesh->blocks.size());
    std::vector<torch::Tensor> mode;
    double B = (ti - to) / (1. / ri - 1. / ro), A = to - B / ro;
    auto integral = [&](torch::Tensor r) {
      return (A * r + B - B * torch::log(A * r + B)) / (A * A);
    };
    for (size_t ib = 0; ib < mesh->blocks.size(); ++ib) {
      auto b = mesh->blocks[ib];
      auto c = b->pcoord;
      int ng = c->options->nghost();
      auto r = c->x1v, temperature = A + B / r;
      auto p =
          tm *
          torch::exp(-gravity / rd *
                     (integral(r) - integral(torch::full_like(r, 1. / eps))));
      auto col = torch::zeros({5, 1, 1, nr}, torch::kFloat64);
      col[IDN][0][0].copy_((p / (rd * temperature)).slice(0, ng, ng + nr));
      col[IPR][0][0].copy_(p.slice(0, ng, ng + nr));
      auto balanced = std::get<0>(
          balance_column(col, c->dx1f.slice(0, ng, ng + nr).contiguous(),
                         gravity, true, 1.e-11, 120, "cartesian"));
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
      auto normal = torch::zeros_like(w.narrow(0, IVX, 3));
      normal[0].fill_(1.);
      cs_contra_to_cart_(normal, av, bv, panel);
      auto z = normal[2], prev = torch::ones_like(z), y = z.clone();
      for (int l = 2; l <= ell; ++l) {
        auto next = ((2. * l - 1.) * z * y - (l - 1.) * prev) / l;
        prev = y;
        y = next;
      }
      auto rr = r.view({1, 1, -1}), W = torch::sin(M_PI * (rr - ri)),
           DW = M_PI * torch::cos(M_PI * (rr - ri));
      auto shape = W * y;
      auto dy = ell * (prev - z * y) / (1. - z.square()).clamp_min(1.e-15);
      auto angular = -normal * z;
      angular[2] += 1.;
      angular *= dy;
      auto velocity = seed * chi * Q *
                      (normal * shape + angular * (2. * W + rr * DW) / degree);
      cs_cart_to_contra_(velocity, av, bv, panel);
      w.narrow(0, IVX, 3).copy_(velocity);
      auto baseline = w[IPR] / (rd * w[IDN]);
      w[IDN].copy_(w[IPR] / (rd * (baseline + seed * sa * shape)));
      vars[ib]["hydro_w"] = w;
      mode.push_back(shape.clone());
    }
    mesh->initialize(vars);
    std::ofstream csv(argv[7]);
    TORCH_CHECK(csv.good(), "cannot open CSV output");
    csv << std::setprecision(16)
        << "time,diffusive_time,amplitude,radial_rms\n";
    std::cout << std::setprecision(12) << "Ra_sa=" << ra << " Pr=1 chi=" << chi
              << " ell=" << ell << " epsilon=" << eps
              << " d/Hp=" << gravity / (rd * tm)
              << " deltaT/T=" << (ti - to) / tm
              << " thin_shell_Ra=" << Q * Q * Q / a2 << std::endl;
    double time = 0.;
    int cycle = 0;
    while (time * chi < end) {
      double dt = std::min(mesh->max_time_step(vars), (end - time * chi) / chi);
      mesh->set_cycle(++cycle);
      for (;;) {
        for (int stage = 0; stage < mesh->blocks[0]->pintg->stages.size();
             ++stage)
          mesh->forward(vars, dt, stage);
        int redo = mesh->check_redo(vars);
        TORCH_CHECK(redo >= 0, "step retry limit");
        if (redo == 0) break;
        dt *= .5;
      }
      time += dt;
      if (cycle % 10 == 0 || time * chi >= end) {
        double numerator = 0., denominator = 0., v2 = 0., volume = 0.;
        for (size_t ib = 0; ib < mesh->blocks.size(); ++ib) {
          auto b = mesh->blocks[ib];
          auto sl = b->part({0, 0, 0}, PartOptions().exterior(false).ndim(3));
          auto u = vars[ib]["hydro_u"], vr = (u[IVX] / u[IDN]).index(sl);
          auto vol = b->pcoord->cell_volume().index(sl), f = mode[ib].index(sl);
          numerator += (vol * f * vr).sum().item<double>();
          denominator += (vol * f * f).sum().item<double>();
          v2 += (vol * vr * vr).sum().item<double>();
          volume += vol.sum().item<double>();
        }
        csv << time << "," << time * chi << "," << numerator / denominator
            << "," << std::sqrt(v2 / volume) << "\n";
        csv.flush();
        if (cycle % 100 == 0)
          std::cout << "cycle=" << cycle << " t_diff=" << time * chi
                    << " A=" << numerator / denominator << std::endl;
      }
    }
  } catch (std::exception const &error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
