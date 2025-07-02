// base
#include <configure.h>

// snap
#include <snap/registry.hpp>

#include "riemann_formatter.hpp"
#include "riemann_solver.hpp"

namespace snap {
void ShallowRoeSolverImpl::reset() {
  // set up equation-of-state model
  peos = register_module_op(this, "eos", options.eos());
}

torch::Tensor ShallowRoeSolverImpl::forward(torch::Tensor wl, torch::Tensor wr,
                                            int dim, torch::Tensor flx) {
  int ivx, ivy;
  if (options.dir() == "xy") {
    ivx = dim == 3 ? 1 : 2;
    ivy = dim == 3 ? 2 : 1;
  } else if (options.dir() == "yz") {
    ivx = dim == 2 ? 2 : 3;
    ivy = dim == 2 ? 3 : 2;
  } else {
    TORCH_CHECK(false,
                "ShallowRoeSolver takes options.dir() = 'xy' or 'yz'"
                " but got options.dir() = ");
  }

  auto sqrtdl = torch::sqrt(wl[0]);
  auto sqrtdr = torch::sqrt(wr[0]);
  auto isdlpdr = 1.0 / (sqrtdl + sqrtdr);

  auto ubar = (wl[ivx] * sqrtdl + wr[ivx] * sqrtdr) * isdlpdr;
  auto vbar = (wl[ivy] * sqrtdl + wr[ivy] * sqrtdr) * isdlpdr;
  auto cbar = torch::sqrt(0.5 * (wl[0] + wr[0]));

  auto del = wr - wl;
  auto hbar = torch::sqrt(wl[0] * wr[0]);

  auto a1 = 0.5 * (cbar * del[0] - hbar * del[ivx]) / cbar;
  auto a2 = hbar * del[ivy];
  auto a3 = 0.5 * (cbar * del[0] + hbar * del[ivx]) / cbar;

  auto wave0 = torch::zeros_like(del);
  auto wave1 = torch::zeros_like(del);
  auto wave2 = torch::zeros_like(del);

  wave0[0] = a1;
  wave0[ivx] = a1 * (ubar - cbar);
  wave0[ivy] = a1 * vbar;

  wave1[0] = 0.;
  wave1[ivx] = 0.;
  wave1[ivy] = a2;

  wave2[0] = a3;
  wave2[ivx] = a3 * (ubar + cbar);
  wave2[ivy] = a3 * vbar;

  auto speed = torch::zeros_like(del);

  speed[0] = torch::abs(ubar - cbar);
  speed[1] = torch::abs(ubar);
  speed[2] = torch::abs(ubar + cbar);

  flx[0] = 0.5 * (wl[0] * wl[ivx] + wr[0] * wr[ivx]);
  flx[ivx] = 0.5 * (wl[0] * wl[ivx].square() + 0.5 * wl[0].square() +
                    wr[0] * wr[ivx].square() + 0.5 * wr[0].square());
  flx[ivy] = 0.5 * (wl[0] * wl[ivx] * wl[ivy] + wr[0] * wr[ivx] * wr[ivy]);

  flx -= 0.5 * (speed[0] * wave0 + speed[1] * wave1 + speed[2] * wave2);

  return flx;
}
}  // namespace snap
