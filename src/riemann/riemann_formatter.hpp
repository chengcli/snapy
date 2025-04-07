#pragma once

// spdlog
#include <spdlog/spdlog.h>

// fvm
#include <fvm/coord/coord_formatter.hpp>
#include <fvm/eos/eos_formatter.hpp>

#include "riemann_solver.hpp"

template <>
struct fmt::formatter<snap::RiemannSolverOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::RiemannSolverOptions& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "(type = {}; eos = {}; coord = {})",
                          p.type(), p.eos(), p.coord());
  }
};
