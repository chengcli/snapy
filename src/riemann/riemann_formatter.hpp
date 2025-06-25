#pragma once

// fmt
#include <fmt/format.h>

// snap
#include <snap/eos/eos_formatter.hpp>

#include "riemann_solver.hpp"

template <>
struct fmt::formatter<snap::RiemannSolverOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::RiemannSolverOptions& p, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "(type = {}; eos = {})", p.type(),
                          p.eos());
  }
};
