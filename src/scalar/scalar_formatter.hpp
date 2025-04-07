#pragma once

// spdlog
#include <spdlog/spdlog.h>

// snap
#include <snap/recon/recon_formatter.hpp>
#include <snap/riemann/riemann_formatter.hpp>

#include "scalar.hpp"

template <>
struct fmt::formatter<snap::ScalarOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::ScalarOptions& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "(riemann = {}; recon = {})", p.riemann(),
                          p.recon());
  }
};
