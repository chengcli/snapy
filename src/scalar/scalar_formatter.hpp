#pragma once

// spdlog
#include <spdlog/spdlog.h>

// fvm
#include <fvm/recon/recon_formatter.hpp>
#include <fvm/riemann/riemann_formatter.hpp>

#include "scalar.hpp"

template <>
struct fmt::formatter<canoe::ScalarOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const canoe::ScalarOptions& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "(riemann = {}; recon = {})", p.riemann(),
                          p.recon());
  }
};
