#pragma once

// spdlog
#include <spdlog/spdlog.h>

// fvm
#include "forcing.hpp"

template <>
struct fmt::formatter<canoe::ConstGravityOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const canoe::ConstGravityOptions& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "(grav1 = {}; grav2 = {}; grav3 = {})",
                          p.grav1(), p.grav2(), p.grav3());
  }
};

template <>
struct fmt::formatter<canoe::CoriolisOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const canoe::CoriolisOptions& p, FormatContext& ctx) {
    return fmt::format_to(
        ctx.out(),
        "(omega1 = {}; omega2 = {}; omega3 = {}; omegax = {}; omegay = {}; "
        "omegaz = {})",
        p.omega1(), p.omega2(), p.omega3(), p.omegax(), p.omegay(), p.omegaz());
  }
};
