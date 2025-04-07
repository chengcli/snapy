#pragma once

// spdlog
#include <spdlog/spdlog.h>

// snap
#include "forcing.hpp"

template <>
struct fmt::formatter<snap::ConstGravityOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::ConstGravityOptions& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "(grav1 = {}; grav2 = {}; grav3 = {})",
                          p.grav1(), p.grav2(), p.grav3());
  }
};

template <>
struct fmt::formatter<snap::CoriolisOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::CoriolisOptions& p, FormatContext& ctx) {
    return fmt::format_to(
        ctx.out(),
        "(omega1 = {}; omega2 = {}; omega3 = {}; omegax = {}; omegay = {}; "
        "omegaz = {})",
        p.omega1(), p.omega2(), p.omega3(), p.omegax(), p.omegay(), p.omegaz());
  }
};
