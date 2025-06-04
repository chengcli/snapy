#pragma once

// spdlog
#include <fmt/format.h>

// snap
#include <kintera/thermo/thermo_formatter.hpp>

#include "equation_of_state.hpp"

template <>
struct fmt::formatter<snap::EquationOfStateOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::EquationOfStateOptions& p, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "(type = {}; thermo = {})", p.type(),
                          p.thermo());
  }
};
