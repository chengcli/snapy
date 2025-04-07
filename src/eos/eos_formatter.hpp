#pragma once

// spdlog
#include <configure.h>
#include <spdlog/spdlog.h>

// fvm
#include <fvm/thermo/thermo_formatter.hpp>
#include "equation_of_state.hpp"

template <>
struct fmt::formatter<canoe::EquationOfStateOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const canoe::EquationOfStateOptions& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "(type = {}; thermo = {})", p.type(),
                          p.thermo());
  }
};
