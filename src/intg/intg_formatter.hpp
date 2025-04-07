#pragma once

// spdlog
#include <spdlog/spdlog.h>

// fvm
#include "integrator.hpp"

template <>
struct fmt::formatter<canoe::IntegratorWeight> {
  // Parse format specifier if any (this example doesn't use custom specifiers)
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  // Define the format function for IntegratorOptions
  template <typename FormatContext>
  auto format(const canoe::IntegratorWeight& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "({}, {}, {})", p.wght0(), p.wght1(),
                          p.wght2());
  }
};

template <>
struct fmt::formatter<canoe::IntegratorOptions> {
  // Parse format specifier if any (this example doesn't use custom specifiers)
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  // Define the format function for IntegratorOptions
  template <typename FormatContext>
  auto format(const canoe::IntegratorOptions& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "(type = {}; cfl = {})", p.type(),
                          p.cfl());
  }
};
