#pragma once

// fmt
#include <fmt/format.h>

// snap
#include "reconstruct.hpp"

template <>
struct fmt::formatter<snap::ReconstructOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::ReconstructOptions& p, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "(shock = {}; interp = {})", p.shock(),
                          p.interp());
  }
};

template <>
struct fmt::formatter<snap::InterpOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::InterpOptions& p, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "(type = {}; scale = {})", p.type(),
                          p.scale());
  }
};
