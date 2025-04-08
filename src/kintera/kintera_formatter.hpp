#pragma once

// fmt
#include <fmt/format.h>

// snap
#include "reaction.hpp"

template <>
struct fmt::formatter<snap::Composition> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::Composition& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "({})", snap::to_string(p));
  }
};

template <>
struct fmt::formatter<snap::Reaction> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::Reaction& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "({})", p.equation());
  }
};
