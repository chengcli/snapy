#pragma once

// spdlog
#include <spdlog/spdlog.h>

// fvm
#include "reaction.hpp"

template <>
struct fmt::formatter<canoe::Composition> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const canoe::Composition& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "({})", canoe::to_string(p));
  }
};

template <>
struct fmt::formatter<canoe::Reaction> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const canoe::Reaction& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "({})", p.equation());
  }
};
