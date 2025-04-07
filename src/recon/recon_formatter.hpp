#pragma once

// spdlog
#include <spdlog/spdlog.h>

// fvm
#include "reconstruct.hpp"

template <>
struct fmt::formatter<canoe::ReconstructOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const canoe::ReconstructOptions& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "(shock = {}; interp = {})", p.shock(),
                          p.interp());
  }
};

template <>
struct fmt::formatter<canoe::InterpOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const canoe::InterpOptions& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "(type = {}; scale = {})", p.type(),
                          p.scale());
  }
};
