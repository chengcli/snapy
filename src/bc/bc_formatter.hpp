#pragma once

// fmt
#include <fmt/format.h>

// snap
#include "bc.hpp"
#include "internal_boundary.hpp"

template <>
struct fmt::formatter<snap::InternalBoundaryOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::InternalBoundaryOptions& p,
              FormatContext& ctx) const {
    return fmt::format_to(
        ctx.out(),
        "(nghost = {}; max_iter = {}, solid_density = {}; solid_pressure = {})",
        p.nghost(), p.max_iter(), p.solid_density(), p.solid_pressure());
  }
};

template <>
struct fmt::formatter<snap::BoundaryFuncOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::BoundaryFuncOptions& p, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "(type = {}; nghost = {})", p.type(),
                          p.nghost());
  }
};
