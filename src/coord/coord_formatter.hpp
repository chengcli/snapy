#pragma once

// fmt
#include <fmt/format.h>

// snap
#include "coordinate.hpp"

template <>
struct fmt::formatter<snap::CoordinateOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::CoordinateOptions& p, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(),
                          "(type = {}; nghost = {}; x1 = {}, {}, {}; x2 = {}, "
                          "{}, {}; x3 = {}, {}, {})",
                          p.type(), p.nghost(), p.x1min(), p.x1max(), p.nc1(),
                          p.x2min(), p.x2max(), p.nc2(), p.x3min(), p.x3max(),
                          p.nc3());
  }
};
