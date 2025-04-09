#pragma once

// fmt
#include <fmt/format.h>

// snap
#include <snap/coord/coord_formatter.hpp>

#include "vertical_implicit.hpp"

template <>
struct fmt::formatter<snap::VerticalImplicitOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::VerticalImplicitOptions& p,
              FormatContext& ctx) const {
    return fmt::format_to(
        ctx.out(),
        "(type = {}; nghost = {}; grav = {}; scheme = {}; coord = {})",
        p.type(), p.nghost(), p.grav(), p.scheme(), p.coord());
  }
};
