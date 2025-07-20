#pragma once

// C/C++
#include <sstream>

// fmt
#include <fmt/format.h>

// snap
#include <snap/coord/coord_formatter.hpp>

#include "implicit.hpp"

template <>
struct fmt::formatter<snap::ImplicitOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::ImplicitOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    p.report(ss);
    return fmt::format_to(ctx.out(), "{}", ss.str());
  }
};
