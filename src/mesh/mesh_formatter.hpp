#pragma once

// fmt
#include <fmt/format.h>

// snap
#include <snap/hydro/hydro_formatter.hpp>
#include <snap/intg/intg_formatter.hpp>

#include "meshblock.hpp"

template <>
struct fmt::formatter<snap::MeshBlockOptions> {
  // Parse format specifier if any (this example doesn't use custom specifiers)
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  // Define the format function for MeshBlockOptions
  template <typename FormatContext>
  auto format(const snap::MeshBlockOptions& p, FormatContext& ctx) const {
    return fmt::format_to(ctx.out(), "(hydro = {}; intg = {})", p.hydro(),
                          p.intg());
  }
};
