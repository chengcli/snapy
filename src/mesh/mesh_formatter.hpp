#pragma once

// fmt
#include <fmt/format.h>

// snap
#include <snap/hydro/hydro_formatter.hpp>
#include <snap/intg/intg_formatter.hpp>
#include <snap/output/output_type.hpp>

#include "meshblock.hpp"

template <>
struct fmt::formatter<snap::MeshBlockOptions> {
  // Parse format specifier if any (this example doesn't use custom specifiers)
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  // Define the format function for MeshBlockOptions
  template <typename FormatContext>
  auto format(const snap::MeshBlockOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    p.report(ss);
    ss << "Distribute info:\n";
    p.dist().report(ss);
    ss << "Output options: [\n";
    for (auto const& out : p.outputs()) {
      out.report(ss);
      ss << ",\n";
    }
    ss << "]\n";
    ss << "Integrator options:\n";
    p.intg().report(ss);
    return fmt::format_to(ctx.out(), "{}\n{}", ss.str(), p.hydro());
  }
};
