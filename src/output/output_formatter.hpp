#pragma once

// spdlog
#include <spdlog/spdlog.h>

template <>
struct fmt::formatter<snap::OutputOptions> {
  // Parse format specifier if any (this example doesn't use custom specifiers)
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  // Define the format function for OutputOptions
  template <typename FormatContext>
  auto format(const snap::OutputOptions& p, FormatContext& ctx) {
    return fmt::format_to(
        ctx.out(),
        "(fid = {}; dt = {}; dcycle = {}; block_name = {}; file_basename = {})",
        p.fid(), p.dt(), p.block_name(), p.file_basename());
  }
};
