#pragma once

// spdlog
#include <spdlog/spdlog.h>

// snap
#include <snap/bc/bc_formatter.hpp>
#include <snap/hydro/hydro_formatter.hpp>

#include "mesh.hpp"
#include "meshblock.hpp"
#include "oct_tree.hpp"

template <>
struct fmt::formatter<snap::MeshOptions> {
  // Parse format specifier if any (this example doesn't use custom specifiers)
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  // Define the format function for MeshOptions
  template <typename FormatContext>
  auto format(const snap::MeshOptions& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "(x1 = {}, {}; x2 = {}, {}; x3 = {}, {})",
                          p.x1min(), p.x1max(), p.x2min(), p.x2max(), p.x3min(),
                          p.x3max());
  }
};

template <>
struct fmt::formatter<snap::MeshBlockOptions> {
  // Parse format specifier if any (this example doesn't use custom specifiers)
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  // Define the format function for MeshBlockOptions
  template <typename FormatContext>
  auto format(const snap::MeshBlockOptions& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "(nghost = {}; hydro = {}; bflags = {})",
                          p.nghost(), p.hydro(), p.bflags());
  }
};

template <>
struct fmt::formatter<snap::LogicalLocation> {
  // Parse format specifier if any (this example doesn't use custom specifiers)
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  // Define the format function for MeshOptions
  template <typename FormatContext>
  auto format(const snap::LogicalLocation& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(),
                          "(level = {}; lx1 = {:b}; lx2 = {:b}; lx3 = {:b})",
                          p->level, p->lx1, p->lx2, p->lx3);
  }
};

template <>
struct fmt::formatter<snap::OctTreeOptions> {
  // Parse format specifier if any (this example doesn't use custom specifiers)
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  // Define the format function for MeshOptions
  template <typename FormatContext>
  auto format(const snap::OctTreeOptions& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(),
                          "(nb1 = {}; nb2 = {}; nb3 = {}; ndim = {})", p.nb1(),
                          p.nb2(), p.nb3(), p.ndim());
  }
};
