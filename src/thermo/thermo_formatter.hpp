#pragma once

// fmt
#include <fmt/format.h>

// snap
#include <snap/kintera/kintera_formatter.hpp>

#include "thermodynamics.hpp"

template <>
struct fmt::formatter<snap::Nucleation> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::Nucleation& p, FormatContext& ctx) {
    return fmt::format_to(ctx.out(), "({}; min_tem = {}; max_tem = {})",
                          p.reaction(), p.min_tem(), p.max_tem());
  }
};

template <>
struct fmt::formatter<snap::CondensationOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::CondensationOptions& p, FormatContext& ctx) {
    std::ostringstream reactions;
    for (size_t i = 0; i < p.react().size(); ++i) {
      reactions << fmt::format("R{}: {}", i + 1, p.react()[i]);
      if (i != p.react().size() - 1) {
        reactions << "; ";
      }
    }

    std::ostringstream species;
    for (size_t i = 0; i < p.species().size(); ++i) {
      species << p.species()[i];
      if (i != p.species().size() - 1) {
        species << ", ";
      }
    }

    return fmt::format_to(ctx.out(), "(react = ({}); species = ({}))",
                          reactions.str(), species.str());
  }
};

template <>
struct fmt::formatter<snap::ThermodynamicsOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::ThermodynamicsOptions& p, FormatContext& ctx) {
    std::ostringstream species;
    for (size_t i = 0; i < p.species().size(); ++i) {
      species << p.species()[i];
      if (i != p.species().size() - 1) {
        species << ", ";
      }
    }

    return fmt::format_to(ctx.out(),
                          "(Rd = {}; gammad_ref = {}; nvapor = {}; ncloud = "
                          "{}; speices = ({}); cond = {})",
                          p.Rd(), p.gammad_ref(), p.nvapor(), p.ncloud(),
                          species.str(), p.cond());
  }
};
