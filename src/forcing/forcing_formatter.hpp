#pragma once

// C/C++
#include <sstream>

// fmt
#include <fmt/format.h>

// snap
#include "forcing.hpp"

template <>
struct fmt::formatter<snap::ConstGravityOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::ConstGravityOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    p.report(ss);
    return fmt::format_to(ctx.out(), "{}", ss.str());
  }
};

template <>
struct fmt::formatter<snap::CoriolisOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::CoriolisOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    return fmt::format_to(ctx.out(), "{}", p.report(ss));
  }
};

template <>
struct fmt::formatter<snap::DiffusionOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::DiffusionOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    return fmt::format_to(ctx.out(), "{}", p.report(ss));
  }
};

template <>
struct fmt::formatter<snap::FricHeatOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::FricHeatOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    return fmt::format_to(ctx.out(), "{}", p.report(ss));
  }
};

template <>
struct fmt::formatter<snap::BodyHeatOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::BodyHeatOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    return fmt::format_to(ctx.out(), "{}", p.report(ss));
  }
};

template <>
struct fmt::formatter<snap::BotHeatOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::BotHeatOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    return fmt::format_to(ctx.out(), "{}", p.report(ss));
  }
};

template <>
struct fmt::formatter<snap::TopCoolOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::TopCoolOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    return fmt::format_to(ctx.out(), "{}", p.report(ss));
  }
};

template <>
struct fmt::formatter<snap::RelaxBotCompOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::RelaxBotCompOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    return fmt::format_to(ctx.out(), "{}", p.report(ss));
  }
};

template <>
struct fmt::formatter<snap::RelaxBotTempOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::RelaxBotTempOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    return fmt::format_to(ctx.out(), "{}", p.report(ss));
  }
};

template <>
struct fmt::formatter<snap::RelaxBotVeloOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::RelaxBotVeloOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    return fmt::format_to(ctx.out(), "{}", p.report(ss));
  }
};

template <>
struct fmt::formatter<snap::TopSpongeLyrOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::TopSpongeLyrOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    return fmt::format_to(ctx.out(), "{}", p.report(ss));
  }
};

template <>
struct fmt::formatter<snap::BotSpongeLyrOptions> {
  constexpr auto parse(fmt::format_parse_context& ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const snap::BotSpongeLyrOptions& p, FormatContext& ctx) const {
    std::stringstream ss;
    return fmt::format_to(ctx.out(), "{}", p.report(ss));
  }
};
