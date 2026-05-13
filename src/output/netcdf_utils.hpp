#pragma once

// C/C++
#include <cctype>
#include <string>

namespace snap {

inline std::string sanitize_netcdf_name(std::string name) {
  for (char& c : name) {
    unsigned char uc = static_cast<unsigned char>(c);
    if (!(std::isalnum(uc) || c == '_')) {
      c = '_';
    }
  }

  if (name.empty()) {
    return "_";
  }

  unsigned char first = static_cast<unsigned char>(name.front());
  if (!(std::isalpha(first) || name.front() == '_')) {
    name.insert(name.begin(), '_');
  }

  return name;
}

}  // namespace snap
