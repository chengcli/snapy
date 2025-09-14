#pragma once

// C/C++
#include <iostream>

// arg
#include <snap/add_arg.h>

namespace snap {

struct DistributeInfo {
  void report(std::ostream &os) const {
    os << "* face = " << face() << "\n";
    os << "* lx1 = " << lx1() << "\n";
    os << "* lx2 = " << lx2() << "\n";
    os << "* lx3 = " << lx3() << "\n";
    os << "* nb1 = " << nb1() << "\n";
    os << "* nb2 = " << nb2() << "\n";
    os << "* nb3 = " << nb3() << "\n";
    os << "* level = " << level() << "\n";
    os << "* gid = " << gid() << "\n";
  }

  ADD_ARG(int, lx1) = 0;
  ADD_ARG(int, lx2) = 0;
  ADD_ARG(int, lx3) = 0;
  ADD_ARG(int, nb1) = 1;
  ADD_ARG(int, nb2) = 1;
  ADD_ARG(int, nb3) = 1;
  ADD_ARG(int, face) = 0;
  ADD_ARG(int, level) = 0;
  ADD_ARG(int, gid) = 0;
};

}  // namespace snap

#undef ADD_ARG
