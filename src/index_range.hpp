#pragma once

// C/C+
#include <iostream>

// spdlog
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>

// base
#include <globals.h>

struct IndexRange {
  int is, ie, js, je, ks, ke;

  IndexRange(int is_, int ie_, int js_, int je_, int ks_, int ke_)
      : is(is_), ie(ie_), js(js_), je(je_), ks(ks_), ke(ke_) {}

  IndexRange(int ie_, int je_, int ke_)
      : is(0), ie(ie_), js(0), je(je_), ks(0), ke(ke_) {}

  void print(std::ostream &stream) const {
    stream << "is = " << is << ", ie = " << ie << ", js = " << js
           << ", je = " << je << ", ks = " << ks << ", ke = " << ke
           << std::endl;
  }

  IndexRange shift(char name, int size) {
    IndexRange r = *this;
    switch (name) {
      case 'i':
        r.is += size;
        r.ie += size;
        break;
      case 'j':
        r.js += size;
        r.je += size;
        break;
      case 'k':
        r.ks += size;
        r.ke += size;
        break;
      default:
        logger->error("IndexRange::shift, unknown dimension name: {}", name);
    }
    return r;
  }

  IndexRange extend(char name, int size) {
    IndexRange r = *this;
    switch (name) {
      case 'i':
        r.is -= size;
        r.ie += size;
        break;
      case 'j':
        r.js -= size;
        r.je += size;
        break;
      case 'k':
        r.ks -= size;
        r.ke += size;
        break;
      default:
        logger->error("IndexRange::extend, unknown dimension name: {}", name);
    }
    return r;
  }
};
