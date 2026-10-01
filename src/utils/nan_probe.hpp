#pragma once

// Diagnostic only (scratch/250-cuda-nan): print a window of a tensor around
// one cell at named checkpoints of the update, plus the count and first
// location of non-finite entries in the whole array. Never modifies data.
//
//   SNAP_NANPROBE=c0:c1:j:i[:r]   cycles c0..c1, window (j +- r, i +- r),
//                                 j, i are full-array indices (ghosts incl.)
//   SNAP_NANPROBE_FILE=path       output file (default stdout)
//   SNAP_NANPROBE_TAGS=a,b        only tags containing one of these
//   SNAP_NANPROBE_STATS=1         also print per-component global min/max

// C/C++
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

// torch
#include <torch/torch.h>

namespace snap {
namespace nanprobe {

struct Config {
  bool on = false;
  int c0 = 0, c1 = 1 << 30, j = 0, i = 0, r = 3;
  FILE* fp = stdout;
  std::vector<std::string> tags;
  bool stats = false;
};

inline Config& cfg() {
  static Config c = [] {
    Config c;
    char const* s = std::getenv("SNAP_NANPROBE");
    if (s) {
      std::sscanf(s, "%d:%d:%d:%d:%d", &c.c0, &c.c1, &c.j, &c.i, &c.r);
      c.on = true;
      char const* f = std::getenv("SNAP_NANPROBE_FILE");
      if (f) c.fp = std::fopen(f, "w");
      if (char const* t = std::getenv("SNAP_NANPROBE_TAGS")) {
        std::string all(t);
        size_t b = 0;
        while (b <= all.size()) {
          size_t e = all.find(',', b);
          if (e == std::string::npos) e = all.size();
          if (e > b) c.tags.push_back(all.substr(b, e - b));
          b = e + 1;
        }
      }
      c.stats = std::getenv("SNAP_NANPROBE_STATS") != nullptr;
    }
    return c;
  }();
  return c;
}

struct State {
  int cycle = 0, stage = -1, redo = 0;
};

inline State& state() {
  static State s;
  return s;
}

inline bool active() {
  auto& c = cfg();
  auto& s = state();
  return c.on && s.cycle >= c.c0 && s.cycle <= c.c1;
}

//! t: [..., nc3, nc2, nc1]; leading dims are flattened into components
inline void probe(char const* tag, torch::Tensor const& t) {
  if (!active() || !t.defined() || t.dim() < 3) return;
  auto& c = cfg();
  if (!c.tags.empty()) {
    bool keep = false;
    for (auto const& x : c.tags) keep = keep || std::strstr(tag, x.c_str());
    if (!keep) return;
  }
  auto& s = state();
  auto x = t.detach().to(torch::kCPU).to(torch::kFloat64).contiguous();
  int64_t n1 = x.size(-1), n2 = x.size(-2), n3 = x.size(-3);
  int64_t nc = x.numel() / (n1 * n2 * n3);
  x = x.reshape({nc, n3, n2, n1});
  auto bad = torch::isfinite(x).logical_not();
  int64_t nbad = bad.sum().item<int64_t>();
  std::string first = "-";
  if (nbad > 0) {
    auto idx = bad.nonzero()[0];
    char buf[96];
    std::snprintf(buf, sizeof(buf), "(c=%ld,k=%ld,j=%ld,i=%ld)",
                  idx[0].item<int64_t>(), idx[1].item<int64_t>(),
                  idx[2].item<int64_t>(), idx[3].item<int64_t>());
    first = buf;
  }
  auto a = x.accessor<double, 4>();
  int64_t nbadw = 0;
  for (int64_t q = 0; q < nc; ++q)
    for (int jj = c.j - c.r; jj <= c.j + c.r; ++jj)
      for (int ii = c.i - c.r; ii <= c.i + c.r; ++ii)
        if (jj >= 0 && jj < n2 && ii >= 0 && ii < n1 &&
            !std::isfinite(a[q][0][jj][ii]))
          ++nbadw;
  std::fprintf(c.fp, "NP %d %d %d %s nc=%ld nbad=%ld nbadwin=%ld first=%s\n",
               s.cycle, s.stage, s.redo, tag, nc, nbad, nbadw, first.c_str());
  if (c.stats) {
    for (int64_t q = 0; q < nc; ++q) {
      auto xq = x[q].flatten();
      int64_t imn = xq.argmin().item<int64_t>();
      int64_t imx = xq.argmax().item<int64_t>();
      std::fprintf(c.fp,
                   "NPS %d %d %d %s %ld min=%.17e at(j=%ld,i=%ld) "
                   "max=%.17e at(j=%ld,i=%ld)\n",
                   s.cycle, s.stage, s.redo, tag, q, xq[imn].item<double>(),
                   (imn / n1) % n2, imn % n1, xq[imx].item<double>(),
                   (imx / n1) % n2, imx % n1);
    }
  }
  for (int64_t q = 0; q < nc; ++q) {
    std::fprintf(c.fp, "NPW %d %d %d %s %ld", s.cycle, s.stage, s.redo, tag, q);
    for (int jj = c.j - c.r; jj <= c.j + c.r; ++jj)
      for (int ii = c.i - c.r; ii <= c.i + c.r; ++ii) {
        if (jj >= 0 && jj < n2 && ii >= 0 && ii < n1)
          std::fprintf(c.fp, " %.17e", a[q][0][jj][ii]);
        else
          std::fprintf(c.fp, " x");
      }
    std::fprintf(c.fp, "\n");
  }
  std::fflush(c.fp);
}

}  // namespace nanprobe
}  // namespace snap
