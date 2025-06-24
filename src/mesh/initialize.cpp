// base
#include <configure.h>

// base
// #include <formatter.hpp>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/input/parameter_input.hpp>

#include "mesh.hpp"
#include "mesh_formatter.hpp"
#include "mesh_functions.hpp"
#include "meshblock.hpp"

namespace snap {
void MeshBlockImpl::initialize(MeshOptions const& mesh_options,
                               OctTree const& tree) {
  auto nghost = options.hydro().coord().nghost();

  // initialize coordinates
  auto lx1 = ploc->lx1;
  auto ll = ploc->level;
  int64_t nrbx_ll = tree->options.nb1() << (ll - tree->root_level());
  auto& mesh_bflags = mesh_options.bflags;

  for (auto& m : named_modules()) {
    if (m.key().find("coord") == std::string::npos) continue;
    auto pcoord = m.value()->as<CoordinateImpl>();
    auto meshgen1 = mesh_options.meshgen1();
    auto meshgen2 = mesh_options.meshgen2();
    auto meshgen3 = mesh_options.meshgen3();
    auto& op = pcoord->options;
    float rghost;

    // calculate physical block size, x1
    if (nc1() > 1) {
      rghost = 1. * nghost / (nc1() - 2 * nghost);
    } else {
      rghost = 0.0;
    }

    if (lx1 == 0) {
      op.x1min(mesh_options.x1min() -
               rghost * (mesh_options.x1max() - mesh_options.x1min()));
      options.bflags()[BoundaryFace::kInnerX1] =
          mesh_bflags[BoundaryFace::kInnerX1];
    } else {
      auto rx = compute_logical_position(torch::tensor({lx1}, torch::kInt64),
                                         nrbx_ll, true);
      op.x1min(meshgen1(rx - rghost, mesh_options.x1min(), mesh_options.x1max())
                   .item<float>());
      options.bflags()[BoundaryFace::kInnerX1] = BoundaryFlag::kExchange;
    }

    if (lx1 == nrbx_ll - 1) {
      op.x1max(mesh_options.x1max() +
               rghost * (mesh_options.x1max() - mesh_options.x1min()));
      options.bflags()[BoundaryFace::kOuterX1] =
          mesh_bflags[BoundaryFace::kOuterX1];
    } else {
      auto rx = compute_logical_position(
          torch::tensor({lx1 + 1}, torch::kInt64), nrbx_ll, true);
      op.x1max(meshgen1(rx + rghost, mesh_options.x1min(), mesh_options.x1max())
                   .item<float>());
      options.bflags()[BoundaryFace::kOuterX1] = BoundaryFlag::kExchange;
    }

    // calculate physical block size, x2
    if (nc2() > 1) {
      rghost = 1. * nghost / (nc2() - 2 * nghost);
    } else {
      rghost = 0.0;
    }

    if (nc2() == 1) {
      op.x2min(mesh_options.x2min());
      op.x2max(mesh_options.x2max());
      options.bflags()[BoundaryFace::kInnerX2] =
          mesh_bflags[BoundaryFace::kInnerX2];
      options.bflags()[BoundaryFace::kOuterX2] =
          mesh_bflags[BoundaryFace::kOuterX2];
    } else {
      auto lx2 = ploc->lx2;
      nrbx_ll = tree->options.nb2() << (ll - tree->root_level());

      if (lx2 == 0) {
        op.x2min(mesh_options.x2min() -
                 rghost * (mesh_options.x2max() - mesh_options.x2min()));
        options.bflags()[BoundaryFace::kInnerX2] =
            mesh_bflags[BoundaryFace::kInnerX2];
      } else {
        auto rx = compute_logical_position(torch::tensor({lx2}, torch::kInt64),
                                           nrbx_ll, true);
        op.x2min(
            meshgen2(rx - rghost, mesh_options.x2min(), mesh_options.x2max())
                .item<float>());
        options.bflags()[BoundaryFace::kInnerX2] = BoundaryFlag::kExchange;
      }

      if (lx2 == nrbx_ll - 1) {
        op.x2max(mesh_options.x2max() +
                 rghost * (mesh_options.x2max() - mesh_options.x2min()));
        options.bflags()[BoundaryFace::kOuterX2] =
            mesh_bflags[BoundaryFace::kOuterX2];
      } else {
        auto rx = compute_logical_position(
            torch::tensor({lx2 + 1}, torch::kInt64), nrbx_ll, true);
        op.x2max(
            meshgen2(rx + rghost, mesh_options.x2min(), mesh_options.x2max())
                .item<float>());
        options.bflags()[BoundaryFace::kOuterX2] = BoundaryFlag::kExchange;
      }
    }

    // calculate physical block size, x3
    if (nc3() > 1) {
      rghost = 1. * nghost / (nc3() - 2 * nghost);
    } else {
      rghost = 0.0;
    }

    if (nc3() == 1) {
      op.x3min(mesh_options.x3min());
      op.x3max(mesh_options.x3max());
      options.bflags()[BoundaryFace::kInnerX3] =
          mesh_bflags[BoundaryFace::kInnerX3];
      options.bflags()[BoundaryFace::kOuterX3] =
          mesh_bflags[BoundaryFace::kOuterX3];
    } else {
      auto lx3 = ploc->lx3;
      nrbx_ll = tree->options.nb3() << (ll - tree->root_level());

      if (lx3 == 0) {
        op.x3min(mesh_options.x3min() -
                 rghost * (mesh_options.x3max() - mesh_options.x3min()));
        options.bflags()[BoundaryFace::kInnerX3] =
            mesh_bflags[BoundaryFace::kInnerX3];
      } else {
        auto rx = compute_logical_position(torch::tensor({lx3}, torch::kInt64),
                                           nrbx_ll, true);
        op.x3min(
            meshgen3(rx - rghost, mesh_options.x3min(), mesh_options.x3max())
                .item<float>());
        options.bflags()[BoundaryFace::kInnerX3] = BoundaryFlag::kExchange;
      }

      if (lx3 == nrbx_ll - 1) {
        op.x3max(mesh_options.x3max() +
                 rghost * (mesh_options.x3max() - mesh_options.x3min()));
        options.bflags()[BoundaryFace::kOuterX3] =
            mesh_bflags[BoundaryFace::kOuterX3];
      } else {
        auto rx = compute_logical_position(
            torch::tensor({lx3 + 1}, torch::kInt64), nrbx_ll, true);
        op.x3max(
            meshgen3(rx + rghost, mesh_options.x3min(), mesh_options.x3max())
                .item<float>());
        options.bflags()[BoundaryFace::kOuterX3] = BoundaryFlag::kExchange;
      }
    }

    pcoord->reset_coordinates({meshgen1, meshgen2, meshgen3});
  }
}

}  // namespace snap
