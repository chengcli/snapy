// fmt
#include <fmt/format.h>

// snap
#include "connectivity.hpp"
#include "layout.hpp"

namespace snap {

void SlabLayoutImpl::reset() {
  // build the ranks
  TORCH_CHECK(options->pz() == 1,
              "SlabLayoutImpl: pz must be 1 for slab layout");

  int px = options->px();
  int py = options->py();

  _coords2.resize(px * py);
  build_zorder_coords2(px, py, _coords2.data());
  build_rank_of2(px, py, _coords2.data(), _rankof.data());

  // build backend
  _init_backend();
}

void SlabLayoutImpl::pretty_print(std::ostream& os) const {
  options->report(os);
  os << " Rank | (rx,ry)\n";
  os << "----------------\n";
  for (int r = 0; r < options->px() * options->py(); ++r) {
    os << fmt::format(" {:>3} | ({:>2},{:>2})\n", r, _coords2[r].x,
                      _coords2[r].y);
  }
}

std::tuple<int, int, int> SlabLayoutImpl::loc_of(int rank) const {
  if (rank < 0 || rank >= options->px() * options->py()) return {-1, -1, 0};
  return {_coords2[rank].x, _coords2[rank].y, 0};
}

int SlabLayoutImpl::neighbor_rank(std::tuple<int, int, int> iloc,
                                  std::tuple<int, int, int> offset) const {
  auto [rx, ry, _1] = iloc;
  auto [dx, dy, _2] = offset;

  int nx = rx + dx;
  int ny = ry + dy;

  if (options->periodic_x()) {
    if (nx < 0)
      nx += options->px();
    else if (nx >= options->px())
      nx -= options->px();
  } else {
    if (nx < 0 || nx >= options->px()) return -1;
  }

  if (options->periodic_y()) {
    if (ny < 0)
      ny += options->py();
    else if (ny >= options->py())
      ny -= options->py();
  } else {
    if (ny < 0 || ny >= options->py()) return -1;
  }

  return _rankof[linear_index2(options->px(), options->py(), ny, nx)];
}

void SlabLayoutImpl::forward(MeshBlockImpl const* pmb, Variables& vars) {
  // Serialize data into send buffers
  serialize(pmb, vars);

  std::vector<c10::intrusive_ptr<c10d::Work>> works;

  // Get my rank
  auto rank = options->rank();

  // Get my logical location
  auto iloc = loc_of(rank);

  for (int x3_offset = -1; x3_offset <= 1; ++x3_offset) {
    for (int x2_offset = -1; x2_offset <= 1; ++x2_offset) {
      // Skip the center (self)
      if (x3_offset == 0 && x2_offset == 0) continue;

      std::tuple<int, int, int> offset(x3_offset, x2_offset, 0);
      int nb = neighbor_rank(iloc, offset);

      int r = get_buffer_id(offset);
      if (nb >= 0) {
        if (nb != rank) {  // different ranks
          // Send operation
          auto send_work = pg->send(send_bufs[r], nb, 0);
          works.push_back(send_work);

          // Receive operation
          auto recv_work = pg->recv(recv_bufs[r], nb, 0);
          works.push_back(recv_work);
        } else {  // self-send
          for (int n = 0; n < recv_bufs[r].size(); ++n)
            recv_bufs[r][n].copy_(send_bufs[r][n]);
        }
      }
    }
  }

  // Wait for all operations to complete
  for (auto& work : works) work->wait();

  // Deserialize received data into ghost zones
  deserialize(pmb, vars);
}

}  // namespace snap
