import torch
import snapy
from exchange import (
        MeshBlockExchange,
        get_buffer_id,
        populate_ranks_3d,
        )

class CubedExchange(MeshBlockExchange):
    def __post_init__(self):
        myrank, world_size = self.__init_dist()

        assert self.px * self.py * self.pz == world_size, \
                f"px1*px2*px3 {self.px}*{self.py}*{self.pz}) != world_size ({world_size})"
        layout = snapy.CubedLayout(self.px, self.py, self.pz,
                                   self.periodic_x3,
                                   self.periodic_x2,
                                   self.periodic_x1)
        self.loc = self.layout.loc_of(myrank)

        self.info = snapy.DistributeInfo()
        self.info.nb3(px)
        self.info.nb2(py)
        self.info.nb1(pz)
        self.info.lx3(loc[0])
        self.info.lx2(loc[1])
        self.info.lx1(loc[2])
        self.info.gid(myrank)

        self.ranks = populate_ranks_3d(self.layout, myrank)

    def init_buffers(self,
                     block: snapy.MeshBlock,
                     block_vars: dict[str, torch.Tensor]):
        self.send_bufs = [None] * 27
        self.recv_bufs = [None] * 27

        for x3_offset in [-1, 0, 1]:
            for x2_offset in [-1, 0, 1]:
                for x1_offset in [-1, 0, 1]:
                    if x3_offset == 0 and x2_offset == 0 and x1_offset == 0:
                        continue
                    offset = (x3_offset, x2_offset, x1_offset)
                    nb = layout.neighbor_rank(*self.loc, *offset)
                    if nb == -1: continue  # no neighbor

                    # invalidate block neighbor
                    block.options.set_bfunc(*offset, None)

                    bid = get_buffer_id(*offset)
                    part = block.part(offset)
                    nhydro, *dims = block_vars["hydro_u"][part].shape

                    self.send_bufs[bid] = torch.empty((nhydro, *dims),
                                                 device=block_vars["hydro_u"].device,
                                                 dtype=block_vars["hydro_u"].dtype)
                    self.recv_bufs[bid] = torch.empty_like(self.send_bufs[bid])

    def serialize(self,
                  block: snapy.MeshBlock,
                  block_vars: dict[str, torch.Tensor]):
        nhydro = block_vars["hydro_u"].shape[0]

        for x3_offset in [-1, 0, 1]:
            for x2_offset in [-1, 0, 1]:
                for x1_offset in [-1, 0, 1]:
                    offset = (x3_offset, x2_offset, x1_offset)
                    bid = get_buffer_id(*offset)
                    if self.send_bufs[bid] is not None:
                        part = block.part(offset)
                        self.send_bufs[bid][:] = block_vars["hydro_u"][part][:]

    def deserialize(self,
                    block: snapy.MeshBlock,
                    block_vars: dict[str, torch.Tensor]):
        nhydro = block_vars["hydro_u"].shape[0]

        for x3_offset in [-1, 0, 1]:
            for x2_offset in [-1, 0, 1]:
                for x1_offset in [-1, 0, 1]:
                    offset = (x3_offset, x2_offset, x1_offset)
                    bid = get_buffer_id(*offset)
                    if self.recv_bufs[bid] is not None:
                        part = block.part(offset)
                        block_vars["hydro_u"][part][:] = self.recv_bufs[bid][:]
