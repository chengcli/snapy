import torch
import snapy
from exchange import (
        MeshBlockExchange,
        get_buffer_id,
        populate_ranks_2d,
        )

class SlabExchange(MeshBlockExchange):
    def __post_init__(self):
        myrank, world_size = self.__init_dist()

        assert xelf.pz == 1, "px1 must be 1 for slab layout"
        assert self.px * self.py == world_size, \
                f"px2*px3 ({self.px}*{self.py}) != world_size ({world_size})"

        self.layout = snapy.SlabLayout(self.px, self.py,
                                       self.periodic_x, self.periodic_y)
        self.loc = self.layout.loc_of(myrank)

        self.info = snapy.DistributeInfo()
        self.info.nb3(self.px)
        self.info.nb2(self.py)
        self.info.lx3(self.loc[0])
        self.info.lx2(self.loc[1])
        self.info.gid(myrank)
        self.info.ranks(populate_ranks_2d(self.layout, myrank))

    def init_buffers(self,
                     block: snapy.MeshBlock,
                     block_vars: dict[str, torch.Tensor]):
        self.block = block

        self.send_bufs = [None] * 9
        self.recv_bufs = [None] * 9

        for x3_offset in [-1, 0, 1]:
            for x2_offset in [-1, 0, 1]:
                if x3_offset == 0 and x2_offset == 0: continue
                offset = (x3_offset, x2_offset, 0)
                nb = self.layout.neighbor_rank(*self.loc, *offset)
                if nb == -1: continue # no neighbor

                # invalidate block neighbor
                self.block.options.set_bfunc(*offset, None)

                bid = get_buffer_id(*offset)
                part = self.block.part(offset)
                nhydro, *dims = block_vars["hydro_u"][part].shape

                self.send_bufs[bid] = torch.empty((nhydro, *dims),
                                             device=block_vars["hydro_u"].device,
                                             dtype=block_vars["hydro_u"].dtype)
                self.recv_bufs[bid] = torch.empty_like(self.send_bufs[bid])

    def serialize(self,
                  block_vars: dict[str, torch.Tensor]):
        nhydro = block_vars["hydro_u"].shape[0]

        for x3_offset in [-1, 0, 1]:
            for x2_offset in [-1, 0, 1]:
                offset = (x3_offset, x2_offset, 0)
                bid = get_buffer_id(*offset)
                if self.send_bufs[bid] is not None:
                    part = self.block.part(offset)
                    self.send_bufs[bid][:nhydro,:].copy_(block_vars["hydro_u"][part])

    def deserialize(self,
                    block_vars: dict[str, torch.Tensor]):
        nhydro = block_vars["hydro_u"].shape[0]

        for x3_offset in [-1, 0, 1]:
            for x2_offset in [-1, 0, 1]:
                offset = (x3_offset, x2_offset, 0)
                bid = get_buffer_id(*offset)
                if self.recv_bufs[bid] is not None:
                    part = self.block.part(offset, exterior=True)
                    block_vars["hydro_u"][part].copy_(self.recv_bufs[bid][:nhydro,:])

    def forward(self, block_vars: dict[str, torch.Tensor]):
        ops = []
        self.serialize(block_vars, self.send_bufs)

        for r in range(1, len(self.info.ranks)):
            if send_bufs[r] is not None:
                ops.append(dist.P2POp(dist.isend, self.send_bufs[r], self.info.ranks[r]))
                ops.append(dist.P2POp(dist.irecv, self.recv_bufs[r], self.info.ranks[r]))

        if ops:
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs: r.wait()

        self.deserialize(block_vars, self.recv_bufs)
        return block_vars
