import torch
import snapy
import torch.distributed as dist
import numpy as np
from typing import List, Optional

@torch.compile
def get_buffer_id(dx: int, dy: int, dz: int=0):
    return dx % 3 + (dy % 3) * 3 + (dz % 3) * 9

@torch.compile
def populate_ranks_2d(layout, myrank):
    ranks = np.zeros(9, dtype=int)

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                offset = (dx, dy, 0)
                bid = get_buffer_id(*offset)
                ranks[bid] = layout.neighbor_rank(*layout.loc_of(myrank),
                                                  *offset)
    # my rank
    ranks[0] = myrank
    return ranks

@torch.compile
def populate_ranks_3d(layout, myrank):
    ranks = np.zeros(27, dtype=int)

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx != 0 or dy != 0 or dz != 0:
                    offset = (dx, dy, dz)
                    bid = get_buffer_id(*offset)
                    ranks[bid] = layout.neighbor_rank(*layout.loc_of(myrank),
                                                      *offset)
    ranks[0] = myrank
    return ranks

@dataclass
class MeshBlockExchange(torch.nn.Module):
    px1: int = 1
    px2: int = 1
    px3: int = 1
    periodic_x1: bool = False
    periodic_x2: bool = False
    periodic_x3: bool = False
    device_name: str = "cpu"

    def __post_init__(self):
        self.__init_dist()

    def __init_dist(self)
        if self.device_name == "cpu":
            dist.init_process_group(backend="gloo", init_method="env://")
        else:
            dist.init_process_group(backend="nccl", init_method="env://")

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if self.device_name == "cuda":
            ngpu = torch.cuda.device_count()
            local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, ngpu)))
            torch.cuda.set_device(local_rank)
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")

        self.px, self.py, self.pz = px3, px2, px1
        self.periodic_x = self.periodic_x3
        self.periodic_y = self.periodic_x2
        self.periodic_z = self.periodic_x1
        return rank, world_size

    def __del__(self):
        dist.destroy_process_group()

    def init_buffers(self,
                     block: snapy.MeshBlock,
                     block_vars: dict[str, torch.Tensor]):
        raise NotImplementedError

    def serialize(self, block_vars: dict[str, torch.Tensor]):
        raise NotImplementedError

    def deserialize(self, block_vars: dict[str, torch.Tensor]):
        raise NotImplementedError

    def forward(self, block_vars: dict[str, torch.Tensor]):
        ops = []
        self.serialize(self.block, block_vars)

        for bid, buf in enumerate(self.send_bufs):
            if buf is not None:
                nb = self.info.ranks[bid]
                ops.append(dist.isend(buf, dst=nb, tag=bid))

        for bid, buf in enumerate(self.recv_bufs):
            if buf is not None:
                nb = self.info.ranks[bid]
                ops.append(dist.irecv(buf, src=nb, tag=bid))

        for op in ops: op.wait()
        self.deserialize(self.block, block_vars)
        return block_vars
