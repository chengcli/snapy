import torch
import snapy
from exchange import (
    populate_ranks_2d,
    )
from slab_exchange import SlabExchange

class CubedSphereExchange(SlabExchange):
    def __post_init__(self):
        myrank, world_size = self.__init_dist()

        assert self.pz == 1, "px1 must be 1 for cubed_sphere layout"
        assert self.px == self.py, "px2 must equal px3 for cubed_sphere layout"
        assert 6 * self.px * self.py == dist.get_world_size(), f"6*px2*px3 ({self.px}*{self.py}) != world_size ({6*dist.get_world_size()})"

        self.layout = snapy.CubedSphereLayout(self.px)
        self.loc = self.layout.loc_of(myrank)

        self.info = snapy.DistributeInfo()
        self.info.nb3(self.px)
        self.info.nb2(self.py)
        self.info.face(self.loc[0])
        self.info.lx3(self.loc[1])
        self.info.lx2(self.loc[2])
        self.info.gid(myrank)

        self.ranks = populate_ranks_2d(self.layout, myrank)
