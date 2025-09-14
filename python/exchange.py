import torch
import snapy
import torch.distributed as dist
from typing import List

def get_buffer_2d(dx: int, dy: int, dz: int = 0):
    return dx % 3 + (dy % 3) * 3

def get_buffer_3d(dx: int, dy: int, dz: int):
    return dx % 3 + (dy % 3) * 3 + (dz % 3) * 9

def init_dist(args,
              periodic_x1: bool=False,
              periodic_x2: bool=False,
              periodic_x3: bool=False):
    if args.device == "cpu":
        dist.init_process_group(backend="gloo", init_method="env://")
    else:
        dist.init_process_group(backend="nccl", init_method="env://")

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if args.device == "cuda":
        ngpu = torch.cuda.device_count()
        local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, ngpu)))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    px, py, pz = args.px3, args.px2, args.px1

    if args.layout == "slab":
        assert pz == 1, "px1 must be 1 for slab layout"
        assert px * py == world_size, f"px2*px3 ({px}*{py}) != world_size ({world_size})"
        layout = snapy.SlabLayout(px, py, periodic_x3, periodic_x2)
    elif args.layout == "cubed":
        assert px * py * pz == world_size, f"px1*px2*px3 ({px}*{py}*{pz}) != world_size ({world_size})"
        layout = snapy.CubedLayout(px, py, pz, periodic_x3, periodic_x2, periodic_x1)
    else: # cubed_sphere
        assert pz == 1, "px1 must be 1 for cubed_sphere layout"
        assert px == py, "px2 must equal px3 for cubed_sphere layout"
        assert 6 * px * py == world_size, f"6*px2*px3 ({px}*{py}) != world_size ({6*world_size})"
        layout = snapy.CubedSphereLayout(px)

    if args.layout == "cubed":  # 3D decomposition
        ranks = [] * 27
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx != 0 or dy != 0 or dz != 0:
                        offset = (dx, dy, dz)
                        bid = get_buffer_3d(*offset)
                        ranks[bid] = layout.neighbor_rank(*layout.loc_of(rank), *offset)
        # my rank
        ranks[get_buffer_3d(0, 0, 0)] = rank
    else:  # 2D decomposition
        neighbor_ranks = [] * 9
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    offset = (dx, dy, 0)
                    bid = get_buffer_2d(*offset)
                    neighbor_ranks[bid] = layout.neighbor_rank(*layout.loc_of(rank), *offset)
        # my rank
        ranks[get_buffer_2d(0, 0, 0)] = rank

    return ranks, device

def init_buffers_2d(block_vars: dict[str, torch.Tensor]):
    send_bufs = [{}] * 9
    recv_bufs = [{}] * 9
    for x3_offset in [-1, 0, 1]:
        for x2_offset in [-1, 0, 1]:
            if x3_offset == 0 and x2_offset == 0: continue
            offset = (x3_offset, x2_offset, 0)
            bid = get_buffer_2d(*offset)
            part = block.part(offset)
            send_bufs[bid]["hydro_u"] = torch.empty_like(block_vars["hydro_u"][part])
            recv_bufs[bid]["hydro_u"] = torch.empty_like(block_vars["hydro_u"][part])
    return send_bufs, recv_bufs

def serialize_2d(send_bufs: List[dict[str, torch.Tensor]],
                 block_vars: dict[str, torch.Tensor]):
    for x3_offset in [-1, 0, 1]:
        for x2_offset in [-1, 0, 1]:
            if x3_offset == 0 and x2_offset == 0: continue
            offset = (x3_offset, x2_offset, 0)
            bid = get_buffer_2d(*offset)
            part = block.part(offset)
            send_bufs[bid]["hydro_u"].copy_(block_vars["hydro_u"][part])

def deserialize_2d(block_vars: dict[str, torch.Tensor],
                   recv_bufs: List[dict[str, torch.Tensor]]):
    for x3_offset in [-1, 0, 1]:
        for x2_offset in [-1, 0, 1]:
            if x3_offset == 0 and x2_offset == 0: continue
            offset = (x3_offset, x2_offset, 0)
            bid = get_buffer_id(*offset)
            part = block.part(tuple([-x for x in offset]), exterior=True)
            block_vars["hydro_u"][part].copy_(recv_bufs[bid]["hydro_u"])

def slab_exchange(block_vars: dict[str, torch.Tensor],
                  ranks: List[int],
                  send_bufs: List[dict[str, torch.Tensor]],
                  recv_bufs: List[dict[str, torch.Tensor]]):
    ops = []
    serialize_2d(send_bufs, block_vars)
    keys = send_bufs[1].keys()

    for r in range(1, len(ranks)):
        for key in keys:
            ops.append(dist.P2POp(dist.isend, send_bufs[r][key], ranks[r]))
            ops.append(dist.P2POp(dist.irecv, recv_bufs[r][key], ranks[r]))

    if ops:
        reqs = dist.batch_isend_irecv(ops)
        for r in reqs: r.wait()

    deserialize_2d(block_vars, recv_bufs)
