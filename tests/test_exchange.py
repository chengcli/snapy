from snapy.exchange import init_dist, slab_exchange

class Args:
    device = 'cpu'
    px1 = 1
    px2 = 2
    px3 = 2
    layout = 'slab'

args = Args()
ranks = init_dist(args, periodic_x1=False, periodic_x2=False, periodic_x3=False)
print("Ranks:", ranks)
