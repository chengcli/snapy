"""Read SNAP_NANPROBE output (scratch/250-cuda-nan, diagnostic only)."""
import sys

R = 3
W = 2 * R + 1


def load(f):
    d, hdr = {}, {}
    for l in open(f):
        p = l.split()
        if l.startswith('NPW '):
            key = (int(p[1]), int(p[2]), int(p[3]), p[4], int(p[5]))
            d[key] = [float(x) if x != 'x' else None for x in p[6:]]
        elif l.startswith('NP '):
            hdr[(int(p[1]), int(p[2]), int(p[3]), p[4])] = p[5:]
    return d, hdr


def at(v, dj, di):
    return v[(dj + R) * W + (di + R)]
