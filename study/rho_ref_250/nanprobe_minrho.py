"""Global interior min density per cycle from SNAP_NANPROBE_STATS output."""
import re
import sys

pat = re.compile(r"NPS (\d+) 9 (\d+) floor_w_interior 0 min=(\S+) "
                 r"at\(j=(\d+),i=(\d+)\)")


def minrho(f):
    out = {}
    for l in open(f):
        m = pat.match(l)
        if m and int(m.group(2)) == 0:
            out[int(m.group(1))] = (float(m.group(3)), int(m.group(4)),
                                    int(m.group(5)))
    return out


if __name__ == "__main__":
    runs = [minrho(f) for f in sys.argv[1:]]
    step = 250
    for c in sorted(runs[0]):
        if c % step == 0 or c >= 7360:
            print(c, "  ".join("%.4e (j=%3d,i=%3d)" % r[c] if c in r else "-"
                               for r in runs))
