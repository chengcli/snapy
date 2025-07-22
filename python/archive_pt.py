#!/usr/bin/env python3

from datetime import datetime
import tarfile
import os
import re

# ──────────────────────────────── CONFIG ────────────────────────────────
INPUT_DIR   = "."

# ──────────────────────── ARCHIVE .pt FILES ─────────────────────────
# if we got here without exception, tar up and remove originals:

# find all .pt files, skip size==0, sort by timestamp
pt_files = []
for fn in os.listdir(INPUT_DIR):
    if not fn.endswith(".pt"):
        continue
    full = os.path.join(INPUT_DIR, fn)
    if os.path.getsize(full) == 0:
        continue
    # expect names like thermo_x_xfrac_to_conc_<epoch>.pt
    m = re.search(r"(\d+)\.pt$", fn)
    if not m:
        continue
    pt_files.append((int(m.group(1)), full))

pt_files.sort(key=lambda x: x[0])

ts_str  = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
tarname = f"xfrac_to_conc_archive_{ts_str}.tar.gz"
with tarfile.open(tarname, "w:gz") as tar:
    for _, path in pt_files:
        tar.add(path, arcname=os.path.basename(path))
        # remove the original .pt
        os.remove(path)

print(f"Archived {len(pt_files)} .pt files into {tarname}")
