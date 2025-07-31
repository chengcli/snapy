#!/usr/bin/env python3
"""
Convert a sequence of PyTorch .pt dumps into a CF‐compliant NetCDF4 file
with dimensions (time, x, y, z) plus a 'species' axis for mole fractions.
"""

import os
import tarfile
import re
import torch
import numpy as np
from datetime import datetime
from netCDF4 import Dataset

# ──────────────────────────────── CONFIG ────────────────────────────────
INPUT_DIR   = "."
OUTPUT_FILE = "thermo_y_intEng_to_temp.nc"
# ────────────────────────────────────────────────────────────────────────

# find all .pt files, skip size==0, sort by timestamp
pt_files = []
for fn in os.listdir(INPUT_DIR):
    if not fn.endswith(".pt"):
        continue
    full = os.path.join(INPUT_DIR, fn)
    if os.path.getsize(full) == 0:
        continue
    # expect names like thermo_y_intEng_to_temp_<epoch>.pt
    m = re.search(r"(\d+)\.pt$", fn)
    if not m:
        continue
    pt_files.append((int(m.group(1)), full))

pt_files.sort(key=lambda x: x[0])
times_epoch = [ts for ts, _ in pt_files]

# load the first file to infer shapes
module = torch.jit.load(pt_files[0][1])
data = {name: param for name, param in module.named_parameters()}

intEng0         = data["intEng"].numpy()
ivol0           = data["ivol"].numpy()
nx3, nx2, nx1   = intEng0.shape
nspecies        = ivol0.shape[3]
nt              = len(pt_files)

# pre‐allocate arrays in (time, x1, x2, x3) order
intEng_arr      = np.empty((nt, nx1, nx2, nx3), dtype=intEng0.dtype)
ivol_arr        = np.empty((nspecies, nt, nx1, nx2, nx3), dtype=ivol0.dtype)

# load all timesteps
for i, (_, path) in enumerate(pt_files):
    module  = torch.jit.load(path)
    data    = {name: param for name, param in module.named_parameters()}
    e_np     = data["intEng"].numpy()        # (z, y, x)
    v_np     = data["ivol"].numpy()        # (z, y, x)

    # reorder to (x, y, z)
    intEng_arr[i]     = e_np.transpose(2, 1, 0)
    for j in range(nspecies):
        ivol_arr[j, i]  = v_np[:,:,:,j].transpose(2, 1, 0)

# create NetCDF4 file
ds = Dataset(OUTPUT_FILE, "w", format="NETCDF4")

# dimensions
ds.createDimension("time", nt)
ds.createDimension("x3",   nx3)
ds.createDimension("x2",   nx2)
ds.createDimension("x1",   nx1)

# coordinate variables
tvar = ds.createVariable("time", "f4", ("time",))
tvar.units    = "seconds since 1970-01-01 00:00:00 UTC"
tvar.calendar = "gregorian"
tvar[:]       = np.array(times_epoch, dtype="f4")

zvar = ds.createVariable("x1", "f4", ("x1",))
yvar = ds.createVariable("x2", "f4", ("x2",))
xvar = ds.createVariable("x3", "f4", ("x3",))

xvar.axis = "X"
yvar.axis = "Y"
zvar.axis = "Z"

xvar[:] = np.arange(nx3)
yvar[:] = np.arange(nx2)
zvar[:] = np.arange(nx1)

# data variables
intEng_v = ds.createVariable("intEng", "f4", ("time","x1","x2","x3"), zlib=True)
intEng_v.units     = "J/m^3"
intEng_v.long_name = "internal energy"

ivol_v = []
for i in range(nspecies):
    ivol_v.append(ds.createVariable(f"ivol{i}", "f4",
                                  ("time","x1","x2","x3"), zlib=True))
    ivol_v[i].units     = "1"
    ivol_v[i].long_name = "mole fraction of each species"

# write the data
intEng_v[:]  = intEng_arr
for i in range(nspecies):
    ivol_v[i][:] = ivol_arr[i]

# global metadata
ds.title       = "Debug fields for thermo_y.intEng_to_temp"
ds.institution = "University of Michigan"
ds.source      = "converted from .pt files"
ds.history     = f"Created {datetime.utcnow().isoformat()}Z"

ds.close()
print(f"Converted file: {OUTPUT_FILE}")
