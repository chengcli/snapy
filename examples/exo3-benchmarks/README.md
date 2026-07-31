# Exo3 benchmark visualizations

These scripts render zonal-wind spin-up animations from Snapy cubed-sphere
NetCDF output. They transform the native contravariant velocity to spherical
components and regrid all six faces as a single point cloud on the unit sphere.

The two scripts are intentionally kept side by side:

- `plot_hs94.py` renders the Held-Suarez benchmark.
- `plot_hjupiter.py` renders the forced dry hot-Jupiter benchmark.

Set the input and output paths with environment variables:

```shell
HS94_INPUT_PATTERN='/path/to/hs94_prod.out0.{:05d}.nc' \
HS94_OUTPUT=hs94_animation.mp4 python plot_hs94.py

HJUPITER_DATA_DIR=/path/to/hjupiter/output \
HJUPITER_OUTPUT=hjupiter_animation.mp4 python plot_hjupiter.py
```

The scripts require NumPy, xarray, PyTorch, SciPy, Matplotlib, imageio, and an
FFmpeg installation available to imageio.
