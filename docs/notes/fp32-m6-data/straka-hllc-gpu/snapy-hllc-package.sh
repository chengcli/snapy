set -eu
export PATH=/home/chengcli/pyenv/bin:/usr/local/cuda-13.1/bin:$PATH
cd /tmp/snapy-hllc-2f321dc/src
MAX_JOBS=8 python setup.py build_ext --build-lib /tmp/snapy-hllc-2f321dc/package --build-temp /tmp/snapy-hllc-2f321dc/extension-build
cp -a python/. /tmp/snapy-hllc-2f321dc/package/snapy/
python - <<'PYLINK'
from pathlib import Path
p=Path('/tmp/snapy-hllc-2f321dc/package/snapy/lib')
if p.is_symlink():
    p.unlink()
p.mkdir(exist_ok=True)
PYLINK
cp build/lib/libsnap*so /tmp/snapy-hllc-2f321dc/package/snapy/lib/
export PYTHONPATH=/tmp/snapy-hllc-2f321dc/package
python - <<'PY'
import snapy, torch
from pathlib import Path
print('LOADED', snapy.__file__, torch.__version__)
print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name())
for line in Path('/proc/self/maps').read_text().splitlines():
    if 'libsnap' in line and 'r-xp' in line:
        print(line)
PY
CUDA_VISIBLE_DEVICES=0 ctest --test-dir build --output-on-failure -R '^test_riemann.release$'
for precision in float64 float32; do
  CUDA_VISIBLE_DEVICES=0 python -u tests/run_fp32_straka.py --precision "$precision" --device cuda --tlim 10 --profile --output "/tmp/snapy-hllc-2f321dc/results/smoke-$precision"
done
