set -eu
export PATH=/home/chengcli/pyenv/bin:$PATH
export PYTHONPATH=/tmp/snapy-hllc-2f321dc/package
export CUDA_VISIBLE_DEVICES=0
cd /tmp/snapy-hllc-2f321dc/src
ctest --test-dir build --output-on-failure -R '^(test_fp32_compute|test_fp32_restart|test_perturbation_flux|test_perturbation_flux_split)_python$'
python /tmp/snapy-hllc-2f321dc/probe-straka-graph.py --precision float32 --output /tmp/snapy-hllc-2f321dc/results/cache-graph-final
for precision in float64 float32; do
 python -u tests/run_fp32_straka.py --precision "$precision" --device cuda --tlim 10 --profile --output "/tmp/snapy-hllc-2f321dc/results/cache-profile-$precision"
 python -u tests/run_fp32_straka.py --precision "$precision" --device cuda --output "/tmp/snapy-hllc-2f321dc/results/cache-full-$precision"
done
python - <<'PY'
from pathlib import Path
import numpy as np
import json
root=Path('/tmp/snapy-hllc-2f321dc/results')
summary={}
for p in ('float64','float32'):
 r=json.loads((root/f'cache-full-{p}/report.json').read_text())
 same=True
 for t in (0,300,600,900):
  a=np.load(root/f'cache-full-{p}/state-{t}.npz')
  b=np.load(root/f'pair-1-{p}/state-{t}.npz')
  same &= all(np.array_equal(a[k],b[k]) for k in a.files)
 summary[p]={'report':r,'all_saved_fields_equal_pre_cache':bool(same)}
 print(p, 'unchanged fields',same, 'seconds',r['elapsed_seconds_including_diagnostics'])
(root/'cache-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
assert all(x['all_saved_fields_equal_pre_cache'] for x in summary.values())
PY
