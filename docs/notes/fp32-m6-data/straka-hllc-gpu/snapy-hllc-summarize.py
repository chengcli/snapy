import hashlib
import json
import statistics
from pathlib import Path
import numpy as np
root = Path('/tmp/snapy-hllc-2f321dc')
results = root / 'results'
summary = {'source_commit': '2f321dc', 'remote_root': str(root), 'gpu': 0,
           'build': 'Release, CUDA 13.1.115, sm_120; UCX/NETCDF/PNETCDF ON; current native libraries and Python bindings rebuilt together',
           'package_version_note': 'snapy version in runner reports comes from installed distribution metadata; source_commit and native hashes identify the isolated tested build',
           'order': ['1-float64', '1-float32', '2-float32', '2-float64', '3-float64', '3-float32'],
           'runs': {}, 'timing': {}, 'repeated_fields_bit_identical': {}}
for precision in ('float64', 'float32'):
    reports = [json.loads((results / f'pair-{i}-{precision}/report.json').read_text()) for i in (1, 2, 3)]
    assert all(r['steps'] == 9000 and r['device'] == 'cuda' and r['history'][-1]['time'] == 900 for r in reports)
    times = [r['elapsed_seconds_including_diagnostics'] for r in reports]
    summary['runs'][precision] = reports
    summary['timing'][precision] = {'seconds': times, 'median_seconds': statistics.median(times), 'min_seconds': min(times), 'max_seconds': max(times), 'sample_stdev_seconds': statistics.stdev(times)}
    same = True
    for t in (0, 300, 600, 900):
        a = np.load(results / f'pair-1-{precision}/state-{t}.npz')
        for i in (2, 3):
            b = np.load(results / f'pair-{i}-{precision}/state-{t}.npz')
            same = same and all(np.array_equal(a[k], b[k]) for k in a.files)
    summary['repeated_fields_bit_identical'][precision] = same
fp64, fp32 = [summary['timing'][p]['median_seconds'] for p in ('float64', 'float32')]
summary['fp32_speedup_vs_fp64'] = fp64 / fp32
summary['fp32_percent_slower_vs_fp64'] = 100 * (fp32 / fp64 - 1)
summary['historical_installed_medians_seconds'] = {'float64': 63.4482360766, 'float32': 70.2005774607}
summary['historical_comparison_note'] = 'Historical installed binaries predate HLLC plus earlier conversion/scratch changes, and the current host has substantial background CPU load; this is not an isolated HLLC speedup measurement or a controlled historical wall-time comparison.'
summary['fp32_percent_faster_than_historical_installed'] = 100 * (1 - fp32 / 70.2005774607)
summary['current_fp64_2x_target_seconds'] = fp64 / 2
summary['initial_2x_target_seconds'] = 31.7241180383
summary['extension_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (root / 'package/snapy').glob('*.so')}
summary['comparisons'] = json.loads((results / 'comparison/comparison.json').read_text())['comparisons']
(results / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k:v for k,v in summary.items() if k != 'runs'}, indent=2))
