import json
import statistics
import subprocess
import sys
from pathlib import Path
import numpy as np

base = Path('/tmp/snapy-hllc-2f321dc')
root = base / 'results/cast-cache'
summary = {}
for variant in ('before', 'after'):
    subprocess.run([sys.executable, str(base / 'analyze-straka-cast.py'), str(root / variant)], check=True)
    kernels = json.loads((root / variant / 'kernel-summary.json').read_text())
    summary[variant] = {}
    for precision in ('float64', 'float32'):
        reports = []
        for trial in (1, 2, 3):
            run = root / variant / f'{trial}-graph-{precision}'
            report = json.loads((run / 'report.json').read_text())
            report.pop('profile', None)
            assert report['steps'] == 9000
            assert report['graph_replays'] == 9000
            for t in (0, 300, 600, 900):
                with np.load(run / f'state-{t}.npz') as a, np.load(root / 'before' / f'1-graph-{precision}' / f'state-{t}.npz') as b:
                    assert set(a.files) == set(b.files)
                    for k in a.files:
                        assert np.array_equal(a[k], b[k]), (variant, precision, trial, t, k)
            reports.append(report)
        elapsed = [r['elapsed_seconds_including_diagnostics'] for r in reports]
        k = kernels[precision]
        summary[variant][precision] = {
            'elapsed_seconds': elapsed,
            'median_seconds': statistics.median(elapsed),
            'median_captured_kernel_ms_per_five_steps': k['median_kernel_ms'],
            'median_captured_conversion_ms_per_five_steps': k['median_groups_ms']['dtype-converting copies'],
            'captured_kernel_counts_per_five_steps': [t['kernel_count'] for t in k['trials']],
            'captured_conversion_counts_per_five_steps': [t['groups_calls']['dtype-converting copies'] for t in k['trials']],
            'all_snapshots_bit_identical_to_baseline': True,
            'reports': reports,
        }
for precision in ('float64', 'float32'):
    a = summary['before'][precision]
    b = summary['after'][precision]
    print(precision, 'seconds', a['median_seconds'], b['median_seconds'], 'speedup', a['median_seconds']/b['median_seconds'])
    print('conversions', a['captured_conversion_counts_per_five_steps'], b['captured_conversion_counts_per_five_steps'])
print('after FP32 / FP64', summary['after']['float32']['median_seconds']/summary['after']['float64']['median_seconds'])
(root / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print('PASS: all twelve full runs and all saved fields bit-identical within each precision')
