import collections
import json
from pathlib import Path
root=Path('/tmp/snapy-hllc-2f321dc/results')
rows={}
for precision in ('float64', 'float32'):
    trace=json.loads((root/f'smoke-{precision}/trace.json').read_text())
    kernels=[e for e in trace['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X']
    by_name=collections.Counter()
    for e in kernels:
        by_name[e['name']] += e['dur']/1000
    report=json.loads((root/f'smoke-{precision}/report.json').read_text())
    rows[precision]={
        'profile_extra_steps': 5,
        'kernel_count': len(kernels),
        'kernel_duration_sum_ms': sum(by_name.values()),
        'hllc_duration_sum_ms': sum(v for k,v in by_name.items() if 'hllc_cuda_impl' in k),
        'operators': {k:report['profile'][k] for k in ('aten::_to_copy', 'aten::copy_', 'aten::empty_strided', 'cudaLaunchKernel')},
        'top_kernel_ms': by_name.most_common(10),
    }
(root/'kernel-summary.json').write_text(json.dumps(rows, indent=2)+'\n')
print(json.dumps({p:{k:v for k,v in r.items() if k!='top_kernel_ms'} for p,r in rows.items()}, indent=2))
