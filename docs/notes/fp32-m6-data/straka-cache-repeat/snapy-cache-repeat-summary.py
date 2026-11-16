import json
import statistics
from pathlib import Path
import numpy as np
root=Path('/tmp/snapy-hllc-2f321dc/results/cache-repeat')
summary={'local_source_revision':'8535ce3','execution':'eager','device':'cuda','gpu':0,
         'provenance':'Reused cache/check-validated native libraries without rebuilding; hashes identify tested binaries.',
         'timing_includes':'9000 timesteps, CFL/retry/finite checks and snapshots; excludes initialization',
         'runs':{},'timing':{},'repeat_fields_equal':{},'pre_cache_fields_equal':{}}
for p in ('float64','float32'):
 reports=[json.loads((root/f'pair-{i}-{p}/report.json').read_text()) for i in (1,2,3)]
 assert all(r['steps']==9000 and r['device']=='cuda' for r in reports)
 summary['runs'][p]=reports
 times=[r['elapsed_seconds_including_diagnostics'] for r in reports]
 summary['timing'][p]={'seconds':times,'median_seconds':statistics.median(times),
                       'range_seconds':[min(times),max(times)],'sample_stdev_seconds':statistics.stdev(times)}
 same=old=True
 for t in (0,300,600,900):
  a=np.load(root/f'pair-1-{p}/state-{t}.npz')
  b=np.load(root.parent/f'pair-1-{p}/state-{t}.npz')
  old &= all(np.array_equal(a[k],b[k]) for k in a.files)
  for i in (2,3):
   b=np.load(root/f'pair-{i}-{p}/state-{t}.npz')
   same &= all(np.array_equal(a[k],b[k]) for k in a.files)
 summary['repeat_fields_equal'][p]=bool(same)
 summary['pre_cache_fields_equal'][p]=bool(old)
 assert same and old
med64,med32=[summary['timing'][p]['median_seconds'] for p in ('float64','float32')]
summary['fp32_speedup_vs_fp64']=med64/med32
summary['fp32_percent_slower']=100*(med32/med64-1)
summary['comparisons']=json.loads((root/'comparison/comparison.json').read_text())['comparisons']
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='runs'},indent=2))
