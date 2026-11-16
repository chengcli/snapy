import json
import statistics
from pathlib import Path
import numpy as np
root=Path('/tmp/snapy-hllc-2f321dc/results/replay-benchmark')
summary={'execution_scope':'fixed dry explicit Straka fixture; identical per-step checks and output schedule',
         'runs':{},'timing':{},'all_fields_bit_identical_to_eager':{}}
for mode in ('eager-float64','cuda-graph-float64','cuda-graph-float32','eager-float32'):
 count=1 if mode=='eager-float32' else 3
 precision=mode.split('-')[-1]
 reports=[json.loads((root/f'{i}-{mode}/report.json').read_text()) for i in range(1,count+1)]
 assert all(r['steps']==9000 for r in reports)
 if mode.startswith('cuda-graph'):
  assert all(r['graph_replays']==9000 for r in reports)
 seconds=[r['elapsed_seconds_including_diagnostics'] for r in reports]
 # Total setup includes initial capture and any recapture already timed.
 # Adding all setup deliberately overestimates total cost, a conservative bound.
 bound=[t+r['graph_setup_seconds'] for t,r in zip(seconds,reports)]
 summary['runs'][mode]=reports
 summary['timing'][mode]={'seconds':seconds,'median_seconds':statistics.median(seconds),
                         'range_seconds':[min(seconds),max(seconds)],
                         'conservative_seconds_including_all_setup':bound,
                         'conservative_median_seconds':statistics.median(bound)}
 same=True
 for i in range(1,count+1):
  for t in (0,300,600,900):
   a=np.load(root/f'{i}-{mode}/state-{t}.npz')
   b=np.load(root/f'1-eager-{precision}/state-{t}.npz')
   same &= all(np.array_equal(a[k],b[k]) for k in a.files)
 summary['all_fields_bit_identical_to_eager'][mode]=bool(same)
 assert same,mode
base=summary['timing']['eager-float64']['median_seconds']
for mode in ('cuda-graph-float64','cuda-graph-float32'):
 summary['timing'][mode]['speedup_vs_eager_fp64']=base/summary['timing'][mode]['median_seconds']
 summary['timing'][mode]['conservative_speedup_vs_eager_fp64']=base/summary['timing'][mode]['conservative_median_seconds']
summary['fp32_vs_fp64_graph_speedup']=summary['timing']['cuda-graph-float64']['median_seconds']/summary['timing']['cuda-graph-float32']['median_seconds']
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='runs'},indent=2))
