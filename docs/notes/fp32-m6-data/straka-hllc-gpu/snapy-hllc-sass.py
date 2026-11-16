import collections
import json
import re
import subprocess
from pathlib import Path
root = Path('/tmp/snapy-hllc-2f321dc')
lib = root / 'package/snapy/lib/libsnap_cuda_release.so'
rows = []
current = None
with (root / 'results/snapy-sm120.sass').open('w') as output:
    subprocess.run(['/usr/local/cuda-13.1/bin/cuobjdump', '--dump-sass', '--gpu-architecture', 'sm_120', str(lib)], stdout=output, check=True)
for line in (root / 'results/snapy-sm120.sass').open():
    if 'Function :' in line:
        symbol = line.split('Function :', 1)[1].strip()
        current = None
        if 'hllc_cuda_impl' in symbol:
            current = {'symbol': symbol, 'demangled': subprocess.check_output(['c++filt', symbol], text=True).strip(), 'fp_ops': collections.Counter()}
            rows.append(current)
    elif current is not None:
        m = re.match(r'\s*/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]+)\s', line)
        if m and (m[1].startswith(('FADD', 'FMUL', 'FFMA', 'F2F', 'DADD', 'DMUL', 'DFMA', 'DSETP'))):
            current['fp_ops'][m[1]] += 1
(root / 'results/hllc-sass-summary.json').write_text(json.dumps(rows, indent=2) + '\n')
for row in rows:
    print(row['demangled'], dict(row['fp_ops']))
assert rows
float_rows = [r for r in rows if '{lambda()#2}' in r['demangled']]
assert float_rows
for row in float_rows:
    assert not any(op.startswith(('DADD', 'DMUL', 'DFMA', 'DSETP', 'F2F.F64.F32', 'F2F.F32.F64')) for op in row['fp_ops']), row
print('PASS: no FP64 arithmetic/conversions in every float HLLC specialization')
