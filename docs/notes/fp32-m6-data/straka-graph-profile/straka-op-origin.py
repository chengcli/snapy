"""Diagnostic eager dispatch tracing; never used for wall-time claims."""
from pathlib import Path
import sys
import json
import collections
import torch
from torch.utils._python_dispatch import TorchDispatchMode
root=Path('/tmp/snapy-hllc-2f321dc/src')
sys.path.insert(0,str(root/'tests'))
from run_fp32_straka import make_case

class Origins(TorchDispatchMode):
    def __init__(self):
        self.rows=collections.Counter()
        self.examples={}
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs=kwargs or {}
        name=str(func)
        if any(s in name for s in ('_to_copy', 'copy_', 'aten.add.', 'aten.mul.', 'aten.div.', 'aten.sum.')):
            bt=torch.utils.get_cpp_backtrace()
            frames=[s for s in bt.splitlines() if 'snap::' in s or 'harp::' in s or 'kintera::' in s]
            origin=frames[0].split(' (')[0] if frames else 'outside-native'
            # Strip addresses/frame numbers, retain readable C++ function.
            if ': ' in origin: origin=origin.split(': ',1)[1]
            desc=[]
            for x in args:
                if isinstance(x,torch.Tensor): desc.append((list(x.shape),str(x.dtype)))
            key=json.dumps([name,origin,desc],sort_keys=True)
            self.rows[key]+=1
            self.examples.setdefault(origin,frames[:8])
        return func(*args,**kwargs)

result={}
for precision in ('float64','float32'):
    block,state,cfg,*_=make_case(precision,torch.device('cuda'))
    for stage in range(3): block.forward(state,.1,stage)
    assert block.check_redo(state)==0
    mode=Origins()
    with mode:
        for stage in range(3): block.forward(state,.1,stage)
        assert block.check_redo(state)==0
    result[precision]={'ops':[{'op':json.loads(k)[0],'origin':json.loads(k)[1],'inputs':json.loads(k)[2],'calls':v} for k,v in mode.rows.items()], 'stacks':mode.examples}
    print(precision,len(mode.rows),sum(mode.rows.values()),flush=True)
p=Path('/tmp/snapy-hllc-2f321dc/results/graph-profile/op-origins.json')
p.write_text(json.dumps(result,indent=2)+'\n')
