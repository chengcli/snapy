import collections
import json
import statistics
from pathlib import Path
root=Path('/tmp/snapy-hllc-2f321dc/results/graph-profile')
def category(name):
    if 'hllc_cuda_impl' in name: return 'HLLC'
    if 'Weno5Op' in name: return 'WENO5'
    if 'hydro_ref_x1_cuda_kernel' in name: return 'hydrostatic x1 reference scan'
    if 'call_interp' in name: return 'other reconstruction'
    if 'direct_copy_kernel_cuda' in name:
        return 'dtype-converting copies' if 'LoadWithCast' in name or 'gpu_kernel_impl<' in name else 'same-dtype copies'
    if name.startswith('memcpy'): return 'graph memcpy kernels'
    if 'CUDAFunctor_add' in name or any(s in name for s in ('MulFunctor','DivFunctor','SubFunctor')):
        if 'LoadWithCast' in name or 'gpu_kernel_impl<' in name: return 'arithmetic with runtime casts'
        if 'CUDAFunctor_add' in name: return 'addition'
        if 'MulFunctor' in name: return 'multiplication'
        if 'DivFunctor' in name: return 'division'
        return 'subtraction'
    if 'FillFunctor' in name: return 'fills/zeroing'
    if 'reduce_kernel' in name: return 'reductions'
    if 'flip' in name: return 'boundary flips'
    return 'other kernels'
summary={}
for precision in ('float64','float32'):
    trials=[]
    for trial in (1,2,3):
        events=json.loads((root/f'{trial}-graph-{precision}/trace.json').read_text())['traceEvents']
        launches={e['args']['correlation'] for e in events if e.get('name')=='cudaGraphLaunch' and e.get('cat')=='cuda_runtime'}
        assert len(launches)==5
        kernels=[e for e in events if e.get('cat')=='kernel' and e.get('ph')=='X']
        by_name=collections.Counter(); grouped=collections.Counter(); counts=collections.Counter()
        regions=collections.Counter(); regioncounts=collections.Counter()
        grouped_graph=collections.Counter()
        for e in kernels:
            ms=e['dur']/1000; cat=category(e['name'])
            by_name[e['name']]+=ms; grouped[cat]+=ms; counts[cat]+=1
            region='captured timestep' if e['args'].get('correlation') in launches else 'outside graph: checks and input copies'
            regions[region]+=ms; regioncounts[region]+=1
            if region=='captured timestep': grouped_graph[cat]+=ms
        trials.append({'kernel_ms':sum(by_name.values()),'kernel_count':len(kernels),
                       'groups_ms':dict(grouped),'groups_calls':dict(counts),
                       'graph_groups_ms':dict(grouped_graph),'regions_ms':dict(regions),'regions_calls':dict(regioncounts),
                       'top_kernels_ms':by_name.most_common(20)})
    cats=set().union(*(t['groups_ms'] for t in trials))
    summary[precision]={'trials':trials,'median_kernel_ms':statistics.median(t['kernel_ms'] for t in trials),
        'median_groups_ms':{k:statistics.median(t['groups_ms'].get(k,0) for t in trials) for k in cats},
        'median_regions_ms':{k:statistics.median(t['regions_ms'][k] for t in trials) for k in trials[0]['regions_ms']}}
a,b=[summary[p]['median_groups_ms'] for p in ('float64','float32')]
summary['ranked_deltas_ms']=[{'group':k,'float64_ms':a.get(k,0),'float32_ms':b.get(k,0),'fp32_minus_fp64_ms':b.get(k,0)-a.get(k,0)} for k in sorted(set(a)|set(b),key=lambda k:b.get(k,0)-a.get(k,0),reverse=True)]
(root/'kernel-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for p in ('float64','float32'):
 print(p,summary[p]['median_kernel_ms'],summary[p]['median_regions_ms'],[t['kernel_count'] for t in summary[p]['trials']],summary[p]['trials'][0]['regions_calls'])
for row in summary['ranked_deltas_ms']: print(row)
