from pathlib import Path
import sys
root=Path('/tmp/snapy-hllc-2f321dc/src')
sys.path.insert(0,str(root/'tests'))
p=root/'tests/run_fp32_straka.py'
s=p.read_text()
start=s.index('    snapshot(0.0)')
end=s.index('\n\nif __name__')
s=s[:start]+'''    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            for stage in range(len(block.intg.stages)):
                block.forward(state, args.dt, stage)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    print("WARMUP PASSED; CAPTURING", flush=True)
    with torch.cuda.graph(graph, stream=stream):
        for stage in range(len(block.intg.stages)):
            block.forward(state, args.dt, stage)
    print("CAPTURE PASSED", flush=True)
'''+s[end:]
exec(compile(s,str(p),'exec'),{'__name__':'__main__','__file__':str(p)})
