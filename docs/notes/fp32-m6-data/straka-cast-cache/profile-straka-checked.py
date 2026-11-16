"""Probe the exact checked timestep, adding CFL/finite work to profile scope."""
from pathlib import Path
import sys
root = Path('/tmp/snapy-hllc-2f321dc/src')
sys.path.insert(0, str(root / 'tests'))
p = root / 'tests/run_fp32_straka.py'
s = p.read_text()
old = '''            for _ in range(5):
                advance(args.dt)
                assert block.check_redo(state) == 0'''
new = '''            for _ in range(5):
                assert args.dt <= block.max_time_step(state) * (1.0 + 1.0e-10)
                advance(args.dt)
                assert block.check_redo(state) == 0
                assert torch.isfinite(state["hydro_u"]).all()'''
assert old in s
s = s.replace(old, new)
exec(compile(s, str(p), 'exec'), {'__name__': '__main__', '__file__': str(p)})
