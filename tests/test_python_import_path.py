import os
import sys
from pathlib import Path

# usage: test_python_import_path.py [SNAPY_TEST_PYTHONPATH [caller PYTHONPATH entry]]
# The *_python ctests must import the snapy under SNAPY_TEST_PYTHONPATH, and keep the
# caller's PYTHONPATH behind it rather than replace it.
if len(sys.argv) < 2 or not sys.argv[1]:
    print("SNAPY_TEST_PYTHONPATH is unset; nothing to check")
    sys.exit(125)

import snapy

root = Path(sys.argv[1]).resolve()
where = Path(snapy.__file__).resolve()
assert root in where.parents, f"snapy imported from {where}, expected under {root}"

entries = os.environ.get("PYTHONPATH", "").split(os.pathsep)
assert Path(entries[0]).resolve() == root, f"PYTHONPATH does not start with {root}: {entries}"
if len(sys.argv) > 2:
    assert sys.argv[2] in entries, f"caller PYTHONPATH entry {sys.argv[2]} was dropped: {entries}"

print("snapy imported from", where)
