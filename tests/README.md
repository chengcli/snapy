# Test suites

Pull requests and pushes run the Linux core suite: numerical regressions,
short integration cases, communication, and restart checks. macOS builds the
library and Python extension, then checks EOS, reconstruction, Gloo, and the
installed Python import path.

Use **Run workflow** on Continuous Integration before a release or when changing
distributed behavior. It runs the broader macOS suite and enables Linux
reference examples and full decomposition tests. The existing
test_shallow_xy_decomp exclusion remains in place.

For a local build with the matching Python package installed:

~~~sh
cmake -S . -B build -DBUILD_TESTS=ON -DFULL_TESTS=OFF
cmake --build build --parallel 4
ctest --test-dir build/tests --output-on-failure
~~~

Set FULL_TESTS=ON for the reference examples and decomposition matrix.
CUDA checks require a CUDA build and a suitable GPU. Set
SNAPY_TEST_PYTHONPATH to the directory containing the matching installed
snapy package when testing outside CI.

Keep a regression in the smallest existing test that exercises the public
behavior. Large uniform grids, print-only demos, and copies of production
algorithms do not belong in the core suite.
