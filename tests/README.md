# Test suites

Pull requests run the Linux core suite, including Straka, multi-block and slab
halo exchange, and restart regressions. macOS builds the library and Python
extension, then checks EOS, reconstruction, Gloo messaging, and the installed
Python import path.

Pushes to main and **Run workflow** on Continuous Integration run the full
Linux suite and the broader macOS suite. Use a manual run before a release
or when changing distributed behavior. The existing test_shallow_xy_decomp
exclusion remains in place.

For a local build with the matching Python package installed:

~~~sh
cmake -S . -B build -DBUILD_TESTS=ON -DFULL_TESTS=OFF
cmake --build build --parallel 4
ctest --test-dir build/tests --output-on-failure
~~~

Set FULL_TESTS=ON for the additional reference examples and decomposition matrix.
CUDA checks require a CUDA build and a suitable GPU. Set
SNAPY_TEST_PYTHONPATH to the directory containing the matching installed
snapy package when testing outside CI.

Keep a regression in the smallest existing test that exercises the public
behavior. Large uniform grids, print-only demos, and copies of production
algorithms do not belong in the core suite.
