set -eu
export PATH=/home/chengcli/pyenv/bin:/usr/local/cuda-13.1/bin:$PATH
cd /tmp/snapy-hllc-2f321dc/src
python - <<'PY'
from pathlib import Path
p = Path('CMakeLists.txt')
s = p.read_text()
assert 'set(CMAKE_CUDA_ARCHITECTURES ${_cuda_arch_list})' in s
p.write_text(s.replace('set(CMAKE_CUDA_ARCHITECTURES ${_cuda_arch_list})', 'set(CMAKE_CUDA_ARCHITECTURES 120)'))
# Use existing dependency sources without downloads or changes to them.
for name in ('fmt', 'gtest', 'yaml-cpp', 'eigen'):
    dst = Path('build/_deps') / (name + '-src')
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(Path('/home/chengcli/scix/repos/snapy/build/_deps') / dst.name)
Path('.cache').mkdir(exist_ok=True)
PY
cmake -S . -B build -DCUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/home/chengcli/pyenv/bin/python -DBUILD_TESTS=ON -DSNAPY_TEST_PYTHONPATH=/tmp/snapy-hllc-2f321dc/package -DFETCHCONTENT_SOURCE_DIR_FMT=/home/chengcli/scix/repos/snapy/build/_deps/fmt-src -DFETCHCONTENT_SOURCE_DIR_GTEST=/home/chengcli/scix/repos/snapy/build/_deps/gtest-src -DFETCHCONTENT_SOURCE_DIR_YAML-CPP=/home/chengcli/scix/repos/snapy/build/_deps/yaml-cpp-src -DFETCHCONTENT_SOURCE_DIR_EIGEN=/home/chengcli/scix/repos/snapy/build/_deps/eigen-src
cmake --build build --parallel 12 --target snap_release snap_cuda_release test_riemann.release
