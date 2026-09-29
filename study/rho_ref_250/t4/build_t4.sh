#!/bin/bash
# Build t4_driver.cpp with exactly the compile and link lines CMake uses for
# examples/straka.cpp (as ../build_driver.sh does for straka_t3.cpp).
#   bash build_t4.sh <snapy build dir>
set -e
B=$(cd "$1" && pwd); HERE=$(cd "$(dirname "$0")" && pwd)
D=$B/examples/CMakeFiles/straka.release.dir
flags() { sed -n "s/^$1 = //p" $D/flags.make; }
cd $B/examples
CXX=$(sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' $B/CMakeCache.txt)
eval "$CXX $(flags CXX_DEFINES) $(flags CXX_INCLUDES) $(flags CXX_FLAGS) \
  -c $HERE/t4_driver.cpp -o $D/t4_driver.cpp.o"
eval "$(sed -e 's|CMakeFiles/straka.release.dir/straka.cpp.o|CMakeFiles/straka.release.dir/t4_driver.cpp.o|' \
            -e 's|-o ../bin/straka.release|-o ../bin/t4_driver.release|' $D/link.txt)"
echo "built $B/bin/t4_driver.release"
