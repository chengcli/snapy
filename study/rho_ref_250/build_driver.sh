#!/bin/bash
# Build straka_t3.cpp with exactly the compile and link lines CMake uses for
# examples/straka.cpp, so the study driver never touches the project build.
#   bash build_driver.sh <snapy build dir>
set -e
B=$(cd "$1" && pwd); HERE=$(cd "$(dirname "$0")" && pwd)
D=$B/examples/CMakeFiles/straka.release.dir
flags() { sed -n "s/^$1 = //p" $D/flags.make; }
cd $B/examples
CXX=$(sed -n 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' $B/CMakeCache.txt)
eval "$CXX $(flags CXX_DEFINES) $(flags CXX_INCLUDES) $(flags CXX_FLAGS) \
  -c $HERE/straka_t3.cpp -o $D/straka_t3.cpp.o"
eval "$(sed -e 's|CMakeFiles/straka.release.dir/straka.cpp.o|CMakeFiles/straka.release.dir/straka_t3.cpp.o|' \
            -e 's|-o ../bin/straka.release|-o ../bin/straka_t3.release|' $D/link.txt)"
echo "built $B/bin/straka_t3.release"
