// C/C++
#include <cstdio>

// snap
#include <snap/layout/distribute_env.hpp>

using namespace snap;

int main(void) {
  int rank = snap::get_rank();
  std::cout << "rank = " << rank << std::endl;
}
