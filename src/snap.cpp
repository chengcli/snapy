// base
#include "snap.h"

namespace snap {

// all of these global variables are set at the start of main():
int my_rank = 0;  // MPI rank of this process
int nranks = 1;   // total number of MPI ranks

}  // namespace snap
