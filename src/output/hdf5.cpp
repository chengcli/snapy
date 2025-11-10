// base
#include <configure.h>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/input/parameter_input.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/vectorize.hpp>

// output
#include "output_formats.hpp"
#include "output_utils.hpp"

// Only proceed if HDF5 output enabled
#ifdef HDF5OUTPUT

// External library headers
#include <hdf5.h>

namespace snap {
HDF5Output::HDF5Output(OutputOptions const &options_) : OutputType(options_) {}

void HDF5Output::write_output_file(MeshBlockImpl *pmb, double current_time,
                                   bool flag) {}
}  // namespace snap

#endif  // HDF5OUTPUT
