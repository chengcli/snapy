// base
#include <configure.h>

#include <input/parameter_input.hpp>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/mesh/mesh.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/util/vectorize.hpp>

// output
#include "output_formats.hpp"
#include "output_utils.hpp"

// Only proceed if HDF5 output enabled
#ifdef HDF5OUTPUT

// External library headers
#include <hdf5.h>

namespace snap {
HDF5Output::HDF5Output(OutputOptions const &options_) : OutputType(options_) {}

void HDF5Output::write_output_file(MeshBlock pmb, float current_time,
                                   OctTreeOptions const &tree, bool flag) {}
}  // namespace snap

#endif  // HDF5OUTPUT
