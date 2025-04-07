// base
#include <configure.h>
#include <input/parameter_input.hpp>

// fvm
#include <fvm/mesh/mesh.hpp>
#include <fvm/mesh/meshblock.hpp>
#include <fvm/coord/coordinate.hpp>
#include <fvm/util/vectorize.hpp>

// output
#include "output_utils.hpp"
#include "output_formats.hpp"

// Only proceed if HDF5 output enabled
#ifdef HDF5OUTPUT

// External library headers
#include <hdf5.h>

namespace canoe {
HDF5Output::HDF5Output(OutputOptions const &options_) : OutputType(options_) {}

void HDF5Output::write_output_file(MeshBlock pmb, float current_time,
                                   OctTreeOptions const &tree, bool flag) {}
}  // namespace canoe

#endif  // HDF5OUTPUT
