//! \file outputs.cpp
//! \brief implements functions for outputs
//!
//! The number and types of outputs are all controlled by the number and values
//! of parameters specified in <output[n]> blocks in the input file.  Each
//! output block must be labelled by a unique integer "n".  Following the
//! convention of the parser implemented in the ParameterInput class, a second
//! output block with the same integer "n" of an earlier block will silently
//! overwrite the values read by the first block. The numbering of the output
//! blocks does not need to be consecutive, and blocks may appear in any order
//! in the input file.  Moreover, unlike the C version of Athena, the total
//! number of <output[n]> blocks does not need to be specified -- in Athena++ a
//! new output type will be created for each and every <output[n]> block in the
//! input file.
//!
//! Required parameters that must be specified in an <output[n]> block are:
//!   - variable     = cons,prim,D,d,E,e,m,m1,m2,m3,v,v1=vx,v2=vy,v3=vz,p,
//!                    bcc,bcc1,bcc2,bcc3,b,b1,b2,b3,phi,uov
//!   - file_type    = rst,tab,vtk,hst,hdf5
//!   - dt           = problem time between outputs
//!
//! EXAMPLE of an <output[n]> block for a VTK dump:
//!
//!     <output3>
//!     file_type   = tab       # Tabular data dump
//!     variable    = prim      # variables to be output
//!     data_format = %12.5e    # Optional data format string
//!     dt          = 0.01      # time increment between outputs
//!     x2_slice    = 0.0       # slice in x2
//!     x3_slice    = 0.0       # slice in x3
//!
//!
//! Each <output[n]> block will result in a new node being created in a linked
//! list of OutputType stored in the Output class.  During a simulation,
//! outputs are made when the simulation time satisfies the criteria implemented
//! in the MakeOutput() function.
//!
//! \note
//! To implement a new output type, write a new OutputType derived class, and
//! construct an object of this class in the Output constructor at the location
//! indicated by the comment text: 'NEW_OUTPUT_TYPES'. Current summary:
//! - outputs.cpp, OutputType:LoadOutputData() (below): conditionally add new
//! OutputData
//!   node to linked list, depending on the user-input 'variable' string.
//!   Provide direction on how to slice a possible 4D source AthenaArray into
//!   separate 3D arrays; automatically enrolls quantity in vtk.cpp,
//!   formatted_table.cpp outputs.
//! - athena_hdf5.cpp, ATHDF5Output::write_output_file(): need to allocate space
//! for the new
//!   OutputData node as an HDF5 "variable" inside an existing HDF5 "dataset"
//!   (cell-centered vs. face-centered data).
//! - restart.cpp, RestartOutput::write_output_file(): memcpy array of quantity
//! to pdata
//!   pointer and increment the pointer. pdata points to an allocated region of
//!   memory whose "datasize" is inferred from MeshBlock::GetBlockSizeInBytes(),
//!   ---->
//! - mesh/meshblock.cpp, MeshBlock::GetBlockSizeInBytes(): increment
//! std::size_t size by
//!   the size of the new quantity's array(s)
//! - mesh/meshblock.cpp, MeshBlock restart constructor: memcpy quantity
//!   (IN THE SAME ORDER AS THE VARIABLES ARE WRITTEN IN restart.cpp)
//!   from the loaded .rst file to the MeshBlock's appropriate physics member
//!   object
//!
//! - history.cpp, HistoryOutput::write_output_file() (3x places):
//!   1) modify NHISTORY_VARS macro
//!      so that the size of data_sum[] can accommodate the new physics, when
//!      active.
//!   2) Compute volume-weighted data_sum[i] for the new quantity + etc. factors
//!   3) Provide short string to serve as the column header description of new
//!   quantity
//!
//! HDF5 note: packing gas velocity into the "prim" HDF5 dataset will cause
//! VisIt to treat the 3x components as independent scalars instead of a
//! physical vector, unlike how it treats .vtk velocity output from Athena++.
//! The workaround is to import the vis/visit/*.xml expressions file, which can
//! pack these HDF5 scalars into a vector.
//
//========================================================================================

// C/C++ headers
#include <cstdio>
#include <cstdlib>
#include <cstring>  // strcmp
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>  // std::string, to_string()

// snap
#include "output.hpp"
#include "output_formats.hpp"

namespace snap {
OutputImpl::OutputImpl(Mesh pm, ParameterInput pin) {
  pfirst_type_ = nullptr;
  std::stringstream msg;
  InputBlock *pib = pin->pfirst_block;
  OutputType *pnew_type;
  OutputType *plast = pfirst_type_;

  // number of history and restart outputs
  int num_hst_outputs = 0, num_rst_outputs = 0;

  // loop over input block names.  Find those that start with "output", read
  // parameters, and construct singly linked list of OutputTypes.
  while (pib != nullptr) {
    if (pib->block_name.compare(0, 6, "output") == 0) {
      OutputOptions op;  // define temporary OutputOptions struct

      // extract integer number of output block.  Save name and number
      std::string outn =
          pib->block_name.substr(6);  // 6 because counting starts at 0!
      op.fid(atoi(outn.c_str()));
      op.block_name().assign(pib->block_name);

      // set time of last output, time between outputs
      op.dt(pin->GetOrAddReal(op.block_name(), "dt", 0.0));
      op.dcycle(pin->GetOrAddInteger(op.block_name(), "dcycle", 0));

      if (op.dt() == 0.0 && op.dcycle() == 0) {
        msg << "### FATAL ERROR in Output constructor" << std::endl
            << "Either dt or dcycle must be specified in " << op.block_name()
            << std::endl;
        throw std::runtime_error(msg.str());
      }
      if (op.dt() > 0.0 && op.dcycle() > 0) {
        msg << "### FATAL ERROR in Output constructor" << std::endl
            << "dt and dcycle cannot be specified simultaneously in "
            << op.block_name() << std::endl;
        throw std::runtime_error(msg.str());
      }
      if (op.dt() > 0.0 ||
          op.dcycle() > 0) {  // only add output if dt > 0 or dycle > 0
        // set file number, basename, id, and format
        op.file_basename(pin->GetString("job", "problem_id"));
        op.file_type(pin->GetString(op.block_name(), "file_type"));

        // read slicing options.  Check that slice is within mesh
        if (pin->DoesParameterExist(op.block_name(), "x1_slice")) {
          auto x1 = pin->GetReal(op.block_name(), "x1_slice");
          if (x1 >= pm->options.x1min() && x1 < pm->options.x1max()) {
            op.x1_slice(x1);
            op.output_slicex1(true);
          } else {
            msg << "### FATAL ERROR in Output constructor" << std::endl
                << "Slice at x1=" << x1 << " in output block '"
                << op.block_name() << "' is out of range of Mesh" << std::endl;
            throw std::runtime_error(msg.str());
          }
        }

        if (pin->DoesParameterExist(op.block_name(), "x2_slice")) {
          auto x2 = pin->GetReal(op.block_name(), "x2_slice");
          if (x2 >= pm->options.x2min() && x2 < pm->options.x2max()) {
            op.x2_slice(x2);
            op.output_slicex2(true);
          } else {
            msg << "### FATAL ERROR in Output constructor" << std::endl
                << "Slice at x2=" << x2 << " in output block '"
                << op.block_name() << "' is out of range of Mesh" << std::endl;
            throw std::runtime_error(msg.str());
          }
        }

        if (pin->DoesParameterExist(op.block_name(), "x3_slice")) {
          auto x3 = pin->GetReal(op.block_name(), "x3_slice");
          if (x3 >= pm->options.x3min() && x3 < pm->options.x3max()) {
            op.x3_slice(x3);
            op.output_slicex3(true);
          } else {
            msg << "### FATAL ERROR in Output constructor" << std::endl
                << "Slice at x3=" << x3 << " in output block '"
                << op.block_name() << "' is out of range of Mesh" << std::endl;
            throw std::runtime_error(msg.str());
          }
        }

        // read sum options.  Check for conflicts with slicing.
        op.output_sumx1(pin->GetOrAddBoolean(op.block_name(), "x1_sum", false));
        if ((op.output_slicex1()) && (op.output_sumx1())) {
          msg << "### FATAL ERROR in Output constructor" << std::endl
              << "Cannot request both slice and sum along x1-direction"
              << " in output block '" << op.block_name() << "'" << std::endl;
          throw std::runtime_error(msg.str());
        }
        op.output_sumx2(pin->GetOrAddBoolean(op.block_name(), "x2_sum", false));
        if ((op.output_slicex2()) && (op.output_sumx2())) {
          msg << "### FATAL ERROR in Output constructor" << std::endl
              << "Cannot request both slice and sum along x2-direction"
              << " in output block '" << op.block_name() << "'" << std::endl;
          throw std::runtime_error(msg.str());
        }
        op.output_sumx3(pin->GetOrAddBoolean(op.block_name(), "x3_sum", false));
        if ((op.output_slicex3()) && (op.output_sumx3())) {
          msg << "### FATAL ERROR in Output constructor" << std::endl
              << "Cannot request both slice and sum along x3-direction"
              << " in output block '" << op.block_name() << "'" << std::endl;
          throw std::runtime_error(msg.str());
        }

        // read ghost cell option
        op.include_ghost_zones(
            pin->GetOrAddBoolean(op.block_name(), "ghost_zones", false));

        // read cartesian mapping option
        op.cartesian_vector(
            pin->GetOrAddBoolean(op.block_name(), "cartesian_vector", false));

        // set output variable and optional data format string used in formatted
        // writes
        if (op.file_type().compare("hst") != 0 &&
            op.file_type().compare("rst") != 0 &&
            op.file_type().compare("dbg") != 0) {
          op.variable(pin->GetString(op.block_name(), "variable"));
        }
        op.data_format(
            pin->GetOrAddString(op.block_name(), "data_format", "%12.5e"));
        // prepend with blank to separate columns
        op.data_format().insert(0, " ");

        // Construct new OutputType according to file format
        // NEW_OUTPUT_TYPES: Add block to construct new types here
        if (op.file_type().compare("hst") == 0) {
          // pnew_type = new HistoryOutput(op);
          num_hst_outputs++;
        } else if (op.file_type().compare("tab") == 0) {
          // pnew_type = new FormattedTableOutput(op);
        } else if (op.file_type().compare("vtk") == 0) {
          // pnew_type = new VTKOutput(op);
        } else if (op.file_type().compare("rst") == 0) {
          // pnew_type = new RestartOutput(op);
          num_rst_outputs++;
        } else if (op.file_type().compare("ath5") == 0 ||
                   op.file_type().compare("hdf5") == 0) {
#ifdef HDF5OUTPUT
          pnew_type = new HDF5Output(op);
#else
          msg << "### FATAL ERROR in Output constructor" << std::endl
              << "Executable not configured for HDF5 outputs, but HDF5 file "
                 "format "
              << "is requested in output block '" << op.block_name() << "'"
              << std::endl;
          throw std::runtime_error(msg.str());
#endif
        } else if (op.file_type().compare("netcdf") == 0) {
#ifdef NETCDFOUTPUT
          pnew_type = new NetcdfOutput(op);
#else
          msg << "### FATAL ERROR in Output constructor" << std::endl
              << "Executable not configured for NETCDF outputs, but NETCDF "
                 "file format "
              << "is requested in output block '" << op.block_name() << "'"
              << std::endl;
          throw std::runtime_error(msg.str());
#endif
        } else if (op.file_type().compare("pnetcdf") == 0) {
#ifdef PNETCDFOUTPUT
          pnew_type = new PnetcdfOutput(op);
#else
          msg << "### FATAL ERROR in Output constructor" << std::endl
              << "Executable not configured for PNETCDF outputs, but PNETCDF "
                 "file format "
              << "is requested in output block '" << op.block_name() << "'"
              << std::endl;
          throw std::runtime_error(msg.str());
#endif
        } else if (op.file_type().compare("fits") == 0) {
#ifdef FITSOUTPUT
          pnew_type = new FITSOutput(op);
#else
          msg << "### FATAL ERROR in Output constructor" << std::endl
              << "Executable not configured for FITS outputs, but FITS file "
                 "format "
              << "is requested in output block '" << op.block_name() << "'"
              << std::endl;
          throw std::runtime_error(msg.str());
#endif
        } else if (op.file_type().compare("dbg") == 0) {
          // pnew_type = new DebugOutput(op);
          // } else if (op.file_type.compare("ptab") == 0) {
          //   pnew_type = new ParticlesTableOutput(op);
        } else {
          msg << "### FATAL ERROR in Output constructor" << std::endl
              << "Unrecognized file format = '" << op.file_type()
              << "' in output block '" << op.block_name() << "'" << std::endl;
          throw std::runtime_error(msg.str());
        }

        // Append type as tail node in singly linked list
        if (pfirst_type_ == nullptr) {
          pfirst_type_ = pnew_type;
        } else {
          plast->pnext_type = pnew_type;
        }
        plast = pnew_type;
      }
    }
    pib = pib->pnext;  // move to next input block name
  }

  // check there were no more than one history or restart files requested
  if (num_hst_outputs > 1 || num_rst_outputs > 1) {
    msg << "### FATAL ERROR in Output constructor" << std::endl
        << "More than one history or restart output block detected in input "
           "file"
        << std::endl;
    throw std::runtime_error(msg.str());
  }

  // Move restarts to the tail end of the OutputType list, so file counters for
  // other output types are up-to-date in restart file
  int pos = 0, found = 0;
  OutputType *pot = pfirst_type_;
  OutputType *prst = pot;
  while (pot != nullptr) {
    if (pot->options.file_type().compare("rst") == 0) {
      prst = pot;
      found = 1;
      if (pot->pnext_type == nullptr) found = 2;
      break;
    }
    pos++;
    pot = pot->pnext_type;
  }
  if (found == 1) {
    // remove the restarting block
    pot = pfirst_type_;
    if (pos == 0) {  // head node/first block
      pfirst_type_ = pfirst_type_->pnext_type;
    } else {
      for (int j = 0; j < pos - 1; j++)  // seek the list
        pot = pot->pnext_type;
      pot->pnext_type = prst->pnext_type;  // remove it
    }
    while (pot->pnext_type != nullptr)
      pot = pot->pnext_type;  // find the tail node
    prst->pnext_type = nullptr;
    pot->pnext_type = prst;
  }
  // if found == 2, do nothing; it's already at the tail node/end of the list
}

// destructor - iterates through singly linked list of OutputTypes and deletes
// nodes

OutputImpl::~OutputImpl() {
  OutputType *ptype = pfirst_type_;
  while (ptype != nullptr) {
    OutputType *ptype_old = ptype;
    ptype = ptype->pnext_type;
    delete ptype_old;
  }
}

void OutputImpl::MakeOutput(Mesh pm, ParameterInput pin, bool wtflag) {
  // wtflag = only true for making final outputs due to signal or
  // wall-time/cycle/time limit. Used by restart file output to change suffix to
  // .final
  bool first = true;
  OutputType *ptype = pfirst_type_;
  while (ptype != nullptr) {
    // output initial conditions, unless next_time set
    if (((pm->current_time == pm->start_time) &&
         (ptype->next_time <= pm->start_time)) ||
        (ptype->options.dt() > 0.0 && pm->current_time >= ptype->next_time) ||
        (ptype->options.dcycle() > 0 &&
         pm->options.ncycle() % ptype->options.dcycle() == 0) ||
        (pm->current_time >= pm->tlim) ||
        (wtflag && ptype->options.file_type() == "rst")) {
      if (first && ptype->options.file_type() != "hst") {
        pm->ApplyUserWorkBeforeOutput();
        first = false;
      }

      for (auto pmb : pm->blocks)
        ptype->write_output_file(pmb, pm->current_time, pm->options.tree(),
                                 wtflag);
      ptype->combine_blocks();

      // increment counters
      ptype->file_number++;
      ptype->next_time += ptype->options.dt();
    }
    // move to next OutputType node in singly linked list
    ptype = ptype->pnext_type;
  }
}
}  // namespace snap
