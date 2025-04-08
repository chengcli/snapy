#pragma once

//! \file io_wrapper.hpp
//! \brief defines a set of small wrapper functions for MPI versus Serial
//! Output.

// C/C++
#include <cstdint>
#include <cstdio>

// base
#include <configure.h>

#ifdef MPI_PARALLEL
#include <mpi.h>
using IOWrapperFile = MPI_File;
#else
using IOWrapperFile = FILE *;
#endif

using IOWrapperSizeT = uint64_t;

namespace snap {
class IOWrapper {
 public:
#ifdef MPI_PARALLEL
  IOWrapper() : fh_(nullptr), comm_(MPI_COMM_WORLD) {}
  void SetCommunicator(MPI_Comm scomm) { comm_ = scomm; }
#else
  IOWrapper() { fh_ = nullptr; }
#endif
  ~IOWrapper() {}
  // nested type definition of strongly typed/scoped enum in class definition
  enum class FileMode { read, write };

  // wrapper functions for basic I/O tasks
  int Open(const char *fname, FileMode rw);
  std::size_t Read(void *buf, IOWrapperSizeT size, IOWrapperSizeT count);
  std::size_t Read_all(void *buf, IOWrapperSizeT size, IOWrapperSizeT count);
  std::size_t Read_at(void *buf, IOWrapperSizeT size, IOWrapperSizeT count,
                      IOWrapperSizeT offset);
  std::size_t Read_at_all(void *buf, IOWrapperSizeT size, IOWrapperSizeT count,
                          IOWrapperSizeT offset);
  std::size_t Write(const void *buf, IOWrapperSizeT size, IOWrapperSizeT count);
  std::size_t Write_at(const void *buf, IOWrapperSizeT size,
                       IOWrapperSizeT count, IOWrapperSizeT offset);
  std::size_t Write_at_all(const void *buf, IOWrapperSizeT size,
                           IOWrapperSizeT count, IOWrapperSizeT offset);
  int Close();
  int Seek(IOWrapperSizeT offset);
  IOWrapperSizeT GetPosition();

 private:
  IOWrapperFile fh_;
#ifdef MPI_PARALLEL
  MPI_Comm comm_;
#endif
};
}  // namespace snap
