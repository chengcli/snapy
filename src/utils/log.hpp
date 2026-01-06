#pragma once

// C/C++
#include <iostream>
#include <sstream>

// snap
#include <snap/layout/layout.hpp>

namespace snap {

//! \brief Extract filename from a full path
//!
//! \param[in] path Full file path
//! \return Filename without directory path
inline std::string get_filename(std::string path) {
  size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return path;
  } else {
    return path.substr(pos + 1);
  }
}

//! \brief Simple logging message class
//!
//! Provides a lightweight logging mechanism that only outputs on rank 0
//! in distributed environments. Messages are buffered and flushed on
//! destruction.
class LogMessage {
 public:
  //! \brief Constructor
  //! \param[in] msg Message tag/prefix
  LogMessage(std::string msg) : msg_(msg), enabled_(get_rank() == 0) {}

  //! \brief Destructor - flushes buffered message
  ~LogMessage() {
    if (enabled_) {
      Flush();
    }
  }

  //! \brief Get stream for writing log message
  //! \return Reference to output stream
  std::ostream& stream() { return stream_; }

 private:
  //! \brief Flush message to stdout
  void Flush() {
    if (!msg_.empty()) {
      std::cout << "[" << msg_ << "] ";
    }
    std::cout << stream_.str();
  }

  std::string msg_;           //!< Message prefix/tag
  bool enabled_;              //!< Whether logging is enabled for this rank
  std::stringstream stream_;  //!< Stream buffer for message
};

}  // namespace snap

//! \brief Logging macro for info-level messages
//!
//! Usage: SINFO(MyTag) << "Message text" << variable;
//! Only outputs on rank 0 in distributed environments.
#define SINFO(msg) LogMessage(#msg).stream()
