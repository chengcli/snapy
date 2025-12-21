#pragma once

// C/C++
#include <iostream>
#include <sstream>

// snap
#include <snap/layout/layout.hpp>

namespace snap {

//! Get filename from path
std::string get_filename(std::string path) {
  size_t pos = path.find_last_of("/\\");
  if (pos == std::string::npos) {
    return path;
  } else {
    return path.substr(pos + 1);
  }
}

class LogMessage {
 public:
  LogMessage(std::string msg) : msg_(msg), enabled_(get_rank() == 0) {}

  ~LogMessage() {
    if (enabled_) {
      Flush();
    }
  }

  std::ostream& stream() { return stream_; }

 private:
  void Flush() {
    if (!msg_.empty()) {
      std::cerr << "[" << msg_ << "] ";
    }
    std::cerr << stream_.str() << std::endl;
  }

  std::string msg_;
  bool enabled_;
  std::stringstream stream_;
};

}  // namespace snap

// Macro to mimic glog style
#define SINFO(msg) LogMessage(#msg).stream()
