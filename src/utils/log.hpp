#pragma once

// C/C++
#include <iostream>
#include <sstream>

namespace snap {

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
    std::cerr << "[" << msg_ << "] ";
    std::cerr << stream_.str() << std::endl;
  }

  std::string msg_;
  bool enabled_;
  std::stringstream stream_;
};

}  // namespace snap

// Macro to mimic glog style
#define LOG(msg) LogMessage(#msg).stream()
