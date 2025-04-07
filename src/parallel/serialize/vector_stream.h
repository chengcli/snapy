#pragma once

#include <streambuf>
#include <vector>

class VectorStream : public std::streambuf {
 public:
  explicit VectorStream(int n) {
    buffer_.resize(n);
    char* base = buffer_.data();
    setp(base, base + n);
  }

  void ExpandBuffer(std::streamsize n) {
    auto offset = pptr() - pbase();
    buffer_.resize(buffer_.size() + n);
    setp(buffer_.data(), buffer_.data() + buffer_.size());
    pbump(static_cast<int>(offset));
  }

  char const* buffer() const { return buffer_.data(); }

  char* buffer() { return buffer_.data(); }

 protected:
  std::streamsize xsputn(const char* s, std::streamsize n) override {
    std::memcpy(pptr(), s, n);
    pbump(static_cast<int>(n));
    return n;
  }

  int overflow(int c) override {
    if (c != EOF) {
      ExpandBuffer(buffer_, 1);
      *pptr() = static_cast<char>(c);
      pbump(1);
    }
    return c;
  }

 private:
  std::vector<char> buffer_;
};
