#pragma once

#include <streambuf>
#include <vector>

//! \brief Stream buffer backed by a std::vector
//!
//! Provides a stream buffer implementation that uses a std::vector<char>
//! as the underlying storage. Useful for in-memory serialization operations.
class VectorStream : public std::streambuf {
 public:
  //! \brief Constructor with initial buffer size
  //! \param[in] n Initial size of buffer in bytes
  explicit VectorStream(int n) {
    buffer_.resize(n);
    char* base = buffer_.data();
    setp(base, base + n);
  }

  //! \brief Expand buffer by specified number of bytes
  //! \param[in] n Number of bytes to add to buffer
  void ExpandBuffer(std::streamsize n) {
    auto offset = pptr() - pbase();
    buffer_.resize(buffer_.size() + n);
    setp(buffer_.data(), buffer_.data() + buffer_.size());
    pbump(static_cast<int>(offset));
  }

  //! \brief Get const pointer to buffer data
  //! \return Const pointer to buffer
  char const* buffer() const { return buffer_.data(); }

  //! \brief Get mutable pointer to buffer data
  //! \return Pointer to buffer
  char* buffer() { return buffer_.data(); }

 protected:
  //! \brief Write sequence of characters
  //! \param[in] s Pointer to character sequence
  //! \param[in] n Number of characters to write
  //! \return Number of characters written
  std::streamsize xsputn(const char* s, std::streamsize n) override {
    std::memcpy(pptr(), s, n);
    pbump(static_cast<int>(n));
    return n;
  }

  //! \brief Handle buffer overflow by expanding buffer
  //! \param[in] c Character to write
  //! \return The character written, or EOF on error
  int overflow(int c) override {
    if (c != EOF) {
      ExpandBuffer(1);
      *pptr() = static_cast<char>(c);
      pbump(1);
    }
    return c;
  }

 private:
  std::vector<char> buffer_;  //!< Underlying buffer storage
};
