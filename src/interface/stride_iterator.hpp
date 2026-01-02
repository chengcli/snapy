#pragma once

// C/C++
#include <cassert>
#include <iterator>

namespace snap {

//! \brief Iterator with custom stride for accessing strided memory
//!
//! A random-access iterator that steps through memory with a custom stride.
//! Useful for accessing elements in multi-dimensional arrays with non-unit
//! strides.
//!
//! \tparam T Iterator type (typically a pointer type)
template <typename T>
class StrideIterator {
 public:
  // public typedefs
  typedef
      typename std::iterator_traits<T>::value_type value_type;  //!< Value type
  typedef typename std::iterator_traits<T>::reference
      reference;  //!< Reference type
  typedef typename std::iterator_traits<T>::difference_type
      difference_type;  //!< Difference type
  typedef typename std::iterator_traits<T>::pointer pointer;  //!< Pointer type
  typedef std::random_access_iterator_tag
      iterator_category;  //!< Iterator category

  // constructors
  //! \brief Default constructor
  StrideIterator() : data(NULL), step(0) {};

  //! \brief Copy constructor
  //! \param[in] x Iterator to copy from
  StrideIterator(const StrideIterator& x) : data(x.data), step(x.step) {}

  //! \brief Constructor with data pointer and stride
  //! \param[in] x Data pointer
  //! \param[in] n Stride value
  StrideIterator(T x, difference_type n) : data(x), step(n) {}

  // operators
  //! \brief Pre-increment operator
  StrideIterator& operator++() {
    data += step;
    return *this;
  }

  //! \brief Post-increment operator
  StrideIterator operator++(int) {
    StrideIterator tmp = *this;
    data += step;
    return tmp;
  }

  //! \brief Addition assignment operator
  //! \param[in] x Number of strides to advance
  StrideIterator& operator+=(difference_type x) {
    data += x * step;
    return *this;
  }

  //! \brief Pre-decrement operator
  StrideIterator& operator--() {
    data -= step;
    return *this;
  }

  //! \brief Post-decrement operator
  StrideIterator operator--(int) {
    StrideIterator tmp = *this;
    data -= step;
    return tmp;
  }

  //! \brief Subtraction assignment operator
  //! \param[in] x Number of strides to retreat
  StrideIterator& operator-=(difference_type x) {
    data -= x * step;
    return *this;
  }

  //! \brief Array subscript operator
  //! \param[in] n Index offset
  //! \return Reference to element at position n*stride from current position
  reference operator[](difference_type n) const { return data[n * step]; }

  //! \brief Dereference operator
  //! \return Reference to current element
  reference operator*() const { return *data; }

  // friend operators
  //! \brief Equality comparison operator
  friend bool operator==(const StrideIterator& x, const StrideIterator& y) {
    assert(x.step == y.step);
    return x.data == y.data;
  }

  //! \brief Inequality comparison operator
  friend bool operator!=(const StrideIterator& x, const StrideIterator& y) {
    assert(x.step == y.step);
    return x.data != y.data;
  }

  //! \brief Less-than comparison operator
  friend bool operator<(const StrideIterator& x, const StrideIterator& y) {
    assert(x.step == y.step);
    return x.data < y.data;
  }

  //! \brief Difference operator (distance between iterators)
  //! \return Number of strides between two iterators
  friend difference_type operator-(const StrideIterator& x,
                                   const StrideIterator& y) {
    assert(x.step == y.step);
    return (x.data - y.data) / x.step;
  }

  //! \brief Addition operator (iterator + offset)
  //! \param[in] x Iterator
  //! \param[in] y Stride offset
  //! \return New iterator advanced by y strides
  friend StrideIterator operator+(const StrideIterator& x, difference_type y) {
    return x += y * x.step;
  }

  //! \brief Addition operator (offset + iterator)
  //! \param[in] x Stride offset
  //! \param[in] y Iterator
  //! \return New iterator advanced by x strides
  friend StrideIterator operator+(difference_type x, const StrideIterator& y) {
    return y += x * y.step;
  }

 private:
  T data;                //!< Pointer to current data element
  difference_type step;  //!< Stride between consecutive elements
};
}  // namespace snap
