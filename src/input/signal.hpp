#pragma once

// C/C++
#include <csignal>  // sigset_t POSIX C extension
#include <cstdint>  // std::int64_t

namespace snap {

//! \brief Signal handler for system signals
//!
//! Singleton class that manages system signals (SIGTERM, SIGINT, SIGALRM)
//! for graceful shutdown and wall-time limits.
class Signal {
 protected:
  //! \brief Protected constructor (Singleton pattern)
  Signal();

 public:
  //! \brief Signal type enumeration
  enum {
    ITERM = 0,    //!< SIGTERM signal index
    IINT = 1,     //!< SIGINT signal index
    IALRM = 2,    //!< SIGALRM signal index
    NSIGNAL = 3,  //!< Total number of signals
  };

  //! \brief Get singleton instance
  //! \return Pointer to Signal instance
  static Signal* GetInstance();

  //! \brief Destroy singleton instance
  static void Destroy();

  //! \brief Set a signal flag (called by signal handlers)
  //! \param[in] s Signal index to set
  static void SetSignalFlag(int s);

  //! \brief Check all signal flags
  //! \return Signal status code
  int CheckSignalFlags();

  //! \brief Get value of a specific signal flag
  //! \param[in] s Signal index to query
  //! \return Signal flag value
  int GetSignalFlag(int s);

  //! \brief Set wall-time alarm
  //! \param[in] t Time in seconds
  void SetWallTimeAlarm(int t);

  //! \brief Cancel wall-time alarm
  void CancelWallTimeAlarm();

 protected:
  static int signalflag_[NSIGNAL];  //!< Array of signal flags
  sigset_t mask_;                   //!< Signal mask for blocking

 private:
  //! Pointer to the single Signal instance
  static Signal* mysig_;
};

}  // namespace snap
