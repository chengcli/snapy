// C/C++ headers
#include <csignal>

namespace snap {

class SignalHandler {
 protected:
  SignalHandler();  // disable direct instantiation

 public:
  static constexpr int nsignal = 3;
  static constexpr int ITERM = 0, IINT = 1, IALRM = 2;

  static SignalHandler *GetInstance();
  static void Destroy();
  static void SetSignalFlag(int s);

  ~SignalHandler() {}
  int CheckSignalFlags();
  int GetSignalFlag(int s);
  void SetWallTimeAlarm(int t);
  void CancelWallTimeAlarm();

 private:
  static SignalHandler *mysignal_;
  static int signalflag_[nsignal];
  static sigset_t mask_;
};

}  // namespace snap
