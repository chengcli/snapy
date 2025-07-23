#pragma once

namespace snap {

class CommandLine {
 protected:
  CommandLine(int argc, char **argv);

 public:
  char *input_filename = nullptr;
  char *restart_filename = nullptr;
  char *prundir = nullptr;
  int res_flag;
  int narg_flag;
  int iarg_flag;
  int mesh_flag;
  int wtlim;
  int argc;
  int nthreads;
  char **argv;

  static CommandLine *ParseArguments(int argc, char **argv);
  static CommandLine *GetInstance();
  static void Destroy();
  ~CommandLine() {}

 private:
  static CommandLine *mycli_;
};

}  // namespace snap
