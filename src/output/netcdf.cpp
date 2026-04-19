// C/C++
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <exception>
#include <filesystem>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// base
#include <configure.h>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/vectorize.hpp>

#include "output_formats.hpp"
#include "output_utils.hpp"

// Only proceed if NETCDF output enabled
#ifdef NETCDFOUTPUT

// External library headers
#include <netcdf.h>

#endif  // NETCDFOUTPUT

namespace snap {
namespace {

struct NetcdfVariableSnapshot {
  std::string grid;
  std::vector<std::string> varnames;
  AthenaArray<double> data;
};

struct NetcdfWriteTask {
  Layout layout;
  std::string output_dir;
  std::string basename;
  std::string file_id;
  int file_number = 0;
  int rank = 0;
  double current_time = 0.0;
  int ncells1 = 0;
  int ncells2 = 0;
  int ncells3 = 0;
  int nfaces1 = 0;
  int nfaces2 = 0;
  int nfaces3 = 0;
  std::vector<float> x1;
  std::vector<float> x1f;
  std::vector<float> x2;
  std::vector<float> x2f;
  std::vector<float> x3;
  std::vector<float> x3f;
  std::vector<NetcdfVariableSnapshot> variables;
  bool combine = false;
};

struct NetcdfCombineTask {
  Layout layout;
  std::string output_dir;
  std::string basename;
  std::string file_id;
  int file_number = 0;
};

std::string make_netcdf_filename(std::string const& output_dir,
                                 std::string const& basename, int rank,
                                 std::string const& file_id, int file_number) {
  char number[6];
  snprintf(number, sizeof(number), "%05d", file_number);
  char blockid[12];
  snprintf(blockid, sizeof(blockid), "block%d", rank);

  std::string fname = output_dir + "/" + basename + "." + blockid + "." +
                      file_id + "." + number + ".nc";
  return fname;
}

#ifdef NETCDFOUTPUT
void check_nc(int status, char const* action,
              std::string const& path = std::string()) {
  if (status == NC_NOERR) return;
  std::ostringstream os;
  os << "NetCDF " << action;
  if (!path.empty()) {
    os << " '" << path << "'";
  }
  os << " failed: " << nc_strerror(status);
  throw std::runtime_error(os.str());
}
#endif

std::vector<std::string> expand_varnames(OutputData const& data,
                                         MetadataTable const* meta) {
  auto names = Vectorize<std::string>(data.name.c_str(), ";");
  std::string grid = meta->GetGridType(names[0]);
  int nvar = get_num_variables(grid, data.data);

  std::vector<std::string> varnames;
  if ((int)names.size() >= nvar) {
    for (int n = 0; n < nvar; ++n) {
      varnames.push_back(names[n]);
    }
    return varnames;
  }

  size_t pos = data.name.find('?');
  for (int n = 0; n < nvar; ++n) {
    if (nvar == 1) {
      if (pos < data.name.length()) {
        varnames.push_back(data.name.substr(0, pos) +
                           data.name.substr(pos + 1));
      } else {
        varnames.push_back(data.name);
      }
    } else {
      char c[16];
      snprintf(c, sizeof(c), "%d", n + 1);
      if (pos < data.name.length()) {
        varnames.push_back(data.name.substr(0, pos) + c +
                           data.name.substr(pos + 1));
      } else {
        varnames.push_back(data.name + c);
      }
    }
  }

  return varnames;
}

#ifdef NETCDFOUTPUT
void write_netcdf_task(NetcdfWriteTask const& task) {
  auto pmeta = MetadataTable::GetInstance();
  auto fname = make_netcdf_filename(task.output_dir, task.basename, task.rank,
                                    task.file_id, task.file_number);

  ensure_output_directory(task.output_dir);
  configure_hdf5_file_locking_for_netcdf();

  int ifile = -1;
  try {
    check_nc(nc_create(fname.c_str(), NC_NETCDF4, &ifile), "create", fname);

    int idt, idx1, idx2, idx3, idx1f = -1, idx2f = -1, idx3f = -1;
    nc_def_dim(ifile, "time", NC_UNLIMITED, &idt);
    nc_def_dim(ifile, "x1", task.ncells1, &idx1);
    if (task.ncells1 > 1) nc_def_dim(ifile, "x1f", task.nfaces1, &idx1f);
    nc_def_dim(ifile, "x2", task.ncells2, &idx2);
    if (task.ncells2 > 1) nc_def_dim(ifile, "x2f", task.nfaces2, &idx2f);
    nc_def_dim(ifile, "x3", task.ncells3, &idx3);
    if (task.ncells3 > 1) nc_def_dim(ifile, "x3f", task.nfaces3, &idx3f);

    int ivt, ivx1, ivx2, ivx3, ivx1f = -1, ivx2f = -1, ivx3f = -1;
    auto [lx2_in, lx3_in, lx1_in] = task.layout->loc_of(task.rank);
    int nb1 = task.layout->options->pz();
    int nb2 = task.layout->options->px();
    int nb3 = task.layout->options->py();
    int face = 0;
    int lx1 = lx1_in;
    int lx2 = lx2_in;
    int lx3 = lx3_in;
    if (task.layout->options->type() == "cubed-sphere") {
      face = lx1;
      lx1 = 0;
      lx2 += (face % 3) * nb2;
      lx3 += (face / 3) * nb3;
      nb2 *= 3;
      nb3 *= 2;
    }

    int pos[4];
    nc_def_var(ifile, "time", NC_FLOAT, 1, &idt, &ivt);
    nc_put_att_text(ifile, ivt, "axis", 1, "T");
    nc_put_att_text(ifile, ivt, "units", 1, "s");
    nc_put_att_text(ifile, ivt, "long_name", 4, "time");

    nc_def_var(ifile, "x1", NC_FLOAT, 1, &idx1, &ivx1);
    nc_put_att_text(ifile, ivx1, "axis", 1, "Z");
    nc_put_att_text(ifile, ivx1, "units", 1, "m");
    nc_put_att_text(ifile, ivx1, "long_name", 27,
                    "Z-coordinate at cell center");
    pos[0] = 1;
    pos[1] = task.ncells1 * nb1;
    pos[2] = task.ncells1 * lx1 + 1;
    pos[3] = task.ncells1 * (lx1 + 1);
    nc_put_att_int(ifile, ivx1, "domain_decomposition", NC_INT, 4, pos);

    if (task.ncells1 > 1) {
      nc_def_var(ifile, "x1f", NC_FLOAT, 1, &idx1f, &ivx1f);
      nc_put_att_text(ifile, ivx1f, "units", 1, "m");
      nc_put_att_text(ifile, ivx1f, "long_name", 25,
                      "Z-coordinate at cell face");
      pos[0]--;
      pos[2]--;
      nc_put_att_int(ifile, ivx1f, "domain_decomposition", NC_INT, 4, pos);
    }

    nc_def_var(ifile, "x2", NC_FLOAT, 1, &idx2, &ivx2);
    nc_put_att_text(ifile, ivx2, "axis", 1, "X");
    nc_put_att_text(ifile, ivx2, "units", 1, "m");
    nc_put_att_text(ifile, ivx2, "long_name", 27,
                    "X-coordinate at cell center");
    pos[0] = 1;
    pos[1] = task.ncells2 * nb2;
    pos[2] = task.ncells2 * lx2 + 1;
    pos[3] = task.ncells2 * (lx2 + 1);
    nc_put_att_int(ifile, ivx2, "domain_decomposition", NC_INT, 4, pos);

    if (task.ncells2 > 1) {
      nc_def_var(ifile, "x2f", NC_FLOAT, 1, &idx2f, &ivx2f);
      nc_put_att_text(ifile, ivx2f, "units", 1, "m");
      nc_put_att_text(ifile, ivx2f, "long_name", 25,
                      "Y-coordinate at cell face");
      pos[0]--;
      pos[2]--;
      nc_put_att_int(ifile, ivx2f, "domain_decomposition", NC_INT, 4, pos);
    }

    nc_def_var(ifile, "x3", NC_FLOAT, 1, &idx3, &ivx3);
    nc_put_att_text(ifile, ivx3, "axis", 1, "Y");
    nc_put_att_text(ifile, ivx3, "units", 1, "m");
    nc_put_att_text(ifile, ivx3, "long_name", 27,
                    "Y-coordinate at cell center");
    pos[0] = 1;
    pos[1] = task.ncells3 * nb3;
    pos[2] = task.ncells3 * lx3 + 1;
    pos[3] = task.ncells3 * (lx3 + 1);
    nc_put_att_int(ifile, ivx3, "domain_decomposition", NC_INT, 4, pos);

    if (task.ncells3 > 1) {
      nc_def_var(ifile, "x3f", NC_FLOAT, 1, &idx3f, &ivx3f);
      nc_put_att_text(ifile, ivx3f, "units", 1, "m");
      nc_put_att_text(ifile, ivx3f, "long_name", 25,
                      "X-coordinate at cell face");
      pos[0]--;
      pos[2]--;
      nc_put_att_int(ifile, ivx3f, "domain_decomposition", NC_INT, 4, pos);
    }

    int nbtotal = nb1 * nb2 * nb3;
    nc_put_att_int(ifile, NC_GLOBAL, "NumFilesInSet", NC_INT, 1, &nbtotal);

    int total_vars = 0;
    for (auto const& var : task.variables) {
      total_vars += var.varnames.size();
    }

    int iaxis[4] = {idt, idx1, idx3, idx2};
    int iaxis1[4] = {idt, idx1f, idx3, idx2};
    int iaxis2[4] = {idt, idx1, idx3, idx2f};
    int iaxis3[4] = {idt, idx1, idx3f, idx2};
    int iaxis_23[3] = {idt, idx3, idx2};
    std::unique_ptr<int[]> var_ids = std::make_unique<int[]>(total_vars);
    int* ivar = var_ids.get();

    for (auto const& var : task.variables) {
      for (auto const& name : var.varnames) {
        if (var.grid == "CCF") {
          nc_def_var(ifile, name.c_str(), NC_FLOAT, 4, iaxis1, ivar);
        } else if ((var.grid == "CFC") && (task.ncells2 > 1)) {
          nc_def_var(ifile, name.c_str(), NC_FLOAT, 4, iaxis2, ivar);
        } else if ((var.grid == "FCC") && (task.ncells3 > 1)) {
          nc_def_var(ifile, name.c_str(), NC_FLOAT, 4, iaxis3, ivar);
        } else if (var.grid == "--C") {
          nc_def_var(ifile, name.c_str(), NC_FLOAT, 2, iaxis, ivar);
        } else if (var.grid == "-CC") {
          nc_def_var(ifile, name.c_str(), NC_FLOAT, 3, iaxis_23, ivar);
        } else if (var.grid == "--F") {
          nc_def_var(ifile, name.c_str(), NC_FLOAT, 2, iaxis1, ivar);
        } else if (var.grid == "---") {
          nc_def_var(ifile, name.c_str(), NC_FLOAT, 1, iaxis, ivar);
        } else {
          nc_def_var(ifile, name.c_str(), NC_FLOAT, 4, iaxis, ivar);
        }

        auto attr = pmeta->GetUnits(name);
        if (!attr.empty()) {
          nc_put_att_text(ifile, *ivar, "units", attr.length(), attr.c_str());
        }

        attr = pmeta->GetLongName(name);
        if (!attr.empty()) {
          nc_put_att_text(ifile, *ivar, "long_name", attr.length(),
                          attr.c_str());
        }
        ivar++;
      }
    }

    nc_enddef(ifile);

    std::unique_ptr<float[]> data =
        std::make_unique<float[]>(task.nfaces1 * task.nfaces3 * task.nfaces2);
    size_t start[4] = {0, 0, 0, 0};
    size_t count[4] = {1, (size_t)task.ncells1, (size_t)task.ncells3,
                       (size_t)task.ncells2};
    size_t count1[4] = {1, (size_t)task.nfaces1, (size_t)task.ncells3,
                        (size_t)task.ncells2};
    size_t count2[4] = {1, (size_t)task.ncells1, (size_t)task.nfaces3,
                        (size_t)task.ncells2};
    size_t count3[4] = {1, (size_t)task.ncells1, (size_t)task.ncells3,
                        (size_t)task.nfaces2};
    size_t count_23[3] = {1, (size_t)task.ncells3, (size_t)task.ncells2};

    float timef = task.current_time;
    nc_put_vara_float(ifile, ivt, start, count, &timef);
    nc_put_var_float(ifile, ivx1, task.x1.data());
    if (task.ncells1 > 1) nc_put_var_float(ifile, ivx1f, task.x1f.data());
    nc_put_var_float(ifile, ivx2, task.x2.data());
    if (task.ncells2 > 1) nc_put_var_float(ifile, ivx2f, task.x2f.data());
    nc_put_var_float(ifile, ivx3, task.x3.data());
    if (task.ncells3 > 1) nc_put_var_float(ifile, ivx3f, task.x3f.data());

    ivar = var_ids.get();
    for (auto const& var : task.variables) {
      int nvar = var.varnames.size();
      if (var.grid == "CCF") {
        for (int n = 0; n < nvar; ++n) {
          float* it = data.get();
          for (int i = 0; i < task.nfaces1; ++i)
            for (int k = 0; k < task.ncells3; ++k)
              for (int j = 0; j < task.ncells2; ++j)
                *it++ = var.data(n, k, j, i);
          nc_put_vara_float(ifile, *ivar++, start, count1, data.get());
        }
      } else if ((var.grid == "CFC") && (task.ncells2 > 1)) {
        for (int n = 0; n < nvar; ++n) {
          float* it = data.get();
          for (int i = 0; i < task.ncells1; ++i)
            for (int k = 0; k < task.ncells3; ++k)
              for (int j = 0; j < task.nfaces2; ++j)
                *it++ = var.data(n, k, j, i);
          nc_put_vara_float(ifile, *ivar++, start, count2, data.get());
        }
      } else if ((var.grid == "FCC") && (task.ncells3 > 1)) {
        for (int n = 0; n < nvar; ++n) {
          float* it = data.get();
          for (int i = 0; i < task.ncells1; ++i)
            for (int k = 0; k < task.nfaces3; ++k)
              for (int j = 0; j < task.ncells2; ++j)
                *it++ = var.data(n, k, j, i);
          nc_put_vara_float(ifile, *ivar++, start, count3, data.get());
        }
      } else if (var.grid == "--C") {
        for (int n = 0; n < nvar; ++n) {
          float* it = data.get();
          for (int i = 0; i < task.ncells1; ++i) *it++ = var.data(n, i);
          nc_put_vara_float(ifile, *ivar++, start, count, data.get());
        }
      } else if (var.grid == "-CC") {
        for (int n = 0; n < nvar; ++n) {
          float* it = data.get();
          for (int k = 0; k < task.ncells3; ++k)
            for (int j = 0; j < task.ncells2; ++j) *it++ = var.data(n, k, j);
          nc_put_vara_float(ifile, *ivar++, start, count_23, data.get());
        }
      } else if (var.grid == "--F") {
        for (int n = 0; n < nvar; ++n) {
          float* it = data.get();
          for (int i = 0; i < task.nfaces1; ++i) *it++ = var.data(n, i);
          nc_put_vara_float(ifile, *ivar++, start, count1, data.get());
        }
      } else if (var.grid == "---") {
        for (int n = 0; n < nvar; ++n) {
          data[0] = var.data(n);
          nc_put_vara_float(ifile, *ivar++, start, count, data.get());
        }
      } else {
        for (int n = 0; n < nvar; ++n) {
          float* it = data.get();
          for (int i = 0; i < task.ncells1; ++i)
            for (int k = 0; k < task.ncells3; ++k)
              for (int j = 0; j < task.ncells2; ++j)
                *it++ = var.data(n, k, j, i);
          nc_put_vara_float(ifile, *ivar++, start, count, data.get());
        }
      }
    }

    check_nc(nc_sync(ifile), "sync", fname);
    check_nc(nc_close(ifile), "close", fname);
    ifile = -1;
  } catch (...) {
    if (ifile >= 0) {
      nc_close(ifile);
    }
    std::error_code ec;
    std::filesystem::remove(fname, ec);
    throw;
  }
}
#endif

}  // namespace

struct NetcdfOutputImpl {
  std::mutex mutex;
  std::condition_variable cv;
  std::condition_variable idle_cv;
  std::deque<NetcdfWriteTask> pending_writes;
  std::deque<NetcdfCombineTask> completed_writes;
  std::exception_ptr async_error;
  bool stop = false;
  bool active = false;
  std::thread worker;

  NetcdfOutputImpl() : worker([this]() { run(); }) {}

  ~NetcdfOutputImpl() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      stop = true;
    }
    cv.notify_all();
    if (worker.joinable()) {
      worker.join();
    }
  }

  void run() {
    while (true) {
      NetcdfWriteTask task;
      {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this]() { return stop || !pending_writes.empty(); });
        if (pending_writes.empty()) {
          if (stop) return;
          continue;
        }
        task = std::move(pending_writes.front());
        pending_writes.pop_front();
        active = true;
      }

      bool ok = false;
      try {
#ifdef NETCDFOUTPUT
        write_netcdf_task(task);
#endif
        ok = true;
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex);
        if (!async_error) {
          async_error = std::current_exception();
        }
      }

      {
        std::lock_guard<std::mutex> lock(mutex);
        if (ok && task.combine) {
          completed_writes.push_back(
              NetcdfCombineTask{task.layout, task.output_dir, task.basename,
                                task.file_id, task.file_number});
        }
        active = false;
        if (pending_writes.empty()) {
          idle_cv.notify_all();
        }
      }
    }
  }
};

NetcdfOutput::NetcdfOutput(OutputOptions const& options_)
    : OutputType(options_), impl_(std::make_unique<NetcdfOutputImpl>()) {}

NetcdfOutput::~NetcdfOutput() = default;

void NetcdfOutput::wait_for_pending_writes() {
  std::deque<NetcdfCombineTask> combine_tasks;
  std::exception_ptr async_error;
  {
    std::unique_lock<std::mutex> lock(impl_->mutex);
    impl_->idle_cv.wait(lock, [this]() {
      return impl_->pending_writes.empty() && !impl_->active;
    });
    combine_tasks.swap(impl_->completed_writes);
    async_error = impl_->async_error;
    impl_->async_error = nullptr;
  }

  for (auto const& task : combine_tasks) {
    combine_netcdf_files(task.layout, task.output_dir, task.basename,
                         task.file_id, task.file_number);
  }

  if (async_error) {
    std::rethrow_exception(async_error);
  }
}

void NetcdfOutput::write_output_file(MeshBlockImpl* pmb_in,
                                     Variables const& vars, double current_time,
                                     bool final_write) {
  if (final_write) {
    wait_for_pending_writes();
    return;
  }

  std::deque<NetcdfCombineTask> combine_tasks;
  std::exception_ptr async_error;
  {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    combine_tasks.swap(impl_->completed_writes);
    async_error = impl_->async_error;
    impl_->async_error = nullptr;
  }
  for (auto const& task : combine_tasks) {
    combine_netcdf_files(task.layout, task.output_dir, task.basename,
                         task.file_id, task.file_number);
  }
  if (async_error) {
    std::rethrow_exception(async_error);
  }

#ifdef NETCDFOUTPUT
  auto pmb = LoadOutputData(pmb_in, vars);
  bool owns_output_pmb = (pmb != pmb_in);

  try {
    NetcdfWriteTask task;
    task.layout = pmb->get_layout();
    task.output_dir = pmb->options->output_dir();
    task.basename = pmb->options->basename();
    task.file_id = options->file_id();
    task.file_number = file_number;
    task.rank = pmb->options->layout()->rank();
    task.current_time = current_time;
    task.combine = options->combine();

    auto pmeta = MetadataTable::GetInstance();
    int nc1 = pmb->options->coord()->nc1();
    int nc2 = pmb->options->coord()->nc2();
    int nc3 = pmb->options->coord()->nc3();
    int nghost = pmb->options->coord()->nghost();

    int out_is = nc1 > 1 ? nghost : 0;
    int out_ie = nc1 > 1 ? nc1 - nghost - 1 : 0;
    int out_js = nc2 > 1 ? nghost : 0;
    int out_je = nc2 > 1 ? nc2 - nghost - 1 : 0;
    int out_ks = nc3 > 1 ? nghost : 0;
    int out_ke = nc3 > 1 ? nc3 - nghost - 1 : 0;

    if (options->include_ghost_zones()) {
      if (out_is != out_ie) {
        out_is -= nghost;
        out_ie += nghost;
      }
      if (out_js != out_je) {
        out_js -= nghost;
        out_je += nghost;
      }
      if (out_ks != out_ke) {
        out_ks -= nghost;
        out_ke += nghost;
      }
    }

    task.ncells1 = out_ie - out_is + 1;
    task.ncells2 = out_je - out_js + 1;
    task.ncells3 = out_ke - out_ks + 1;
    task.nfaces1 = task.ncells1 + (task.ncells1 > 1 ? 1 : 0);
    task.nfaces2 = task.ncells2 + (task.ncells2 > 1 ? 1 : 0);
    task.nfaces3 = task.ncells3 + (task.ncells3 > 1 ? 1 : 0);

    auto [lx2, lx3, lx1] = task.layout->loc_of(task.rank);
    int face = 0;
    if (task.layout->options->type() == "cubed-sphere") {
      face = lx1;
    }

    task.x1.reserve(task.ncells1);
    for (int i = out_is; i <= out_ie; ++i) {
      task.x1.push_back(pmb->pcoord->x1v[i].item<float>());
    }
    if (task.ncells1 > 1) {
      task.x1f.reserve(task.nfaces1);
      for (int i = out_is; i <= out_ie + 1; ++i) {
        task.x1f.push_back(pmb->pcoord->x1f[i].item<float>());
      }
    }

    task.x2.reserve(task.ncells2);
    for (int j = out_js; j <= out_je; ++j) {
      task.x2.push_back(pmb->pcoord->x2v[j].item<float>() +
                        (face % 3) * M_PI / 2.);
    }
    if (task.ncells2 > 1) {
      task.x2f.reserve(task.nfaces2);
      for (int j = out_js; j <= out_je + 1; ++j) {
        task.x2f.push_back(pmb->pcoord->x2f[j].item<float>() +
                           (face % 3) * M_PI / 2.);
      }
    }

    task.x3.reserve(task.ncells3);
    for (int k = out_ks; k <= out_ke; ++k) {
      task.x3.push_back(pmb->pcoord->x3v[k].item<float>() +
                        (face / 3) * M_PI / 2.);
    }
    if (task.ncells3 > 1) {
      task.x3f.reserve(task.nfaces3);
      for (int k = out_ks; k <= out_ke + 1; ++k) {
        task.x3f.push_back(pmb->pcoord->x3f[k].item<float>() +
                           (face / 3) * M_PI / 2.);
      }
    }

    for (OutputData* pdata = pfirst_data_; pdata != nullptr;
         pdata = pdata->pnext) {
      auto names = Vectorize<std::string>(pdata->name.c_str(), ";");
      NetcdfVariableSnapshot snapshot;
      snapshot.grid = pmeta->GetGridType(names[0]);
      snapshot.varnames = expand_varnames(*pdata, pmeta);
      snapshot.data = AthenaArray<double>(pdata->data);
      task.variables.push_back(std::move(snapshot));
    }

    ClearOutputData();
    if (owns_output_pmb) delete pmb;

    std::exception_ptr late_async_error;
    {
      std::lock_guard<std::mutex> lock(impl_->mutex);
      late_async_error = impl_->async_error;
      impl_->async_error = nullptr;
      if (!late_async_error) {
        impl_->pending_writes.push_back(std::move(task));
      }
    }
    if (late_async_error) {
      std::rethrow_exception(late_async_error);
    }
    impl_->cv.notify_one();
  } catch (...) {
    ClearOutputData();
    if (owns_output_pmb) delete pmb;
    throw;
  }
#endif
}

}  // namespace snap
