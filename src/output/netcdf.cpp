// C/C++
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// base
#include <configure.h>

// snap
#include <snap/coord/coordinate.hpp>
#include <snap/input/parameter_input.hpp>
#include <snap/mesh/meshblock.hpp>
#include <snap/utils/vectorize.hpp>

// output
#include "output_formats.hpp"
#include "output_utils.hpp"

// Only proceed if NETCDF output enabled
#ifdef NETCDFOUTPUT

// External library headers
#include <netcdf.h>

#endif  // NETCDFOUTPUT

namespace snap {
NetcdfOutput::NetcdfOutput(OutputOptions const &options_)
    : OutputType(options_) {}

void NetcdfOutput::write_output_file(MeshBlock pmb, float current_time,
                                     OctTreeOptions const &tree, bool flag) {
#ifdef NETCDFOUTPUT
  auto pmeta = MetadataTable::GetInstance();
  auto phydro = pmb->phydro;

  int nc1 = pmb->options.hydro().coord().nc1();
  int nc2 = pmb->options.hydro().coord().nc2();
  int nc3 = pmb->options.hydro().coord().nc3();
  int nghost = pmb->options.hydro().coord().nghost();

  // set start/end array indices depending on whether ghost zones are included
  out_is = nc1 > 1 ? nghost : 0;
  out_ie = nc1 > 1 ? nc1 - nghost - 1 : 0;
  out_js = nc2 > 1 ? nghost : 0;
  out_je = nc2 > 1 ? nc2 - nghost - 1 : 0;
  out_ks = nc3 > 1 ? nghost : 0;
  out_ke = nc3 > 1 ? nc3 - nghost - 1 : 0;

  // FIXME: include_ghost zones probably doesn't work with grids other than
  // CCC
  if (options.include_ghost_zones()) {
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

  // set ptrs to data in OutputData linked list, then slice/sum as needed
  LoadOutputData(pmb);

  // create filename: "file_basename"+
  // "."+"blockid"+"."+"fileid"+"."+XXXXX+".nc", where XXXXX = 5-digit
  // file_number
  std::string fname;
  char number[6];
  snprintf(number, sizeof(number), "%05d", file_number);
  char blockid[12];
  snprintf(blockid, sizeof(blockid), "block%d", pmb->options.gid());

  fname.assign(options.file_basename());
  fname.append(".");
  fname.append(blockid);
  fname.append(".");
  fname.append(options.file_id());
  fname.append(".");
  fname.append(number);
  fname.append(".nc");

  // 1. open file for output
  std::stringstream msg;
  int ifile;

  nc_create(fname.c_str(), NC_NETCDF4, &ifile);

  // 2. coordinate structure
  int ncells1 = out_ie - out_is + 1;
  int ncells2 = out_je - out_js + 1;
  int ncells3 = out_ke - out_ks + 1;

  int nfaces1 = ncells1;
  if (ncells1 > 1) nfaces1++;
  int nfaces2 = ncells2;
  if (ncells2 > 1) nfaces2++;
  int nfaces3 = ncells3;
  if (ncells3 > 1) nfaces3++;

  // 2. define coordinate
  int idt, idx1, idx2, idx3, idx1f, idx2f, idx3f, iray;
  // time
  nc_def_dim(ifile, "time", NC_UNLIMITED, &idt);

  nc_def_dim(ifile, "x1", ncells1, &idx1);
  if (ncells1 > 1) nc_def_dim(ifile, "x1f", nfaces1, &idx1f);

  nc_def_dim(ifile, "x2", ncells2, &idx2);
  if (ncells2 > 1) nc_def_dim(ifile, "x2f", nfaces2, &idx2f);

  nc_def_dim(ifile, "x3", ncells3, &idx3);
  if (ncells3 > 1) nc_def_dim(ifile, "x3f", nfaces3, &idx3f);

  // 3. define variables
  int ivt, ivx1, ivx2, ivx3, ivx1f, ivx2f, ivx3f, imu, iphi;
  int loc[4] = {pmb->options.lx1(), pmb->options.lx2(), pmb->options.lx3(),
                pmb->options.level()};
  int pos[4];

  nc_def_var(ifile, "time", NC_FLOAT, 1, &idt, &ivt);
  nc_put_att_text(ifile, ivt, "axis", 1, "T");
  nc_put_att_text(ifile, ivt, "units", 1, "s");
  nc_put_att_text(ifile, ivt, "long_name", 4, "time");

  nc_def_var(ifile, "x1", NC_FLOAT, 1, &idx1, &ivx1);
  nc_put_att_text(ifile, ivx1, "axis", 1, "Z");
  nc_put_att_text(ifile, ivx1, "units", 1, "m");
  nc_put_att_text(ifile, ivx1, "long_name", 27, "Z-coordinate at cell center");

  pos[0] = 1;
  pos[1] = ncells1 * tree.nb1();
  pos[2] = ncells1 * loc[0] + 1;
  pos[3] = ncells1 * (loc[0] + 1);
  nc_put_att_int(ifile, ivx1, "domain_decomposition", NC_INT, 4, pos);

  if (ncells1 > 1) {
    nc_def_var(ifile, "x1f", NC_FLOAT, 1, &idx1f, &ivx1f);
    nc_put_att_text(ifile, ivx1f, "units", 1, "m");
    nc_put_att_text(ifile, ivx1f, "long_name", 25, "Z-coordinate at cell face");
    pos[0]--;
    pos[2]--;
    nc_put_att_int(ifile, ivx1f, "domain_decomposition", NC_INT, 4, pos);
  }

  nc_def_var(ifile, "x2", NC_FLOAT, 1, &idx2, &ivx2);
  nc_put_att_text(ifile, ivx2, "axis", 1, "Y");
  nc_put_att_text(ifile, ivx2, "units", 1, "m");
  nc_put_att_text(ifile, ivx2, "long_name", 27, "Y-coordinate at cell center");

  pos[0] = 1;
  pos[1] = ncells2 * tree.nb2();
  pos[2] = ncells2 * loc[1] + 1;
  pos[3] = ncells2 * (loc[1] + 1);
  nc_put_att_int(ifile, ivx2, "domain_decomposition", NC_INT, 4, pos);

  if (ncells2 > 1) {
    nc_def_var(ifile, "x2f", NC_FLOAT, 1, &idx2f, &ivx2f);
    nc_put_att_text(ifile, ivx2f, "units", 1, "m");
    nc_put_att_text(ifile, ivx2f, "long_name", 25, "Y-coordinate at cell face");
    pos[0]--;
    pos[2]--;
    nc_put_att_int(ifile, ivx2f, "domain_decomposition", NC_INT, 4, pos);
  }

  nc_def_var(ifile, "x3", NC_FLOAT, 1, &idx3, &ivx3);
  nc_put_att_text(ifile, ivx3, "axis", 1, "X");
  nc_put_att_text(ifile, ivx3, "units", 1, "m");
  nc_put_att_text(ifile, ivx3, "long_name", 27, "X-coordinate at cell center");

  pos[0] = 1;
  pos[1] = ncells3 * tree.nb3();
  pos[2] = ncells3 * loc[2] + 1;
  pos[3] = ncells3 * (loc[2] + 1);
  nc_put_att_int(ifile, ivx3, "domain_decomposition", NC_INT, 4, pos);

  if (ncells3 > 1) {
    nc_def_var(ifile, "x3f", NC_FLOAT, 1, &idx3f, &ivx3f);
    nc_put_att_text(ifile, ivx3f, "units", 1, "m");
    nc_put_att_text(ifile, ivx3f, "long_name", 25, "X-coordinate at cell face");
    pos[0]--;
    pos[2]--;
    nc_put_att_int(ifile, ivx3f, "domain_decomposition", NC_INT, 4, pos);
  }

  int nbtotal = tree.nb1() * tree.nb2() * tree.nb3();
  nc_put_att_int(ifile, NC_GLOBAL, "NumFilesInSet", NC_INT, 1, &nbtotal);

  OutputData *pdata = pfirst_data_;

  // count total variables (vector variables are expanded into flat scalars)
  int total_vars = 0;
  while (pdata != nullptr) {
    auto names = Vectorize<std::string>(pdata->name.c_str(), ";");
    std::string grid = pmeta->GetGridType(names[0]);
    int nvar = get_num_variables(grid, pdata->data);

    total_vars += nvar;
    pdata = pdata->pnext;
  }

  int iaxis[4] = {idt, idx1, idx2, idx3};
  int iaxis1[4] = {idt, idx1f, idx2, idx3};
  int iaxis2[4] = {idt, idx1, idx2f, idx3};
  int iaxis3[4] = {idt, idx1, idx2, idx3f};
  int iaxisr[4] = {idt, iray, idx2, idx3};
  int *var_ids = new int[total_vars];
  int *ivar = var_ids;

  pdata = pfirst_data_;
  while (pdata != nullptr) {
    auto names = Vectorize<std::string>(pdata->name.c_str(), ";");
    std::string grid = pmeta->GetGridType(names[0]);
    int nvar = get_num_variables(grid, pdata->data);

    std::vector<std::string> varnames;
    if (names.size() >= nvar) {
      for (int n = 0; n < nvar; ++n) {
        varnames.push_back(names[n]);
      }
    } else {
      for (int n = 0; n < nvar; ++n) {
        size_t pos = pdata->name.find('?');
        if (nvar == 1) {                     // SCALARS
          if (pos < pdata->name.length()) {  // find '?'
            varnames.push_back(pdata->name.substr(0, pos) +
                               pdata->name.substr(pos + 1));
          } else {
            varnames.push_back(pdata->name);
          }
        } else {  // VECTORS
          char c[16];
          snprintf(c, sizeof(c), "%d", n + 1);
          if (pos < pdata->name.length()) {  // find '?'
            varnames.push_back(pdata->name.substr(0, pos) + c +
                               pdata->name.substr(pos + 1));
          } else {
            varnames.push_back(pdata->name + c);
          }
        }
      }
    }

    for (int n = 0; n < nvar; ++n) {
      auto name = varnames[n];

      if (grid == "CCF")
        nc_def_var(ifile, name.c_str(), NC_FLOAT, 4, iaxis1, ivar);
      else if ((grid == "CFC") && (ncells2 > 1))
        nc_def_var(ifile, name.c_str(), NC_FLOAT, 4, iaxis2, ivar);
      else if ((grid == "FCC") && (ncells3 > 1))
        nc_def_var(ifile, name.c_str(), NC_FLOAT, 4, iaxis3, ivar);
      else if (grid == "--C")
        nc_def_var(ifile, name.c_str(), NC_FLOAT, 2, iaxis, ivar);
      else if (grid == "--F")
        nc_def_var(ifile, name.c_str(), NC_FLOAT, 2, iaxis1, ivar);
      else if (grid == "---")
        nc_def_var(ifile, name.c_str(), NC_FLOAT, 1, iaxis, ivar);
      else
        nc_def_var(ifile, name.c_str(), NC_FLOAT, 4, iaxis, ivar);

      // set units
      auto attr = pmeta->GetUnits(name);
      if (attr != "") {
        nc_put_att_text(ifile, *ivar, "units", attr.length(), attr.c_str());
      }

      // set long_name
      attr = pmeta->GetLongName(name);
      if (attr != "") {
        nc_put_att_text(ifile, *ivar, "long_name", attr.length(), attr.c_str());
      }

      ivar++;
    }
    pdata = pdata->pnext;
  }

  nc_enddef(ifile);

  // 4. write variables
  float *data = new float[nfaces1 * nfaces2 * nfaces3];
  size_t start[4] = {0, 0, 0, 0};
  size_t count[4] = {1, (size_t)ncells1, (size_t)ncells2, (size_t)ncells3};
  size_t count1[4] = {1, (size_t)nfaces1, (size_t)ncells2, (size_t)ncells3};
  size_t count2[4] = {1, (size_t)ncells1, (size_t)nfaces2, (size_t)ncells3};
  size_t count3[4] = {1, (size_t)ncells1, (size_t)ncells2, (size_t)nfaces3};

  nc_put_vara_float(ifile, ivt, start, count, &current_time);

  for (int i = out_is; i <= out_ie; ++i)
    data[i - out_is] = phydro->pcoord->x1v[i].item<float>();
  nc_put_var_float(ifile, ivx1, data);

  if (ncells1 > 1) {
    for (int i = out_is; i <= out_ie + 1; ++i)
      data[i - out_is] = phydro->pcoord->x1f[i].item<float>();
    nc_put_var_float(ifile, ivx1f, data);
  }

  for (int j = out_js; j <= out_je; ++j)
    data[j - out_js] = phydro->pcoord->x2v[j].item<float>();
  nc_put_var_float(ifile, ivx2, data);

  if (ncells2 > 1) {
    for (int j = out_js; j <= out_je + 1; ++j)
      data[j - out_js] = phydro->pcoord->x2f[j].item<float>();
    nc_put_var_float(ifile, ivx2f, data);
  }

  for (int k = out_ks; k <= out_ke; ++k)
    data[k - out_ks] = phydro->pcoord->x3v[k].item<float>();
  nc_put_var_float(ifile, ivx3, data);

  if (ncells3 > 1) {
    for (int k = out_ks; k <= out_ke + 1; ++k)
      data[k - out_ks] = phydro->pcoord->x3f[k].item<float>();
    nc_put_var_float(ifile, ivx3f, data);
  }

  ivar = var_ids;
  pdata = pfirst_data_;
  while (pdata != nullptr) {
    auto names = Vectorize<std::string>(pdata->name.c_str(), ",");
    std::string grid = pmeta->GetGridType(names[0]);
    int nvar = get_num_variables(grid, pdata->data);

    if (grid == "CCF") {
      for (int n = 0; n < nvar; n++) {
        float *it = data;
        for (int i = out_is; i <= out_ie + 1; ++i)
          for (int j = out_js; j <= out_je; ++j)
            for (int k = out_ks; k <= out_ke; ++k)
              *it++ = pdata->data(n, k, j, i);
        nc_put_vara_float(ifile, *ivar++, start, count1, data);
      }
    } else if ((grid == "CFC") && (ncells2 > 1)) {
      for (int n = 0; n < nvar; n++) {
        float *it = data;
        for (int i = out_is; i <= out_ie; ++i)
          for (int j = out_js; j <= out_je + 1; ++j)
            for (int k = out_ks; k <= out_ke; ++k)
              *it++ = pdata->data(n, k, j, i);
        nc_put_vara_float(ifile, *ivar++, start, count2, data);
      }
    } else if ((grid == "FCC") && (ncells3 > 1)) {
      for (int n = 0; n < nvar; n++) {
        float *it = data;
        for (int i = out_is; i <= out_ie; ++i)
          for (int j = out_js; j <= out_je; ++j)
            for (int k = out_ks; k <= out_ke + 1; ++k)
              *it++ = pdata->data(n, k, j, i);
        nc_put_vara_float(ifile, *ivar++, start, count3, data);
      }
    } else if (grid == "--C") {
      for (int n = 0; n < nvar; n++) {
        float *it = data;
        for (int i = out_is; i <= out_ie; ++i) *it++ = pdata->data(n, i);
        nc_put_vara_float(ifile, *ivar++, start, count, data);
      }
    } else if (grid == "--F") {
      for (int n = 0; n < nvar; n++) {
        float *it = data;
        for (int i = out_is; i <= out_ie + 1; ++i) *it++ = pdata->data(n, i);
        nc_put_vara_float(ifile, *ivar++, start, count1, data);
      }
    } else if (grid == "---") {
      for (int n = 0; n < nvar; n++) {
        float *it = data;
        *it++ = pdata->data(n);
        nc_put_vara_float(ifile, *ivar++, start, count, data);
      }
    } else {
      for (int n = 0; n < nvar; n++) {
        float *it = data;
        for (int i = out_is; i <= out_ie; ++i)
          for (int j = out_js; j <= out_je; ++j)
            for (int k = out_ks; k <= out_ke; ++k)
              *it++ = pdata->data(n, k, j, i);
        nc_put_vara_float(ifile, *ivar++, start, count, data);
      }
    }

    // doesn't work
    // nc_put_att_text(ifile, *(ivar-1), "output",
    //  output_params.variable.length(), output_params.variable.c_str());
    pdata = pdata->pnext;
  }

  // 5. close nc file
  nc_close(ifile);

  ClearOutputData();  // required when LoadOutputData() is used.
  delete[] data;
  delete[] var_ids;
#endif  // NETCDFOUTPUT
}
}  // namespace snap
