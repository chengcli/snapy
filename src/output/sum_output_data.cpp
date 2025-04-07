// snap
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "output_type.hpp"

namespace snap {
void OutputType::SumOutputData(MeshBlock pmb, int dim) {
  // For each node in OutputData doubly linked list, sum arrays containing
  // output data
  OutputData *pdata = pfirst_data_;
  while (pdata != nullptr) {
    OutputData *pnew = new OutputData;
    pnew->type = pdata->type;
    pnew->name = pdata->name;
    int nx4 = pdata->data.GetDim4();
    int nx3 = pdata->data.GetDim3();
    int nx2 = pdata->data.GetDim2();
    int nx1 = pdata->data.GetDim1();

    // Loop over variables and dimensions, sum over specified dimension
    if (dim == 3) {
      pnew->data.NewAthenaArray(nx4, 1, nx2, nx1);
      for (int n = 0; n < nx4; ++n) {
        for (int k = out_ks; k <= out_ke; ++k) {
          for (int j = out_js; j <= out_je; ++j) {
            for (int i = out_is; i <= out_ie; ++i) {
              pnew->data(n, 0, j, i) += pdata->data(n, k, j, i);
            }
          }
        }
      }
    } else if (dim == 2) {
      pnew->data.NewAthenaArray(nx4, nx3, 1, nx1);
      for (int n = 0; n < nx4; ++n) {
        for (int k = out_ks; k <= out_ke; ++k) {
          for (int j = out_js; j <= out_je; ++j) {
            for (int i = out_is; i <= out_ie; ++i) {
              pnew->data(n, k, 0, i) += pdata->data(n, k, j, i);
            }
          }
        }
      }
    } else {
      pnew->data.NewAthenaArray(nx4, nx3, nx2, 1);
      for (int n = 0; n < nx4; ++n) {
        for (int k = out_ks; k <= out_ke; ++k) {
          for (int j = out_js; j <= out_je; ++j) {
            for (int i = out_is; i <= out_ie; ++i) {
              pnew->data(n, k, j, 0) += pdata->data(n, k, j, i);
            }
          }
        }
      }
    }

    ReplaceOutputDataNode(pdata, pnew);
    pdata = pnew->pnext;
  }

  // modify array indices
  if (dim == 3) {
    out_ks = 0;
    out_ke = 0;
  } else if (dim == 2) {
    out_js = 0;
    out_je = 0;
  } else {
    out_is = 0;
    out_ie = 0;
  }
  return;
}
}  // namespace snap
