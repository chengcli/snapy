// snap
#include <snap/hydro/hydro.hpp>
#include <snap/mesh/meshblock.hpp>

#include "output_type.hpp"

namespace snap {
bool OutputType::SliceOutputData(MeshBlock pmb, int dim) {
  int islice(0), jslice(0), kslice(0);

  auto x1min = pmb->phydro->pcoord->options.x1min() auto x1max =
      pmb->phydro->pcoord->options.x1max() auto x2min =
          pmb->phydro->pcoord->options.x2min() auto x2max =
              pmb->phydro->pcoord->options.x2max() auto x3min =
                  pmb->phydro->pcoord->options.x3min() auto x3max =
                      pmb->phydro->pcoord->options.x3max()

                      // Compute i,j,k indices of slice; check if in range of
                      // data in this block
                      if (dim == 1) {
    if (options.x1_slice() >= x1min() && options.x1_slice() < pmb->x1max()) {
      for (int i = pmb->is() + 1; i <= pmb->ie() + 1; ++i) {
        if (pmb->phydro->pcoord->x1f[i].item<float>() > options.x1_slice()) {
          islice = i - 1;
          break;
        }
      }
    } else {
      return false;
    }
  }
  else if (dim == 2) {
    if (options.x2_slice() >= pmb->x2min() &&
        options.x2_slice() < pmb->x2max()) {
      for (int j = pmb->js() + 1; j <= pmb->je() + 1; ++j) {
        if (pmb->phydro->pcoord->x2f[j].item<float>() > options.x2_slice()) {
          jslice = j - 1;
          break;
        }
      }
    } else {
      return false;
    }
  }
  else {
    if (options.x3_slice() >= pmb->x3min() &&
        options.x3_slice() < pmb->x3max()) {
      for (int k = pmb->ks() + 1; k <= pmb->ke() + 1; ++k) {
        if (pmb->phydro->pcoord->x3f[k].item<float>() > options.x3_slice()) {
          kslice = k - 1;
          break;
        }
      }
    } else {
      return false;
    }
  }

  // For each node in OutputData doubly linked list, slice arrays containing
  // output data
  OutputData *pdata, *pnew;
  pdata = pfirst_data_;

  while (pdata != nullptr) {
    pnew = new OutputData;
    pnew->type = pdata->type;
    pnew->name = pdata->name;
    int nx4 = pdata->data.GetDim4();
    int nx3 = pdata->data.GetDim3();
    int nx2 = pdata->data.GetDim2();
    int nx1 = pdata->data.GetDim1();

    // Loop over variables and dimensions, extract slice
    if (dim == 3) {
      pnew->data.NewAthenaArray(nx4, 1, nx2, nx1);
      for (int n = 0; n < nx4; ++n) {
        for (int j = out_js; j <= out_je; ++j) {
          for (int i = out_is; i <= out_ie; ++i) {
            pnew->data(n, 0, j, i) = pdata->data(n, kslice, j, i);
          }
        }
      }
    } else if (dim == 2) {
      pnew->data.NewAthenaArray(nx4, nx3, 1, nx1);
      for (int n = 0; n < nx4; ++n) {
        for (int k = out_ks; k <= out_ke; ++k) {
          for (int i = out_is; i <= out_ie; ++i) {
            pnew->data(n, k, 0, i) = pdata->data(n, k, jslice, i);
          }
        }
      }
    } else {
      pnew->data.NewAthenaArray(nx4, nx3, nx2, 1);
      for (int n = 0; n < nx4; ++n) {
        for (int k = out_ks; k <= out_ke; ++k) {
          for (int j = out_js; j <= out_je; ++j) {
            pnew->data(n, k, j, 0) = pdata->data(n, k, j, islice);
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
  return true;
}
}  // namespace snap
