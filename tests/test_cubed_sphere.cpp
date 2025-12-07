// C/C++
#include <cstdio>

// snap
#include <snap/coord/cubed_sphere_utils.hpp>
#include <snap/layout/cubed_sphere_layout.hpp>
#include <snap/layout/layout.hpp>

using namespace snap;

void run_demo(CubedSphereLayoutImpl const &cs, int rx, int ry, int face) {
  printf(
      "Demo cubed-sphere Z-order connectivity pxy=%u face=%d (rx,ry)=(%u,%u)\n",
      cs.pxy(), face, rx, ry);

  int g_self = cs.neighbor_rank({rx, ry, face}, {0, 0, 0});
  int g_left = cs.neighbor_rank({rx, ry, face}, {-1, 0, 0});
  int g_right = cs.neighbor_rank({rx, ry, face}, {1, 0, 0});
  int g_down = cs.neighbor_rank({rx, ry, face}, {0, -1, 0});
  int g_up = cs.neighbor_rank({rx, ry, face}, {0, 1, 0});
  int g_ul = cs.neighbor_rank({rx, ry, face}, {-1, 1, 0}); /* corner */
  int g_dr = cs.neighbor_rank({rx, ry, face}, {1, -1, 0}); /* corner */

  printf("self=%d L=%d R=%d D=%d U=%d UL=%d DR=%d\n", g_self, g_left, g_right,
         g_down, g_up, g_ul, g_dr);
}

/* --------- Analytic test fields on the sphere --------- */

static inline double field_const(Vec3 v) {
  (void)v;
  return 1.0;
}

static inline double field_linear(Vec3 v) {
  /* choose some arbitrary vector a = (1, 0.5, -0.25) */
  return v.x + 0.5 * v.y - 0.25 * v.z;
}

/* --------- Fill interior panel data from analytic field --------- */
/* face_data[f] is N×N row-major [y][x], no ghosts */

typedef double (*FieldFn)(Vec3);

static void fill_interior_from_field(int N, double *face_data[6], FieldFn f) {
  for (int face = 0; face < 6; ++face) {
    double *data = face_data[face];
    for (int j = 0; j < N; ++j) {
      double beta = cs_equ_center(N, j);
      for (int i = 0; i < N; ++i) {
        double alpha = cs_equ_center(N, i);
        Vec3 v = cs_ab_to_xyz(CS_FACE_NAMES[face], alpha, beta);
        data[j * N + i] = f(v);
      }
    }
  }
}

/*static void run_test(int N, int nghost, const double *usrc, FieldFn f)
{
  // Allocate interior fields per face
  double *face_data[6];
  for (int fidx = 0; fidx < 6; ++fidx) {
    face_data[fidx] = (double*)malloc((size_t)N * (size_t)N * sizeof(double));
    if (!face_data[fidx]) {
      fprintf(stderr, "Allocation failed\n");
      exit(1);
    }
  }

  fill_interior_from_field(N, face_data, f);

  double max_err = 0.0;
  size_t n_total = 0;

  for (int face_t = 0; face_t < 6; ++face_t)
    for (int side_t = SIDE_L; side_t <= SIDE_T; ++side_t) {
      std::vector<double> usrc(nghost * N);
      cs_build_ghost_usrc(usrc.data(), N, nghost, face_t, side_t);

      for (int depth = 1; depth <= nghost; ++depth) {
        for (int j = 0; j < N; ++j) {
          int idx = (depth - 1) * N + j;
          const double *src_face = face_data[e->face_s];

          // interpolated ghost value from source interior line
          double g_interp = sample_source_line(src_face, N, e->side_s,
usrc[idx]);

          // analytic "truth" at target ghost center
          double alpha_t, beta_t;
          cs_equ_ghost_center(side_t, N, j, depth, &alpha_t, &beta_t);
          Vec3 vt = cs_ab_to_xyz(CS_FACE_NAMES[face_t], alpha_t, beta_t);
          double g_true = f(vt);

          double err = fabs(g_interp - g_true);
          if (err > max_err) {
            max_err = err;
          }
          ++n_total;
        }
      }
    }

  printf("Max |error| over %zu ghost points = %.3e\n", n_total, max_err);

  for (int fidx = 0; fidx < 6; ++fidx) {
    free(face_data[fidx]);
  }
}
*/

void run_ghost_usrc() {
  int N = 8;       // cells per dim per face
  int nghost = 2;  // ghost layers

  std::vector<double> usrc(nghost * N);

  cs_build_ghost_usrc(usrc.data(), N, nghost);

  std::cout << "Ghost usrc values (N=" << N << ", nghost=" << nghost << "):\n";

  for (int depth = 1; depth <= nghost; ++depth) {
    std::cout << " depth " << depth << ": ";
    for (int j = 0; j < N; ++j) {
      int idx = (depth - 1) * N + j;
      std::cout << usrc[idx] << " ";
    }
    std::cout << "\n";
  }
}

void run_ghost() {
  int N = 32;      // cells per dim per face
  int nghost = 2;  // ghost layers

  printf("Cubed-sphere interpolation test: N=%d, nghost=%d\n", N, nghost);

  /* Run tests */
  // run_test(N, nghost, usrc.data(), field_const);
  // run_test(N, nghost, usrc.data(), field_linear);
}

int main(void) {
  auto op = LayoutOptionsImpl::create();
  op->px(4);
  op->py(4);

  CubedSphereLayoutImpl cs(op);

  for (int n = 0; n < 6; ++n) {
    printf("\nface %d tests:\n", n);
    run_demo(cs, 0, 0, n);
    run_demo(cs, 0, 1, n);
    run_demo(cs, 1, 0, n);
    run_demo(cs, 1, 1, n);
  }

  run_ghost_usrc();

  return 0;
}
