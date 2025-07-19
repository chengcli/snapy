// C/C++
#include <cstdio>
#include <cstdlib>
#include <cstring>

// torch
#include <torch/torch.h>

// snap
#include <snap/utils/flip_zero.h>

#include <snap/bc/internal_boundary.hpp>

using namespace snap;

int test1(int argc, char *argv[]) {
  // Check if the user provided a command-line argument
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <01-string>\n", argv[0]);
    return 1;
  }

  char *input = argv[1];
  int length = strlen(input);

  // Validate that the string contains only '0' and '1'
  for (int i = 0; i < length; i++) {
    if (input[i] != '0' && input[i] != '1') {
      fprintf(stderr, "Error: Input string must contain only '0' and '1'.\n");
      return 1;
    }
  }

  int minRun0 = 3;
  int minRun1 = 2;
  int allowBothFlips = 0;
  int stride = 1;

  int n = strlen(input);
  int *arr = (int *)malloc(n * sizeof(int));
  for (int i = 0; i < n; i++) {
    arr[i] = input[i] - '0';
  }

  // initialize dp arrays
  constexpr int MAXRUN = InternalBoundaryOptions::MAXRUN;
  int *dp = (int *)malloc((n + 1) * MAXRUN * 2 * sizeof(int));
  int *fromLen = (int *)malloc((n + 1) * MAXRUN * 2 * sizeof(int));
  int *fromBit = (int *)malloc((n + 1) * MAXRUN * 2 * sizeof(int));
  int *usedFlip = (int *)malloc((n + 1) * MAXRUN * 2 * sizeof(int));

  // We'll run the DP
  int minFlips = compute_min_flips(arr, n, minRun0, minRun1, allowBothFlips,
                                   stride, dp, fromLen, fromBit, usedFlip);

  if (minFlips < 0) {
    // impossible
    printf("Impossible to fix.\n");
    free(arr);
    free(dp);
    free(fromLen);
    free(fromBit);
    free(usedFlip);
    return 0;
  }

  // Reconstruct final solution
  int *fixedSeq = (int *)malloc(n * sizeof(int));
  reconstruct_solution(fixedSeq, arr, n, minRun0, minRun1, allowBothFlips,
                       stride, fromLen, fromBit, usedFlip);

  // Print results
  printf("Minimum flips = %d\n", minFlips);
  printf("Fixed sequence: ");
  for (int i = 0; i < n; i++) {
    printf("%d", fixedSeq[i]);
  }
  printf("\n");

  free(arr);
  free(fixedSeq);
  free(dp);
  free(fromLen);
  free(fromBit);
  free(usedFlip);
  return 0;
}

void test2() {
  int flips = 0;
  auto solid = torch::zeros({8, 8, 8}, torch::kInt32);
  if (torch::cuda::is_available()) solid = solid.to(torch::kCUDA);

  solid[4][4][4] = 1;
  solid[2][5][3] = 1;

  // std::cout << "solid = " << solid << std::endl;

  InternalBoundary pib;

  auto out = pib->rectify_solid(solid, flips);
  std::cout << "out = " << out << std::endl;
  std::cout << "flips = " << flips << std::endl;
}

int main(int argc, char *argv[]) { test2(); }
