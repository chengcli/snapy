#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#define INF (INT_MAX / 2)

/* ---------------------------------------------------------------------------
   Data Structures for DP
   ---------------------------------------------------------------------------
   dp[i][len][bit] = minimal flips to fix prefix s[0..i-1],
                     ending in a run of `bit` (0 or 1),
                     with "effective" run-length = len.

   We'll store parent pointers to reconstruct the solution:
   fromLen[i][len][bit], fromBit[i][len][bit], usedFlip[i][len][bit].
 --------------------------------------------------------------------------- */

// We'll define a global or static size limit for demonstration:
#define MAXN 100

static int dp[MAXN + 1][4][2];
static int fromLen[MAXN + 1][4][2];
static int fromBit[MAXN + 1][4][2];
static int usedFlip[MAXN + 1][4][2];

/*
   We'll pass these around:
   - s[] : input sequence
   - outSeq[] : reconstructed fixed sequence
   - n : length of s
   - minRun0 : minimum run length for 0
   - minRun1 : minimum run length for 1
   - allowBothFlips : 0 => only 0->1 flips, 1 => 0->1 or 1->0 flips
*/

/* ---------------------------------------------------------------------------
   Function: readInput
   Reads from stdin:
   1) int n  (length of sequence)
   2) the sequence of bits
   3) int minRun0
   4) int minRun1
   5) int allowBothFlips (0 or 1)
 --------------------------------------------------------------------------- */
int readInput(int **arr, int *n, int *minRun0, int *minRun1,
              int *allowBothFlips) {
  scanf("%d", n);
  if ((*n) <= 0 || (*n) > MAXN) {
    return -1;  // invalid n
  }

  // allocate array
  *arr = (int *)malloc((*n) * sizeof(int));

  for (int i = 0; i < (*n); i++) {
    scanf("%d", &((*arr)[i]));
  }

  scanf("%d", minRun0);
  scanf("%d", minRun1);
  scanf("%d", allowBothFlips);

  return 0;  // OK
}

/* ---------------------------------------------------------------------------
   Function: initializeDP
   Resets dp[][][] arrays to INF, fromLen/fromBit/usedFlip to -1.
   We also need to know the max possible "effective run length"
   for 0 and for 1. Those are minRun0, minRun1, but for DP we
   usually track "up to" that min (meaning once we exceed the required
   length, we can store it as "already good" or ">= minRun").

   We'll do something like:
     - For bit=0, we track runLength in [0..minRun0] where
       runLength==minRun0 means ">= minRun0".
     - For bit=1, we track runLength in [0..minRun1] likewise.

   But the code is simpler if we keep them separate. We'll pass them in
   as maxLen0, maxLen1:
     maxLen0 = minRun0
     maxLen1 = minRun1

   If the runs can be *very large*, to avoid huge memory,
   we cap runLength at maxLen0 or maxLen1.
   Example: if minRun0=2, once we have a run of length 2 (or more),
   we store length=2 as ">=2".
 --------------------------------------------------------------------------- */
void initializeDP(int n, int maxLen0, int maxLen1) {
  for (int i = 0; i <= n; i++) {
    for (int bit = 0; bit < 2; bit++) {
      // For bit=0, we only care about runLen up to maxLen0
      // For bit=1, we only care about runLen up to maxLen1
      int lim = (bit == 0 ? maxLen0 : maxLen1);
      for (int len = 0; len <= lim; len++) {
        dp[i][len][bit] = INF;
        fromLen[i][len][bit] = -1;
        fromBit[i][len][bit] = -1;
        usedFlip[i][len][bit] = -1;
      }
    }
  }

  // Start states: dp[0][0][0] = 0, dp[0][0][1] = 0
  // Means "no bits used, no run in progress"
  dp[0][0][0] = 0;
  dp[0][0][1] = 0;
}

/* ---------------------------------------------------------------------------
   Helper: canFlip(originalBit, candidateBit, allowBothFlips)
   Returns the flip cost (0 or 1) if allowed, or INF if not allowed.

   If allowBothFlips == 0 => only 0->1 flips
      -> that means:
         if originalBit=0 && candidateBit=0 => cost=0
         if originalBit=0 && candidateBit=1 => cost=1
         if originalBit=1 && candidateBit=1 => cost=0
         if originalBit=1 && candidateBit=0 => not allowed => INF

   If allowBothFlips == 1 => 0->1 or 1->0 allowed
      -> cost=1 if there's a flip, cost=0 if no flip
 --------------------------------------------------------------------------- */
int canFlip(int originalBit, int candidateBit, int allowBothFlips) {
  if (allowBothFlips == 0) {
    // Only 0->1 allowed
    if (originalBit == candidateBit) {
      // no flip
      return 0;
    } else {
      // originalBit != candidateBit
      // valid only if originalBit=0, candidateBit=1
      if (originalBit == 0 && candidateBit == 1) {
        return 1;
      } else {
        // This is 1->0, not allowed
        return INF;
      }
    }
  } else {
    // Both 0->1 and 1->0 allowed
    // cost=0 if same, cost=1 if different
    if (originalBit == candidateBit) {
      return 0;
    } else {
      return 1;
    }
  }
}

/* ---------------------------------------------------------------------------
   Function: computeMinFlips
   - s: the input sequence
   - n: length of s
   - minRun0: required run length for 0
   - minRun1: required run length for 1
   - allowBothFlips: 0 => only 0->1, 1 => both 0->1 and 1->0
   - outMinFlips: pointer to an int that will store the minimal #flips
                  or -1 if impossible
   This function fills the dp[][][] arrays and determines the minimal flips.
 --------------------------------------------------------------------------- */
void computeMinFlips(const int *s, int n, int minRun0, int minRun1,
                     int allowBothFlips, int *outMinFlips) {
  // "Effective" max lengths for DP
  int maxLen0 = minRun0;  // once we reach minRun0, store as ">= minRun0"
  int maxLen1 = minRun1;

  // Initialize the DP
  initializeDP(n, maxLen0, maxLen1);

  // Main DP
  for (int i = 0; i < n; i++) {
    for (int bit = 0; bit < 2; bit++) {
      int lim = (bit == 0 ? maxLen0 : maxLen1);
      for (int len = 0; len <= lim; len++) {
        int currCost = dp[i][len][bit];
        if (currCost == INF) continue;  // not reachable

        // We'll try to assign "candidate" for s[i] as either 0 or 1
        for (int candidate = 0; candidate <= 1; candidate++) {
          // Check if flipping from s[i] to candidate is allowed
          int flipCost = canFlip(s[i], candidate, allowBothFlips);
          if (flipCost == INF) {
            continue;  // not allowed
          }
          int newCost = currCost + flipCost;
          int newBit = candidate;

          // Next run length
          if (newBit == bit) {
            // Extend same run
            int newLen = len + 1;
            // Cap it at the maximum for that bit
            if (bit == 0 && newLen > maxLen0) newLen = maxLen0;
            if (bit == 1 && newLen > maxLen1) newLen = maxLen1;

            // Update dp
            if (newCost < dp[i + 1][newLen][bit]) {
              dp[i + 1][newLen][bit] = newCost;
              fromLen[i + 1][newLen][bit] = len;
              fromBit[i + 1][newLen][bit] = bit;
              usedFlip[i + 1][newLen][bit] = (flipCost > 0 ? 1 : 0);
            }
          } else {
            // Switch run from bit -> newBit
            // Must close the old run (unless len=0 => no run yet)
            if (len == 0) {
              // Starting new run of length 1
              if (newCost < dp[i + 1][1][newBit]) {
                dp[i + 1][1][newBit] = newCost;
                fromLen[i + 1][1][newBit] = 0;
                fromBit[i + 1][1][newBit] = bit;
                usedFlip[i + 1][1][newBit] = (flipCost > 0 ? 1 : 0);
              }
            } else {
              // To close a run of bit=0, we need len >= minRun0
              // To close a run of bit=1, we need len >= minRun1
              int needed = (bit == 0 ? minRun0 : minRun1);
              if (len >= needed) {
                // We can start a new run with length 1
                if (newCost < dp[i + 1][1][newBit]) {
                  dp[i + 1][1][newBit] = newCost;
                  fromLen[i + 1][1][newBit] = len;
                  fromBit[i + 1][1][newBit] = bit;
                  usedFlip[i + 1][1][newBit] = (flipCost > 0 ? 1 : 0);
                }
              }
            }
          }
        }
      }
    }
  }

  // Find the best valid endpoint
  // We want a final run of 0 that has length >= minRun0,
  // or a final run of 1 that has length >= minRun1.
  int ans = INF;

  // We'll store the best final (len, bit) as well
  int bestLen = -1, bestBit = -1;

  // Check final runs for bit=0
  for (int len = minRun0; len <= maxLen0; len++) {
    if (dp[n][len][0] < ans) {
      ans = dp[n][len][0];
      bestLen = len;
      bestBit = 0;
    }
  }

  // Check final runs for bit=1
  for (int len = minRun1; len <= maxLen1; len++) {
    if (dp[n][len][1] < ans) {
      ans = dp[n][len][1];
      bestLen = len;
      bestBit = 1;
    }
  }

  if (ans >= INF) {
    *outMinFlips = -1;  // impossible
  } else {
    *outMinFlips = ans;

    // Store bestLen/bestBit into fromLen[?][?][?] "special"
    // or we can store them in some global variable,
    // but let's keep them in the usedFlip array or so.
    // Actually, we can store them in usedFlip[n][0][0],
    // but let's not override it.
    // We'll store them in two static variables or pass them globally.
    // For clarity, let's store them in usedFlip as a "hack."
    // Or simpler: we'll just store them in fromLen[n][0][0]
    // so we know how to start reconstruction.
    // We'll do fromLen[n][0][0] = bestLen, fromBit[n][0][0] = bestBit
    fromLen[n][0][0] = bestLen;
    fromBit[n][0][0] = bestBit;
  }
}

/* ---------------------------------------------------------------------------
   Function: reconstructSolution
   Reconstruct the fixed sequence of length n into outSeq[],
   given that we've computed the DP and found the best final state
   fromLen[n][0][0], fromBit[n][0][0].
 --------------------------------------------------------------------------- */
void reconstructSolution(const int *s, int n, int minRun0, int minRun1,
                         int allowBothFlips, int *outSeq) {
  // We stored the best final (len, bit) in fromLen[n][0][0] / fromBit[n][0][0]
  int bestLen = fromLen[n][0][0];
  int bestBit = fromBit[n][0][0];

  if (bestLen < 0 || bestBit < 0) {
    // Means no solution or we never set it; do nothing
    return;
  }

  int curI = n;
  int curLen = bestLen;
  int curBit = bestBit;

  // "Effective" max lengths for DP
  int maxLen0 = minRun0;
  int maxLen1 = minRun1;

  // Reconstruct backwards
  while (curI > 0) {
    int flip = usedFlip[curI][curLen][curBit];
    int pLen = fromLen[curI][curLen][curBit];
    int pBit = fromBit[curI][curLen][curBit];

    // We want to figure out what bit was actually chosen at index curI - 1
    // If flip=0 => candidateBit == originalBit
    // If flip=1 => candidateBit == 1 - originalBit (if allowBothFlips=1)
    // or candidateBit == 1 if originalBit=0 (if allowBothFlips=0)
    // But we have to be careful if both flips are allowed. Let's do:
    //   originalBit = s[curI - 1]
    //   if flip=0 => finalBit = originalBit
    //   if flip=1 => finalBit = 1 - originalBit (both flips allowed)
    //                      or = 1 if only 0->1 flips allowed
    int originalBit = s[curI - 1];
    int finalBit;

    if (allowBothFlips == 0) {
      // only 0->1 allowed
      if (flip == 0) {
        finalBit = originalBit;  // no flip
      } else {
        // flip=1 => must be 0->1
        finalBit = 1;
      }
    } else {
      // both 0->1 and 1->0 allowed
      if (flip == 0) {
        finalBit = originalBit;
      } else {
        finalBit = (originalBit == 0) ? 1 : 0;
      }
    }

    outSeq[curI - 1] = finalBit;

    // Move to predecessor
    curI--;
    curLen = pLen;
    curBit = pBit;
  }
}

/* ---------------------------------------------------------------------------
   Function: printSequence
   Prints the resulting sequence outSeq[] of length n to stdout.
 --------------------------------------------------------------------------- */
void printSequence(int *seq, int n) {
  for (int i = 0; i < n; i++) {
    printf("%d", seq[i]);
  }
  printf("\n");
}

/* ---------------------------------------------------------------------------
   main
   ---------------------------------------------------------------------------
 */
int main(void) {
  int *arr = NULL;
  int n;
  int minRun0, minRun1;
  int allowBothFlips;

  if (readInput(&arr, &n, &minRun0, &minRun1, &allowBothFlips) < 0) {
    printf("Error reading input.\n");
    return 1;
  }

  // We'll run the DP
  int minFlips;
  computeMinFlips(arr, n, minRun0, minRun1, allowBothFlips, &minFlips);

  if (minFlips < 0) {
    // impossible
    printf("Impossible to fix.\n");
    free(arr);
    return 0;
  }

  // Reconstruct final solution
  int *fixedSeq = (int *)malloc(n * sizeof(int));
  reconstructSolution(arr, n, minRun0, minRun1, allowBothFlips, fixedSeq);

  // Print results
  printf("Minimum flips = %d\n", minFlips);
  printf("Fixed sequence: ");
  printSequence(fixedSeq, n);

  free(arr);
  free(fixedSeq);
  return 0;
}
