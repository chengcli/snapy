#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#define INF (INT_MAX / 2)

/*
  We'll store DP plus "parent pointers" to reconstruct the final sequence.

  dp[i][len][bit]:
    minimal flips to fix prefix s[0..i-1],
    ending with a run of `bit` whose "effective" length is `len`.

  bit = 0 => run of zeros, len in [0..2] (2 means ">=2")
  bit = 1 => run of ones,  len in [0..3] (3 means ">=3")

  fromLen[i][len][bit], fromBit[i][len][bit]:
    store which dp state (oldLen, oldBit) led to dp[i][len][bit].
  usedFlip[i][len][bit]:
    store whether we flipped the (i-1)-th element (0 => no flip, 1 => flip).
*/
int minFlipsFixRunsWithEndpoint(const int *s, int n, int *fixedSeq) {
  static int dp[100001][4][2];
  static int fromLen[100001][4][2];
  static int fromBit[100001][4][2];
  static int usedFlip[100001][4][2];

  // Initialize dp with INF
  for (int i = 0; i <= n; i++) {
    for (int len = 0; len < 4; len++) {
      for (int bit = 0; bit < 2; bit++) {
        dp[i][len][bit] = INF;
        fromLen[i][len][bit] = -1;
        fromBit[i][len][bit] = -1;
        usedFlip[i][len][bit] = -1;
      }
    }
  }

  // Start: no bits taken, no run => dp[0][0][0] = 0, dp[0][0][1] = 0
  dp[0][0][0] = 0;
  dp[0][0][1] = 0;

  // Main DP loop
  for (int i = 0; i < n; i++) {
    for (int bit = 0; bit < 2; bit++) {
      // maxLen for bit=0 is 2; for bit=1 is 3
      int maxLen = (bit == 0 ? 2 : 3);

      for (int len = 0; len <= maxLen; len++) {
        int currCost = dp[i][len][bit];
        if (currCost == INF) continue;  // not reachable

        // We consider valid "candidate" bits at position i
        // Under the constraint: we can flip 0->1, but NOT 1->0.
        // So if s[i] == 1, only candidate=1 is possible; if s[i]==0,
        // we can do candidate=0 (cost 0) or candidate=1 (cost 1).

        // 1) Keep as is (flip=0)
        //    Only valid if candidate == s[i]
        //    cost = 0
        {
          int candidate = s[i];
          int flipCost = 0;
          int newCost = currCost + flipCost;
          int newBit = candidate;  // final assigned bit

          // Extend or switch run
          if (newBit == bit) {
            // Extending the same run
            int newLen = (len + 1);
            if (newLen > maxLen) newLen = maxLen;

            if (newCost < dp[i + 1][newLen][bit]) {
              dp[i + 1][newLen][bit] = newCost;
              fromLen[i + 1][newLen][bit] = len;
              fromBit[i + 1][newLen][bit] = bit;
              usedFlip[i + 1][newLen][bit] = 0;  // no flip
            }
          } else {
            // Switching from bit -> newBit
            // Must close the old run (unless len=0 => no real run yet)
            if (len == 0) {
              // just start a new run of length 1
              if (newCost < dp[i + 1][1][newBit]) {
                dp[i + 1][1][newBit] = newCost;
                fromLen[i + 1][1][newBit] = 0;
                fromBit[i + 1][1][newBit] = bit;
                usedFlip[i + 1][1][newBit] = 0;  // no flip
              }
            } else {
              // If bit=0 => need len=2 to close
              // If bit=1 => need len=3 to close
              int needed = (bit == 0 ? 2 : 3);
              if (len == needed) {
                if (newCost < dp[i + 1][1][newBit]) {
                  dp[i + 1][1][newBit] = newCost;
                  fromLen[i + 1][1][newBit] = len;
                  fromBit[i + 1][1][newBit] = bit;
                  usedFlip[i + 1][1][newBit] = 0;  // no flip
                }
              }
            }
          }
        }

        // 2) Flip 0->1 (flip=1) — only valid if s[i] == 0
        if (s[i] == 0) {
          int candidate = 1;  // flipping 0->1
          int flipCost = 1;
          int newCost = currCost + flipCost;
          int newBit = candidate;  // final assigned bit => 1

          if (newBit == bit) {
            // Extending the same run
            int newLen = (len + 1);
            if (newLen > maxLen) newLen = maxLen;

            if (newCost < dp[i + 1][newLen][bit]) {
              dp[i + 1][newLen][bit] = newCost;
              fromLen[i + 1][newLen][bit] = len;
              fromBit[i + 1][newLen][bit] = bit;
              usedFlip[i + 1][newLen][bit] = 1;  // we flipped
            }
          } else {
            // Switching from bit -> newBit
            if (len == 0) {
              if (newCost < dp[i + 1][1][newBit]) {
                dp[i + 1][1][newBit] = newCost;
                fromLen[i + 1][1][newBit] = 0;
                fromBit[i + 1][1][newBit] = bit;
                usedFlip[i + 1][1][newBit] = 1;  // we flipped
              }
            } else {
              int needed = (bit == 0 ? 2 : 3);
              if (len == needed) {
                if (newCost < dp[i + 1][1][newBit]) {
                  dp[i + 1][1][newBit] = newCost;
                  fromLen[i + 1][1][newBit] = len;
                  fromBit[i + 1][1][newBit] = bit;
                  usedFlip[i + 1][1][newBit] = 1;  // we flipped
                }
              }
            }
          }
        }
      }
    }
  }

  // Find the best valid endpoint among dp[n][..][..].
  // - If bit=0 => need len=2
  // - If bit=1 => need len=3
  int ans = INF;
  int bestBit = -1, bestLen = -1;

  // final run of 0: len must be 2
  if (dp[n][2][0] < ans) {
    ans = dp[n][2][0];
    bestBit = 0;
    bestLen = 2;
  }
  // final run of 1: len must be 3
  if (dp[n][3][1] < ans) {
    ans = dp[n][3][1];
    bestBit = 1;
    bestLen = 3;
  }

  if (ans >= INF) {
    // impossible
    return -1;
  }

  // Reconstruct the resulting sequence by walking backward
  int curI = n;
  int curLen = bestLen;
  int curBit = bestBit;

  while (curI > 0) {
    int flip = usedFlip[curI][curLen][curBit];
    int pLen = fromLen[curI][curLen][curBit];
    int pBit = fromBit[curI][curLen][curBit];

    // If flip=0 => we kept s[curI-1]
    // If flip=1 => we flipped s[curI-1] (which must have been 0 -> 1)
    int finalBit;
    if (flip == 0) {
      finalBit = s[curI - 1];
    } else {
      // We only do 0->1 flips
      finalBit = 1;
    }

    fixedSeq[curI - 1] = finalBit;

    curI--;
    curLen = pLen;
    curBit = pBit;
  }

  return ans;
}

int main(void) {
  int n;
  scanf("%d", &n);
  int *arr = (int *)malloc(n * sizeof(int));
  for (int i = 0; i < n; i++) {
    scanf("%d", &arr[i]);
  }

  // We'll store the final fixed sequence here
  int *fixedSeq = (int *)malloc(n * sizeof(int));

  int answer = minFlipsFixRunsWithEndpoint(arr, n, fixedSeq);
  if (answer < 0) {
    printf("Impossible to fix.\n");
  } else {
    printf("Minimum flips = %d\n", answer);
    printf("Fixed sequence: ");
    for (int i = 0; i < n; i++) {
      printf("%d", fixedSeq[i]);
    }
    printf("\n");
  }

  free(arr);
  free(fixedSeq);
  return 0;
}
