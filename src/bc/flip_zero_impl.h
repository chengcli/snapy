// C/C++
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// snap
#include <configure.h>

#define INF (INT_MAX / 2)
#define MAXRUN 4
#define DP(i, len, bit) dp[(i) * stride * MAXRUN * 2 + (len) * 2 + (bit)]
#define FROMLEN(i, len, bit) \
  fromLen[(i) * stride * MAXRUN * 2 + (len) * 2 + (bit)]
#define FROMBIT(i, len, bit) \
  fromBit[(i) * stride * MAXRUN * 2 + (len) * 2 + (bit)]
#define USEDFLIP(i, len, bit) \
  usedFlip[(i) * stride * MAXRUN * 2 + (len) * 2 + (bit)]
#define INSEQ(i) inSeq[(i) * stride]

namespace snap {

template <typename T>
DISPATCH_MACRO void initialize_dp(int n, T *dp, T *fromLen, T *fromBit,
                                  T *usedFlip, int dir, int stride) {
  for (int i = 0; i <= n; i++) {
    for (int bit = 0; bit < 2; bit++) {
      // For bit=0, we only care about runLen up to maxLen0
      // For bit=1, we only care about runLen up to maxLen1
      for (int len = 0; len < MAXRUN; len++) {
        DP(i, len, bit) = (T)INF;
        FROMLEN(i, len, bit) = -1;
        FROMBIT(i, len, bit) = -1;
        USEDFLIP(i, len, bit) = -1;
      }
    }
  }

  int is = dir > 0 ? 0 : n;

  // Start states: dp[0][0][0] = 0, dp[0][0][1] = 0
  // Means "no bits used, no run in progress"
  DP(is, 0, 0) = 0;
  DP(is, 0, 1) = 0;
}

template <typename T>
DISPATCH_MACRO int can_flip(T originalBit, T candidateBit, int allowBothFlips) {
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

template <typename T>
DISPATCH_MACRO int compute_min_flips(const T *inSeq, int n, int minRun0,
                                     int minRun1, int allowBothFlips, int dir,
                                     int stride,
                                     /* store the DP arrays */
                                     T *dp, T *fromLen, T *fromBit,
                                     T *usedFlip) {
  // "Effective" max lengths for DP
  auto maxLen0 = minRun0;  // once we reach minRun0, store as ">= minRun0"
  auto maxLen1 = minRun1;

  // Initialize the DP
  initialize_dp<T>(n, dp, fromLen, fromBit, usedFlip, dir, stride);

  // Main DP
  int is = dir > 0 ? 0 : n;
  int ie = dir > 0 ? n : 0;
  int shift = dir > 0 ? 0 : -1;
  for (int i = is; i != ie; i += dir) {
    for (int bit = 0; bit < 2; bit++) {
      for (int len = 0; len < MAXRUN; len++) {
        auto currCost = DP(i, len, bit);
        if (currCost == INF) continue;  // not reachable

        // We'll try to assign "candidate" for s[i] as either 0 or 1
        for (T candidate = 0; candidate <= 1; candidate++) {
          // Check if flipping from s[i] to candidate is allowed
          auto flipCost = can_flip(INSEQ(i + shift), candidate, allowBothFlips);
          if (flipCost == INF) {
            continue;  // not allowed
          }
          auto newCost = currCost + flipCost;
          T newBit = candidate;

          // Next run length
          if (newBit == bit) {
            // Extend same run
            auto newLen = len + 1;
            // Cap it at the maximum for that bit
            if (bit == 0 && newLen > maxLen0) newLen = maxLen0;
            if (bit == 1 && newLen > maxLen1) newLen = maxLen1;

            // Update dp
            if (newCost < DP(i + dir, newLen, bit)) {
              DP(i + dir, newLen, bit) = newCost;
              FROMLEN(i + dir, newLen, bit) = len;
              FROMBIT(i + dir, newLen, bit) = bit;
              USEDFLIP(i + dir, newLen, bit) = (flipCost > 0 ? 1 : 0);
            }
          } else {
            // Switch run from bit -> newBit
            // Must close the old run (unless len=0 => no run yet)
            if (len == 0) {
              // Starting new run of length 1
              if (newCost < DP(i + dir, 1, newBit)) {
                DP(i + dir, 1, newBit) = newCost;
                FROMLEN(i + dir, 1, newBit) = 0;
                FROMBIT(i + dir, 1, newBit) = bit;
                USEDFLIP(i + dir, 1, newBit) = (flipCost > 0 ? 1 : 0);
              }
            } else {
              // To close a run of bit=0, we need len >= minRun0
              // To close a run of bit=1, we need len >= minRun1
              auto needed = (bit == 0 ? minRun0 : minRun1);
              if (len >= needed) {
                // We can start a new run with length 1
                if (newCost < DP(i + dir, 1, newBit)) {
                  DP(i + dir, 1, newBit) = newCost;
                  FROMLEN(i + dir, 1, newBit) = len;
                  FROMBIT(i + dir, 1, newBit) = bit;
                  USEDFLIP(i + dir, 1, newBit) = (flipCost > 0 ? 1 : 0);
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
    if (DP(ie, len, 0) < ans) {
      ans = DP(ie, len, 0);
      bestLen = len;
      bestBit = 0;
    }
  }

  // Check final runs for bit=1
  for (int len = minRun1; len <= maxLen1; len++) {
    if (DP(ie, len, 1) < ans) {
      ans = DP(ie, len, 1);
      bestLen = len;
      bestBit = 1;
    }
  }

  if (ans >= INF) {
    return -1;  // impossible
  } else {
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
    FROMLEN(ie, 0, 0) = bestLen;
    FROMBIT(ie, 0, 0) = bestBit;

    return ans;
  }
}

template <typename T>
DISPATCH_MACRO void reconstruct_solution(T *inSeq, int n, int minRun0,
                                         int minRun1, int allowBothFlips,
                                         int dir, int stride,
                                         /* store the DP arrays */
                                         T *fromLen, T *fromBit, T *usedFlip) {
  int is = dir > 0 ? 0 : n;
  int ie = dir > 0 ? n : 0;
  int shift = dir > 0 ? 0 : -1;

  // We stored the best final (len, bit) in fromLen[ie][0][0] /
  // fromBit[ie][0][0]
  auto bestLen = FROMLEN(ie, 0, 0);
  auto bestBit = FROMBIT(ie, 0, 0);

  if (bestLen < 0 || bestBit < 0) {
    // Means no solution or we never set it; do nothing
    return;
  }

  auto curI = ie;
  auto curLen = bestLen;
  auto curBit = bestBit;

  // Reconstruct backwards
  while (dir * (curI - is) > 0) {
    auto flip = USEDFLIP(curI, curLen, curBit);
    auto pLen = FROMLEN(curI, curLen, curBit);
    auto pBit = FROMBIT(curI, curLen, curBit);

    // We want to figure out what bit was actually chosen at index curI - 1
    // If flip=0 => candidateBit == originalBit
    // If flip=1 => candidateBit == 1 - originalBit (if allowBothFlips=1)
    // or candidateBit == 1 if originalBit=0 (if allowBothFlips=0)
    // But we have to be careful if both flips are allowed. Let's do:
    //   originalBit = s[curI - 1]
    //   if flip=0 => finalBit = originalBit
    //   if flip=1 => finalBit = 1 - originalBit (both flips allowed)
    //                      or = 1 if only 0->1 flips allowed
    T originalBit = INSEQ(curI - dir + shift);
    T finalBit;

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

    INSEQ(curI - dir + shift) = finalBit;

    // Move to predecessor
    curI -= dir;
    curLen = pLen;
    curBit = pBit;
  }
}

}  // namespace snap

#undef INF
#undef DP
#undef FROMLEN
#undef FROMBIT
#undef USEDFLIP
#undef INSEQ
#undef MAXRUN
