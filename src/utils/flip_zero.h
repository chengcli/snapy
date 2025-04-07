#pragma once

// snap
#include <configure.h>

namespace snap {

/*!
 * Resets dp[][][] arrays to INF, fromLen/fromBit/usedFlip to -1.
 * We also need to know the max possible "effective run length"
 * for 0 and for 1. Those are minRun0, minRun1, but for DP we
 * usually track "up to" that min (meaning once we exceed the required
 * length, we can store it as "already good" or ">= minRun").
 *
 * We'll do something like:
 *   - For bit=0, we track runLength in [0..minRun0] where
 *     runLength==minRun0 means ">= minRun0".
 *   - For bit=1, we track runLength in [0..minRun1] likewise.
 *
 * But the code is simpler if we keep them separate. We'll pass them in
 * as maxLen0, maxLen1:
 *   maxLen0 = minRun0
 *   maxLen1 = minRun1
 *
 * If the runs can be *very large*, to avoid huge memory,
 * we cap runLength at maxLen0 or maxLen1.
 * Example: if minRun0=2, once we have a run of length 2 (or more),
 * we store length=2 as ">=2".
 */
DISPATCH_MACRO void initialize_dp(int n, int *dp, int *fromLen, int *fromBit,
                                  int *usedFlip);

/*!
 * Helper: canFlip(originalBit, candidateBit, allowBothFlips)
 * Returns the flip cost (0 or 1) if allowed, or INF if not allowed.
 *
 * If allowBothFlips == 0 => only 0->1 flips
 *    -> that means:
 *       if originalBit=0 && candidateBit=0 => cost=0
 *       if originalBit=0 && candidateBit=1 => cost=1
 *       if originalBit=1 && candidateBit=1 => cost=0
 *       if originalBit=1 && candidateBit=0 => not allowed => INF
 *
 * If allowBothFlips == 1 => 0->1 or 1->0 allowed
 *    -> cost=1 if there's a flip, cost=0 if no flip
 */
DISPATCH_MACRO int can_flip(int originalBit, int candidateBit,
                            int allowBothFlips);

/*!
 * Function: computeMinFlips
 * - s: the input sequence
 * - n: length of s
 * - minRun0: required run length for 0
 * - minRun1: required run length for 1
 * - allowBothFlips: 0 => only 0->1, 1 => both 0->1 and 1->0
 * - outMinFlips: pointer to an int that will store the minimal #flips
 *                or -1 if impossible
 * This function fills the dp[][][] arrays and determines the minimal flips.
 */
DISPATCH_MACRO int compute_min_flips(const int *inSeq, int n, int minRun0,
                                     int minRun1, int allowBothFlips,
                                     int stride,
                                     /* store the DP arrays */
                                     int *dp, int *fromLen, int *fromBit,
                                     int *usedFlip);

/*!
 * Function: reconstructSolution
 * Reconstruct the fixed sequence of length n into outSeq[],
 * given that we've computed the DP and found the best final state
 * fromLen[n][0][0], fromBit[n][0][0].
 */
DISPATCH_MACRO void reconstruct_solution(int *outSeq, const int *inSeq, int n,
                                         int minRun0, int minRun1,
                                         int allowBothFlips, int stride,
                                         int *fromLen, int *fromBit,
                                         int *usedFlip);

}  // namespace snap
