#pragma once

// base
#include <configure.h>

namespace snap {

enum VicProfilePhase {
  kVicForwardSweep = 0,
  kVicInverse = 1,
  kVicBackwardSubstitution = 2,
  kVicProfileCount = 3,
};

#if defined(__CUDACC__)
extern __device__ unsigned long long g_vic_profile_cycles[kVicProfileCount];
extern __device__ unsigned long long g_vic_profile_calls[kVicProfileCount];
extern __device__ int g_vic_profile_enabled;

__host__ __device__ inline bool vic_profile_enabled() {
#if defined(__CUDA_ARCH__)
  return g_vic_profile_enabled != 0;
#else
  return false;
#endif
}

__host__ __device__ inline unsigned long long vic_clock() {
#if defined(__CUDA_ARCH__)
  return clock64();
#else
  return 0;
#endif
}

__host__ __device__ inline void vic_profile_add(int phase,
                                                unsigned long long cycles,
                                                unsigned long long calls = 0) {
#if defined(__CUDA_ARCH__)
  if (!vic_profile_enabled()) return;
  atomicAdd(&g_vic_profile_cycles[phase], cycles);
  if (calls > 0) atomicAdd(&g_vic_profile_calls[phase], calls);
#else
  (void)phase;
  (void)cycles;
  (void)calls;
#endif
}
#else
inline bool vic_profile_enabled() { return false; }
inline unsigned long long vic_clock() { return 0; }
inline void vic_profile_add(int, unsigned long long, unsigned long long = 0) {}
#endif

}  // namespace snap
