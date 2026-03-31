// Copyright © 2023 Apple Inc.
// Patched by Localmee: void casts silence unused variable warnings on 5-core
// GPU variants (iPad mini A17 Pro). Without this, Metal JIT treats warnings
// as errors, invalidates shader cache, and triggers infinite recompilation.
// Upstream issue: Metal kernel caching regression on binned A17 Pro chips.
// Safe to remove when ml-explore/mlx fixes this upstream.

#pragma once

#if defined __METAL__ || defined MLX_METAL_JIT
#define MTL_CONST constant
#else
#define MTL_CONST
#endif

static MTL_CONST constexpr int MAX_REDUCE_SPECIALIZED_DIMS = 4;
static MTL_CONST constexpr int REDUCE_N_READS = 4;
static MTL_CONST constexpr int REDUCE_N_WRITES = 4;
static MTL_CONST constexpr int SOFTMAX_N_READS = 4;
static MTL_CONST constexpr int RMS_N_READS = 4;
static MTL_CONST constexpr int RMS_LOOPED_LIMIT = 4096;

// Silence "unused variable" warnings on GPU variants where these constants
// are not referenced by the active kernel specialization.
// Without this, Metal JIT compiler logs warnings → "Errors found! Invalidating cache..."
// → infinite recompile loop → CPU 213% → freeze (iPad mini A17 Pro 5-core GPU).
#ifdef __METAL__
namespace _mlx_silence_warnings {
    constant constexpr auto _0 [[maybe_unused]] = MAX_REDUCE_SPECIALIZED_DIMS;
    constant constexpr auto _1 [[maybe_unused]] = REDUCE_N_READS;
    constant constexpr auto _2 [[maybe_unused]] = REDUCE_N_WRITES;
    constant constexpr auto _3 [[maybe_unused]] = SOFTMAX_N_READS;
    constant constexpr auto _4 [[maybe_unused]] = RMS_N_READS;
    constant constexpr auto _5 [[maybe_unused]] = RMS_LOOPED_LIMIT;
}
#endif

// Instantiate a templated kernel.
// Extra args are used as template parameters:
// e.g. instantiate_kernel(binary_int, binary, a, b) ->
// [[host_name(binary_int)]] [kernel] binary<a, b>
#define instantiate_kernel(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;
