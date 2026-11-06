// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Parts of this file are adapted from ggml (llama.cpp), which is under the MIT License. That project's notice, which
// applies to those parts:
// MIT License
//
// Copyright (c) 2023-2024 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Parts of this file follow gguf-py (gguf/quants.py), which is under the MIT License. That project's notice:
// MIT License
//
// Copyright (c) 2023 Georgi Gerganov
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Converts a GGUF tensor (Q4_K, Q5_K, Q6_K, binary32, Q8_0, F16 or BF16) to the integers that the
// program cp_commit commits to ("fold" in identifiers; summary: weight_sums/README.md, "Which models,
// and the rule").
// Before scaling by 2^e, each floating-point step is in binary32, rounded to nearest even, never
// fused. A second implementation, prover/weights_blocks/fold_rule.h, must give the same integers.
// The rule of this conversion, and as far as we can tell only the rule, grew out of VerInf's: see fold_rule.h's head.
// Public source: the decoding in this file is adapted from ggml's public code: scale_min() from get_scale_min_k4. The
// byte layout of the block formats Q4_K, Q5_K, Q6_K and Q8_0, and the rule that decodes them, are those of ggml, the
// tensor library of the llama.cpp project (github.com/ggml-org/llama.cpp; MIT licence, "Copyright (c) 2023-2024 The
// ggml authors"; the notice is in NOTICE and THIRD_PARTY.md): ggml/src/ggml-common.h (block_q4_K, block_q5_K,
// block_q6_K) and ggml/src/ggml-quants.c (get_scale_min_k4; dequantize_row_q4_K, _q5_K, _q6_K); for Q8_0 see
// fold_rule.h, value8().
#pragma once
#include "weights_commitment/gguf_file.h"
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>

namespace fold {
// Bits that the kernels set in *fault. They never clear it, and they write their output regardless.
enum Fault : unsigned { NOT_FINITE = 1, OUT_OF_RANGE = 2, SMALL_GAIN = 4 };

__device__ __forceinline__ float half_at(const uint8_t *p) {
  return __half2float(__ushort_as_half((unsigned short)(p[0] | (p[1] << 8))));
}

// The j-th pair (scale, minimum; 6 bits each; j < 8) of the twelve scale bytes of Q4_K and Q5_K.
__device__ __forceinline__ void scale_min(const uint8_t *s, int j, int &sc, int &m) {
  if (j < 4) {
    sc = s[j] & 63;
    m = s[j + 4] & 63;
  } else {
    sc = (s[j + 4] & 15) | ((s[j - 4] >> 6) << 4);
    m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
  }
}

// Decodes element e (0 to 255) of the block b: Q5_K if `five` is set, else Q4_K. Bytes: binary16
// factors at 0 and 2, scale bytes at 4, q's low 4 bits at 16 (Q5_K: at 48, its fifth bits at 16).
// The products and the subtraction are __fmul_rn and __fsub_rn, so that each is rounded by itself.
// Public sources for the techniques of this routine.
// The idea is public: rounding-mode intrinsics are never fused: N. Whitehead and A. Fit-Florea,
// "Floating Point and IEEE 754 Compliance for NVIDIA GPUs", NVIDIA white paper, 2011, 4.3
// "Controlling Fused Multiply-add", https://docs.nvidia.com/cuda/floating-point/index.html.
// The idea is public: NVIDIA's CUDA Math API reference, "Single Precision Intrinsics", says so:
// https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__SINGLE.html.
__device__ __forceinline__ float value_q45(const uint8_t *b, int e, bool five) {
  const int j = e >> 5, i = e & 31, c = j >> 1;
  int sc, m;
  scale_min(b + 4, j, sc, m);
  const uint8_t byte = b[(five ? 48 : 16) + 32 * c + i];
  int q = (j & 1) ? byte >> 4 : byte & 15;
  if (five) q |= ((b[16 + i] >> j) & 1) << 4;
  const float ds = __fmul_rn(half_at(b), (float)sc), dm = __fmul_rn(half_at(b + 2), (float)m);
  return __fsub_rn(__fmul_rn(ds, (float)q), dm);
}

// Decodes element e of the Q6_K block b. Bytes: low 4 bits of q + 32 at 0, high 2 at 128, scales
// at 192, factor at 208. Each product is written as __fmul_rn. The idea is public: see value_q45().
__device__ __forceinline__ float value_q6(const uint8_t *b, int e) {
  const int h = e >> 7, t = (e >> 5) & 3, c = e & 31;
  const int lo = (b[64 * h + 32 * (t & 1) + c] >> (4 * (t >> 1))) & 15, hi = (b[128 + 32 * h + c] >> (2 * t)) & 3;
  const int q = (lo | (hi << 4)) - 32, s = (int8_t)b[192 + (e >> 4)];
  return __fmul_rn(__fmul_rn(half_at(b + 208), (float)s), (float)q);
}
// Any type but the six below is read as binary32; `bytes` must then be 4-byte aligned.
__device__ __forceinline__ float value(const uint8_t *bytes, uint32_t type, uint64_t i) {
  switch (type) {
  case gguf::Q4_K:
    return value_q45(bytes + i / 256 * 144, (int)(i % 256), false);
  case gguf::Q5_K:
    return value_q45(bytes + i / 256 * 176, (int)(i % 256), true);
  case gguf::Q6_K:
    return value_q6(bytes + i / 256 * 210, (int)(i % 256));
  case gguf::Q8_0: { // a block of 34 bytes: d (binary16), 32 signed codes q; v = RN(d q). Layout, origin: fr::value8()
    const uint8_t *b = bytes + i / 32 * 34;
    return __fmul_rn(half_at(b), (float)(int)(int8_t)b[2 + i % 32]);
  }
  case gguf::F16: // F16 and BF16: layout and origin: fr::value16()
    return half_at(bytes + 2 * i);
  case gguf::BF16:
    return __uint_as_float((uint32_t)(bytes[2 * i] | (bytes[2 * i + 1] << 8)) << 16);
  default:
    return ((const float *)bytes)[i];
  }
}

// gamma[j] = g[j] * c for j < n. g: an RMS norm's weights ("gains"); c_word: c's binary32 bits.
// Public sources for the techniques of this routine.
// The product is written as __fmul_rn. The idea is public: see value_q45(). The intrinsic
// __uint_as_float is public: NVIDIA's CUDA Math API reference, "Type Casting Intrinsics",
// https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__CAST.html.
// The loop form is public: the grid-stride loop:
// "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops", a post on NVIDIA's developer blog,
// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
__global__ void gains(const float *g, uint32_t c_word, uint64_t n, float *gamma, unsigned *fault) {
  for (uint64_t j = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; j < n; j += (uint64_t)gridDim.x * blockDim.x) {
    const float x = __fmul_rn(g[j], __uint_as_float(c_word));
    if (!isfinite(g[j]) || !isfinite(x)) atomicOr(fault, NOT_FINITE);
    if (x != 0.f && fabsf(x) < 7.8886090522101181e-31f /* 2^-100 */) atomicOr(fault, SMALL_GAIN);
    gamma[j] = x;
  }
}

// w[i] is the integer for the tensor's element first + i, i < n: the decoded element, times
// gamma[(first + i) % row] unless gamma is null, times 2^e (exact in binary64 if |e| <= 896),
// rounded to the nearest integer, ties to even. `row` is the length of the tensor's rows. `bytes`
// points at element `first`, which in a block format must begin a block. `limit` must fit in Int.
// Public sources for the techniques of this routine.
// The product is written as __fmul_rn. The idea is public: see value_q45().
// The loop form is public: the grid-stride loop: the post on grid-stride loops, cited at gains().
template <class Int>
__global__ void integers(const uint8_t *bytes, uint32_t type, uint64_t first, uint64_t n, uint64_t row,
                         const float *gamma, int e, long long limit, Int *w, unsigned *fault) {
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
    const float v = value(bytes, type, i);
    const float p0 = gamma ? __fmul_rn(gamma[(first + i) % row], v) : v;
    if (!isfinite(v) || !isfinite(p0)) atomicOr(fault, NOT_FINITE);
    const long long x = __double2ll_rn(ldexp((double)p0, e));
    if (x > limit || x < -limit) atomicOr(fault, OUT_OF_RANGE);
    w[i] = (Int)x;
  }
}
// c_0, the c of gains(), as a binary32 bit pattern: FlashProve's general rule (scales/README.md,
// "The constants of a norm"). With t the largest integer with 4^t <= d, c = sqrt(d) / 2^t (so
// 1 <= c < 2); c_0 = RN32(c), and for a query matrix c_q = RN32(c / sqrt(dh)) carries attention's
// scale 1 / sqrt(dh). At d 5,120 dh 128 these are 0x3F8F1BBD and 0x3DCA62C2, at d 2,048 dh 64
// 0x3FB504F3 and 0x3E3504F3 (docs/tinyllama/ARITHMETIC_CONTRACT.md 4.3). This generalizes the
// former literals, which were right for d 5,120 and dh 128 only.
inline uint32_t fold_c_word(uint64_t d, uint64_t dh, bool query) {
  int t = 0;
  while ((uint64_t)1 << (2 * (t + 1)) <= d) ++t; // 4^(t+1) <= d
  const double c = std::sqrt((double)d) / (double)((uint64_t)1 << t);
  const float f = query ? (float)(c / std::sqrt((double)dh)) : (float)c;
  uint32_t w;
  std::memcpy(&w, &f, sizeof w);
  return w;
}
// The norm whose gains multiply a tensor ("": none), and c_0 (c_q for attn_q), for a model of this
// width d and head dimension dh.
struct Reads {
  const char *norm;
  uint32_t c_word;
};
inline Reads reads(const std::string &kind, uint64_t d, uint64_t dh) {
  if (kind == "attn_q") return {"attn_norm", fold_c_word(d, dh, true)};
  if (kind == "attn_k" || kind == "attn_v") return {"attn_norm", fold_c_word(d, dh, false)};
  for (const char *k :
       {"ffn_gate", "ffn_up", "ffn_gate_inp", "ffn_gate_shexp", "ffn_up_shexp", "ffn_gate_exps", "ffn_up_exps"})
    if (kind == k) return {"ffn_norm", fold_c_word(d, dh, false)};
  if (kind == "output") return {"output_norm", fold_c_word(d, dh, false)};
  return {"", 0};
}

} // namespace fold
