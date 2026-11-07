// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// AOutput computes the attention output o of one layer, by a_forward() (forward.cuh) on a one-layer
// plan. A sub-proof (`FORK` in identifiers) is the argument for one part of the model. The layer
// sub-proof's forward pass needs o before the attention sub-proof runs; that sub-proof later
// computes the same o from the same q, k and v.
#pragma once
#include "common/end_name.h" // name_the_end (M5g)
#include "attn/forward.cuh"
#include "hiding/place.cuh"
// o[t][h * dh + c] = oh[h][t][c]. 8 must divide dh, and o and oh must be 16-byte aligned.
// The idea is public: vectorized memory access through cast pointers: a post on NVIDIA's developer
// blog, "CUDA Pro Tip: Increase Performance with Vectorized Memory Access",
// https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/.
// The post's elements are of type int.
__global__ void k_attn_output_turn(const i16 *__restrict__ oh, int T, int H, int dh, i16 *__restrict__ o) {
  const size_t n = (size_t)T * H * dh / 8;
  GSA(x, n) {
    const size_t e = x * 8;
    const int c = (int)(e % dh);
    const size_t r = e / dh;
    const int h = (int)(r % H);
    const size_t t = r / H;
    *(uint4 *)(o + e) = *(const uint4 *)(oh + ((size_t)h * T + t) * dh + c);
  }
}
struct AOutput {
  APlan P;
  size_t bytes = 0; // device memory taken by init()

  // Call init() once, before run(). AOutput does not free its buffers. The arguments are the fields
  // of AP (plan.cuh) with the same names; Sword is the parameter file's line `S` (AP::S before
  // build()). Prints a message and exits with code 2 if APlan::build() or the forward pass does not
  // support them.
  void init(int lt, int lB, int H, int lgk, int ldh, int Sword, int rx, int kFM, int kFD) {
    AP &p = P.p;
    p.lt = lt;
    p.lB = lB;
    p.H = H;
    p.lgk = lgk;
    p.ldh = ldh;
    p.ll = 0;
    p.S = Sword;
    p.rx = rx;
    p.kFM = kFM;
    p.kFD = kFD;
    // a_forward() reads none of the fields set below; their values only need to pass build().
    // a_forward_ok() bounds a sum that grows with the shift, so P.sh is the largest shift, 12.
    // Therefore build() also exits if lgk + lt + lB < lmS = 11, and it tests kFM, kFD and the
    // exponential table's length at shift 12, not at the layers' shifts.
    p.D = 2048;
    p.lmS = 11;
    p.lwH = p.lwO = p.lwQ = 5;
    p.la = lt;
    p.lmb = 11;
    p.iota = 54;
    p.nmin = 20;
    P.form = 1;
    P.sh = {12};
    P.build();
    if (!a_forward_ok(P) || p.HA > 16) {
      printf("attention: the forward pass supports 32, 64 or 128 channels per head, 64 to 32,768 tokens, tiles of at "
             "least 16 tokens and at most 16 query heads per key-value head; the requested shape is outside these "
             "limits\n");
      name_the_end("attn-output-shape"); // serving: p258 end 1 attn-output-shape (M5i)
      exit(2);
    }
    const size_t nr = (size_t)H * p.T;
    // Buffers that only run() uses go to hz::FORK: other code may reuse their memory between runs.
    // The others go to hz::OWN.
    mo = al<int>(nr, hz::FORK);
    io = al<int>(nr, hz::FORK);
    Zo = al<int>(nr, hz::FORK);
    oh = al<i16>(nr * p.dh, hz::FORK);
    rem = al<u32>(nr * p.dh, hz::FORK);
    err = al<u32>(4);
    ws = al<u8>(a_forward_scratch(p), hz::FORK);
    shAll = al<int>(16);
    Tt = al<int>(P.Tt.size());
    CK(cudaMemcpy(Tt, P.Tt.data(), P.Tt.size() * 4, cudaMemcpyHostToDevice));
    std::vector<int> s16(16);
    for (int i = 0; i < 16; i++) s16[i] = i;
    CK(cudaMemcpy(shAll, s16.data(), 64, cudaMemcpyHostToDevice));
    CK(cudaMemset(err, 0, 16));
  }
  // q and o are [t][H * dh]. k and v are [t][G * dh], where G is the number of key-value heads.
  // All four must be 16-byte aligned. sh is the layer's attention shift (sh in `KL.bin`), 4 to 12;
  // it is not checked here. a_forward() reads it from shAll[sh] in device memory.
  // Does not wait for the device. Runs on different streams must not overlap: they share
  // AOutput's buffers.
  void run(const i16 *q, const i16 *k, const i16 *v, int sh, i16 *o, cudaStream_t st = 0) {
    const AP &p = P.p;
    const AOff none{}; // keys = false: only oh is needed, and neither cells nor keys are written
    a_forward(P, shAll + sh, q, k, v, Tt, mo, io, Zo, oh, rem, nullptr, none, none, nullptr, nullptr, nullptr, nullptr,
              err, ws, st, false);
    k_attn_output_turn<<<nbk((size_t)p.T * p.dq / 8, 256, 4096), 256, 0, st>>>(oh, p.T, p.H, p.dh, o);
  }
  // Returns non-zero if, in any run() since the error_words() buffer was last cleared, a division
  // remainder may not fit in its AP::wd bytes (2 Z - 1 >= 2^(8 wd), Z the softmax denominator).
  // init() clears the buffer. Waits only for the calling thread's default stream
  // (nvcc --default-stream per-thread): wait first for any other stream given to run().
  u32 errors() {
    u32 e = 0;
    CK(cudaMemcpy(&e, err, 4, cudaMemcpyDeviceToHost));
    return e;
  }
  // Returns the 16-byte device buffer whose first word errors() reads. A caller that proves several
  // requests must clear it when each begins and, if it interleaves them, save and restore it with
  // each request's state.
  u32 *error_words() const { return err; }

private:
  int *mo = nullptr, *io = nullptr, *Zo = nullptr, *shAll = nullptr, *Tt = nullptr;
  i16 *oh = nullptr;
  u32 *rem = nullptr, *err = nullptr;
  void *ws = nullptr;
  template <class X> X *al(size_t n, hz::Place place = hz::OWN) {
    X *p_ = nullptr;
    const size_t b = (n * sizeof(X) + 255) & ~(size_t)255;
    hz::alloc(p_, b, place);
    bytes += b;
    return p_;
  }
};
