// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Gram passes: one pass over the two vectors of a product layer gives the messages of three
// sumcheck rounds, in place of one pass per round. Its sums are computed as products of byte
// matrices by cublasLtMatmul (cuBLASLt, NVIDIA's library of matrix products).
//
// For the whole file: set_up() must run before the first product(). The sizes, bounds, alignments
// and launch shapes that the comments below require of a caller are not checked in this file. A
// cuBLASLt or CUDA call that fails (LT_CK, CK) prints a line, and the process exits with code 1. In
// the production build of cp_prove the line is not seen during a proof: its output is closed from
// just before the first secret is read (prover/prover/closed_front.h). In cp_commit, and in the
// unit of cp_prove that owns the weight table, CK has another form, which calls exit(5); in
// cp_prove the exit handler then chooses the code (prover/weights_commitment/common.cuh); by a
// search no code of these two calls this file.
//
// A product layer of a lookup argument reduces a claim on a vector u = L R (entry by entry) to
// claims on L and R, its left and right side, by a sumcheck on the sum over x of
// eq(rho, x) L(x) R(x) for a point rho, where eq is the multilinear equality polynomial.
// Three bits of x cut each vector into 8 equal parts, the slabs L_0 .. L_7 and R_0 .. R_7. A pass
// computes the 8 x 8 matrix
//   G[a][a'] = sum over c of eq(c) * L_a(c) * R_a'(c),   a, a' < 8,
// where the index c is the other bits of x, and eq(c) is their factor of eq(rho, x). From G,
// k_g2msgT (prover/hiding/product_test.cuh) computes the messages of the three rounds that bind
// the three bits. Public source for several rounds from one pass: [Gruen].
//
// An entry, a slab's element at one index, is in K3 = F_p[X] / (X^3 - 2), p = 2^64 - 2^32 + 1, the
// field E of docs/format/FORMAT.md, R1.4, and has three 64-bit coordinates. The gf_
// functions and gram_times_2_to() work modulo this p; gf_times_nonresidue(v) is 2 v.
// The blk::lazy:: functions are those of prover/field/lazy_gl64.cuh.
//
// The weights. The callers cut the indices into chunks of at most 2^15; nothing here checks that.
// eq(c) = eqlo[c mod chunk] * eqhi[c / chunk], for two tables that the callers fill.
//
// Method. Each coordinate is cut into 8 signed bytes (digits), so an entry has 24 digits. A kernel
// writes each vector's digits to a byte matrix (a digit matrix) with one column per index; the
// entries of L are multiplied by eqlo first. cublasLtMatmul multiplies the two matrices, with the
// indices as the summed dimension, into 32-bit sums, one for each pair of a row of the first matrix
// and a row of the second. A second kernel (k_join) multiplies each sum by its power of 256, adds
// the results, reduces them modulo p and X^3 - 2, and multiplies by eqhi. A chunk is cut into
// pieces, and one sum covers one piece. It cannot overflow, because 2^15 * 128^2 < 2^31.
//
// Layers. A stored layer has its entries in GPU memory. The other layers are on the key side of a
// lookup call, the product tree over the call's keys; their entries are computed from the keys and
// are not stored. (In a lookup call every key of its lists is to occur in a public table.) A key
// is 16 bits, the number k of a row of that table; its value is the row value tau(k), and alpha is
// the call's challenge.
// Layer A is the last layer of the key side for a range table, where tau(k) = k. Its entries are
// alpha - k. Layer F is the same layer for a function table with the tau of struct Tau. Both begin
// with three passes (pass 1, 2 and 3), which bind bits 2 1 0, 5 4 3 and 8 7 6 of x. In pass 1 the
// second matrix holds unsigned bytes minus 128 (see WIDE). k_fold64 stores pass 3's folded entries.
//
// This file includes library headers only. prover/hiding/product_test.cuh includes it after the
// headers that define, among others:
// - T (= K3) and F (= FE, the arithmetic of K3): prover/hiding/transcript.cuh;
// - FE (= FK2) and M3L, a factor prepared for many products (its coordinates k0, k1, k2 with 2 k1
//   and 2 k2): prover/hiding/pt_field.h;
// - Chal, a layer's point rho and its round challenges: prover/hiding/pt_transcript.h.
// Host code that launches the kernels is in four files of prover/hiding/: gram_engine.inc,
// gram_keys.inc, forest.cuh, forest_keys.cuh.
//
// Public sources. Each is cited by its tag in square brackets where its idea is used.
// [TensorFHE] S. Fan, Z. Wang, W. Xu, R. Hou, D. Meng, M. Zhang, "TensorFHE: Achieving Practical
//   Computation on Encrypted Data Using GPGPU", arXiv:2212.14191, section IV-C: "For each 32-bit
//   integer element m_{i,j} in the matrix M, we segment every consecutive 8-bits into an 8-bit
//   integer"; "TCU output is restricted as s32"; the outputs are fused into one matrix, as partial
//   accumulations; "Then, the fused matrix is moduloed element-wise by q". They cut 32-bit residues
//   into 4 bytes; this file cuts 64-bit coordinates into 8.
// [Ozaki-int8] numbers cut into slices, multiplied by cublasGemmEx: H. Ootomo, K. Ozaki, R. Yokota,
//   "DGEMM on Integer Matrix Multiplication Unit", arXiv:2306.11975, 4.1 (Algorithm 3).
// [Multiword] Berthomieu, Graillat, Lesnoff, Mary, "Multiword matrix multiplication over large
//   finite fields in floating-point arithmetic", arXiv:2601.07508, section 3.2, before Algorithm
//   3.3: "This variant concatenates the matrices B_j in order to compute the products A_i B_j ...
//   as a single contiguous matrix product". They stack the words of one side or of the other, in
//   floating point; this file stacks the digits of both. For that no public source was found
//   (searched: public papers; public code was not searched). For integers cut into single bits the
//   idea is public: [Ma], [ABQ-LLM]. For floating-point numbers cut into pieces it is public too,
//   though not all products are computed: [Parikh].
// [Ma] the bits of both operands stacked, so that one product holds the sums for all pairs of bits:
//   S. Ma, C. Fang, H. Shao, Z. Wang, "Efficient Arbitrary Precision Acceleration for Large
//   Language Models on GPU Tensor Cores", ASPDAC '25, arXiv:2409.17870, sections 3.2 and 4.2. There
//   the digits are single bits: 1-bit matrices made beforehand (section 4.1) and concatenated in
//   shared memory within the paper's kernel (section 4.2); here the digits are signed bytes of
//   field elements, and a library multiplies the two matrices.
// [ABQ-LLM] C. Zeng, S. Liu, Y. Xie, H. Liu, X. Wang, M. Wei, S. Yang, F. Chen, X. Mei, "ABQ-LLM:
//   Arbitrary-Bit Quantized Inference Acceleration for Large Language Models", arXiv:2408.08554,
//   section 3.4. For p- and q-bit operands and a product of shape M x N x K, "the actual computing
//   task undertaken by ABQKernel is" p*M x q*N x K; the parts of the output are scaled and added
//   ("Bit Reduction"). There the digits are single bits; here, signed bytes of field elements.
// [Parikh] D. N. Parikh, R. A. van de Geijn, G. M. Henry, "Cascading GEMM: High Precision from Low
//   Precision", arXiv:2303.04353, sections 3.3, 3.4, 4.3 and 5.1. Section 3.3, for vectors x and y
//   of FP64x2 numbers: "Viewing splits of x as rows of a" 4 x k "matrix and splits of y and related
//   vectors as columns in a" k x 7 matrix, the paper writes the dot product as the product of these
//   two matrices, "followed by the summation of the elements or the appropriate contributions into
//   bins". Section 3.4 treats two whole matrices of FP64x2 numbers in the same way. Sections 6.2.4
//   and 6.2.5 name INT8 splits for the same method as work for the future: they count the splits
//   and the products ("at least 21 multiplies"; "about 105 to 120 INT8 multiplies") and do not say
//   how the splits are arranged. There the splits are floating-point numbers, the routine is the
//   matrix product (GEMM) of the BLIS library with its packing and two of its loops changed (its
//   micro-kernels "can be reused as-is"), and ten of the 28 products are computed; here the digits
//   are signed bytes of field elements, and a library multiplies the two matrices whole.
// [FFLAS] J-G. Dumas, P. Giorgi, C. Pernet, "Dense Linear Algebra over Word-Size Prime Fields: the
//   FFLAS and FFPACK packages", arXiv:cs/0601133, 2.1.2: "Divisions intervene either just before an
//   overflow occurs within the hardware data, or only after a fixed numbers of accumulations."
//   Their bound is on products of residues below p; here it is on products of byte digits.
// [Gruen] Gruen, "Some Improvements for the PIOP for ZeroCheck", ePrint 2024/108, section 5: one
//   pass, several rounds.
// [Signed digits] drouyang, "Signed Bucket Indexes for Multi-Scalar Multiplication",
//   https://hackmd.io/@drouyang/signed-bucket-index: "whenever s_i >= L/2, we replace it with
//   s_i - L and add 1 to the next slice s_{i+1}". Here L = 256. The note uses the recoding for sums
//   of elliptic-curve points.
// [cuBLAS] NVIDIA, cuBLAS documentation, https://docs.nvidia.com/cuda/cublas/: cublasLtMatmul()
//   (for CUBLAS_COMPUTE_32I the operands are CUDA_R_8I; each matrix address must be a multiple of
//   4, and a multiple of 16 is faster; 4 must also divide A's, B's and C's leading dimensions;
//   no format but 'TN' is accepted); 3.1.2 advises one call of cublasLtMatmulAlgoGetHeuristic(),
//   whose result is reused; CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: 0 unless set (no workspace);
//   CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET.
// [oneDNN] oneDNN Developer Guide, "Nuances of int8 Computations", section "2. Inputs of the same
//   type: s8": "at run-time, it adds 128 to one of the s8 input to make it of type u8 instead. Once
//   the result is computed, oneDNN subtracts the extra value it added ... This subtracted value
//   sometimes referred as a compensation." Their instructions multiply an unsigned byte by a signed
//   one. Here the shift goes the other way: 128 is subtracted from unsigned bytes, to make them
//   signed, and the compensation is added. The quoted words are also at
//   https://oneapi-src.github.io/oneDNN/v2/dev_guide_int8_computations.html.
// [gemmlowp] gemmlowp, doc/low-precision.md (in gemmlowp's repository): "we can compute the second
//   term, lhs_offset * P * rhs, by summing each column of rhs ... This is just a rank one update of
//   the result". Also B. Jacob et al., "Quantization and Training of Neural Networks for Efficient
//   Integer-Arithmetic-Only Inference", arXiv:1712.05877, section 2.3, equations (7) and (8) (here
//   the zero-point Z_2 is -128), and Appendix B (uint8 to int8 by subtracting 128). They lower both
//   factors and add up the row sums outside the product.
// [Dakkak] sums by a product with a row of ones: A. Dakkak, C. Li, J. Xiong, I. Gelado, W. Hwu,
//   "Accelerating Reduction and Scan Using Tensor Core Units", ICS 2019, arXiv:1811.09736,
//   section 4 and Figure 5. Their matrix P holds nothing but that row, and the product is made for
//   the sums alone; here the row of ones is one more row of a matrix that holds data.
// [CUDA guide] values passed between the threads of a warp: NVIDIA, CUDA C++ Programming Guide,
//   https://docs.nvidia.com/cuda/cuda-c-programming-guide/: "Warp Shuffle Functions". See also:
//   NVIDIA, CUDA Programming Guide, "C/C++ Language Extensions", "Warp Shuffle Functions",
//   https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html.
// [CUTLASS] NVIDIA, CUTLASS documentation, "Efficient GEMM in CUDA",
//   https://docs.nvidia.com/cutlass/media/docs/cpp/efficient_gemm.html: K is cut into parts; one
//   kernel multiplies each part and a second reduces the results (split K, parallel reduction).
// [Harris] M. Harris, "Optimizing Parallel Reduction in CUDA", NVIDIA Developer Technology: sums of
//   a thread block through shared memory.
// [Grid-stride] a grid-stride loop: "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops",
//   NVIDIA Technical Blog,
//   https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
// [Best practices] __launch_bounds__(maxThreadsPerBlock), an upper limit for the block size:
//   NVIDIA, CUDA C++ Best Practices Guide,
//   https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/, "Occupancy".
// [Milakov] #pragma unroll lets registers hold a thread's short arrays: M. Milakov, "Fast Dynamic
//   Indexing of Private Arrays in CUDA", NVIDIA Technical Blog, 2015, section "Static indexing",
//   https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/.
// [Kepler guide] __ldg() for data that nothing writes while the kernel runs: NVIDIA, Kepler Tuning
//   Guide, https://docs.nvidia.com/cuda/kepler-tuning-guide/, "Read-Only Data Cache". The guide
//   says this of GK110, a variant of the Kepler architecture.
// [Harvey] values that depend on a fixed factor alone, precomputed in a table: D. Harvey, "Faster
//   arithmetic for number-theoretic transforms", arXiv:1205.2926, https://arxiv.org/pdf/1205.2926,
//   sections 2 and 5. There the value is W' = floor(W beta / p), where p is a word-sized prime and
//   beta describes the machine word size. The paper gives this form as that of Shoup's library NTL
//   ("NTL butterfly implementation"). M3L, the prepared form that this file reads, holds no such
//   quotient: see [Poly1305] and [NEON Crypto].
// [Poly1305] a fixed factor stored with multiples of its pieces: D. J. Bernstein, "The Poly1305-AES
//   message-authentication code", 2005.03.29, https://cr.yp.to/mac/poly1305-20050329.pdf,
//   section 4. There the multiplier, 5 * 2^-130, comes from the integer modulus 2^130 - 5; the
//   factor 2 of M3L comes from the polynomial modulus X^3 - 2.
// [NEON Crypto] the multiples 5g1, 5g2, 5g3, 5g4 of a reused factor, precomputed: D. J. Bernstein,
//   P. Schwabe, "NEON Crypto", CHES 2012, LNCS 7428, pp. 320-339,
//   https://link.springer.com/content/pdf/10.1007/978-3-642-33027-8_19.pdf, section 4, on a product
//   modulo 2^130 - 5.
// [Prefetching] the block-stride loop (i starts at threadIdx.x and grows by blockDim.x); unrolling
//   a long loop in part: R. Van der Wijngaart, F. Oh, "Boosting Application Performance with GPU
//   Memory Prefetching", NVIDIA Technical Blog, Mar 23, 2022,
//   https://developer.nvidia.com/blog/boosting-application-performance-with-gpu-memory-prefetching/
#pragma once
#include "common/end_name.h"
#include <cublasLt.h>
#include <map>
#include <mutex>
#include <tuple>
namespace grm { // Gram passes: the code that computes G, the slabs' weighted inner products
typedef signed char i8;
constexpr int ROWS = 192, SUMS = ROWS * ROWS; // the rows of a digit matrix (put_digits); the 32-bit sums of one piece
// Rewrites x in signed digits, because cublasLtMatmul multiplies signed bytes only: byte m of the
// result is d_m, where x = sum_m d_m 256^m with -128 <= d_m <= 127. A digit of 128 or more is
// lowered by 256 and carries 1 into the next. Adding 128 to every byte at once produces exactly
// these carries, and flipping each byte's top bit subtracts the 128 again. x is any 64-bit word.
// Above 0x7F7F7F7F7F7F7F7F, eight digits' largest value, it gets the digits of x - p, the same
// residue (blk::lazy::signed_digits(), prover/field/lazy_gl64.cuh).
// Public source: [cuBLAS]. Compare: [Signed digits]. The idea is public: making all signed digits
// at once by one addition of half the base at each digit position is the TFHE library's gadget
// decomposition for base 2^k (src/libtfhe/tgsw-functions.cpp, tgsw.cpp; Apache 2.0; there for the
// top l digits, modulo 2^32), and Knuth, The Art of Computer Programming, vol. 2, section 4.1, for
// balanced ternary. H. S. Warren, "Hacker's Delight", chapter 2, does not state it. This is the
// idea's base-256 case over all eight bytes, exact while the addition does not overflow 64 bits;
// the subtraction of 128 from each byte is one xor (Wikipedia, "Offset binary", for a single byte).
__device__ __forceinline__ u64 signed_digits(u64 x) {
  return blk::lazy::signed_digits(blk::lazy::Loose{x});
}
// For m < 4, w[m] holds byte m of a[0], a[1], a[2], a[3], in this order from its lowest byte.
// The idea is public: NVIDIA, CUDA Math API Reference Manual, "Integer Intrinsics", __byte_perm().
// The same eight __byte_perm calls, with the same operands and selectors, are in MegEngine,
// dnn/src/cuda/memory_utils.cuh, transpose_int8_4x4_impl (Copyright (c) 2014-2021 Megvii Inc.; Apache License 2.0; see
// NOTICE); there the second and the third call stand in the other order.
__device__ __forceinline__ void transpose_bytes(const u32 (&a)[4], u32 *w) {
  const u32 t0 = __byte_perm(a[0], a[1], 0x5140), t1 = __byte_perm(a[2], a[3], 0x5140),
            t2 = __byte_perm(a[0], a[1], 0x7362), t3 = __byte_perm(a[2], a[3], 0x7362);
  w[0] = __byte_perm(t0, t1, 0x5410), w[1] = __byte_perm(t0, t1, 0x7632);
  w[2] = __byte_perm(t2, t3, 0x5410), w[3] = __byte_perm(t2, t3, 0x7632);
}
// w[m] = signed digit m of x[0 .. 3], one coordinate of four entries, packed into one 32-bit word.
// The compiler hint is public: #pragma unroll [Milakov].
__device__ __forceinline__ void digit_words(const u64 (&x)[4], u32 (&w)[8]) {
  u32 lo[4], hi[4];
#pragma unroll
  for (int k = 0; k < 4; k++) {
    const u64 z = signed_digits(x[k]);
    lo[k] = (u32)z, hi[k] = (u32)(z >> 32);
  }
  transpose_bytes(lo, w);
  transpose_bytes(hi, w + 4);
}
// Writes the digits of four entries e[0 .. 3], which are indices c .. c + 3 of slab a, to a digit
// matrix whose rows are ld bytes apart. Row a * 24 + 8 j + m holds signed digit m of coordinate j.
// `at` points to column c of row 0; `at` and ld must be multiples of 4, which is not checked here.
// k_digits_1 also calls it with a = 8: rows 192 .. 215, the digits of the weights eqlo.
// Public sources: [TensorFHE], [Ozaki-int8]. The matrices of both sides of a product stack digits
// as rows: see [Multiword], at the top of this file.
// The compiler hint is public: #pragma unroll [Milakov].
__device__ __forceinline__ void put_digits(i8 *at, size_t ld, int a, const K3 *e) {
#pragma unroll
  for (int j = 0; j < 3; j++) {
    const u64 x[4] = {j == 0 ? e[0].k0 : j == 1 ? e[0].k1 : e[0].k2, j == 0 ? e[1].k0 : j == 1 ? e[1].k1 : e[1].k2,
                      j == 0 ? e[2].k0 : j == 1 ? e[2].k1 : e[2].k2, j == 0 ? e[3].k0 : j == 1 ? e[3].k1 : e[3].k2};
    u32 w[8];
    digit_words(x, w);
#pragma unroll
    for (int m = 0; m < 8; m++) *reinterpret_cast<u32 *>(at + (size_t)(a * 24 + 8 * j + m) * ld) = w[m];
  }
}
// Stored layer, with slabs `stride` entries apart. Writes the digits of n indices, starting at
// index c0 (a multiple of chunk): A receives the digits of eqlo * L, B those of R. n and chunk must
// be multiples of 4. Like the other k_digits kernels and k_fold64, it needs blocks of 256 threads.
// Public sources: [TensorFHE], [Ozaki-int8]; for the form of the loop, [Grid-stride].
// A and B both hold stacked digits: see [Multiword], at the top of this file.
// - The weights eqlo are read from a table in the prepared form M3L. The idea is public: [Harvey],
//   [Poly1305], [NEON Crypto].
// - The compiler hints are public: __launch_bounds__ [Best practices]; #pragma unroll [Milakov].
__global__ void __launch_bounds__(256)
    k_digits(const K3 *__restrict__ L, const K3 *__restrict__ R, size_t stride, size_t c0, int n, int chunk,
             const M3L *__restrict__ eqlo, i8 *__restrict__ A, i8 *__restrict__ B, size_t ld) {
  const int a = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int c = blockIdx.x * 128 + lane * 4; c < n; c += gridDim.x * 128) {
    const K3 *l = L + a * stride + c0 + c, *r = R + a * stride + c0 + c;
    const M3L *w = eqlo + c % chunk;
    K3 e[4];
#pragma unroll
    for (int k = 0; k < 4; k++) e[k] = FK2::mulp(w[k], l[k]);
    put_digits(A + c, ld, a, e);
#pragma unroll
    for (int k = 0; k < 4; k++) e[k] = r[k];
    put_digits(B + c, ld, a, e);
  }
}
// 2^TURNS = 1 modulo p. gram_times_2_to() and signed_turn() want an exponent s of 2 with 0 <= s < TURNS: some
// callers take it modulo TURNS first, the others have it below TURNS by construction (by one
// reading); nothing checks it.
constexpr int TURNS = 192;
// Returns x * 2^s mod p for a canonical x and 0 <= s < TURNS: shifts and one reduction (2^96 = -1).
__device__ __forceinline__ u64 gram_times_2_to(u64 x, int s) {
  return blk::lazy::times_2_to(blk::F{x}, s).v;
}
// Returns v * 2^s mod p for |v| < 2^lazy::JOIN_BITS and 0 <= s < TURNS: lazy::join_digit.
__device__ __forceinline__ u64 signed_turn(long long v, int s) {
  return blk::lazy::join_digit(v, s).v;
}
// Combines the 32-bit sums into one field element per chunk and pair (a, a'), numbered 8 a + a'.
// For each chunk, C holds `pieces` matrices of ROWS x ROWS sums, column-major: the sum for row r of
// A and row q of B is at q * ROWS + r. Each of a pair's 576 sums is added over the pieces and
// multiplied by 2^(8 (m + m')) X^(j + j'), for digits m and m' of coordinates j and j', with
// X^3 = 2 (gf_times_nonresidue() multiplies by 2). Gc[chunk * 64 + pair] = the element times eqhi[chunk].
// Public sources: [TensorFHE] for putting the sums together; [CUTLASS] (split K) for adding the
// pieces' partial sums in a second kernel; [CUDA guide] for the shuffles; [Harris] for the sums
// through shared memory. Each matrix of sums holds a sum for every pair of digits; here the sums
// are multiplied by their powers of 2 and added. For integers cut into single bits the idea is
// public: [Ma] (sections 3.2 and 4.2), [ABQ-LLM] ("Bit Reduction").
// The compiler hints and the loop form are public: __launch_bounds__ [Best practices]; unrolling in
// full [Milakov] and in part [Prefetching]; the block-stride loop [Prefetching].
__global__ void __launch_bounds__(192)
    k_join(const int *__restrict__ C, int pieces, const T *__restrict__ eqhi, T *__restrict__ Gc) {
  __shared__ u64 sh[6][5];
  const int ch = blockIdx.x >> 6, pair = blockIdx.x & 63, aa = pair >> 3, ab = pair & 7, t = threadIdx.x;
  const int *D = C + (size_t)ch * pieces * SUMS;
  u64 acc[5] = {0, 0, 0, 0, 0};
#pragma unroll
  for (int q = 0; q < 3; q++) {
    const int e = t + 192 * q, rb = e % 24, cb = e / 24;
    const int *p = D + (ab * 24 + cb) * ROWS + aa * 24 + rb;
    long long v = 0;
#pragma unroll 4
    for (int s = 0; s < pieces; s++) v += p[(size_t)s * SUMS];
    const u64 x = signed_turn(v, (8 * ((rb & 7) + (cb & 7))) % TURNS);
    const int dg = (rb >> 3) + (cb >> 3);
#pragma unroll
    for (int k = 0; k < 5; k++)
      if (dg == k) acc[k] = gf_add(acc[k], x);
  }
#pragma unroll
  for (int k = 0; k < 5; k++) {
    for (int o = 16; o; o >>= 1) acc[k] = gf_add(acc[k], __shfl_down_sync(~0u, acc[k], o));
    if ((t & 31) == 0) sh[t >> 5][k] = acc[k];
  }
  __syncthreads();
  if (t == 0) {
#pragma unroll
    for (int k = 0; k < 5; k++)
      for (int w = 1; w < 6; w++) acc[k] = gf_add(acc[k], sh[w][k]);
    Gc[blockIdx.x] =
        F::mul(K3{gf_add(acc[0], gf_times_nonresidue(acc[3])), gf_add(acc[1], gf_times_nonresidue(acc[4])), acc[2]}, eqhi[ch]);
  }
}
// Launches k_join, which needs one block of 192 threads for each chunk and pair.
static inline void join(const int *C, int pieces, int nchunks, const T *eqhi, T *Gc, cudaStream_t st) {
  k_join<<<nchunks * 64, 192, 0, st>>>(C, pieces, eqhi, Gc);
  CK(cudaGetLastError());
}
// Computes G[p] = sum over the chunks ch of Gc[ch * 64 + p] for every pair p < 64. If dka > 0, the
// layer is on a lookup call's table side, where the lowest dlk bits of x are a bit position j of
// the multiplicities (how often the keys hit each table row). Only entries with j < dka are stored
// (dka <= 32, which is not checked); every other entry is 1. The kernel adds the others' share,
// 1 - sum_{j < dka} eq(rho[da .. da + dlk - 1], j), to every G[p]. Launch 64 blocks of 128 threads.
// Public source: [Harris]. The loop form is public: the block-stride loop [Prefetching].
__global__ void k_sum(const T *__restrict__ Gc, int nchunks, T *__restrict__ G, const Chal<F> *chl, int dka, int dlk,
                      int da) {
  __shared__ T sh[128];
  const int p = blockIdx.x, t = threadIdx.x;
  T s = F::zero();
  for (int ch = t; ch < nchunks; ch += 128) s = F::add(s, Gc[(size_t)ch * 64 + p]);
  if (dka && t < 32) {
    if (t == 0) s = F::add(s, F::one());
    if (t < dka) {
      T v = F::one();
      for (int i = 0; i < dlk; i++) {
        const T rh = chl->rho[da + i];
        v = F::mul(v, ((t >> (dlk - 1 - i)) & 1) ? rh : F::sub(F::one(), rh));
      }
      s = F::sub(s, v);
    }
  }
  sh[t] = s;
  __syncthreads();
  for (int o = 64; o; o >>= 1) {
    if (t < o) sh[t] = F::add(sh[t], sh[t + o]);
    __syncthreads();
  }
  if (t == 0) G[p] = sh[0];
}
#define LT_CK(x)                                                                                                       \
  do {                                                                                                                 \
    const cublasStatus_t s_ = (x);                                                                                     \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                                                 \
      printf("cublasLt: status %d at %s:%d\n", (int)s_, __FILE__, __LINE__);                                           \
      fflush(stdout);                                                                                                  \
      exit(1);                                                                                                         \
    }                                                                                                                  \
  } while (0)
// The descriptors and the algorithm for products of one shape: created at its first product,
// reused, never destroyed. Public source: [cuBLAS] (algo is kept, so the heuristic is asked once).
struct Plan {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a, b, c;
  cublasLtMatmulAlgo_t algo;
};
// The cuBLASLt handle, which set_up() creates, and the plans. Each translation unit that includes
// this file has its own copy. Public source: [cuBLAS].
struct Lib {
  cublasLtHandle_t h = nullptr;
  std::mutex m;
  std::map<std::tuple<int, int, int, int, size_t, size_t>, Plan> plans;
};
static Lib g_lib;
// Computes C[i] = A[i] B[i]^T for i < batch. A[i] has m rows of k digits, ld bytes apart, from
// A + i * step on; B[i] has n such rows, from B + i * step on. The results, each m x n sums in
// column-major order, follow one another in C. In cuBLAS's column-major terms, A is k x m with
// leading dimension ld and is transposed, and B is k x n and is not transposed.
// The library is given no workspace: only an algorithm that needs none is used. So apart from C it
// writes to no GPU memory of the caller, and products on two streams share none. The sums are exact
// 32-bit integers, so every algorithm gives the same C. The first product of a shape (m, n, k,
// batch, ld, step) creates its Plan. That must not happen during a stream capture; pt_graphed()
// (prover/hiding/product_test.cuh) by default runs its launches once before it captures them.
// The algorithm is the library's first answer to one query; nothing is timed. If the query fails or
// its answer needs workspace, the process exits with code 1. The query keeps the default alignment,
// so the addresses A, B and C must be multiples of 256. No code checks an address: each rests on
// its caller's sizes. spread() (prover/hiding/forest.cuh) tests that its sizes are multiples of
// 256; lib_wide() (prover/hiding/gram_keys.inc) does not. There a result is an odd multiple of 128
// bytes long, and a later call for the same chunks begins ld / k results further on. ld / k is even
// for grm::Key, which PtPL::prove() uses (not checked: computed from callers' sizes for 2^17 to 2^32 keys; for grm::Tau
// it is odd at 2^24 and 2^25), so each call's C is at a multiple of 256. Inside a batch results are at multiples of 128
// only and A[i], B[i] at multiples of step; [cuBLAS] does not say if the default applies there.
// Public source: [cuBLAS] ('TN' format, alignment, strided batch; cublasLtMatmulHeuristicResult_t,
// workspaceSize; cublasLtMatmulAlgoGetHeuristic(), fastest estimate first; 256 by default:
// cublasLtMatmulPreferenceAttributes_t, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, _B_, _C_, _D_).
// The callers pass both sides' stacked digits as A and B: see [Multiword], at the top of this file.
static inline void product(const i8 *A, const i8 *B, int *C, int m, int n, int k, int batch, size_t ld, size_t step,
                           cudaStream_t st) {
  const Plan *pl;
  {
    std::lock_guard<std::mutex> l(g_lib.m);
    const auto key = std::make_tuple(m, n, k, batch, ld, step);
    auto it = g_lib.plans.find(key);
    if (it == g_lib.plans.end()) {
      Plan p;
      const cublasOperation_t tr = CUBLAS_OP_T, no = CUBLAS_OP_N;
      LT_CK(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32I, CUDA_R_32I));
      LT_CK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &tr, sizeof tr));
      LT_CK(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &no, sizeof no));
      LT_CK(cublasLtMatrixLayoutCreate(&p.a, CUDA_R_8I, k, m, ld));
      LT_CK(cublasLtMatrixLayoutCreate(&p.b, CUDA_R_8I, k, n, ld));
      LT_CK(cublasLtMatrixLayoutCreate(&p.c, CUDA_R_32I, m, n, m));
      const int32_t bc = batch;
      const int64_t sa = (int64_t)step, sc = (int64_t)m * n;
      for (cublasLtMatrixLayout_t y : {p.a, p.b, p.c})
        LT_CK(cublasLtMatrixLayoutSetAttribute(y, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof bc));
      LT_CK(cublasLtMatrixLayoutSetAttribute(p.a, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sa, sizeof sa));
      LT_CK(cublasLtMatrixLayoutSetAttribute(p.b, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sa, sizeof sa));
      LT_CK(cublasLtMatrixLayoutSetAttribute(p.c, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sc, sizeof sc));
      cublasLtMatmulPreference_t pref;
      cublasLtMatmulHeuristicResult_t first;
      int got = 0;
      const uint64_t no_workspace = 0;
      LT_CK(cublasLtMatmulPreferenceCreate(&pref));
      LT_CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &no_workspace,
                                                 sizeof no_workspace));
      const cublasStatus_t found =
          cublasLtMatmulAlgoGetHeuristic(g_lib.h, p.op, p.a, p.b, p.c, p.c, pref, 1, &first, &got);
      LT_CK(cublasLtMatmulPreferenceDestroy(pref));
      if (found != CUBLAS_STATUS_SUCCESS || got < 1 || first.state != CUBLAS_STATUS_SUCCESS || first.workspaceSize) {
        printf("cublasLt has no algorithm for signed bytes at m %d n %d k %d batch %d ld %zu\n", m, n, k, batch, ld);
        fflush(stdout);
        exit(1);
      }
      p.algo = first.algo;
      it = g_lib.plans.emplace(key, p).first;
    }
    pl = &it->second;
  }
  const int one = 1, zero = 0;
  LT_CK(
      cublasLtMatmul(g_lib.h, pl->op, &one, A, pl->a, B, pl->b, &zero, C, pl->c, C, pl->c, &pl->algo, nullptr, 0, st));
  CK(cudaGetLastError());
}
// The GPU memory of one pass of one Eng, a sequence of layers (prover/hiding/product_test.cuh): two
// adjacent digit matrices of ROWS rows x cap columns and the sums of pieces_for(cap) pieces. lay()
// places the sums after the matrices; Eng::gram_slab_in() may place the matrices in another buffer,
// with a larger cap. `Slab` means this memory, not a part of a vector. cap_for() returns at most
// 2^g_log; g_log must be at least 15, as a chunk has up to 2^15 indices. piece() aims at <= g_k
// indices. Under the bound of 2^15 no 32-bit sum overflows (2^15 * 128^2 = 2^29). Public source for
// the principle, a reduction before a sum can overflow: [FFLAS]; its bound is on other products.
constexpr int g_log = 16, g_k = 4096;
struct Slab {
  i8 *A = nullptr, *B = nullptr;
  int *C = nullptr;
  size_t cap = 0; // the columns of a digit matrix of ROWS rows; for matrices with other numbers of rows, see columns()
  // Returns the number of indices of one piece of a chunk. The chunk is halved until a piece has at
  // most g_k indices, and further until `total` indices make at least 16 pieces or a piece has at
  // most 512, but never to a half that is no multiple of 16. Public source: [CUTLASS] (split K).
  static int piece(size_t chunk, size_t total) {
    size_t k = chunk;
    while ((k > (size_t)g_k || (total / k < 16 && k > 512)) && k % 32 == 0) k /= 2;
    return (int)k;
  }
  // Returns cap: the smallest power of 2 >= big (a vector's entries), clamped to 2^15 .. 2^g_log.
  static size_t cap_for(size_t big) {
    size_t c = (size_t)1 << 15;
    while (c < big && c < ((size_t)1 << g_log)) c *= 2;
    return c;
  }
  static size_t pieces_for(size_t cap) { return 2 * cap / g_k + 1; }
  // Returns the size in bytes of one buffer for lay(): the two digit matrices (384 cap), the sums
  // (147,456 per piece) and 512 spare bytes, which no code uses. That is 15,090,176 at cap = 2^15
  // and 30,032,384 at cap = 2^16.
  static size_t bytes(size_t cap) { return 2 * ROWS * cap + pieces_for(cap) * SUMS * sizeof(int) + 512; }
  // Returns how many chunks, in pieces of k indices, fit in this memory, by its columns and its
  // space for sums, if the digit matrices have ROWS rows. It returns 1 even if none fits.
  int chunks(size_t chunk, int k) const {
    return (int)std::max<size_t>(1, std::min(cap / chunk, pieces_for(cap) / (chunk / k)));
  }
  // Returns the number of columns, rounded down to a multiple of k, of two digit matrices that
  // together have `rows` rows and occupy the memory of A and B.
  size_t columns(int rows, int k) const { return 2 * ROWS * cap / rows / k * k; }
  // Returns how many pieces of `ints` sums each fit in the space for sums at C.
  size_t sums_hold(int ints) const { return pieces_for(cap) * SUMS / ints; }
  void lay(void *p, size_t cap_) {
    cap = cap_;
    A = (i8 *)p;
    B = A + ROWS * cap;
    C = (int *)(B + ROWS * cap);
  }
};
// Returns w * v for any v, as a one-term FE::SAcc, a sum of products that FE::sred() reduces once.
__device__ __forceinline__ K3 times_small(const K3 &w, u32 v) {
  FE::SAcc acc = FE::szero();
  FE::smac(acc, FE::sprep(w), v);
  return FE::sred(acc);
}
// ---- Layers made from 16-bit keys: layer A (struct Key) and layer F (struct Tau) ----
// value(k) = tau(k): k for Key (NB = 2), (k & 255) + 2^20 (k >> 8) + 2^40 tab[k] for Tau (NB = 3).
// It equals sum_{b < NB} byte(k, b) 2^turn(b), and lo(k) + 2^24 hi(k) with both parts below 2^24.
struct Key {
  static constexpr int NB = 2;
  static constexpr bool HI = false;
  __device__ __forceinline__ u32 byte(u32 k, int b) const { return b ? k >> 8 : k & 255; }
  static __host__ __device__ constexpr int turn(int b) { return 8 * b; }
  __device__ __forceinline__ u64 value(u32 k) const { return k; }
  __device__ __forceinline__ K3 times(const K3 &w, u32 k) const { return times_small(w, k); }
  __device__ __forceinline__ u32 lo(u32 k) const { return k; }
  __device__ __forceinline__ u32 hi(u32) const { return 0; }
};
struct Tau {
  const u8 *tab;
  static constexpr int NB = 3;
  static constexpr bool HI = true;
  // tab needs a byte for each of the 2^16 keys, in GPU memory. byte() and value() read it by
  // __ldg(). The idea is public: [Kepler guide], which is about GK110.
  __device__ __forceinline__ u32 byte(u32 k, int b) const {
    return b == 0 ? k & 255 : b == 1 ? k >> 8 : __ldg(tab + k);
  }
  static __host__ __device__ constexpr int turn(int b) { return 20 * b; }
  __device__ __forceinline__ u64 value(u32 k) const {
    return (u64)(k & 255) | (u64)(k >> 8) << 20 | (u64)__ldg(tab + k) << 40;
  }
  __device__ __forceinline__ K3 times(const K3 &w, u32 k) const { return F::mulb(w, value(k)); }
  __device__ __forceinline__ u32 lo(u32 k) const { return (u32)value(k) & 0xFFFFFF; }
  __device__ __forceinline__ u32 hi(u32 k) const { return (u32)(value(k) >> 24); }
};
__device__ __forceinline__ u32 key_of(const uint4 &q, int a) {
  const u32 w = a < 2 ? q.x : a < 4 ? q.y : a < 6 ? q.z : q.w;
  return (a & 1) ? w >> 16 : w & 0xFFFFu;
}
// Pass 1 computes the sums over c of w vL_a vR_a', of w vL_a and of w vR_a', from which k_finA80
// (prover/hiding/product_test.cuh) computes G. Here w is eq(c), of which the matrices hold only the
// factor eqlo. For N keys, vL_a(c) = value(key[8 c + a]), vR_a(c) = value(key[N / 2 + 8 c + a]).
// - The wide matrix has 216 rows of signed digits, ordered as in put_digits(): w vL_a is slab a,
//   and w, for the sums of w vR_a', is slab 8, rows 192 .. 215.
// - The narrow matrix has 8 NB + 4 rows. Row a' * NB + b holds byte(k, b) - 128 for the keys k of
//   vR_a', a signed byte, because the library multiplies signed bytes only [cuBLAS]. [gemmlowp]
//   (Jacob et al., Appendix B) subtracts 128 in the same way; compare [oneDNN], which adds 128 to
//   signed bytes. Row 8 NB holds ones; three rows of zeros make the number of rows a multiple of 4.
// A wide row times the row of ones is the sum of its digits [Dakkak]. To undo the subtraction,
// k_join_1 adds 128 times this sum to the wide row's product with each row of bytes [gemmlowp].
constexpr int WIDE = 216;
template <class V> __host__ __device__ constexpr int narrow() {
  return 8 * V::NB + 4;
}
// Writes pass 1's matrices for n indices (a multiple of 4) from index c0 on: the wide one to A, the
// narrow one to B. SC = N / 16. Public sources: [TensorFHE], [Ozaki-int8]; the loop: [Grid-stride].
// For digits or bytes stacked on both sides of a product, see [Multiword], at the top of this file.
// The compiler hints are public: __launch_bounds__ [Best practices]; #pragma unroll [Milakov].
template <class V>
__global__ void __launch_bounds__(256)
    k_digits_1(const uint16_t *__restrict__ key, size_t SC, size_t c0, int n, int chunk, const K3 *__restrict__ eqlo,
               i8 *__restrict__ A, i8 *__restrict__ B, size_t ld, const V val) {
  const int a = threadIdx.x >> 5, lane = threadIdx.x & 31;
  for (int c = blockIdx.x * 128 + lane * 4; c < n; c += gridDim.x * 128) {
    K3 w[4], e[4];
    u32 l[4], r[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
      l[k] = key[8 * (c0 + c + k) + a], r[k] = key[8 * (SC + c0 + c + k) + a];
      w[k] = eqlo[(c0 + c + k) % chunk];
      e[k] = val.times(w[k], l[k]);
    }
    put_digits(A + c, ld, a, e);
    if (a == 0) put_digits(A + c, ld, 8, w);
#pragma unroll
    for (int b = 0; b < V::NB; b++) {
      u32 right = 0;
#pragma unroll
      for (int k = 0; k < 4; k++) right |= (val.byte(r[k], b) ^ 128) << (8 * k);
      *reinterpret_cast<u32 *>(B + (size_t)(a * V::NB + b) * ld + c) = right;
    }
    if (a < 4) *reinterpret_cast<u32 *>(B + (size_t)(8 * V::NB + a) * ld + c) = a ? 0u : 0x01010101u;
  }
}
// Combines pass 1's sums. Each chunk has 80 outputs, Gc[chunk * 80 + p]: p = 8 a + a' is
// w vL_a vR_a', p = 64 + a is w vL_a, and p = 72 + a' is w vR_a'. For each chunk, C holds `pieces`
// matrices of WIDE x narrow() sums, column-major. Launch one warp per output.
// Public sources: [TensorFHE], [CUTLASS] (split K), [CUDA guide] (shuffles), as at k_join;
// [gemmlowp] (128 * ones). The compiler hint is public: __launch_bounds__ [Best practices].
template <class V>
__global__ void __launch_bounds__(128)
    k_join_1(const int *__restrict__ C, int pieces, int nchunks, const T *__restrict__ eqhi, T *__restrict__ Gc) {
  constexpr int size = WIDE * narrow<V>();
  const int id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, l = threadIdx.x & 31;
  if (id >= nchunks * 80) return;
  const int ch = id / 80, p = id % 80, m = l & 7;
  const bool alone = p >= 64 && p < 72; // w vL_a: the right factor is 1
  const int row0 = p < 64 ? (p >> 3) * 24 : alone ? (p - 64) * 24 : 192, col0 = (p < 64 ? p & 7 : p - 72) * V::NB;
  const int *r = C + (size_t)ch * pieces * size + row0 + l;
  u64 x = 0;
  // A warp has 24 rows of digits. Its last eight lanes read nothing, and their sum is not used.
  if (l < 24) {
    long long ones = 0; // the digit row's product with the row of ones, which is the sum of the row's digits
    for (int i = 0; i < pieces; i++) ones += r[(size_t)i * size + 8 * V::NB * WIDE];
    if (alone) x = signed_turn(ones, 8 * m);
    for (int b = 0; b < (alone ? 0 : V::NB); b++) {
      long long v = 128 * ones;
      for (int i = 0; i < pieces; i++) v += r[(size_t)i * size + (col0 + b) * WIDE];
      x = gf_add(x, signed_turn(v, (8 * m + V::turn(b)) % TURNS));
    }
  }
  for (int o = 4; o; o >>= 1) x = gf_add(x, __shfl_down_sync(~0u, x, o, 8));
  const u64 c0 = __shfl_sync(~0u, x, 0), c1 = __shfl_sync(~0u, x, 8), c2 = __shfl_sync(~0u, x, 16);
  if (l == 0) Gc[id] = F::mul(K3{c0, c1, c2}, eqhi[ch]);
}
// Adds sum_j K[j] lo(k_j) to acc and, if V::HI, sum_j K[j] hi(k_j) to acc2, for q's eight keys k_j.
template <class V>
__device__ __forceinline__ void fold_keys(FE::SAcc &acc, FE::SAcc &acc2, const uint4 &q, const FE::SP *K,
                                          const V &val) {
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const u32 k = key_of(q, j);
    FE::smac(acc, K[j], val.lo(k));
    if (V::HI) FE::smac(acc2, K[j], val.hi(k));
  }
}
template <class V> __device__ __forceinline__ K3 fold_end(const K3 &alpha, const FE::SAcc &acc, const FE::SAcc &acc2) {
  K3 s = FE::sred(acc);
  if (V::HI) {
    const K3 h = FE::sred(acc2);
    s = F::add(s, K3{gram_times_2_to(h.k0, 24), gram_times_2_to(h.k1, 24), gram_times_2_to(h.k2, 24)});
  }
  return F::sub(alpha, s);
}
// Pass 2. An entry is U_a(c) = alpha - sum_{j < 8} K[j] value(key[side + 64 c + 8 a + j]): side is
// 0 (left) or N / 2 (right), and K holds the eq weights of pass 1's three round challenges. A side
// is M8 = N / 16 words of 8 keys. c0, n, chunk: as in k_digits. A receives the digits of eqlo * U
// (left), B those of U (right). Public sources: [TensorFHE], [Ozaki-int8]; the loop: [Grid-stride].
template <class V>
__global__ void __launch_bounds__(256)
    k_digits_2(const uint4 *__restrict__ key, size_t M8, size_t c0, int n, int chunk, const M3L *__restrict__ eqlo,
               const K3 *__restrict__ K, const K3 *__restrict__ alpha, i8 *__restrict__ A, i8 *__restrict__ B,
               size_t ld, const V val) {
  FE::SP Ks[8]; // the fold's weights K
#pragma unroll
  for (int j = 0; j < 8; j++) Ks[j] = FE::sprep(K[j]);
  const K3 al = *alpha;
  const int a = threadIdx.x & 7;
  for (int c = (blockIdx.x * 32 + (threadIdx.x >> 3)) * 4; c < n; c += gridDim.x * 128) {
    K3 e[4];
#pragma unroll
    for (int sd = 0; sd < 2; sd++) {
#pragma unroll
      for (int k = 0; k < 4; k++) {
        FE::SAcc acc = FE::szero(), acc2 = FE::szero();
        fold_keys(acc, acc2, key[sd * M8 + (c0 + c + k) * 8 + a], Ks, val);
        const K3 u = fold_end<V>(al, acc, acc2);
        e[k] = sd ? u : FK2::mulp(eqlo[c % chunk + k], u);
      }
      put_digits((sd ? B : A) + c, ld, a, e);
    }
  }
}
// Pass 3's fold: U_a(c) = alpha - sum_{j < 64} K[j] value(key[side + 64 (8 c + a) + j]), where K
// holds the 64 eq weights of the six round challenges of passes 1 and 2. UL (left) and UR (right)
// receive U_a(c) at a * CO + c, for c < CO; rows = 8 CO. Eight lanes compute one entry.
// Public sources: [CUDA guide] (shuffles); for the form of the loop, [Grid-stride].
// The lanes' sums of limb products (FE::SAcc) are added before the reduction in fold_end(). This is
// specific to the field's arithmetic (modulo p). No sum overflows, as lo(k), hi(k) < 2^24.
// The compiler hints are public: __launch_bounds__ [Best practices]; #pragma unroll [Milakov].
template <class V>
__global__ void __launch_bounds__(256)
    k_fold64(const uint4 *__restrict__ key, size_t rows, size_t CO, const K3 *__restrict__ K,
             const K3 *__restrict__ alpha, K3 *__restrict__ UL, K3 *__restrict__ UR, const V val) {
  const K3 al = *alpha;
  const int jj = threadIdx.x & 7;
  FE::SP Ks[8]; // the weights K[8 jj .. 8 jj + 7]
#pragma unroll
  for (int j = 0; j < 8; j++) Ks[j] = FE::sprep(K[jj * 8 + j]);
  for (size_t g = blockIdx.x * 32 + (threadIdx.x >> 3); g < 2 * rows; g += (size_t)gridDim.x * 32) {
    FE::SAcc acc = FE::szero(), acc2 = FE::szero();
    fold_keys(acc, acc2, key[g * 8 + jj], Ks, val);
#pragma unroll
    for (int o = 4; o; o >>= 1)
#pragma unroll
      for (int i = 0; i < 3; i++) {
        acc.l[i] += __shfl_down_sync(~0u, acc.l[i], o, 8);
        acc.h[i] += __shfl_down_sync(~0u, acc.h[i], o, 8);
        if (V::HI) {
          acc2.l[i] += __shfl_down_sync(~0u, acc2.l[i], o, 8);
          acc2.h[i] += __shfl_down_sync(~0u, acc2.h[i], o, 8);
        }
      }
    if (jj == 0) {
      const size_t r = g >= rows ? g - rows : g;
      (g >= rows ? UR : UL)[(r & 7) * CO + (r >> 3)] = fold_end<V>(al, acc, acc2);
    }
  }
}
// Creates the library's handle. Call it before the first product(), outside every stream capture.
// Later calls do nothing. Eng::init() and Prover::make_tables() (prover/prover/inputs.cuh), among others, call it.
static inline void set_up() {
  std::lock_guard<std::mutex> l(g_lib.m);
  if (g_lib.h) return;
  // Same check as LT_CK, but this startup handle creation is named for a serving build's last line.
  // Only this make_tables-time failure is cublaslt-start; later product() LT_CK failures stay unnamed.
  const cublasStatus_t s_ = cublasLtCreate(&g_lib.h);
  if (s_ != CUBLAS_STATUS_SUCCESS) {
    printf("cublasLt: status %d at %s:%d\n", (int)s_, __FILE__, __LINE__);
    fflush(stdout);
    name_the_end("cublaslt-start");
    exit(1);
  }
}
} // namespace grm
