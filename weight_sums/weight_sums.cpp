// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// decode_q8_0() and the case BF16 of decode() are adapted from gguf-py 0.19.0 (gguf/quants.py: dequantize_blocks of the
// classes Q8_0 and BF16); the case F16 follows its dequantize(). It is under the MIT License. That project's notice:
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

// This program reads the files (shards) of one model, which are in GGUF, ggml's container format.
// It converts the stored weights of the committed tensors to the integers that cp_commit, the
// committer program, commits to, and writes the sums file. cp_commit compares the digests in that
// file with those of the integers that it computes. A committed tensor is one that the weight
// commitment covers: the embedding, the output head and, in each layer, the tensors whose kind has
// the role matrix in kLayerKinds. A norm, the vector of gains of a normalization layer, is
// multiplied into the matrices that read it and has no line in the sums file.
//
// The sums file has one line for each committed tensor: name, element count, exponent of its
// power-of-two scale, number of bits that its widest integer needs, and the SHA-256 digest of its
// integers. A last line, END and the number of those lines, closes it. The conversion rule is
// defined in docs/model.md, "11 From the model's files to the integers", and summarized in
// weight_sums/README.md, "Which models, and the rule". kUsage gives the command line.
// Start reading at run(): read_shard() reads a shard's tensor table, plan() finds the committed
// tensors, and convert() applies the rule. The comments' symbols are defined under "Numbers" (RN32,
// RNE) and "Which tensor is what" (v, g, c, gamma_j, p, mx, e, w).
//
// Name every shard in one command, the model's first file first. The first shard's metadata
// describe the model, and a shard that is left out of the command can go unnoticed.
//
// Anyone who has the sums file can test a guess at a tensor against its digest, so keep the file
// where you keep the weights. The lines that the program prints give paths, names of tensors of a
// known kind, counts and durations, and never a stored value, exponent, number of bits or digest.
// An error message can still state a fact about one, for example that every value of a tensor is 0.
//
// weight_sums.py applies the same rule in Python, so that the two sums files can be compared. This
// program shares no source file with it or with cp_commit. It has its own reader of the container,
// its own block decoder and its own SHA-256, and weight_sums.py uses the reader and the decoder of
// the package gguf and the arithmetic of numpy. For the K types both decoders go back to ggml's
// public code: one is adapted from it in this file, and the other is part of that package. For Q8_0
// and BF16 this file's decoder is adapted from gguf-py, and for F16 it follows gguf-py (see the
// head).
//
// Public sources. All paths of ggml in this file are those of llama.cpp
// (github.com/ggml-org/llama.cpp) as carried in llama-cpp-python 0.3.7, whose changelog names the
// commit 794fe23f29fb40104975c91fe19f23798f7c726e.
//   GGUF      The container format of the ggml and llama.cpp projects. The comment at the top of
//             ggml/include/gguf.h describes the whole layout. The reader in ggml/src/gguf.cpp shows
//             which files ggml itself rejects. The format's specification is docs/gguf.md in
//             github.com/ggml-org/ggml.
//   Blocks    The byte layout of the block formats Q4_K, Q5_K and Q6_K, and the rule that decodes
//             them, are those of ggml, the tensor library of the llama.cpp project
//             (github.com/ggml-org/llama.cpp). They are defined in ggml/src/ggml-common.h
//             (block_q4_K, block_q5_K, block_q6_K) and in ggml/src/ggml-quants.c (get_scale_min_k4;
//             dequantize_row_q4_K, _q5_K, _q6_K). The decoder in this file follows ggml's public
//             code and is adapted from it. Its function scale_and_minimum has the two cases and the
//             four expressions of get_scale_min_k4, and so have the decoders of tests/oracle.py and
//             of the prover. ggml and llama.cpp have the MIT licence, in that copy with the notice
//             "Copyright (c) 2023-2024 The ggml authors". THIRD_PARTY.md, section "ggml", has its
//             text. The full path of tests/oracle.py is weight_sums/tests/oracle.py. The decoder's
//             comments keep the letters of ggml. The sources for Q8_0, F16 and BF16 are named at
//             decode_q8_0 and decode.
//   SHA-256   SHA-256 is defined in FIPS 180-4, "Secure Hash Standard". RFC 6234, "US Secure Hash
//             Algorithms", gives the same algorithm with test vectors, and the section numbers in
//             the comments below are those of the RFC: 4.1 (padding), 5.1 (functions and
//             constants), 6.1 (initial value) and 6.2 (processing).
//   Numbers   IEEE 754 for binary16, binary32, binary64 and rounding to nearest with ties to even.
//   The rule  The rule grew out of VerInf's rule for converting stored weights to integers:
//             J. Petrie, VerInf, source code and README, github.com/JamesPetrie/VerInf,
//             commit 89903c96 (28 August 2026), MIT License. The present text of this program has
//             not been compared with VerInf's code; earlier forms were. THIRD_PARTY.md, "VerInf",
//             lists the files that apply the rule, and NOTICE has VerInf's licence
//             notice. The rule's steps are common practice in the quantization of neural networks.
//             Multiplying the gains of a norm into the matrices that read the norm: S. Ashkboos et
//             al., "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs", arXiv:2404.00456,
//             sections 3.4 and 4; S. Ashkboos et al., "SliceGPT: Compress Large Language Models by
//             Deleting Rows and Columns", arXiv:2401.15024, sections 2.1 and 3.2. One power-of-two
//             scale per tensor and no zero point: M. Nagel et al., "A White Paper on Neural Network
//             Quantization", arXiv:2106.08295, section 2.2. The range taken from the largest
//             absolute value: H. Wu et al., "Integer Quantization for Deep Learning Inference:
//             Principles and Empirical Evaluation", arXiv:2004.09602, section 3.4, and S. R. Jain
//             et al., "Trained Quantization Thresholds for Accurate and Efficient Fixed-Point
//             Inference of Deep Neural Networks", arXiv:1903.08066, section 4.2. Rounding of ties
//             to even: the same paper of Jain et al., section 3.2.

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <new>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

// Beyond the C++17 standard library, the program uses a few POSIX calls and, where the C library
// has it, Linux's renameat2(), all of them for the output file only (class OutputFile).
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace weight_sums {

// The conversion rule rounds to binary32 after every float operation. One condition for this is not
// asserted here: compile with -ffp-contract=off, as build.sh does, so that the compiler never
// merges a product and a sum into one rounding.
static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4, "float must be IEEE 754 binary32");
static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8, "double must be IEEE 754 binary64");
static_assert(FLT_EVAL_METHOD == 0, "float operations must be evaluated in binary32");

// ---------------------------------------------------------------------------------------------------------------------
// Exit statuses
// ---------------------------------------------------------------------------------------------------------------------

// The exit statuses other than 0. Status 0 means that the sums file is complete.
enum class Status {
  command_or_file = 21, // a wrong command line, or a file that cannot be read or written
  container = 22,       // the GGUF container: its bytes, its metadata, or the names and shapes of its tensors
  type = 23,            // the storage type of a committed tensor or of a norm
  not_finite = 24,      // a value, a multiplier or a product that is not finite
  bound = 25,           // a bound of the conversion rule, or a model shape for which the rule is not defined
  memory = 26,          // not enough memory
};

// Defaults and limits of the options --eres (the embedding's exponent) and --threads.
constexpr int kDefaultEres = 11;
constexpr int kLowestEres = -100;
constexpr int kHighestEres = 100;
constexpr int kDefaultThreads = 8;
constexpr int kHighestThreads = 256;

// main() prints the message and returns the status.
struct Failure {
  Status status;
  std::string message;
};

[[noreturn]] inline void fail(Status status, std::string message) {
  throw Failure{status, std::move(message)};
}

// ---------------------------------------------------------------------------------------------------------------------
// SHA-256 (FIPS 180-4; section numbers of RFC 6234)
// ---------------------------------------------------------------------------------------------------------------------

class Sha256 {
public:
  void update(const std::uint8_t *data, std::size_t size) {
    length_ += size;
    if (pending_size_ != 0) {
      const std::size_t taken = std::min(size, pending_.size() - pending_size_);
      std::memcpy(pending_.data() + pending_size_, data, taken);
      pending_size_ += taken;
      data += taken;
      size -= taken;
      if (pending_size_ < pending_.size()) return;
      compress(pending_.data());
      pending_size_ = 0;
    }
    for (; size >= 64; data += 64, size -= 64) compress(data);
    std::memcpy(pending_.data(), data, size);
    pending_size_ = size;
  }

  // Pads the message (section 4.1) and returns its digest. Call it once, after the last update().
  std::array<std::uint8_t, 32> finish() {
    const std::uint64_t bits = length_ * 8;
    std::array<std::uint8_t, 72> padding{};
    padding[0] = 0x80;
    // The padded message is a multiple of 64 bytes long, and its last 8 bytes hold the length of
    // the message in bits.
    const std::size_t zeros = (64 + 55 - pending_size_) % 64;
    for (int k = 0; k < 8; ++k) padding[1 + zeros + k] = static_cast<std::uint8_t>(bits >> (56 - 8 * k));
    update(padding.data(), 1 + zeros + 8);
    return hash_value();
  }

  // The hash value H(i) of section 6.2 after the complete 64-byte blocks so far, as 32 bytes. It is
  // public so that a test can check the padding.
  std::array<std::uint8_t, 32> hash_value() const {
    std::array<std::uint8_t, 32> bytes{};
    for (int k = 0; k < 32; ++k) bytes[k] = static_cast<std::uint8_t>(state_[k / 4] >> (24 - 8 * (k % 4)));
    return bytes;
  }

private:
  static std::uint32_t rotate_right(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

  // Processes one block of 64 bytes (section 6.2). s0, s1, sum0, sum1, choice and majority are the
  // functions SSIG0, SSIG1, BSIG0, BSIG1, CH and MAJ of section 5.1.
  void compress(const std::uint8_t *block) {
    // The round constants of section 5.1: "the first 32 bits of the fractional parts of the cube
    // roots of the first" 64 primes.
    static constexpr std::uint32_t kRound[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    std::uint32_t w[64];
    for (int t = 0; t < 16; ++t) {
      w[t] = std::uint32_t{block[4 * t]} << 24 | std::uint32_t{block[4 * t + 1]} << 16 |
             std::uint32_t{block[4 * t + 2]} << 8 | std::uint32_t{block[4 * t + 3]};
    }
    for (int t = 16; t < 64; ++t) {
      const std::uint32_t s0 = rotate_right(w[t - 15], 7) ^ rotate_right(w[t - 15], 18) ^ (w[t - 15] >> 3);
      const std::uint32_t s1 = rotate_right(w[t - 2], 17) ^ rotate_right(w[t - 2], 19) ^ (w[t - 2] >> 10);
      w[t] = s1 + w[t - 7] + s0 + w[t - 16];
    }
    std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
    std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
    for (int t = 0; t < 64; ++t) {
      const std::uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
      const std::uint32_t choice = (e & f) ^ (~e & g);
      const std::uint32_t t1 = h + sum1 + choice + kRound[t] + w[t];
      const std::uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t t2 = sum0 + majority;
      h = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
    }
    state_[0] += a, state_[1] += b, state_[2] += c, state_[3] += d;
    state_[4] += e, state_[5] += f, state_[6] += g, state_[7] += h;
  }

  // The initial hash value of section 6.1: "the first 32 bits of the fractional parts of the
  // square roots of the first" 8 primes.
  std::array<std::uint32_t, 8> state_{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::uint64_t length_ = 0;             // the length of the message so far, in bytes
  std::array<std::uint8_t, 64> pending_; // the bytes that do not fill a block yet
  std::size_t pending_size_ = 0;
};

inline std::string to_hex(const std::array<std::uint8_t, 32> &digest) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string text;
  for (const std::uint8_t byte : digest) {
    text += kDigits[byte >> 4];
    text += kDigits[byte & 15];
  }
  return text;
}

// ---------------------------------------------------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------------------------------------------------

// RN32(x) is the binary32 number nearest to x, and RNE(x) is the integer nearest to x, both with
// ties to even.

inline std::uint16_t load_le16(const std::uint8_t *bytes) {
  return static_cast<std::uint16_t>(bytes[0] | bytes[1] << 8);
}

inline std::uint32_t load_le32(const std::uint8_t *bytes) {
  return std::uint32_t{bytes[0]} | std::uint32_t{bytes[1]} << 8 | std::uint32_t{bytes[2]} << 16 |
         std::uint32_t{bytes[3]} << 24;
}

inline std::uint64_t load_le64(const std::uint8_t *bytes) {
  return std::uint64_t{load_le32(bytes)} | std::uint64_t{load_le32(bytes + 4)} << 32;
}

inline float float_from_bits(std::uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof value);
  return value;
}

// Widens a binary16 bit pattern to binary32. The widening is exact: every binary16 number is a
// binary32 number. binary16 has 1 sign bit, 5 exponent bits with bias 15 and 10 fraction bits.
// binary32 has 1 sign bit, 8 exponent bits with bias 127 and 23 fraction bits.
inline float widen_binary16(std::uint16_t half) {
  const std::uint32_t sign = std::uint32_t{half & 0x8000u} << 16;
  const std::uint32_t exponent = (half >> 10) & 31;
  std::uint32_t fraction = half & 1023;
  if (exponent == 31) return float_from_bits(sign | 0x7F800000u | fraction << 13); // an infinity or a NaN
  if (exponent != 0) return float_from_bits(sign | (exponent + (127 - 15)) << 23 | fraction << 13);
  if (fraction == 0) return float_from_bits(sign);
  // A subnormal binary16 number is fraction * 2^-24, which is a normal binary32 number. Shift the
  // fraction until its leading 1 reaches the place of the implicit bit. After s shifts the number
  // is 1.f * 2^(-14 - s).
  std::uint32_t shifts = 0;
  while ((fraction & 1024) == 0) {
    fraction <<= 1;
    ++shifts;
  }
  return float_from_bits(sign | (127 - 14 - shifts) << 23 | (fraction & 1023) << 13);
}

// 2^e as a binary64 number, for -1022 <= e <= 1023.
inline double power_of_two(int e) {
  return std::ldexp(1.0, e);
}

// RNE(2^e * p) for a finite binary32 number p, given scale = 2^e. The product is exact in binary64:
// p has at most 24 significant bits, and the product stays far inside the range of binary64. |p| is
// 0 or from 2^-149 to 2^128, and |e| <= 161, because choose_exponent(mx) starts at e = 12 -
// floor(log2(mx)) for a binary32 number mx. So the only rounding is the one to an integer.
// std::nearbyint rounds ties to even in the rounding mode to nearest (cppreference.com,
// "std::nearbyint"). That mode is the default (IEEE 754 and the C standard), and the program never
// changes it.
inline double round_scaled(float p, double scale) {
  return std::nearbyint(static_cast<double>(p) * scale);
}

// ---------------------------------------------------------------------------------------------------------------------
// The GGUF container
// ---------------------------------------------------------------------------------------------------------------------

// The value types of metadata that the program tells apart, by their numbers in enum gguf_type
// (ggml/include/gguf.h).
constexpr std::uint32_t kValueUint8 = 0;
constexpr std::uint32_t kValueUint16 = 2;
constexpr std::uint32_t kValueUint32 = 4;
constexpr std::uint32_t kValueString = 8;
constexpr std::uint32_t kValueArray = 9;
constexpr std::uint32_t kValueUint64 = 10;

// Storage types of tensors (enum ggml_type in ggml/include/ggml.h). These seven are the ones that
// the rule reads: binary32; three block formats, which store 256 elements and the scales that they
// share as one super-block; and F16, Q8_0 and BF16, whose numbers were read in the package gguf,
// version 0.19.0: gguf/constants.py, GGMLQuantizationType.
constexpr std::uint32_t kTypeF32 = 0;
constexpr std::uint32_t kTypeF16 = 1;
constexpr std::uint32_t kTypeQ80 = 8;
constexpr std::uint32_t kTypeBF16 = 30;
constexpr std::uint32_t kTypeQ4K = 12;
constexpr std::uint32_t kTypeQ5K = 13;
constexpr std::uint32_t kTypeQ6K = 14;

constexpr std::size_t kSuperBlock = 256; // the number of elements in a super-block (QK_K in ggml/src/ggml-common.h)
constexpr std::size_t kBlockQ80 = 32;    // the number of elements in a block of Q8_0 (QK8_0; see decode_q8_0)

// The largest general.alignment and the largest number of elements in a tensor that are accepted.
// Both are limits of this program and not of the format.
constexpr std::uint64_t kHighestAlignment = std::uint64_t{1} << 20;
constexpr std::uint64_t kMostElements = std::uint64_t{1} << 40;

struct MetadataValue {
  std::uint32_t type = 0;
  std::uint64_t number = 0; // the value, if the type is UINT8, UINT16, UINT32 or UINT64
  std::string text;         // the value, if the type is STRING; kept only for the key general.architecture

  bool is_unsigned() const {
    return type == kValueUint8 || type == kValueUint16 || type == kValueUint32 || type == kValueUint64;
  }
};

// One entry of a shard's tensor table. `elements` is the product of `dims`, and `type` is a number
// of enum ggml_type.
struct Tensor {
  std::string name;
  std::array<std::uint64_t, 4> dims{1, 1, 1, 1}; // a dimension that the file does not store counts as 1
  std::uint64_t elements = 1;
  std::uint32_t type = 0;
  std::uint64_t offset = 0;   // in bytes, counted from the start of the shard's data section
  std::string path;           // the path of the shard that holds the tensor
  std::uint64_t position = 0; // of the tensor's first byte, counted from the start of the shard; locate() sets it
};

// What read_shard() keeps of one shard. The tensor data start at byte `data_start` of the file.
struct Shard {
  std::string path;
  std::uint64_t size = 0;
  std::map<std::string, MetadataValue> metadata;
  std::uint64_t alignment = 32; // the default is GGUF_DEFAULT_ALIGNMENT in ggml/include/gguf.h
  std::uint64_t data_start = 0;
  std::vector<Tensor> tensors;
};

// Reads, in order, the part of a shard that comes before the tensor data. Every read and skip first
// checks its length against the bytes left in the file. So a length field that points past the end
// gives status 22 (container) and never causes a large allocation.
class ShardHead {
public:
  explicit ShardHead(const std::string &path) : path_(path) {
    std::error_code error;
    if (!std::filesystem::is_regular_file(path, error)) {
      fail(Status::command_or_file, "There is no regular file at " + path + ". Apart from the three options and " +
                                        "their values, the command line holds only paths of GGUF files.");
    }
    size_ = std::filesystem::file_size(path, error);
    file_.open(path, std::ios::binary);
    if (error || !file_) fail(Status::command_or_file, path + " cannot be opened. Check its permissions.");
  }

  std::uint64_t size() const { return size_; }
  std::uint64_t position() const { return position_; }
  std::uint64_t remaining() const { return size_ - position_; }

  // Names the part of the file that is being read, for the message of ends_early().
  void enter(const char *part) { part_ = part; }

  void read(void *to, std::uint64_t count) {
    claim(count);
    file_.read(static_cast<char *>(to), static_cast<std::streamsize>(count));
    check();
  }

  void skip(std::uint64_t count) {
    claim(count);
    file_.ignore(static_cast<std::streamsize>(count));
    check();
  }

  std::uint32_t u32() {
    std::uint8_t bytes[4];
    read(bytes, sizeof bytes);
    return load_le32(bytes);
  }

  std::uint64_t u64() {
    std::uint8_t bytes[8];
    read(bytes, sizeof bytes);
    return load_le64(bytes);
  }

  // A string is a 64-bit length and that many bytes, with no terminating zero.
  std::string string() {
    const std::uint64_t length = u64();
    if (length > remaining()) ends_early();
    std::string text(static_cast<std::size_t>(length), '\0');
    read(text.data(), length);
    return text;
  }

  void skip_string() { skip(u64()); }

  [[noreturn]] void malformed(const std::string &what) const {
    fail(Status::container, path_ + " is not a GGUF file that this program reads: " + what + ".");
  }

private:
  [[noreturn]] void ends_early() const { malformed(std::string("the file ends inside its ") + part_); }

  void claim(std::uint64_t count) {
    if (count > remaining()) ends_early();
    position_ += count;
  }

  void check() {
    if (!file_) fail(Status::command_or_file, path_ + " cannot be read: the system reported an input error.");
  }

  std::string path_;
  std::ifstream file_;
  std::uint64_t size_ = 0;
  std::uint64_t position_ = 0;
  const char *part_ = "header";
};

// The size in bytes of a metadata value of a fixed-size type, by its number in enum gguf_type:
// UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64.
// STRING and ARRAY have no fixed size and have a 0 in the table.
inline std::uint64_t fixed_value_size(ShardHead &head, std::uint32_t type) {
  static constexpr std::uint64_t kSizes[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
  if (type >= 13 || kSizes[type] == 0) head.malformed("a metadata value has an unknown type, or is an array of arrays");
  return kSizes[type];
}

// Steps over one metadata value. An array is a 32-bit element type, a 64-bit count, and the
// elements. An array of arrays is rejected, as ggml's own reader rejects it
// (gguf_init_from_file_impl in ggml/src/gguf.cpp).
inline void skip_value(ShardHead &head, std::uint32_t type) {
  if (type == kValueString) {
    head.skip_string();
  } else if (type == kValueArray) {
    const std::uint32_t element_type = head.u32();
    const std::uint64_t count = head.u64();
    if (element_type == kValueString) {
      for (std::uint64_t i = 0; i < count; ++i) head.skip_string();
    } else {
      const std::uint64_t size = fixed_value_size(head, element_type);
      if (count > head.remaining() / size) head.malformed("an array of the metadata is longer than the file");
      head.skip(count * size);
    }
  } else {
    head.skip(fixed_value_size(head, type));
  }
}

// Reads the header, the metadata and the tensor table of one shard. The layout is the one that the
// comment at the top of ggml/include/gguf.h lists. All numbers are little-endian.
inline Shard read_shard(const std::string &path) {
  ShardHead head(path);
  Shard shard;
  shard.path = path;
  shard.size = head.size();

  if (head.u32() != 0x46554747u) head.malformed("it does not start with the four bytes \"GGUF\"");
  if (head.u32() != 3) head.malformed("its GGUF version is not 3");
  const std::uint64_t tensor_count = head.u64();
  const std::uint64_t pair_count = head.u64();

  head.enter("metadata");
  for (std::uint64_t i = 0; i < pair_count; ++i) {
    std::string key = head.string();
    MetadataValue value;
    value.type = head.u32();
    if (value.is_unsigned()) {
      std::uint8_t bytes[8] = {};
      head.read(bytes, fixed_value_size(head, value.type));
      value.number = load_le64(bytes);
    } else if (value.type == kValueString && key == "general.architecture") {
      value.text = head.string();
    } else {
      skip_value(head, value.type);
    }
    // If a key appears twice, the later value counts.
    shard.metadata.insert_or_assign(std::move(key), std::move(value));
  }
  if (const auto found = shard.metadata.find("general.alignment"); found != shard.metadata.end()) {
    if (found->second.type != kValueUint32) {
      fail(Status::container, path + ": the metadata key general.alignment must have the value type UINT32.");
    }
    shard.alignment = found->second.number;
    if (shard.alignment < 1 || shard.alignment > kHighestAlignment) {
      head.malformed("general.alignment is not between 1 and 1048576");
    }
  }

  // In a damaged file the bytes that are read as a name can be tensor data. So a message about an
  // entry of the table gives the entry's number and not its name.
  head.enter("tensor table");
  for (std::uint64_t i = 0; i < tensor_count; ++i) {
    const std::string entry = "entry " + std::to_string(i + 1) + " of the tensor table ";
    Tensor tensor;
    tensor.path = path;
    tensor.name = head.string();
    const std::uint32_t dimensions = head.u32();
    if (dimensions < 1 || dimensions > tensor.dims.size()) {
      head.malformed(entry + "does not have one to four dimensions");
    }
    for (std::uint32_t k = 0; k < dimensions; ++k) {
      tensor.dims[k] = head.u64();
      if (tensor.dims[k] == 0) head.malformed(entry + "has a dimension of 0");
      // The comparison detects a product above the limit without computing it, so nothing
      // overflows.
      if (tensor.dims[k] > kMostElements / tensor.elements) head.malformed(entry + "has more than 2^40 elements");
      tensor.elements *= tensor.dims[k];
    }
    tensor.type = head.u32();
    tensor.offset = head.u64();
    shard.tensors.push_back(std::move(tensor));
  }

  // The data section starts at the next multiple of the alignment.
  shard.data_start = head.position() + (shard.alignment - head.position() % shard.alignment) % shard.alignment;
  return shard;
}

// The value of the metadata key <architecture>.<name>, which the model must have with an unsigned
// integer type.
inline std::uint64_t required_number(const Shard &shard, const std::string &architecture, const std::string &name) {
  const auto found = shard.metadata.find(architecture + "." + name);
  if (found == shard.metadata.end() || !found->second.is_unsigned()) {
    fail(Status::container, shard.path + ": the metadata key <architecture>." + name +
                                " is absent or is not an unsigned integer. <architecture> is the value of the key " +
                                "general.architecture. Name the model's first file first.");
  }
  return found->second.number;
}

// The block of a storage type: its number of elements and its number of bytes. F32, F16 and BF16
// count as blocks of one element. The sizes of Q4_K, Q5_K and Q6_K are those of block_q4_K,
// block_q5_K and block_q6_K in ggml/src/ggml-common.h. Those of Q8_0, F16 and BF16 are in the table
// GGML_QUANT_SIZES of the package gguf, version 0.19.0 (gguf/constants.py): 32 elements in 2 + 32
// bytes, and 1 element in 2 bytes.
struct Storage {
  std::uint64_t block_elements;
  std::uint64_t block_bytes;
};

inline Storage storage_of(std::uint32_t type) {
  switch (type) {
  case kTypeQ4K:
    return {kSuperBlock, 144};
  case kTypeQ5K:
    return {kSuperBlock, 176};
  case kTypeQ6K:
    return {kSuperBlock, 210};
  case kTypeQ80:
    return {kBlockQ80, 34};
  case kTypeF16:
  case kTypeBF16:
    return {1, 2};
  default:
    return {1, 4}; // F32
  }
}

// Checks the first dimension, offset and size of a committed tensor or a norm, and sets the
// tensor's position in the file. The type must be one of the seven storage types, which plan()
// checks first: storage_of() treats any other type as F32. plan() does not call locate() for an
// ignored tensor.
inline void locate(Tensor &tensor, const Shard &shard) {
  const Storage storage = storage_of(tensor.type);
  if (tensor.dims[0] % storage.block_elements != 0) {
    fail(Status::container, shard.path + ": the first dimension of the tensor " + tensor.name +
                                " is not a multiple of " + std::to_string(storage.block_elements) +
                                ", as its storage type requires.");
  }
  if (tensor.offset % shard.alignment != 0) {
    fail(Status::container, shard.path + ": the offset of the tensor " + tensor.name +
                                " is not a multiple of the file's alignment. The file is damaged.");
  }
  // The comparisons are arranged so that no sum or product can overflow.
  const std::uint64_t data_bytes = shard.size > shard.data_start ? shard.size - shard.data_start : 0;
  if (tensor.offset > data_bytes ||
      tensor.elements / storage.block_elements > (data_bytes - tensor.offset) / storage.block_bytes) {
    fail(Status::container, shard.path + ": the data of the tensor " + tensor.name +
                                " run past the end of the file. The file is incomplete or damaged.");
  }
  tensor.position = shard.data_start + tensor.offset;
}

// ---------------------------------------------------------------------------------------------------------------------
// Which tensor is what (docs/model.md, "11.3 Which tensor is which matrix")
// ---------------------------------------------------------------------------------------------------------------------

// The rule in symbols. v is an element's decoded value. A norm is the vector of gains g of a
// normalization layer, and a matrix reads a norm if its input is that layer's output. The rule
// multiplies the gains into such a matrix: gamma_j = RN32(g_j * c) is the multiplier of input index
// j, where the constant c depends on the matrix's kind, and p = RN32(gamma_j * v) is the element's
// product. In a tensor that reads no norm, p = v. mx is the largest |p| of a tensor. e is the
// tensor's exponent: choose_exponent(mx) for a matrix, and e_res (option --eres) for the embedding.
// w = RNE(2^e * p) is the element's integer. The page docs/model.md writes pi for p, c_0 for c and,
// in the block formats, h and h_min for d and dmin, mn for m, and c for the code q.

// The head dimension (metadata key attention.key_length) that the rule assumes when a file omits
// it (docs/model.md, "11.1 The container").
constexpr std::uint64_t kHeadDimension = 128;

// The bit pattern of a binary32 number.
inline std::uint32_t bits_of(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof bits);
  return bits;
}

// The two gain-folding constants (c_0, c_q) as binary32 bit patterns, for a model of this width and
// head dimension, by FlashProve's own rule (scales/README.md, "The constants of a norm"): with t
// the largest integer with 4^t <= width, c = sqrt(width) / 2^t (so 1 <= c < 2), c_0 = RN32(c) and
// c_q = RN32(c / sqrt(head_dimension)), c_q carrying attention's scale 1 / sqrt(head_dimension). At
// width 5120 head 128 these are 0x3F8F1BBD and 0x3DCA62C2; at 2048 and 64, 0x3FB504F3 and
// 0x3E3504F3 (docs/tinyllama/ARITHMETIC_CONTRACT.md 4.3). Generalizes the former literals, which
// were right for width 5120 and head 128 only.
inline std::pair<std::uint32_t, std::uint32_t> fold_constants(std::uint64_t width, std::uint64_t head_dimension) {
  int t = 0;
  while ((std::uint64_t)1 << (2 * (t + 1)) <= width) ++t; // 4^(t+1) <= width
  const double c = std::sqrt((double)width) / (double)((std::uint64_t)1 << t);
  return {bits_of((float)c), bits_of((float)(c / std::sqrt((double)head_dimension)))};
}

// A tensor with the role matrix or embedding is committed. The output head has the role matrix.
enum class Role { ignored, norm, matrix, embedding };

// What the rule does with a tensor.
struct Use {
  Role role = Role::ignored;
  std::string norm; // for a matrix: the name of the norm whose gains it reads, or empty
  bool query = false; // for a matrix that reads a norm: true if it uses c_q (c_0 / sqrt(dh)), else c_0
};

// One kind of tensor in a layer, "blk.<layer>.<kind>.weight". A kind with role matrix is committed.
// For `query`, see `Use`.
struct LayerKind {
  const char *kind;
  Role role;
  const char *norm; // the kind of the norm that the matrix reads, or nullptr
  bool query;
};

constexpr LayerKind kLayerKinds[] = {
    {"attn_norm", Role::norm, nullptr, false},
    {"ffn_norm", Role::norm, nullptr, false},
    {"attn_q", Role::matrix, "attn_norm", true},
    {"attn_k", Role::matrix, "attn_norm", false},
    {"attn_v", Role::matrix, "attn_norm", false},
    {"attn_output", Role::matrix, nullptr, false},
    {"ffn_gate", Role::matrix, "ffn_norm", false},
    {"ffn_up", Role::matrix, "ffn_norm", false},
    {"ffn_down", Role::matrix, nullptr, false},
    {"ffn_gate_inp", Role::matrix, "ffn_norm", false},
    {"ffn_gate_shexp", Role::matrix, "ffn_norm", false},
    {"ffn_up_shexp", Role::matrix, "ffn_norm", false},
    {"ffn_down_shexp", Role::matrix, nullptr, false},
    {"ffn_gate_exps", Role::matrix, "ffn_norm", false},
    {"ffn_up_exps", Role::matrix, "ffn_norm", false},
    {"ffn_down_exps", Role::matrix, nullptr, false},
};

// Returns how the rule uses the tensor `name` in a model with `layers` layers. A layer number has
// one to nine decimal digits (a limit of this program) and no leading zero, except in "0". A name
// of any other form, or of an unknown kind, is ignored. A name of a known kind whose layer number
// is `layers` or more is an error.
inline Use classify(const std::string &name, std::uint64_t layers) {
  if (name == "token_embd.weight") return {Role::embedding, "", false};
  if (name == "output.weight") return {Role::matrix, "output_norm.weight", false};
  if (name == "output_norm.weight") return {Role::norm, "", false};

  const std::string prefix = "blk.";
  if (name.compare(0, prefix.size(), prefix) != 0) return {};
  const std::size_t dot = name.find('.', prefix.size());
  if (dot == std::string::npos) return {};
  const std::string digits = name.substr(prefix.size(), dot - prefix.size());
  if (digits.empty() || digits.size() > 9 || digits.find_first_not_of("0123456789") != std::string::npos) return {};
  if (digits.size() > 1 && digits[0] == '0') return {};
  const std::string rest = name.substr(dot + 1);

  for (const LayerKind &known : kLayerKinds) {
    if (rest != std::string(known.kind) + ".weight") continue;
    if (std::stoull(digits) >= layers) {
      fail(Status::container, "The layer number of the tensor " + name + " is block_count or more. block_count comes " +
                                  "from the first file named. Check that all files belong to one model and that its " +
                                  "first file is named first.");
    }
    Use use;
    use.role = known.role;
    if (known.norm != nullptr) use.norm = prefix + digits + "." + known.norm + ".weight";
    use.query = known.query;
    return use;
  }
  return {};
}

// ---------------------------------------------------------------------------------------------------------------------
// The stored blocks and their values (docs/model.md, "11.2 Stored blocks and their values")
// ---------------------------------------------------------------------------------------------------------------------

// The 6-bit scale and the 6-bit minimum of sub-block j (0 to 7; each has 32 elements) of a Q4_K or
// Q5_K super-block, from its 12 scale bytes `s`. The function follows get_scale_min_k4 in
// ggml/src/ggml-quants.c: it has the same two cases and the same four expressions (MIT licence; see
// the head of this file).
inline void scale_and_minimum(const std::uint8_t *s, int j, std::uint8_t &scale, std::uint8_t &minimum) {
  if (j < 4) {
    scale = s[j] & 63;
    minimum = s[j + 4] & 63;
  } else {
    scale = static_cast<std::uint8_t>((s[j + 4] & 15) | (s[j - 4] >> 6) << 4);
    minimum = static_cast<std::uint8_t>((s[j + 4] >> 4) | (s[j] >> 6) << 4);
  }
}

// Decodes one super-block of Q4_K (144 bytes: d, dmin, 12 scale bytes, 128 code bytes) or of Q5_K
// (176 bytes: d, dmin, 12 scale bytes, 32 bytes of fifth bits, 128 bytes of low code bits). The two
// differ only in the fifth bits, so fifth_bits is nullptr for Q4_K. Sub-block j holds elements 32 j
// to 32 j + 31. Its codes are in the 32 bytes that start at codes[32 * floor(j / 2)], in the low
// half-bytes for even j and in the high half-bytes for odd j. Bit j of fifth_bits[i] is the fifth
// bit of element i of sub-block j. With d and dmin widened from binary16, sc and m the scale and
// the minimum of the sub-block, and q the element's code:
//
//   v = RN32(RN32(RN32(d * sc) * q) - RN32(dmin * m))
inline void decode_q4_or_q5(const std::uint8_t *block, const std::uint8_t *fifth_bits, const std::uint8_t *codes,
                            float *out) {
  const float d = widen_binary16(load_le16(block));
  const float dmin = widen_binary16(load_le16(block + 2));
  for (int j = 0; j < 8; ++j) {
    std::uint8_t scale, minimum;
    scale_and_minimum(block + 4, j, scale, minimum);
    const float step = d * static_cast<float>(scale);
    const float lowest = dmin * static_cast<float>(minimum);
    const std::uint8_t *bytes = codes + 32 * (j / 2);
    const int shift = 4 * (j % 2);
    for (int i = 0; i < 32; ++i) {
      int q = (bytes[i] >> shift) & 15;
      if (fifth_bits != nullptr) q += ((fifth_bits[i] >> j) & 1) << 4;
      const float product = step * static_cast<float>(q);
      out[32 * j + i] = product - lowest;
    }
  }
}

// Decodes one super-block of Q6_K (210 bytes: 128 bytes ql, 64 bytes qh, 16 signed one-byte scales,
// d). Element 128 a + 32 t + c (a < 2, t < 4, c < 32) has the four low bits of its code in
// ql[64 a + 32 (t mod 2) + c], in the low half-byte for t < 2 and in the high half-byte otherwise,
// and the two high bits in bits 2 t and 2 t + 1 of qh[32 a + c]. The code q is that 6-bit number
// minus 32. Each run of 16 elements has its own scale s. With d widened from binary16:
//
//   v = RN32(RN32(d * s) * q)
inline void decode_q6(const std::uint8_t *block, float *out) {
  const std::uint8_t *ql = block;
  const std::uint8_t *qh = block + 128;
  const std::uint8_t *scales = block + 192;
  const float d = widen_binary16(load_le16(block + 208));
  float steps[16];
  for (int k = 0; k < 16; ++k) steps[k] = d * static_cast<float>(static_cast<std::int8_t>(scales[k]));
  for (int a = 0; a < 2; ++a) {
    for (int t = 0; t < 4; ++t) {
      for (int c = 0; c < 32; ++c) {
        const int low = (ql[64 * a + 32 * (t % 2) + c] >> (4 * (t / 2))) & 15;
        const int high = (qh[32 * a + c] >> (2 * t)) & 3;
        const int n = 128 * a + 32 * t + c;
        out[n] = steps[n / 16] * static_cast<float>((low | high << 4) - 32);
      }
    }
  }
}

// Decodes one block of Q8_0 (34 bytes: d as binary16, then 32 signed 8-bit codes q).
//
//   v = RN32(d * q)
//
// The product is exact in binary32: d has at most 11 significant bits and q at most 8.
// Adapted from the package gguf, version 0.19.0, gguf/quants.py, Q8_0.dequantize_blocks, which splits a block after its
// first 2 bytes, widens d from float16 and the codes from int8 to float32, and multiplies. Public sources: llama.cpp
// (github.com/ggml-org/llama.cpp), ggml-quants.h at tag b2261, read in a third party's public mirror,
// not in the project's own repository:
// "#define QK8_0 32", and the structure block_q8_0, whose two members are "ggml_fp16_t d;" and "int8_t  qs[QK8_0];"; the
// project's wiki page "Tensor Encoding Schemes" and Hugging Face, "GGUF" (https://huggingface.co/docs/hub/gguf): "Each
// block has 32 weights", "w = q * block_scale".
inline void decode_q8_0(const std::uint8_t *block, float *out) {
  const float d = widen_binary16(load_le16(block));
  for (std::size_t i = 0; i < kBlockQ80; ++i) out[i] = d * static_cast<float>(static_cast<std::int8_t>(block[2 + i]));
}

// Decodes a run of elements to their values v. For Q4_K, Q5_K and Q6_K, `elements` must be a
// multiple of kSuperBlock, and for Q8_0 of kBlockQ80, which is not checked here. Like storage_of(),
// it treats a `type` that is none of the seven as F32.
//
// F16 stores binary16 numbers and BF16 the upper 16 bits of binary32 numbers, both little-endian, and v is the stored
// number, widened exactly. Public sources: the package gguf, version 0.19.0, gguf/quants.py: dequantize() views F16
// data as float16 and converts them to float32, and BF16.dequantize_blocks, from which the case BF16 is adapted, shifts
// the 16 bits left by 16 and views the result as float32; Hugging Face, "GGUF" (https://huggingface.co/docs/hub/gguf):
// F16, "16-bit standard IEEE 754 half-precision floating-point number"; BF16, "16-bit shortened version of the 32-bit
// IEEE 754 single-precision floating-point number".
inline void decode(std::uint32_t type, const std::uint8_t *bytes, std::size_t elements, float *out) {
  const std::size_t blocks = elements / kSuperBlock;
  switch (type) {
  case kTypeQ4K:
    for (std::size_t b = 0; b < blocks; ++b, bytes += 144, out += kSuperBlock) {
      decode_q4_or_q5(bytes, nullptr, bytes + 16, out);
    }
    break;
  case kTypeQ5K:
    for (std::size_t b = 0; b < blocks; ++b, bytes += 176, out += kSuperBlock) {
      decode_q4_or_q5(bytes, bytes + 16, bytes + 48, out);
    }
    break;
  case kTypeQ6K:
    for (std::size_t b = 0; b < blocks; ++b, bytes += 210, out += kSuperBlock) decode_q6(bytes, out);
    break;
  case kTypeQ80:
    for (std::size_t b = 0; b < elements / kBlockQ80; ++b, bytes += 34, out += kBlockQ80) decode_q8_0(bytes, out);
    break;
  case kTypeF16:
    for (std::size_t i = 0; i < elements; ++i) out[i] = widen_binary16(load_le16(bytes + 2 * i));
    break;
  case kTypeBF16:
    for (std::size_t i = 0; i < elements; ++i) out[i] = float_from_bits(std::uint32_t{load_le16(bytes + 2 * i)} << 16);
    break;
  default: // F32
    for (std::size_t i = 0; i < elements; ++i) out[i] = float_from_bits(load_le32(bytes + 4 * i));
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// From a tensor to its line of the sums file
// ---------------------------------------------------------------------------------------------------------------------

// The integers are defined in docs/model.md, "11.4 From a decoded value to an integer".
// The line's form is defined in weight_sums/README.md, "The file's form".

// One committed tensor, with the norm that it reads, if any. For `constant`, see `Use`.
struct Job {
  const Tensor *tensor = nullptr;
  const Tensor *norm = nullptr;
  std::uint32_t constant = 0;
  bool embedding = false;
};

// A tensor's line of the sums file. `bits` is width_in_bits() of its integers, and `hash` is their
// SHA-256 digest in hexadecimal.
struct Line {
  std::string name;
  std::uint64_t elements = 0;
  int exponent = 0;
  int bits = 0;
  std::string hash;
};

// Fails because of the model's own values or shape, which the user cannot change.
[[noreturn]] inline void outside_the_rule(Status status, const std::string &what) {
  fail(status, what + " The conversion rule does not cover such a model.");
}

inline bool all_finite(const float *values, std::size_t count) {
  bool finite = true;
  for (std::size_t i = 0; i < count; ++i) finite &= std::isfinite(values[i]);
  return finite;
}

// A tensor is read in chunks of this many elements, a multiple of kSuperBlock, so that memory use
// does not grow with the tensor. If the embedding has both a value that is not finite and an
// integer that is too large, the chunk size can decide which error is reported.
constexpr std::size_t kChunk = 1024 * kSuperBlock;

// Calls consume(products, count) for each chunk of the tensor's products p, in the file's element
// order. If `multipliers` is empty, p = v. Otherwise multipliers[j] is gamma_j, where j is the
// element's index along the first dimension, which varies fastest in the file and must have
// multipliers.size() elements (not checked here). Fails if a value or a product is not finite, or
// if the file cannot be read.
template <typename Consumer>
void for_each_product(const Tensor &tensor, const std::vector<float> &multipliers, Consumer consume) {
  std::ifstream file(tensor.path, std::ios::binary);
  file.seekg(static_cast<std::streamoff>(tensor.position));
  const Storage storage = storage_of(tensor.type);
  std::vector<std::uint8_t> bytes(kChunk / storage.block_elements * storage.block_bytes);
  std::vector<float> values(kChunk);
  std::size_t j = 0;
  for (std::uint64_t done = 0; done < tensor.elements;) {
    const auto count = static_cast<std::size_t>(std::min<std::uint64_t>(kChunk, tensor.elements - done));
    file.read(reinterpret_cast<char *>(bytes.data()),
              static_cast<std::streamsize>(count / storage.block_elements * storage.block_bytes));
    if (!file) fail(Status::command_or_file, tensor.path + " cannot be read: the system reported an input error.");
    decode(tensor.type, bytes.data(), count, values.data());
    if (!all_finite(values.data(), count)) {
      outside_the_rule(Status::not_finite, "The tensor " + tensor.name + " holds a value that is not finite.");
    }
    if (!multipliers.empty()) {
      for (std::size_t i = 0; i < count; ++i) {
        values[i] = multipliers[j] * values[i];
        if (++j == multipliers.size()) j = 0;
      }
      if (!all_finite(values.data(), count)) {
        outside_the_rule(Status::not_finite, "A value of the tensor " + tensor.name +
                                                 " is not finite after the multiplication by its norm.");
      }
    }
    consume(values.data(), count);
    done += count;
  }
}

// The multipliers gamma_j of a norm, for the constant c.
inline std::vector<float> read_multipliers(const Tensor &norm, float constant) {
  std::vector<float> multipliers;
  for_each_product(norm, {}, [&](const float *gains, std::size_t count) {
    multipliers.insert(multipliers.end(), gains, gains + count);
  });
  for (float &multiplier : multipliers) {
    multiplier = multiplier * constant;
    if (!std::isfinite(multiplier)) {
      outside_the_rule(Status::not_finite, "The norm " + norm.name + " gives a multiplier that is not finite.");
    }
    if (multiplier != 0.0f && std::fabs(multiplier) < 0x1p-100f) {
      outside_the_rule(Status::bound, "The norm " + norm.name +
                                          " gives a multiplier that is not 0 but is below 2^-100 in absolute value.");
    }
  }
  return multipliers;
}

// The largest integer e with RNE(2^e * mx) <= 4,095, for a finite mx > 0. RNE(2^e * mx) never
// decreases as e grows. With k = floor(log2(mx)), the number 2^(12 - k) * mx is at least 4,096, so
// 12 - k and every larger e fail. The search goes down from 12 - k, and the first e that passes is
// the largest one.
inline int choose_exponent(float mx) {
  int e = 12 - std::ilogb(mx);
  while (round_scaled(mx, power_of_two(e)) > 4095.0) --e;
  return e;
}

// The smallest b >= 1 with -2^(b-1) <= w < 2^(b-1) for every w from `lowest` to `highest`.
inline int width_in_bits(std::int64_t lowest, std::int64_t highest) {
  int bits = 1;
  while (lowest < -(std::int64_t{1} << (bits - 1)) || highest >= (std::int64_t{1} << (bits - 1))) ++bits;
  return bits;
}

// Applies the rule to one committed tensor. A matrix is read twice: once for mx, which fixes the
// exponent, and once for the integers. The embedding takes e_res as its exponent and is read once.
inline Line convert(const Job &job, int eres) {
  const Tensor &tensor = *job.tensor;
  std::vector<float> multipliers;
  if (job.norm != nullptr) multipliers = read_multipliers(*job.norm, float_from_bits(job.constant));

  int exponent = eres;
  if (!job.embedding) {
    float mx = 0.0f;
    for_each_product(tensor, multipliers, [&](const float *products, std::size_t count) {
      for (std::size_t i = 0; i < count; ++i) mx = std::max(mx, std::fabs(products[i]));
    });
    if (mx == 0.0f) {
      outside_the_rule(Status::bound, "Every value of the tensor " + tensor.name + " is 0" +
                                          (job.norm != nullptr ? " after the multiplication by its norm." : "."));
    }
    exponent = choose_exponent(mx);
    if (exponent < -100 || exponent > 100) {
      outside_the_rule(Status::bound, "The exponent of the tensor " + tensor.name + " is not between -100 and 100.");
    }
  }

  // The hash input is each integer in two's complement, little-endian: 2 bytes for a matrix, 4 for
  // the embedding. The bounds checked below make the integers fit.
  const std::size_t integer_bytes = job.embedding ? 4 : 2;
  const double scale = power_of_two(exponent);
  std::vector<std::uint8_t> hash_input(kChunk * integer_bytes);
  Sha256 hash;
  std::int32_t lowest = 0, highest = 0;
  // choose_exponent() keeps a matrix within 4,095, so one of its integers reaches `bound` only if
  // the file has changed since the first reading. A change that leaves every integer below the
  // bound is not caught. An integer of the embedding must be below 2^30 in absolute value.
  const double bound = job.embedding ? 0x1p30 : 4096.0;
  const char *advice = job.embedding ? "The bound is 2^30. Use a lower --eres."
                                     : "The bound is 4,095. The file must have changed during the run. Run the "
                                       "program again when nothing writes to the file.";
  for_each_product(tensor, multipliers, [&](const float *products, std::size_t count) {
    std::uint8_t *to = hash_input.data();
    for (std::size_t i = 0; i < count; ++i) {
      const double rounded = round_scaled(products[i], scale);
      if (!(std::fabs(rounded) < bound)) {
        fail(Status::bound, "An integer of the tensor " + tensor.name + " is too large. " + advice);
      }
      const auto w = static_cast<std::int32_t>(rounded);
      lowest = std::min(lowest, w);
      highest = std::max(highest, w);
      const auto word = static_cast<std::uint32_t>(w);
      for (std::size_t k = 0; k < integer_bytes; ++k) *to++ = static_cast<std::uint8_t>(word >> (8 * k));
    }
    hash.update(hash_input.data(), count * integer_bytes);
  });

  return {tensor.name, tensor.elements, exponent, width_in_bits(lowest, highest), to_hex(hash.finish())};
}

// ---------------------------------------------------------------------------------------------------------------------
// The whole run
// ---------------------------------------------------------------------------------------------------------------------

struct Options {
  int eres = kDefaultEres;
  int threads = kDefaultThreads;
  std::string out;
  std::vector<std::string> shards;
};

constexpr char kUsage[] =
    "usage: weight_sums [--eres <integer>] [--threads <integer>] --out <new file> <shard.gguf> [<shard.gguf> ...]\n"
    "  --eres     the exponent of the embedding, from -100 to 100 (default 11)\n"
    "  --threads  the number of threads, from 1 to 256 (default 8)\n"
    "  --out      the sums file to write; nothing may exist at this path or at <path>.part\n"
    "  Name every GGUF file of the model, its first file first.";

[[noreturn]] inline void wrong_command(const std::string &what) {
  fail(Status::command_or_file, what + "\n" + kUsage);
}

// Reads a decimal integer between `lowest` and `highest`, in the form that std::strtol reads: white
// space and a sign may come first, and nothing may follow the digits.
inline int parse_integer(const std::string &option, const std::string &word, int lowest, int highest) {
  char *end = nullptr;
  errno = 0;
  const long value = std::strtol(word.c_str(), &end, 10);
  if (end == word.c_str() || *end != '\0' || errno == ERANGE || value < lowest || value > highest) {
    wrong_command(option + " takes an integer from " + std::to_string(lowest) + " to " + std::to_string(highest) + ".");
  }
  return static_cast<int>(value);
}

// Options may come before, between or after the shards. Apart from the options and their values,
// an argument that begins with "--" is rejected, and one that begins with a single "-" is the path
// of a shard.
inline Options parse_command_line(const std::vector<std::string> &words) {
  Options options;
  std::vector<std::string> given;
  for (std::size_t i = 0; i < words.size(); ++i) {
    const std::string &word = words[i];
    if (word == "--eres" || word == "--threads" || word == "--out") {
      if (std::find(given.begin(), given.end(), word) != given.end()) wrong_command(word + " is given twice.");
      given.push_back(word);
      if (i + 1 == words.size()) wrong_command(word + " needs a value after it.");
      const std::string &value = words[++i];
      if (value.empty()) wrong_command(word + " needs a value that is not empty.");
      if (word == "--eres") options.eres = parse_integer(word, value, kLowestEres, kHighestEres);
      if (word == "--threads") options.threads = parse_integer(word, value, 1, kHighestThreads);
      if (word == "--out") options.out = value;
    } else if (word.empty()) {
      wrong_command("An argument is empty.");
    } else if (word.compare(0, 2, "--") == 0) {
      wrong_command(word + " is not an option of this program. The options are --eres, --threads and --out, each " +
                    "followed by its value as a separate argument.");
    } else {
      options.shards.push_back(word);
    }
  }
  if (options.out.empty()) wrong_command("--out <new file> is required.");
  if (options.shards.empty()) wrong_command("Name at least one GGUF file.");
  return options;
}

// Set when a line could not be written to standard error. print() then prints nothing more.
inline std::atomic<bool> g_silent{false};

// Prints one line to standard error. Threads may call it at the same time, because since C++17 a
// stream has a lock of its own (cppreference.com, "std::FILE"; the C standard).
inline void print(const std::string &sentence) {
  if (g_silent) return;
  const bool printed = std::fprintf(stderr, "weight_sums: %s\n", sentence.c_str()) >= 0;
  if (!printed || std::fflush(stderr) != 0) g_silent = true; // the flush matters if something made the stream buffered
}

// The file <out>.part, which exists and stays open from the constructor until complete(). It is
// created at the start, so that two runs cannot work on the same output path, and it becomes
// <out> at the end. A run that stops with an error removes it. A run that is killed leaves it.
// The program removes and overwrites only what it created itself, as long as no one else
// replaces <out>.part during the run. remove_part() checks that the name still leads to this
// run's file, but someone can replace the file between that check and the removal, and complete()
// renames or links whatever is at <out>.part. complete() creates <out> by a call that fails if
// anything is there. If no such call succeeds although nothing is at <out>, it renames <out>.part,
// and only this rename can replace a file, one that has appeared at <out> in the meantime.
//
// The caller must print a line before it creates this object. In a process that was started with
// standard error closed, the file can get the file descriptor of standard error. The earlier line
// has then failed and has made print() silent, so no progress line goes into the file.
class OutputFile {
public:
  explicit OutputFile(const std::string &out) : out_(out), part_(out + ".part") {
    // Neither message advises removing a file: if the user left out the new name after --out,
    // <out> is one of the model's files.
    require_free(out_, "Give --out the name of a new file. If " + out +
                           " is one of the model's files, the new file's name is missing after --out.");
    require_free(part_,
                 "Either another run is writing to this --out, or a run that was killed left the file. Give --out "
                 "another name.");
    // With O_EXCL the call fails if the name is taken. Anyone who reads the file can test any guess
    // at a tensor against it, so the file has mode 0600 from the moment it exists: only its owner
    // can read or write it, whatever the umask.
    const int made = ::open(part_.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    part_exists_ = made >= 0;
    file_ = made < 0 || ::fstat(made, &as_made_) != 0 ? nullptr : ::fdopen(made, "w");
    // If fdopen() has no memory for the stream, the new file is removed. If fstat() fails,
    // as_made_ identifies no file, so the new file stays.
    if (file_ == nullptr && made >= 0) {
      ::close(made);
      remove_part();
    }
    if (file_ == nullptr) {
      fail(Status::command_or_file,
           part_.string() + " cannot be created. Check that its folder exists and can be written.");
    }
  }

  OutputFile(const OutputFile &) = delete;
  OutputFile &operator=(const OutputFile &) = delete;

  ~OutputFile() {
    if (file_ != nullptr) std::fclose(file_);
    remove_part();
  }

  // Writes the text and gives the file its final name, <out>. Call it once.
  void complete(const std::string &text) {
    const bool written = std::fwrite(text.data(), 1, text.size(), file_) == text.size();
    const bool closed = std::fclose(file_) == 0;
    file_ = nullptr;
    if (!written || !closed) {
      fail(Status::command_or_file, part_.string() + " cannot be written. Check the free space on the disk.");
    }
    // Two calls create a name and fail if it is taken: Linux's renameat2() with RENAME_NOREPLACE,
    // which some network file systems reject, and link(), which some other file systems reject.
#ifdef RENAME_NOREPLACE
    if (::renameat2(AT_FDCWD, part_.c_str(), AT_FDCWD, out_.c_str(), RENAME_NOREPLACE) == 0) {
      part_exists_ = false;
      return;
    }
#endif
    if (::link(part_.c_str(), out_.c_str()) == 0) {
      remove_part();
      return;
    }
    require_free(out_, "It appeared during this run, so the sums file is not written.");
    // Nothing is at <out>, so the calls above failed for another reason, such as a file system
    // that supports neither.
    std::error_code error;
    std::filesystem::rename(part_, out_, error);
    if (error) fail(Status::command_or_file, part_.string() + " cannot be renamed to " + out_.string() + ".");
    part_exists_ = false;
  }

private:
  // Removes <out>.part if part_exists_ and the name still leads to the file that this run created.
  // A file that someone else has put under the name stays.
  void remove_part() {
    struct stat now;
    const bool there = part_exists_ && ::lstat(part_.c_str(), &now) == 0;
    if (there && now.st_dev == as_made_.st_dev && now.st_ino == as_made_.st_ino) ::unlink(part_.c_str());
    part_exists_ = false;
  }

  // Fails if anything exists at the path, including a directory (the messages say "folder") or a
  // symbolic link that points nowhere, or if the path cannot be examined.
  static void require_free(const std::filesystem::path &path, const std::string &advice) {
    std::error_code error;
    const std::filesystem::file_type type = std::filesystem::symlink_status(path, error).type();
    if (type == std::filesystem::file_type::not_found) return;
    if (type == std::filesystem::file_type::none) {
      fail(Status::command_or_file,
           path.string() + " cannot be examined. Check the permissions of the folders on its path.");
    }
    fail(Status::command_or_file, path.string() + " exists already, and this program overwrites nothing. " + advice);
  }

  // The paths are built before the file is created. So after open() the constructor makes no
  // allocation that can throw until it has returned or called remove_part(): a throw there would
  // leave the file behind. The destructor needs no allocation to remove the file.
  std::filesystem::path out_;
  std::filesystem::path part_;
  std::FILE *file_ = nullptr;
  bool part_exists_ = false; // true while this run still has <out>.part to remove or rename
  struct stat as_made_ {};   // the device and the file number of <out>.part, taken from the open file when it was made
};

inline std::string seconds_since(std::chrono::steady_clock::time_point start) {
  const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
  char text[32];
  std::snprintf(text, sizeof text, "%.1f s", elapsed.count());
  return text;
}

// Finds the committed tensors and their norms in the shards. Checks their types, shapes and
// positions and the model's shape, without reading tensor data. The first shard's metadata
// describe the model. Completeness is not checked: a committed tensor that is in no shard gets no
// line (explained in weight_sums/README.md, "What the file does not show").
inline std::vector<Job> plan(std::vector<Shard> &shards) {
  const Shard &first = shards.front();
  const auto architecture = first.metadata.find("general.architecture");
  if (architecture == first.metadata.end() || architecture->second.type != kValueString ||
      architecture->second.text.empty()) {
    fail(Status::container, first.path + ": the metadata key general.architecture is absent, or is not a string, or " +
                                "is empty. Name the model's first file first.");
  }
  const std::string &prefix = architecture->second.text;
  const std::uint64_t width = required_number(first, prefix, "embedding_length");
  const std::uint64_t layers = required_number(first, prefix, "block_count");
  // By the rule, an absent attention.key_length means 128 (docs/model.md, "11.1 The container").
  // This program also treats a value that is not an unsigned integer as absent.
  const auto key_length = first.metadata.find(prefix + ".attention.key_length");
  const bool has_head_dimension = key_length != first.metadata.end() && key_length->second.is_unsigned();
  const std::uint64_t head_dimension = has_head_dimension ? key_length->second.number : kHeadDimension;
  // The rule's two constants are a function of the width and head dimension (fold_constants), so the
  // rule is defined for any positive whole width and head dimension, not only width 5120 and head
  // 128. A scale fit (scales/README.md) may still need a narrower range.
  if (width < 1) outside_the_rule(Status::bound, first.path + ": the embedding_length is not at least 1.");
  if (head_dimension < 1) outside_the_rule(Status::bound, first.path + ": the attention.key_length is not at least 1.");
  const auto [norm_constant, query_constant] = fold_constants(width, head_dimension);

  std::map<std::string, const Tensor *> by_name;
  std::vector<std::pair<const Tensor *, Use>> committed;
  for (Shard &shard : shards) {
    for (std::size_t i = 0; i < shard.tensors.size(); ++i) {
      Tensor &tensor = shard.tensors[i];
      if (!by_name.emplace(tensor.name, &tensor).second) {
        fail(Status::container, shard.path + ": entry " + std::to_string(i + 1) + " of the tensor table has the same " +
                                    "name as an earlier tensor of this command. Check that no file is named twice.");
      }
      const Use use = classify(tensor.name, layers);
      if (use.role == Role::ignored) continue;
      // From here on a message may give the tensor's name, because the name has one of the
      // forms that classify() knows. (The message in classify() gives only such a name.) In a
      // damaged file, any other name can be tensor data: see read_shard().
      if (use.role == Role::norm) {
        if (tensor.type != kTypeF32) fail(Status::type, "The norm " + tensor.name + " must have the storage type F32.");
        if (tensor.elements != width) {
          fail(Status::container, "The norm " + tensor.name + " does not have " + std::to_string(width) + " elements.");
        }
      } else {
        if (tensor.type != kTypeF32 && tensor.type != kTypeQ4K && tensor.type != kTypeQ5K && tensor.type != kTypeQ6K &&
            tensor.type != kTypeQ80 && tensor.type != kTypeF16 && tensor.type != kTypeBF16) {
          fail(Status::type, "The tensor " + tensor.name + " has a storage type that this program does not read. It " +
                                 "reads F32, F16, BF16, Q8_0, Q4_K, Q5_K and Q6_K.");
        }
        committed.emplace_back(&tensor, use);
      }
      locate(tensor, shard);
    }
  }

  std::vector<Job> jobs;
  for (const auto &[tensor, use] : committed) {
    Job job;
    job.tensor = tensor;
    job.embedding = use.role == Role::embedding;
    if (!use.norm.empty()) {
      if (tensor->dims[0] != width) {
        fail(Status::container, "The tensor " + tensor->name + " reads a norm, so its first dimension must be " +
                                    std::to_string(width) + ".");
      }
      const auto norm = by_name.find(use.norm);
      if (norm == by_name.end()) {
        fail(Status::container, "The tensor " + tensor->name + " reads the norm " + use.norm +
                                    ", which is in none of the files. Name every file of the model in one command.");
      }
      job.norm = norm->second;
      job.constant = use.query ? query_constant : norm_constant;
    }
    jobs.push_back(job);
  }
  // The order of the sums file: by name, as byte strings.
  std::sort(jobs.begin(), jobs.end(), [](const Job &a, const Job &b) { return a.tensor->name < b.tensor->name; });
  return jobs;
}

// Converts all jobs on up to `threads` threads and returns their lines in the order of the jobs.
// For a given input, the lines, or the error that is reported, do not depend on the number of
// threads or on their timing. (Whether a read error or a failed allocation happens can depend on
// the run.)
//
// If several tensors have an error, the first of them in the order of the jobs is reported. A
// thread reads the flag `stop` only before it takes a tensor, and it always finishes a tensor that
// it has taken. The tensors are taken in the order of the jobs, so when any tensor fails, every
// tensor before it has been taken and will be finished.
inline std::vector<Line> convert_all(const std::vector<Job> &jobs, int eres, std::size_t threads) {
  std::vector<Line> lines(jobs.size());
  std::vector<std::exception_ptr> errors(jobs.size());
  std::atomic<std::size_t> next{0};
  std::atomic<std::size_t> finished{0};
  std::atomic<bool> stop{false};

  const auto work = [&] {
    while (!stop) {
      const std::size_t i = next++;
      if (i >= jobs.size()) return;
      try {
        const auto start = std::chrono::steady_clock::now();
        lines[i] = convert(jobs[i], eres);
        print(std::to_string(++finished) + " of " + std::to_string(jobs.size()) + ": " + lines[i].name + ", " +
              std::to_string(lines[i].elements) + " elements, " + seconds_since(start));
      } catch (...) {
        errors[i] = std::current_exception();
        stop = true;
      }
    }
  };

  // This thread works too, so it starts one thread fewer than the given number.
  std::vector<std::thread> others;
  others.reserve(threads);
  try {
    while (others.size() + 1 < threads) others.emplace_back(work);
  } catch (...) {
    // The system cannot start another thread, so fewer threads do the work.
  }
  work();
  for (std::thread &thread : others) thread.join();
  for (const std::exception_ptr &error : errors) {
    if (error) std::rethrow_exception(error);
  }
  return lines;
}

// The text of the sums file. plan() has already put the lines in the order of their names.
inline std::string sums_file(const std::vector<Line> &lines) {
  std::string text;
  for (const Line &line : lines) {
    text += line.name + ' ' + std::to_string(line.elements) + ' ' + std::to_string(line.exponent) + ' ' +
            std::to_string(line.bits) + ' ' + line.hash + '\n';
  }
  text += "END " + std::to_string(lines.size()) + '\n';
  return text;
}

inline void run(const std::vector<std::string> &words) {
  const auto start = std::chrono::steady_clock::now();
  const Options options = parse_command_line(words);
  print("shards: " + std::to_string(options.shards.size())); // before the output file is made: see OutputFile
  OutputFile output(options.out);
  std::vector<Shard> shards;
  for (const std::string &path : options.shards) shards.push_back(read_shard(path));
  const std::vector<Job> jobs = plan(shards);
  std::size_t tensors = 0;
  for (const Shard &shard : shards) tensors += shard.tensors.size();
  // More threads than tensors would have nothing to do.
  const std::size_t threads = std::clamp<std::size_t>(jobs.size(), 1, static_cast<std::size_t>(options.threads));
  print("tensors: " + std::to_string(tensors) + ", committed tensors: " + std::to_string(jobs.size()) +
        ", threads: " + std::to_string(threads));
  const std::vector<Line> lines = convert_all(jobs, options.eres, threads);
  // The last line is built first, so that no allocation can fail once the sums file is in place.
  const std::string last_line = "wrote " + options.out + ", " + seconds_since(start);
  output.complete(sums_file(lines));
  print(last_line);
}

} // namespace weight_sums

// weight_sums/tests/unit_test.cpp includes this file for its functions and defines its own main().
//
// main() skips argv[0], which is absent if argc is 0. It reports a lack of memory without print(),
// whose std::string argument would need memory.
#ifndef WEIGHT_SUMS_WITHOUT_MAIN
int main(int argc, char **argv) {
  using namespace weight_sums;
  try {
    run(std::vector<std::string>(argv + std::min(argc, 1), argv + argc));
    return 0;
  } catch (const Failure &failure) {
    print(failure.message);
    return static_cast<int>(failure.status);
  } catch (const std::bad_alloc &) {
    std::fputs("weight_sums: There is not enough memory. Try a lower --threads.\n", stderr);
    return static_cast<int>(Status::memory);
  }
}
#endif
