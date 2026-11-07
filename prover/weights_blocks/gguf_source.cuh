// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// GgufSource: the BlockSource (weight reader's interface) that reads a model from the GGUF files
// ("shards") in one directory, checks them against `shape` and copies tensors to the GPU as
// stored. wsums.txt there gives a tensor's exponent e, fold.txt the embedding's: the conversion to
// integers (prover/weights_blocks/fold_rule.h) scales by 2^e.
// BlockSource and the tensor numbers (`stream`; 13 is the embedding, 14 the output head) are in
// prover/weights_blocks/block_weights.h, `fold::` in prover/weights_commitment/fold_weights.cuh.
// Include prover/prover/fork_memory.h first, for its allocators.
#pragma once
#include "common/end_name.h"
#include "weights_commitment/fold_weights.cuh"
#include "weights_blocks/block_weights.h"
#include <cerrno>

namespace wb {

struct GgufSource : BlockSource {
  Shape shape; // the expected shape, from the prover's parameter file; set it before open()

  // Exits with a message if a file is missing or malformed, or the model and `shape` disagree.
  // fold::reads() computes its constants from the width and head dimension, so any shape is read;
  // the check below still requires the model to match the parameter file's shape.
  // All files are opened through one directory handle (gguf::Model::open_folder(),
  // prover/weights_commitment/gguf_file.h), so `shards`, an earlier listing, is used only in
  // messages; it must not be empty.
  void open(const std::string &setup_dir, const std::vector<std::string> &shards) {
    file_.open_folder(setup_dir);
    if ((uint64_t)shape.d != file_.embedding_length || (uint64_t)shape.dh != file_.key_length ||
        (uint64_t)shape.layers != file_.block_count)
      stop("the model has " + std::to_string(file_.block_count) + " layers of width " +
               std::to_string(file_.embedding_length) + ", but the parameter file has " + std::to_string(shape.layers) +
               " layers of width " + std::to_string(shape.d) + ", or the head widths differ",
           shards[0]);
    for (int l = 0; l < shape.layers; l++)
      if (file_.has("blk." + std::to_string(l) + ".ffn_gate_exps.weight") != shape.has_experts(l))
        stop(shape.has_experts(l) ? "the parameter file gives the layer experts, but the model has none"
                                  : "the model has experts in the layer, but the parameter file gives it none",
             "layer " + std::to_string(l));
    read_exponents(setup_dir + "/wsums.txt");
    read_fold_words(setup_dir + "/fold.txt");
    for (auto &s : stage_) CK(fork_malloc_host(&s, PIECE));
    for (auto &e : staged_) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    CK(fork_alloc(&fault_, 4, Place::Own));
    CK(cudaMemset(fault_, 0, 4));
  }

  int type(int stream, int layer) override {
    switch (file_.at(name(stream, layer)).type) {
    case gguf::Q4_K:
      return Q4_K;
    case gguf::Q5_K:
      return Q5_K;
    case gguf::Q6_K:
      return Q6_K;
    case gguf::F32:
      return F32;
    case gguf::Q8_0: // this type and the next two: their numbers' and sizes' source is named in gguf_file.h, at Type
      return Q8_0;
    case gguf::F16:
      return F16;
    case gguf::BF16:
      return BF16;
    default:
      stop("the tensor's type is not one of Q4_K, Q5_K, Q6_K, F32, Q8_0, F16 and BF16", name(stream, layer));
    }
  }

  bool keeps_on_host() override { return true; }

  FileTensor tensor(int stream, int layer, uint8_t *into, size_t room, cudaStream_t reader, uint8_t *keep) override {
    const std::string n = name(stream, layer);
    const gguf::Tensor &t = file_.at(n);
    const size_t bytes = t.bytes();
    for (uint64_t axis : t.dims)
      if (axis > (1u << 30)) stop("the tensor has an axis longer than 2^30", n);
    if (t.dims[3] != 1) stop("the tensor has four axes; at most three are supported", n);
    if (t.dims[0] % 256) stop("the length of the tensor's rows is not a multiple of 256 elements", n);
    if (into && bytes != room)
      stop("the tensor has " + std::to_string(bytes) + " bytes, but the parameter file's shape implies " +
               std::to_string(room),
           n);
    if (!into) {
      CK(cudaStreamSynchronize(reader));
      if (bytes > room_bytes_) {
        if (room_) CK(fork_free(room_));
        CK(fork_alloc(&room_, bytes, Place::Own));
        room_bytes_ = bytes;
      }
      into = room_;
    }
    // Reading overlaps copying: each piece goes into its own part of `keep` if given, else into one
    // of two staging buffers, refilled only after its last copy has finished.
    int s = 0;
    for (size_t at = 0; at < bytes; at += PIECE, s ^= 1) {
      const size_t piece = std::min(PIECE, bytes - at);
      if (keep) {
        file_.read(t, at, piece, keep + at);
        CK(cudaMemcpyAsync(into + at, keep + at, piece, cudaMemcpyHostToDevice, reader));
        continue;
      }
      CK(cudaEventSynchronize(staged_[s]));
      file_.read(t, at, piece, stage_[s]);
      CK(cudaMemcpyAsync(into + at, stage_[s], piece, cudaMemcpyHostToDevice, reader));
      CK(cudaEventRecord(staged_[s], reader));
    }
    file_.drop_from_page_cache(t); // once per tensor, not per piece: a page that two pieces share would stay
    FileTensor r;
    r.bytes = into, r.type = type(stream, layer), r.rows = t.dims[1] * t.dims[2], r.inputs = (int)t.dims[0];
    return r;
  }

  // Returns, in GPU memory, the multipliers gamma[j] = g[j] * c_0, rounded to binary32, or null if
  // the tensor reads no norm. g is the norm's weights ("gains"), c_0 a constant; fold::reads()
  // gives the norm and c_word, c_0's binary32 bit pattern. Cached per (norm, c_word).
  // gamma is defined in docs/model.md, "11.4 From a decoded value to an integer" (there gamma_k).
  const float *gain(int stream, int layer) override {
    const fold::Reads reads = fold::reads(kind(stream, layer), shape.d, shape.dh);
    if (!reads.norm[0]) return nullptr;
    const std::string n =
        (stream == 14 ? std::string() : "blk." + std::to_string(layer) + ".") + reads.norm + ".weight";
    float *&gamma = gains_[{n, reads.c_word}];
    if (gamma) return gamma;
    const gguf::Tensor &g = file_.at(n);
    if (g.type != gguf::F32 || g.cells() != file_.embedding_length)
      stop("the norm must have " + std::to_string(file_.embedding_length) + " values of type F32", n);
    // The gains pass through a page-locked staging buffer and are zeroed there: they are weights.
    CK(cudaEventSynchronize(staged_[0]));
    file_.read(g, 0, g.bytes(), stage_[0]);
    float *stored = nullptr;
    unsigned fault = 0;
    CK(fork_alloc(&stored, g.bytes(), Place::Own));
    CK(fork_alloc(&gamma, g.bytes(), Place::Own));
    // 0xFF bytes: every float a NaN. If the launch below never runs and no call reports it,
    // BlockWeights::fill() rejects the NaNs (k_gain_range()). Zeros would pass there, and the
    // prover would start with wrong weights.
    CK(cudaMemset(gamma, 0xFF, g.bytes()));
    CK(cudaMemcpy(stored, stage_[0], g.bytes(), cudaMemcpyHostToDevice));
    explicit_bzero(stage_[0], g.bytes());
    fold::gains<<<64, 256>>>(stored, reads.c_word, g.cells(), gamma, fault_);
    CK(cudaMemcpy(&fault, fault_, 4, cudaMemcpyDeviceToHost));
    CK(fork_free(stored));
    if (fault)
      stop("a gain of the norm, or its product with the constant c_0, is not finite, or the product is not "
           "zero but smaller than 2^-100 in magnitude",
           n);
    return gamma;
  }

  float scale(int stream, int layer) override {
    if (stream == 13) return ldexpf(1.f, embedding_exponent_);
    auto it = exponent_.find(name(stream, layer));
    if (it == exponent_.end()) stop("wsums.txt has no line for the tensor", name(stream, layer));
    return ldexpf(1.f, it->second);
  }

  void all_read() override {
    if (room_) CK(fork_free(room_));
    room_ = nullptr, room_bytes_ = 0;
  }

private:
  static constexpr size_t PIECE = (size_t)64 << 20;

  [[noreturn]] static void stop(const std::string &what, const std::string &where) {
    printf("weights: %s: %s\n", gguf::shown(where).c_str(), what.c_str());
    name_the_end("cp-weights-files");
    exit(3);
  }
  // The tensor's name without "blk.<layer>." and ".weight". In a layer with experts, streams 6 to 8
  // (the shared expert's) end in "_shexp". `stream` (2 to 14) is not range-checked.
  std::string kind(int stream, int layer) const {
    static const char *names[15] = {
        "",       "",         "attn_q",       "attn_k",        "attn_v",      "attn_output",   "ffn_gate",
        "ffn_up", "ffn_down", "ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "token_embd",
        "output"};
    return std::string(names[stream]) + (stream >= 6 && stream <= 8 && shape.has_experts(layer) ? "_shexp" : "");
  }
  std::string name(int stream, int layer) const {
    return (stream >= 13 ? std::string() : "blk." + std::to_string(layer) + ".") + kind(stream, layer) + ".weight";
  }
  // wsums.txt has lines "name elements e bits sha256", then "END <number of lines>"
  // (weight_sums/README.md, "The file's form"). Only e is stored; bits and sha256 are not checked.
  void read_exponents(const std::string &path) {
    FILE *f = file_.side("wsums.txt");
    if (!f) stop(std::string("cannot be opened: ") + strerror(errno), path);
    char n[256], sum[80];
    unsigned long long cells = 0;
    int e, bits;
    bool ended = false;
    while (fscanf(f, "%255s", n) == 1) {
      if (!strcmp(n, "END")) {
        ended = fscanf(f, "%llu", &cells) == 1 && cells == exponent_.size();
        break;
      }
      if (fscanf(f, "%llu %d %d %79s", &cells, &e, &bits, sum) != 4 || !file_.has(n) || file_.at(n).cells() != cells ||
          !exponent_.insert({n, e}).second)
        stop("line " + std::to_string(exponent_.size() + 1) +
                 " is malformed, names a tensor for the second time, or names a tensor that the model does not have "
                 "or that has another number of elements",
             path);
    }
    fclose(f);
    if (!ended) stop("the last line must be \"END <number of lines before it>\"", path);
  }
  // `eres` is the embedding's exponent. `qkrows asis` states that the rows of attn_q and attn_k are
  // used in the GGUF file's order (weight_sums/README.md, "Make the file").
  void read_fold_words(const std::string &path) {
    FILE *f = file_.side("fold.txt");
    if (!f) stop(std::string("cannot be opened: ") + strerror(errno), path);
    char word[64], value[64];
    int rows_as_stored = 0, exponent = 0;
    while (fscanf(f, "%63s %63s", word, value) == 2) {
      char *end = value;
      if (!strcmp(word, "eres")) embedding_exponent_ = (int)strtol(value, &end, 10), exponent += *end ? 2 : 1;
      if (!strcmp(word, "qkrows")) rows_as_stored += strcmp(value, "asis") ? 2 : 1;
    }
    fclose(f);
    if (exponent != 1 || rows_as_stored != 1)
      stop("it must contain the lines \"eres <whole number>\" and \"qkrows asis\", each exactly once", path);
  }

  gguf::Model file_;
  std::map<std::string, int> exponent_;
  int embedding_exponent_ = 0;
  uint8_t *stage_[2] = {nullptr, nullptr}, *room_ = nullptr;
  cudaEvent_t staged_[2];
  size_t room_bytes_ = 0;
  unsigned *fault_ = nullptr;
  std::map<std::pair<std::string, uint32_t>, float *> gains_;
};

} // namespace wb
