// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Two of the committer program cp_commit's sources of weights (-weights): a directory with a GGUF
// model's files, wsums.txt and fold.txt, and the synthetic model ("made_up" in identifiers) of
// prover/weights_blocks/made_up_blocks.h. This file also names the tensors and checks their sizes.
// Its caller hf::PinnedWeights (prover/hash_commitment/hf_commit_weights.cuh) converts the tensors
// to the committed integers ("fold" in identifiers) and compares their SHA-256 with wsums.txt.
#pragma once
#include <condition_variable>
#include <mutex>
#include "weights_blocks/made_up_blocks.h"
#include "weights_commitment/fold_weights.cuh"
#include "weights_commitment/sha256.h"
namespace rf {
// The synthetic model of -weights made-up-blocks, for benchmarks and tests. Its block bytes, norm
// gains and sizes must match wb::MadeUpSource (prover/weights_blocks/made_up_source.cuh), which the
// prover cp_prove reads for a commitment to it. A stream is a kind of tensor, numbered as there:
// 2 q, 3 k, 4 v, 5 o, 6 gate, 7 up, 8 down, 9 router, 10 to 12 the routed gate, up and down, 13 embedding, 14 head.
// H, G: query and key/value heads; f, fe: dense and expert feed-forward widths; NE: routed experts.
// Public source: synthetic weights: vLLM's load format "dummy": vLLM documentation, v0.4.0.post1,
// "Engine Arguments", --load-format, https://docs.vllm.ai/en/v0.4.0.post1/models/engine_args.html.
// Public source: likewise trtllm-bench's --no_weights_loading: NVIDIA, TensorRT-LLM documentation,
// "trtllm-bench", build options, https://nvidia.github.io/TensorRT-LLM/commands/trtllm-bench.html.
struct MadeUpModel {
  u64 seed = 0;
  size_t d = 0, H = 0, G = 0, dh = 0, f = 0, fe = 0, NE = 0, V = 0;
  bool experts = false; // if set, every odd layer has experts
  // A stream's tensor in a layer: output rows (of one expert) and input columns, as in wb::Shape.
  size_t rows(int stream, u64 layer) const {
    switch (stream) {
    case 2:
      return H * dh;
    case 3:
    case 4:
      return G * dh;
    case 6:
    case 7:
      return experts && (layer & 1) ? fe : f;
    case 9:
      return NE;
    case 10:
    case 11:
      return fe;
    case 13:
    case 14:
      return V;
    default: // streams 5, 8, 12
      return d;
    }
  }
  size_t inputs(int stream, u64 layer) const {
    return stream == 5 ? H * dh : stream == 8 ? (experts && (layer & 1) ? fe : f) : stream == 12 ? fe : d;
  }
  // Parses [blk.<layer>.]<kind>[_shexp].weight ("blk." exactly when stream < 13; <layer> unchecked). Else false.
  static bool place(const std::string &name, int &stream, u64 &layer) {
    static const char *kinds[15] = {
        "",       "",         "attn_q",       "attn_k",        "attn_v",      "attn_output",   "ffn_gate",
        "ffn_up", "ffn_down", "ffn_gate_inp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "token_embd",
        "output"};
    layer = 0;
    std::string kind = name;
    if (kind.size() < 8 || kind.compare(kind.size() - 7, 7, ".weight")) return false;
    kind.resize(kind.size() - 7);
    const bool in_a_layer = !kind.compare(0, 4, "blk.");
    if (in_a_layer) {
      const size_t dot = kind.find('.', 4);
      if (dot == std::string::npos || dot == 4) return false;
      layer = strtoull(kind.c_str() + 4, nullptr, 10);
      kind = kind.substr(dot + 1);
    }
    if (kind.size() > 6 && !kind.compare(kind.size() - 6, 6, "_shexp")) kind.resize(kind.size() - 6);
    for (stream = 2; stream < 15; stream++)
      if (kind == kinds[stream]) return in_a_layer == (stream < 13);
    return false;
  }
};
// Writes a tensor's blocks (type 0: Q4_K, 1: Q5_K; in a build with -DMADE_UP_WIDTH=8, 16 or 17 every tensor has instead
// the type 4 (Q8_0), 5 (F16) or 6 (BF16): type_of() in prover/weights_blocks/made_up_blocks.h) expert by expert, like wb::k_made_up_blocks().
// The loop form is public: the grid-stride loop: M. Harris, "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops",
// NVIDIA's developer blog, https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/.
__global__ void k_made_up_tensor(int type, u64 seed, int stream, u64 layer, size_t rows_per_expert, size_t per_row,
                                 size_t blocks, uint8_t *out) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < blocks; i += (size_t)gridDim.x * blockDim.x) {
    const size_t row = i / per_row;
    ::made_up::block(type, seed, stream, layer, row / rows_per_expert, row % rows_per_expert, i % per_row,
                     out + i * ::made_up::block_bytes(type));
  }
}
// Public source for gains made as bit patterns: see prover/weights_blocks/made_up_blocks.h,
// gain_bits(). The loop form is public: the grid-stride loop (cited at k_made_up_tensor()).
__global__ void k_made_up_gains(u64 seed, int norm, u64 layer, size_t n, float *out) {
  for (size_t j = blockIdx.x * (size_t)blockDim.x + threadIdx.x; j < n; j += (size_t)gridDim.x * blockDim.x)
    out[j] = __uint_as_float(::made_up::gain_bits(seed, norm, layer, j));
}
// Writes the router's, embedding's or head's committed integers: signed values of `bits` bits from
// g5::v13m() (prover/weights_commitment/kernels.cuh). cp_prove gets these tensors in binary32 from
// wb::k_made_up_f32(), as synth_weight() (prover/prover/forward.cuh) / 4096. synth_weight() equals v13m() at
// every width but 16, and the caller passes only 13 or 20. So an integer n stands for cp_prove's
// value n * 2^-12: the caller sets the exponent e = 12 for every tensor of the synthetic model, the
// embedding too (FoldSource::eres is for a model's files only). out is [row][input]; if transposed,
// v13m() counts [input][row].
// The loop form is public: the grid-stride loop (cited at k_made_up_tensor()).
template <class Int>
__global__ void k_made_up_integers(u64 seed, int stream, int bits, u64 base, size_t rows, size_t inputs, bool transposed,
                                   Int *out) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < rows * inputs;
       i += (size_t)gridDim.x * blockDim.x) {
    const size_t row = i / inputs, in = i % inputs;
    out[i] = (Int)::g5::v13m(seed, stream, base + (transposed ? in * rows + row : i), bits);
  }
}
// The opened source. A line of wsums.txt: tensor, elements, exponent e (an integer n represents
// n * 2^-e), bits (unused here), SHA-256 of the integers. fold.txt's line eres: the embedding's e;
// open() requires the line, so it replaces the default.
// fault: 4 bytes in GPU memory where fold::gains() sets bits of fold::Fault (fold_weights.cuh).
struct FoldSource {
  ::gguf::Model file;
  std::map<std::string, std::pair<int, std::string>> sums; // from a tensor's name to (e, SHA-256)
  int eres = 11;
  unsigned *fault = nullptr;
  // Set if the source is the synthetic model. Then no file is opened, and wsums.txt does not exist.
  bool made_up = false;
  MadeUpModel model;
  void open_made_up(const MadeUpModel &model_) {
    made_up = true, model = model_;
    CK(cudaMalloc(&fault, 4));
    CK(cudaMemset(fault, 0, 4));
  }
  // Opens the directory `setup` once, and the GGUF files, wsums.txt and fold.txt through that one
  // descriptor (gguf::Model::open_folder(), gguf_file.h). Exits with code 2 if it rejects a file.
  void open(const std::string &setup) {
    file.open_folder(setup);
    // The conversion's two constants are a function of the width and head dimension
    // (fold::fold_c_word in fold_weights.cuh), so any positive whole width and head dimension are
    // read; the parameter file's shape check ties the model to its policy.
    if (file.embedding_length < 1 || file.key_length < 1) {
      fprintf(stderr, "the model's width and head dimension must each be at least 1\n");
      exit(2);
    }
    FILE *f = file.side("wsums.txt");
    if (!f) {
      fprintf(stderr, "%s/wsums.txt cannot be opened\n", setup.c_str());
      exit(2);
    }
    char nm[256], hx[80];
    unsigned long long cells;
    int e, bits;
    bool ended = false;
    while (fscanf(f, "%255s", nm) == 1) {
      if (!strcmp(nm, "END")) {
        ended = fscanf(f, "%llu", &cells) == 1 && cells == sums.size();
        break;
      }
      if (fscanf(f, "%llu %d %d %79s", &cells, &e, &bits, hx) != 4 || !file.has(nm) || file.at(nm).cells() != cells ||
          !sums.insert({nm, {e, hx}}).second) {
        fprintf(stderr, "wsums.txt: a line is malformed, occurs twice, names a tensor that the model file does not "
                        "have, or gives another number of elements than the model file\n");
        exit(2);
      }
    }
    fclose(f);
    if (!ended) {
      fprintf(stderr, "wsums.txt: the last line must be END followed by the number of lines before it\n");
      exit(2);
    }
    f = file.side("fold.txt");
    char w[64], v[64];
    int rows_as_stored = 0, exponent = 0; // the same rule as in cp_prove's reader (weights_blocks/gguf_source.cuh)
    while (f && fscanf(f, "%63s %63s", w, v) == 2) {
      char *end = v;
      if (!strcmp(w, "eres")) eres = (int)strtol(v, &end, 10), exponent += *end ? 2 : 1;
      if (!strcmp(w, "qkrows")) rows_as_stored += strcmp(v, "asis") ? 2 : 1;
    }
    if (f) fclose(f);
    if (exponent != 1 || rows_as_stored != 1) {
      fprintf(stderr, "fold.txt cannot be read, or does not contain the lines \"eres <whole number>\" and \"qkrows asis\", "
                      "each exactly once\n");
      exit(2);
    }
    CK(cudaMalloc(&fault, 4));
    CK(cudaMemset(fault, 0, 4));
  }
};
// Opens the source; the synthetic model gets the parameter file's seed and shape. Only a production
// build (BUILT_FOR_TESTS_ALONE is 0: prover/common/setting.h) compiles the call that opens a
// model's files, in case cp_commit_main()'s rejection "tests-build-on-a-model" is lost or changed.
#ifndef BUILT_FOR_TESTS_ALONE
#error "common/setting.h comes first: an undefined mark would read as 0, the open side"
#endif
static void open_source(FoldSource &source, const Opt &O, const PolC &PO, const ShapeW &sh) {
#if BUILT_FOR_TESTS_ALONE
  if (!O.made_up_blocks) {
    fprintf(stderr, "cp_commit: internal error (tests-build-fence): a build for tests alone has come to a model's files, which it refuses long before: nothing of them is opened. The program has a fault; nothing of this run may be used\n");
    exit(4);
  }
#else
  if (!O.made_up_blocks) return source.open(O.fold);
#endif
  MadeUpModel model;
  model.seed = PO["seed_lo"] | ((u64)PO["seed_hi"] << 32);
  model.d = sh.d, model.H = PO["H"], model.G = PO["G"], model.dh = sh.dh, model.f = sh.f, model.fe = sh.fe;
  model.NE = sh.NE, model.V = sh.V, model.experts = sh.moeMode != 0;
  source.open_made_up(model);
}
/* Exits (code 2) unless the tensors of unit u (prover/common/weight_layout.h: a block of rows of
 * the weight matrix W) have the sizes that u.g (the unit's index groups; `R` is an index's extent),
 * H, G and dh imply: `g5::k_cellsT`, which fills W, indexes them by these numbers only. Axes in the
 * file: input index, output row, expert. */
static void hold_sizes(const FoldSource &source, const UnitW &u, const vector<std::string> &names, u64 H, u64 G,
                       u64 dh) {
  auto is = [&](const std::string &n, u64 d0, u64 d1, u64 d2) {
    u64 dims[4] = {0, 0, 0, 1};
    if (source.made_up) {
      int stream;
      u64 layer;
      if (MadeUpModel::place(n, stream, layer)) {
        dims[0] = source.model.inputs(stream, layer), dims[1] = source.model.rows(stream, layer);
        dims[2] = stream >= 10 && stream <= 12 ? source.model.NE : 1;
      }
    } else {
      const ::gguf::Tensor &t = source.file.at(n);
      for (int i = 0; i < 4; i++) dims[i] = t.dims[i];
    }
    if (dims[0] != d0 || dims[1] != d1 || dims[2] != d2 || dims[3] != 1) {
      fprintf(stderr, "%s has other dimensions than the parameter file's shape parameters give it\n", n.c_str());
      exit(2);
    }
  };
  const u64 R0 = u.g[0].R, R1 = u.g[1].R, R2 = u.g.size() > 2 ? u.g[2].R : 1;
  switch (u.X) {
  case 0:
    is(names[0], R2, H * dh, 1);
    is(names[1], R2, G * dh, 1);
    break;
  case 7:
  case 8:
  case 10:
    is(names[0], R0, R2, R1);
    break;
  case 12:
  case 13:
    is(names[0], R1, R0, 1);
    break;
  default:
    is(names[0], R0, R1, 1);
  }
}
/* Returns the names of the GGUF tensors that make up unit u. u.X, 0 to 13, is the unit's kind, not
   a stream of MadeUpModel. For X == 9, moeLayer selects the shared expert's down projection. */
static vector<std::string> unit_tensors(const UnitW &u, bool moeLayer) {
  const std::string b = "blk." + std::to_string(u.l) + ".";
  switch (u.X) {
  case 0:
    return {b + "attn_q.weight", b + "attn_k.weight"};
  case 1:
    return {b + "attn_v.weight"};
  case 2:
    return {b + "attn_output.weight"};
  case 3:
    return {b + "ffn_gate.weight"};
  case 4:
    return {b + "ffn_up.weight"};
  case 5:
    return {b + "ffn_gate_shexp.weight"};
  case 6:
    return {b + "ffn_up_shexp.weight"};
  case 7:
    return {b + "ffn_gate_exps.weight"};
  case 8:
    return {b + "ffn_up_exps.weight"};
  case 9:
    return {b + (moeLayer ? "ffn_down_shexp.weight" : "ffn_down.weight")};
  case 10:
    return {b + "ffn_down_exps.weight"};
  case 11:
    return {b + "ffn_gate_inp.weight"};
  case 12:
    return {"token_embd.weight"};
  default:
    return {"output.weight"};
  }
}
} // namespace rf
