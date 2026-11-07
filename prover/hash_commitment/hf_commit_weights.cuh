// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// PinnedWeights stores the model's weights for the committer program, cp_commit, and implements
// WeightWalker (prover/hash_commitment/hf_commit_walk.h). cp_commit goes over all the weights many
// times, so it cannot read the model's files each time, and the committed integers (two bytes per
// weight) do not fit into host memory. So each tensor is stored once, in page-locked host memory,
// as the model's file stores it (quantized blocks). At full size (Llama 4 Maverick's shape) the
// integers are 806 GB, the public model's files 232 GB, the synthetic model 237 GB. A pass over the
// weights (`walk` in identifiers) copies the tensors to the GPU chunk by chunk (`piece`) and
// converts each chunk there (fold::integers(), prover/weights_commitment/fold_weights.cuh) to the
// committed integers, as defined in docs/model.md, "11 From the model's files to the integers".
// The prover, cp_prove, streams its weights from host memory the same way:
// prover/prover/weight_stream.cuh, which cites public sources: DeepSpeed's ZeRO-Inference, FlexGen.
// The weights come from a model's files (-weights <folder>), the synthetic model in quantized
// blocks (-weights made-up-blocks) or the tests' generator (-weights generated). The last two are
// computed on the GPU once and then stored like a file's tensors, so a pass treats all three alike.
// What they give as integers (all of the generator's tensors; the synthetic model's router,
// embedding and head) is stored as integers.
// A file of a wider type takes more host memory: 1.06 bytes per weight (Q8_0) or 2 (F16, BF16). The
// committed integers have the same width for every type.
//
// A unit (rf::UnitW) is the rows of the committed matrix W that hold one tensor or, for kind 0, a
// layer's q and k; its kind is u.X (rf::unit_tensors(), prover/weights_commitment/fold_source.cuh).
// Call load() once. A pass (walk_on()) calls visit(piece, stream) for each chunk or whole unit, in
// rising unit order. reserve() allocates its GPU buffers (`rooms`) in advance; release() zeroes and
// frees them. The checking pass is in check_the_weights() (prover/hash_commitment/hf_commit.cu).
//
// The host memory and the GPU buffers hold weights. The tensors' host memory is never zeroed or
// freed. The norms' gains (normalization weights) are also in host vectors: not page-locked,
// zeroed only when freed (prover/common/zeroing_heap.h), and cp_commit never frees all_gains.
#pragma once
#include "weights_commitment/fold_source.cuh"
#include "weights_commitment/worst_case.cuh" // `k_generated`: the tests' generator, for one tensor
#include "weights_blocks/decode.cuh"         // `decode_planes`: cp_prove's decoder of blocks
#include "hash_commitment/hf_commit_walk.h"
#include "hash_commitment/commit_lie.cuh"
#include "hash_commitment/hf_commit_doors.h"
#include <mutex>
namespace hf {
#ifdef SERVING_BUILD
namespace making {
// Defined in public_weights_markers.h, which hf_commit.cu includes later, after hf_commit_making.h.
static int check_the_digests(const ::gguf::Model &model, const std::string &directory);
} // namespace making
#endif
struct PinnedWeights : WeightWalker {
  // A tensor's format in host memory: a GGUF type number (gguf::Type) or the integers themselves.
  static constexpr uint32_t INT16 = 0x10000, INT32 = 0x10001;
  static constexpr size_t NO_GAINS = ~(size_t)0;
  struct Tensor {
    std::string name;
    uint32_t type = 0;
    size_t cells = 0, inputs = 0, bytes = 0; // inputs: a row's length, the file's fastest index
    uint8_t *host = nullptr;
    size_t gains = NO_GAINS; // where the row's multipliers begin in all_gains
    int e = 0;               // integer = round(2^e * multiplier * value); |integer| > limit: fault
    long long limit = 4095;
    std::string sum; // the SHA-256 of the integers that wsums.txt gives; empty if there is none
    bool wide() const { return type == INT32 || limit > 32767; } // int32: the embedding
  };
  static size_t bytes_of(uint32_t type, size_t cells) {
    switch (type) {
    case ::gguf::Q4_K: return cells / 256 * 144;
    case ::gguf::Q5_K: return cells / 256 * 176;
    case ::gguf::Q6_K: return cells / 256 * 210;
    case ::gguf::Q8_0: return cells / 32 * 34;
    case ::gguf::F16:
    case ::gguf::BF16:
    case INT16: return cells * 2;
    default: return cells * 4; // binary32, INT32: describe_file() has refused a file's tensor of any other type
    }
  }
  // The elements of a type's block, which no cut goes through. 1: the elements are stored one by one.
  static size_t block_cells(uint32_t type) {
    return type == ::gguf::Q4_K || type == ::gguf::Q5_K || type == ::gguf::Q6_K ? 256 : type == ::gguf::Q8_0 ? 32 : 1;
  }
  // The type's number in weights_blocks/decode.cuh, cp_prove's decoder. -1: that file does not read the type.
  static int decoder_type(uint32_t type) {
    switch (type) {
    case ::gguf::Q4_K: return 0;
    case ::gguf::Q5_K: return 1;
    case ::gguf::Q8_0: return 4;
    case ::gguf::F16: return 5;
    case ::gguf::BF16: return 6;
    default: return -1;
    }
  }

  const PlantedInteger planted; // test hook FORMF_PLANTED_INTEGER (hf_commit_doors.h); unset: inert
  const GeneratedValues generated_values; // test hook FORMF_GENERATED_VALUES, likewise
  const rf::HfShape *shape = nullptr;
  std::vector<Tensor> tensors;               // two for a unit of kind 0, one for every other
  std::vector<std::pair<int, int>> of_unit;  // a unit's tensors: indices into `tensors`; -1: none
  std::vector<float> all_gains;
  rf::FoldSource source; // the model's open files, wsums.txt and fold.txt; or the synthetic model
  size_t host_bytes = 0;

  // ---- loading ----
  // Fills the host memory from the source that the options name. It uses the calling thread's
  // current GPU and exits with an error if a file does not fit the parameter file (`policy.txt`).
  void load(const rf::Opt &O, const rf::PolC &PO, const rf::HfShape &shape_) {
    shape = &shape_, huge_pages = O.huge_pages;
    generated_values.hold(O.generator);
    const rf::ShapeW &sh = shape->sh;
    const auto begin = std::chrono::steady_clock::now();
    if (O.by_tensors()) {
      rf::open_source(source, O, PO, sh);
#ifdef SERVING_BUILD
      // With -public-weights nothing is hidden. So a commitment is created from a model's files only if each of them has the SHA-256 that its marker file declares public.
      if (O.public_seed && !source.made_up && making::check_the_digests(source.file, O.weights)) exit(2);
#endif
      if (!source.made_up && source.file.block_count != (u64)sh.L) {
        fprintf(stderr, "cp_commit: the model file and the parameter file give different numbers of layers\n");
        exit(2);
      }
    }
    for (const rf::UnitW &u : shape->Y.U) {
      const auto names = rf::unit_tensors(u, sh.moeMode && (u.l & 1));
      if (O.by_tensors()) rf::hold_sizes(source, u, names, PO["H"], PO["G"], sh.dh);
      of_unit.push_back({(int)tensors.size(), names.size() > 1 ? (int)tensors.size() + 1 : -1});
      for (size_t i = 0; i < names.size(); i++) {
        Tensor t;
        t.name = names[i];
        if (!O.by_tensors())
          describe_generated(t, u, i == 1);
        else if (source.made_up)
          describe_made_up(t);
        else
          describe_file(t);
        t.bytes = bytes_of(t.type, t.cells);
        host_bytes += t.bytes;
        tensors.push_back(t);
      }
    }
    // Page-locked before any weight is written: no tensor can be swapped out. The norms' gains can.
    {
      size_t all = 0;
      for (const Tensor &t : tensors) all += (t.bytes + HUGE_PAGE - 1) / HUGE_PAGE * HUGE_PAGE;
      uint8_t *at = touched_host_memory(all);
      for (Tensor &t : tensors) { // one range per tensor: a copy must not cross a range's end
        const size_t bytes = (t.bytes + HUGE_PAGE - 1) / HUGE_PAGE * HUGE_PAGE;
        if (bytes) CK(cudaHostRegister(at, bytes, cudaHostRegisterPortable));
        t.host = at, at += bytes;
      }
    }
    printf("cp_commit: %.1f GB of host memory page-locked for %zu tensors: %.1f s\n", host_bytes / 1e9, tensors.size(),
           seconds_since(begin));
    const auto fill = std::chrono::steady_clock::now();
    if (O.by_tensors() && !source.made_up) {
      // pread() on shared descriptors, so the readers need no lock. Eight keep a fast disk busy.
      std::atomic<size_t> next{0};
      std::vector<std::thread> readers;
      for (int r = 0; r < 8; r++) readers.emplace_back(&PinnedWeights::read_tensors, this, &next);
      for (auto &r : readers) r.join();
    } else {
      for (size_t i = 0; i < shape->Y.U.size(); i++)
        for (int second = 0; second < 2; second++) {
          const int at = second ? of_unit[i].second : of_unit[i].first;
          if (at < 0) continue;
          Tensor &t = tensors[at];
          uint8_t *made;
          CK((cudaMalloc)((void **)&made, t.bytes));
          if (source.made_up)
            make_made_up(t, made);
          else
            make_generated(t, shape->Y.U[i], second, made), generated_values.lay(i, made, t.cells, t.type == INT32);
          CK(cudaMemcpyAsync(t.host, made, t.bytes, cudaMemcpyDeviceToHost, cudaStreamPerThread));
          CK(cudaStreamSynchronize(cudaStreamPerThread));
          CK((cudaFree)(made));
        }
    }
    printf("cp_commit: the weights are in host memory: %.1f s\n", seconds_since(fill));
    // No-op in this repository. The lambda locates a weight code as in wb::MadeUpSource::tensor().
    commit_lie::source([&](int stream, long long layer, long long row, long long input) -> uint8_t * {
      if (!source.made_up) return nullptr;
      for (const Tensor &t : tensors) {
        int of_stream;
        u64 of_layer;
        if (!rf::MadeUpModel::place(t.name, of_stream, of_layer) || of_stream != stream || (long long)of_layer != layer) continue;
        if ((t.type != ::gguf::Q4_K && t.type != ::gguf::Q5_K) || row < 0 || (size_t)row >= t.cells / t.inputs || input < 0 || (size_t)input >= t.inputs || input % 64 >= 32) return nullptr;
        return t.host + ((size_t)row * (t.inputs / 256) + (size_t)input / 256) * (t.type == ::gguf::Q5_K ? 176 : 144) + (t.type == ::gguf::Q5_K ? 48 : 16) + 32 * ((size_t)input % 256 / 64) + (size_t)input % 32;
      }
      return nullptr;
    });
  }
  // Host memory for weights: mapped (huge pages are requested only under -huge-pages), zeroed by up
  // to 32 threads, then registered with CUDA, which page-locks it. It is marked to be left out of
  // core dumps (a failure is ignored) and lasts until unlock_host_memory() or the process's end.
  // The idea is public: mmap(2), madvise(2) (MADV_HUGEPAGE, MADV_DONTDUMP),
  // https://man7.org/linux/man-pages/man2/madvise.2.html; NVIDIA, CUDA Runtime API,
  // cudaHostRegister (cudaHostRegisterPortable).
  // A copy between host and GPU must not cross the end of a registered range (cudaMemcpyAsync()
  // then returns cudaErrorInvalidValue), so whatever one call copies is registered as one range.
  static constexpr size_t HUGE_PAGE = (size_t)2 << 20;
  static void touch_host_memory(uint8_t *first, size_t bytes) { memset(first, 0, bytes); } // a thread: host code alone
  static uint8_t *touched_host_memory(size_t bytes) { // mapped and zeroed, not yet registered
    bytes = std::max(HUGE_PAGE, (bytes + HUGE_PAGE - 1) / HUGE_PAGE * HUGE_PAGE);
    uint8_t *const base = (uint8_t *)mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
      fprintf(stderr, "cp_commit: %zu bytes of host memory cannot be had: %s\n", bytes, strerror(errno));
      exit(2);
    }
    if (huge_pages) (void)madvise(base, bytes, MADV_HUGEPAGE); // a hint: a failure changes speed only
    (void)madvise(base, bytes, MADV_DONTDUMP);
    const size_t threads =
        std::max<size_t>(1, std::min<size_t>({(size_t)std::thread::hardware_concurrency(), 32, bytes / HUGE_PAGE}));
    const size_t share = (bytes / threads + HUGE_PAGE - 1) / HUGE_PAGE * HUGE_PAGE;
    std::vector<std::thread> touching;
    for (size_t first = 0; first < bytes; first += share)
      touching.emplace_back(touch_host_memory, base + first, std::min(share, bytes - first));
    for (auto &t : touching) t.join();
    return base;
  }
  static uint8_t *locked_host_memory(size_t bytes) { // one registered range
    bytes = std::max(HUGE_PAGE, (bytes + HUGE_PAGE - 1) / HUGE_PAGE * HUGE_PAGE);
    uint8_t *const base = touched_host_memory(bytes);
    CK(cudaHostRegister(base, bytes, cudaHostRegisterPortable));
    return base;
  }
  static void unlock_host_memory(uint8_t *base, size_t bytes) {
    bytes = std::max(HUGE_PAGE, (bytes + HUGE_PAGE - 1) / HUGE_PAGE * HUGE_PAGE);
    explicit_bzero(base, bytes);
    CK(cudaHostUnregister(base));
    munmap(base, bytes);
  }
  // A reader thread: host code only.
  void read_tensors(std::atomic<size_t> *next) {
    for (size_t i = (*next)++; i < tensors.size(); i = (*next)++) {
      source.file.read(source.file.at(tensors[i].name), 0, tensors[i].bytes, tensors[i].host);
      source.file.drop_from_page_cache(source.file.at(tensors[i].name));
    }
  }
  static double seconds_since(std::chrono::steady_clock::time_point then) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - then).count();
  }

  // A tensor of a model's file. wsums.txt gives its exponent (fold.txt the embedding's) and `sum`.
  // Its multipliers, if any: one per input, a norm's gain times the constant c_0 (fold::reads(kind, d, dh)).
  void describe_file(Tensor &t) {
    const ::gguf::Tensor &g = source.file.at(t.name);
    const auto line = source.sums.find(t.name);
    if (line == source.sums.end()) {
      fprintf(stderr, "wsums.txt has no line for %s\n", t.name.c_str());
      exit(2);
    }
    if (!g.bytes()) {
      fprintf(stderr, "cp_commit: %s has a GGUF type that the conversion to integers does not read (it reads F32, F16, BF16, "
                      "Q8_0, Q4_K, Q5_K and Q6_K); nothing is written\n", t.name.c_str());
      exit(2);
    }
    // cp_prove asks of every matrix that its rows be whole runs of 256 elements (weights_blocks/block_weights.cuh, read in
    // fill()). The file's reader checks that for Q4_K, Q5_K and Q6_K; for Q8_0 it checks a multiple of 32, and for the
    // 16-bit types no length at all.
    if (decoder_type(g.type) >= 4 && g.dims[0] % 256) {
      fprintf(stderr, "cp_commit: %s: its rows are not a multiple of 256 elements long, and cp_prove takes no such matrix; "
                      "nothing is written\n", t.name.c_str());
      exit(2);
    }
    const bool embedding = t.name == "token_embd.weight";
    t.type = g.type, t.cells = g.cells(), t.inputs = g.dims[0];
    t.e = embedding ? source.eres : line->second.first;
    // cp_prove rejects an exponent outside 0 .. 40 (BlockWeights::fill()), so none is committed to.
    if (t.e < 0 || t.e > 40) {
      fprintf(stderr, "%s: its exponent must be a whole number from 0 to 40\n", t.name.c_str());
      exit(2);
    }
    t.limit = embedding ? (1ll << 30) - 1 : 4095;
    t.sum = line->second.second;
    std::string kind = t.name.substr(0, t.name.size() - 7);
    const std::string blk = kind.rfind("blk.", 0) == 0 ? kind.substr(0, kind.find('.', 4) + 1) : "";
    kind = kind.substr(blk.size());
    // cp_prove stores a layer's matrix of type Q4_K or Q5_K, except the router's, as the file's
    // blocks and rejects an exponent above 24 for it (`stays_blocks` in BlockWeights::fill()). So too
    // a matrix of type Q8_0, F16 or BF16 (weights_blocks/blocks.h, decoded_from_blocks()).
    if (!blk.empty() && kind != "ffn_gate_inp" && decoder_type(g.type) >= 0 && t.e > 24) {
      fprintf(stderr, "cp_commit: %s: its exponent in wsums.txt is %d, but cp_prove stores this tensor as blocks and "
                      "takes at most 24; nothing is written\n", t.name.c_str(), t.e);
      exit(2);
    }
    const ::fold::Reads rd = ::fold::reads(kind, source.file.embedding_length, source.file.key_length);
    if (!rd.norm[0]) return;
    const ::gguf::Tensor &norm = source.file.at(blk + rd.norm + ".weight");
    if (norm.type != ::gguf::F32 || norm.cells() != g.dims[0]) {
      fprintf(stderr, "a norm's tensor is not of type F32 or does not have the length of the matrix's input\n");
      exit(2);
    }
    std::vector<float> gains(norm.cells());
    source.file.read(norm, 0, norm.bytes(), gains.data());
    float *in, *out;
    CK((cudaMalloc)((void **)&in, norm.bytes()));
    CK((cudaMalloc)((void **)&out, norm.bytes()));
    // Fills `out` with NaNs. If the launch below never runs and no call reports that, the loop over
    // the multipliers rejects the NaNs, so stale memory is never used.
    CK(cudaMemset(out, 0xFF, norm.bytes()));
    CK(cudaMemcpy(in, gains.data(), norm.bytes(), cudaMemcpyHostToDevice));
    ::fold::gains<<<64, 256>>>(in, rd.c_word, norm.cells(), out, source.fault);
    CK(cudaGetLastError()); // finds a rejected launch (prover/docs/cuda-status.md, "The rule")
    keep_gains(t, out, norm.cells());
    CK(cudaMemset(in, 0, norm.bytes())); // both hold weights
    CK(cudaMemset(out, 0, norm.bytes()));
    CK((cudaFree)(in));
    CK((cudaFree)(out));
    hold_no_fault(source.fault, t.name.c_str());
    // cp_prove's range for a multiplier (k_gain_range(), prover/weights_blocks/block_weights.cuh).
    // The check in fold::gains() is wider: finite, and zero or at least 2^-100 in magnitude.
    for (size_t j = 0; j < norm.cells(); j++) {
      const float size = fabsf(all_gains[t.gains + j]);
      if (size == 0.f || (size >= 0x1p-64f && size <= 0x1p64f)) continue;
      fprintf(stderr, "cp_commit: %s: one of its multipliers (a gain in %s%s.weight times the constant c_0) is neither "
                      "zero nor of magnitude 2^-64 to 2^64, the range that cp_prove takes; nothing is written\n",
              t.name.c_str(), blk.c_str(), rd.norm);
      exit(2);
    }
  }
  void keep_gains(Tensor &t, const float *device, size_t n) {
    t.gains = all_gains.size();
    all_gains.resize(all_gains.size() + n);
    CK(cudaMemcpyAsync(all_gains.data() + t.gains, device, n * 4, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CK(cudaStreamSynchronize(cudaStreamPerThread));
  }
  // A tensor of the synthetic model. Streams: 9 router, 10-12 routed experts, 13 embedding, 14 head
  void describe_made_up(Tensor &t) {
    int stream;
    u64 layer;
    if (!rf::MadeUpModel::place(t.name, stream, layer)) {
      fprintf(stderr, "the synthetic model has no tensor %s\n", t.name.c_str());
      exit(2);
    }
    const rf::MadeUpModel &model = source.model;
    t.inputs = model.inputs(stream, layer);
    t.cells = model.rows(stream, layer) * (stream >= 10 && stream <= 12 ? model.NE : 1) * t.inputs;
    t.e = 12;
    if (stream == 13 || stream == 9 || stream == 14) {
      t.type = stream == 13 ? INT32 : INT16;
      return;
    }
    if (t.inputs % 256) {
      fprintf(stderr, "the synthetic model's blocks hold 256 inputs each: %s has %zu inputs per row\n", t.name.c_str(),
              t.inputs);
      exit(2);
    }
    // made_up::type_of() gives a number of weights_blocks/decode.cuh: 0, 1 or, in a build with MADE_UP_WIDTH, 4, 5 or 6.
    static const uint32_t in_a_file[7] = {::gguf::Q4_K, ::gguf::Q5_K, 0, 0, ::gguf::Q8_0, ::gguf::F16, ::gguf::BF16};
    const int made = ::made_up::type_of(stream, layer);
    if (made < 0 || made > 6 || made == 2 || made == 3) {
      fprintf(stderr, "cp_commit: internal error: the synthetic model gives %s the type number %d, which is none of 0 (Q4_K), "
                      "1 (Q5_K), 4 (Q8_0), 5 (F16) and 6 (BF16)\n", t.name.c_str(), made);
      exit(4);
    }
    t.type = in_a_file[made];
    const int norm = ::made_up::norm_of(stream);
    if (norm < 0) return;
    float *gains;
    CK((cudaMalloc)((void **)&gains, t.inputs * 4));
    // NaNs, as in describe_file(). Here fold::integers() reports them, in the checking pass.
    CK(cudaMemset(gains, 0xFF, t.inputs * 4));
    rf::k_made_up_gains<<<64, 256>>>(model.seed, norm, layer, t.inputs, gains);
    CK(cudaGetLastError()); // as in describe_file()
    keep_gains(t, gains, t.inputs);
    CK((cudaFree)(gains));
  }
  void make_made_up(const Tensor &t, uint8_t *out) {
    int stream;
    u64 layer;
    rf::MadeUpModel::place(t.name, stream, layer);
    const rf::MadeUpModel &model = source.model;
    const size_t rows = t.cells / t.inputs;
    if (stream == 13)
      rf::k_made_up_integers<int><<<8192, 256>>>(model.seed, stream, 20, 0, rows, t.inputs, false, (int *)out);
    else if (stream == 9 || stream == 14)
      rf::k_made_up_integers<short><<<8192, 256>>>(model.seed, stream, 13, stream == 9 ? layer * model.d * model.NE : 0,
                                                   rows, t.inputs, true, (short *)out);
    else
      rf::k_made_up_tensor<<<8192, 256>>>(decoder_type(t.type), model.seed, stream, layer, model.rows(stream, layer),
                                          t.inputs / 256, t.cells / 256, out);
    CK(cudaGetLastError()); // as in describe_file()
  }
  // A tensor of the tests' generator, with its axes in a model file's order: input, row, expert.
  void describe_generated(Tensor &t, const rf::UnitW &u, bool second) {
    const bool by_expert = u.X == 7 || u.X == 8 || u.X == 10, by_token = u.X == 12 || u.X == 13;
    t.inputs = u.X == 0 ? u.g[2].R : by_token ? u.g[1].R : u.g[0].R;
    const size_t rows = u.X == 0     ? (second ? shape->sh.dkv : shape->sh.dq)
                        : by_expert  ? u.g[2].R
                        : by_token   ? u.g[0].R
                                     : u.g[1].R;
    t.cells = t.inputs * rows * (by_expert ? u.g[1].R : 1);
    t.type = u.X == 12 ? INT32 : INT16;
  }
  void make_generated(const Tensor &t, const rf::UnitW &u, bool second, uint8_t *out) {
    const bool by_expert = u.X == 7 || u.X == 8 || u.X == 10;
    const long long rows = (long long)(t.cells / t.inputs / (by_expert ? u.g[1].R : 1));
    if (u.X == 12)
      rf::worst_case::k_generated<int>
          <<<8192, 256>>>(shape->model, u.X, u.l, false, t.cells, (long long)t.inputs, rows, (int *)out);
    else
      rf::worst_case::k_generated<short>
          <<<8192, 256>>>(shape->model, u.X, u.l, second, t.cells, (long long)t.inputs, rows, (short *)out);
    CK(cudaGetLastError()); // as in describe_file()
  }
  static void hold_no_fault(unsigned *fault, const char *where) {
    unsigned word;
    CK(cudaMemcpyAsync(&word, fault, 4, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CK(cudaStreamSynchronize(cudaStreamPerThread));
    if (!word) return;
    fprintf(stderr, "the conversion of %s to integers failed (fault word %u, the sum of: 1 a value is not finite, 2 an "
                    "integer is out of range, 4 a norm's gain is too small)\n", where, word);
    exit(3);
  }

  // ---- passes over the weights ----
  // A pass uses ROOMS buffers for packed bytes (chunks in their host format), so the copies from
  // host memory, which limit its speed, continue while earlier chunks wait for the GPU. One buffer
  // for the integers suffices, as visits must work on the decoding stream. Do not overlap passes.
  static constexpr int ROOMS = 3;
  struct Gpu {
    float *gains = nullptr;
    unsigned *fault = nullptr;
    uint8_t *packed[ROOMS] = {};
    uint8_t *integers = nullptr;
    size_t packed_bytes = 0, integer_bytes = 0;
    cudaStream_t copies = nullptr;
    cudaEvent_t loaded[ROOMS] = {}, decoded[ROOMS] = {};
  };
  Gpu the_gpu;
  Gpu &this_gpu() { return the_gpu; }
  // A chunk before it is loaded; its fields are as in Piece.
  struct Cut {
    size_t unit, first, count;
    bool whole;
  };
  // grains: experts or vocabulary rows a unit can split into (1: no split); cells: elements of each
  void grain(size_t unit, size_t &grains, size_t &cells) const {
    const rf::UnitW &u = shape->Y.U[unit];
    const Tensor &t = tensors[of_unit[unit].first];
    grains = 1, cells = t.cells;
    if (u.X == 7 || u.X == 8 || u.X == 10) grains = u.g[1].R;
    if (u.X == 12 || u.X == 13) grains = t.cells / t.inputs;
    cells = t.cells / grains;
    if (cells % block_cells(t.type)) grains = 1, cells = t.cells; // a cut would split a block
  }
  std::vector<Cut> cuts(size_t piece_bytes) const {
    std::vector<Cut> list;
    for (size_t unit = 0; unit < shape->Y.U.size(); unit++) {
      size_t grains, cells;
      grain(unit, grains, cells);
      const size_t width = tensors[of_unit[unit].first].wide() ? 4 : 2;
      const size_t step = std::min(grains, std::max<size_t>(1, piece_bytes / (cells * width)));
      for (size_t first = 0; first < grains; first += step)
        list.push_back({unit, first, std::min(step, grains - first), step == grains});
    }
    return list;
  }
  // In a buffer of packed bytes each tensor's share begins at a multiple of 256 bytes.
  static size_t rounded(size_t bytes) { return (bytes + 255) & ~(size_t)255; }

  size_t joined_cut_bytes = (size_t)1 << 30; // of integers; hf_commit.cu sets it to -piece-bytes
  // Allocates the GPU buffers that a pass over these chunks needs, unless those present are large
  // enough. Their sizes follow from the list and the tensors' shapes and types only.
  void take_rooms(Gpu &g, const std::vector<Cut> &list, bool joined) {
    size_t packed = 0, integers = 0;
    for (const Cut &c : list) {
      size_t p = 0, n = 0;
      for (int at : {of_unit[c.unit].first, of_unit[c.unit].second}) {
        if (at < 0) continue;
        const Tensor &t = tensors[at];
        size_t grains, cells;
        grain(c.unit, grains, cells);
        const size_t here = c.whole ? t.cells : cells * c.count;
        p += rounded(bytes_of(t.type, here)), n += (joined ? t.cells : here) * (t.wide() ? 4 : 2);
      }
      packed = std::max(packed, p), integers = std::max(integers, rounded(n));
    }
    take_gains(g);
    if (packed > g.packed_bytes || integers > g.integer_bytes) {
      release_rooms(g);
      for (auto &room : g.packed) {
        CK((cudaMalloc)((void **)&room, packed));
        CK(::lent::poison(room, packed)); // test hook: fills it with the byte LENT_POISON, if set
      }
      CK((cudaMalloc)((void **)&g.integers, integers));
      CK(::lent::poison(g.integers, integers));
      g.packed_bytes = packed, g.integer_bytes = integers;
    }
  }
  // Copies the norms' gains to the GPU. Only end() zeroes them: a run that stops earlier leaves
  // them there. On first use it also creates the copy stream, events and fault word, never freed.
  void take_gains(Gpu &g) {
    if (!g.copies) {
      CK(cudaStreamCreateWithFlags(&g.copies, cudaStreamNonBlocking));
      for (int s = 0; s < ROOMS; s++) {
        CK(cudaEventCreateWithFlags(&g.loaded[s], cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&g.decoded[s], cudaEventDisableTiming));
      }
      CK((cudaMalloc)((void **)&g.fault, 4));
      CK(cudaMemset(g.fault, 0, 4));
    }
    if (!g.gains) {
      CK((cudaMalloc)((void **)&g.gains, all_gains.size() * 4 + 256));
      CK(cudaMemcpy(g.gains, all_gains.data(), all_gains.size() * 4, cudaMemcpyHostToDevice));
      // This copy reads pageable memory, so it may end after the call returns (NVIDIA, CUDA Runtime
      // API, "API synchronization behavior"), and a pass on another stream does not wait for it.
      CK(cudaDeviceSynchronize());
    }
  }
  void reserve(size_t piece_bytes) override {
    const bool joined = piece_bytes == WHOLE_UNITS;
    take_rooms(this_gpu(), cuts(joined ? joined_cut_bytes : piece_bytes), joined);
  }
  void walk(size_t piece_bytes, const Visit &visit) override { walk_on(cudaStreamPerThread, piece_bytes, visit); }
  void walk_units(cudaStream_t stream, size_t piece_bytes, size_t unit_from, size_t unit_to, bool planes,
                  const Visit &visit) override {
    walk_on(stream, piece_bytes, visit, false, planes, unit_from, unit_to);
  }
  bool blocks(size_t unit, PackedBlocks &out) override {
    if (of_unit[unit].second >= 0 || planted.acts_on(unit)) return false;
    const Tensor &t = tensors[of_unit[unit].first];
    if (decoder_type(t.type) < 0 || t.wide() || t.inputs % 256) return false;
    Gpu &g = this_gpu();
    take_gains(g);
    out.host = t.host, out.type = decoder_type(t.type), out.inputs = t.inputs, out.file_rows = t.cells / t.inputs;
    out.gains = t.gains == NO_GAINS ? nullptr : g.gains + t.gains;
    out.scale = ldexpf(1.f, t.e);
    return true;
  }
  // One pass over the units unit_from .. unit_to - 1. With by_name it waits after each chunk and
  // exits on a fault at once, naming the unit's first tensor; otherwise at its end, without a name.
  // With `planes`, unless piece_bytes is WHOLE_UNITS, the chunks that can come as byte planes do
  // (see Piece): qblocks::decode_planes() is several times faster than fold::integers() but reports
  // no fault. It gives the same integers unless a binary32 product overflows or is subnormal
  // (not proved). RangeWindows (hf_range_windows.cuh) uses it after the checking pass.
  // Pieces of the wider types that the same decoder reads (Q8_0, F16, BF16) come as planes too, under the same
  // conditions, where a row is a whole number of runs of 256 inputs.
  void walk_on(cudaStream_t stream, size_t piece_bytes, const Visit &visit, bool by_name = false, bool planes = false,
               size_t unit_from = 0, size_t unit_to = ~(size_t)0) {
    Gpu &g = this_gpu();
    // A pass over whole units (`joined`) still loads a large unit's packed bytes in chunks and
    // decodes each chunk to its place in the integers' buffer. So the packed buffers stay small
    // (1.4 GB instead of 11 GB at full size), and the visit is called once the unit is complete.
    const bool joined = piece_bytes == WHOLE_UNITS;
    const std::vector<Cut> list = cuts(joined ? joined_cut_bytes : piece_bytes);
    take_rooms(g, list, joined);
    size_t copied = 0; // the chunks of this pass that have gone through a buffer
    for (size_t j = 0; j < list.size(); j++) {
      const Cut &c = list[j];
      if (c.unit < unit_from || c.unit >= unit_to) continue;
      const int s = (int)(copied % ROOMS);
      const rf::UnitW &u = shape->Y.U[c.unit];
      size_t grains, cells;
      grain(c.unit, grains, cells);
      Piece piece;
      piece.unit = c.unit, piece.u = &u, piece.first = c.first, piece.count = c.count, piece.whole = c.whole;
      // The buffer is free once the chunk that was in it has been decoded.
      if (copied >= ROOMS) CK(cudaStreamWaitEvent(g.copies, g.decoded[s], 0));
      size_t p = 0;
      size_t at_packed[2] = {0, 0};
      int i = 0;
      for (int at : {of_unit[c.unit].first, of_unit[c.unit].second}) {
        if (at < 0) continue;
        const Tensor &t = tensors[at];
        const size_t first = c.whole ? 0 : cells * c.first, here = c.whole ? t.cells : cells * c.count;
        at_packed[i++] = p;
        CK(cudaMemcpyAsync(g.packed[s] + p, t.host + bytes_of(t.type, first), bytes_of(t.type, here),
                           cudaMemcpyHostToDevice, g.copies));
        p += rounded(bytes_of(t.type, here));
      }
      CK(cudaEventRecord(g.loaded[s], g.copies));
      CK(cudaStreamWaitEvent(stream, g.loaded[s], 0));
      size_t n = 0;
      i = 0;
      for (int at : {of_unit[c.unit].first, of_unit[c.unit].second}) {
        if (at < 0) continue;
        const Tensor &t = tensors[at];
        const size_t first = c.whole ? 0 : cells * c.first, here = c.whole ? t.cells : cells * c.count;
        const uint8_t *from = g.packed[s] + at_packed[i];
        uint8_t *to = g.integers + n + (joined ? first * (t.wide() ? 4 : 2) : 0);
        (i++ ? piece.w1 : piece.w0) = to;
        const float *gains = t.gains == NO_GAINS ? nullptr : g.gains + t.gains;
        if (planes && !joined && u.X >= 1 && u.X <= 11 && decoder_type(t.type) >= 0 && t.inputs % 256 == 0) {
          // The chunk's blocks are [expert][row][block]; qblocks::Where's window is the whole row.
          const size_t experts = grains > 1 ? c.count : 1, rows = here / experts / t.inputs;
          const int in_a_row = (int)(t.inputs / 256);
          piece.w0 = nullptr, piece.ld = experts * t.inputs;
          piece.lo = (const int8_t *)to, piece.hi = piece.lo + here;
          ::qblocks::decode_planes(::qblocks::Where{from, in_a_row, in_a_row, 0, 0, 0, rows}, decoder_type(t.type),
                                   (int)rows, (int)experts * in_a_row, gains, ldexpf(1.f, t.e), (int8_t *)to,
                                   (int8_t *)to + here, piece.ld, stream);
        } else if (t.type == INT16 || t.type == INT32)
          CK(cudaMemcpyAsync(to, from, t.type == INT32 ? here * 4 : here * 2, cudaMemcpyDeviceToDevice, stream));
        else if (t.wide())
          ::fold::integers<int><<<8192, 256, 0, stream>>>(from, t.type, first, here, t.inputs, gains, t.e, t.limit,
                                                          (int *)to, g.fault);
        else
          ::fold::integers<short><<<8192, 256, 0, stream>>>(from, t.type, first, here, t.inputs, gains, t.e, t.limit,
                                                            (short *)to, g.fault);
        n += here * (t.wide() ? 4 : 2);
      }
      CK(cudaEventRecord(g.decoded[s], stream));
      copied++;
      if (by_name) {
        CK(cudaStreamSynchronize(stream));
        hold_no_fault(g.fault, tensors[of_unit[c.unit].first].name.c_str());
      }
      if (joined && !c.whole) { // the visit comes with a unit's last chunk and sees the whole unit
        if (c.first + c.count < grains) continue;
        piece.w0 = g.integers, piece.first = 0, piece.count = grains, piece.whole = true;
      }
      // by_name marks the checking pass for the test hook: only check_the_weights() sets it.
      planted.lay(piece, tensors[of_unit[c.unit].first].wide(), by_name, shape->Y.U.size(), stream);
      visit(piece, stream);
    }
    CK(cudaStreamSynchronize(stream));
    CK(cudaGetLastError());
    hold_no_fault(g.fault, "a tensor");
  }
  static void release_rooms(Gpu &g) {
    for (auto &room : g.packed) {
      if (room) CK(cudaMemset(room, 0, g.packed_bytes));
      CK((cudaFree)(room));
      room = nullptr;
    }
    if (g.integers) CK(cudaMemset(g.integers, 0, g.integer_bytes));
    CK((cudaFree)(g.integers));
    g.integers = nullptr;
    g.packed_bytes = g.integer_bytes = 0;
  }
  void release() override {
    CK(cudaDeviceSynchronize());
    release_rooms(this_gpu());
  }
  // Call it after the weights' last use on the GPU: it zeroes and frees the norms' gains. release()
  // leaves them, because a PackedBlocks from blocks() points to them.
  // cp_commit_main() (hf_commit.cu) calls end() once, late in the run. An earlier return or exit()
  // skips that call, and cp_commit's clean-up removes files and zeroes host keys only: the gains
  // then stay in the clear in GPU memory at least until the process ends, and so do a pass's
  // buffers (packed weights and integers) if the exit comes before release().
  void end() {
    Gpu &g = this_gpu();
    if (!g.gains) return;
    CK(cudaDeviceSynchronize());
    CK(cudaMemset(g.gains, 0, all_gains.size() * 4 + 256));
    CK((cudaFree)(g.gains));
    g.gains = nullptr;
  }

  // ---- the digests of wsums.txt ----
  // take() copies a tensor's integers to the host, where a thread hashes them. After finish(),
  // `held` counts the tensors and `equal` those with the SHA-256 in wsums.txt. The integers come
  // from a pass: from the bytes in host memory, not a second read of the file. The idea is public:
  // Saltzer, Reed, Clark, 'End-to-End Arguments in System Design', ACM Transactions on Computer
  // Systems 2(4), 1984, 277-288, section 'End-to-end caretaking': checksum the final copy.
  struct Sums {
    static constexpr int MOST_BUFFERS = 6;
    std::mutex lock;
    std::condition_variable freed;
    std::vector<char *> free_buffers, all_buffers;
    size_t buffer_bytes = 0, held = 0, equal = 0;
    std::vector<std::thread> hashers;
    void take(const void *device, size_t bytes, const std::string &want, cudaStream_t stream) {
      char *host = nullptr;
      {
        std::unique_lock<std::mutex> hold(lock);
        if (free_buffers.empty() && (int)all_buffers.size() < MOST_BUFFERS) {
          host = (char *)locked_host_memory(buffer_bytes);
          all_buffers.push_back(host);
        } else {
          freed.wait(hold, [&] { return !free_buffers.empty(); });
          host = free_buffers.back();
          free_buffers.pop_back();
        }
      }
      CK(cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, stream));
      CK(cudaStreamSynchronize(stream));
      hashers.emplace_back([this, host, bytes, want] {
        ::Sha256 s;
        s.update(host, bytes);
        const bool same = s.hex() == want;
        explicit_bzero(&s, sizeof s); // its buffer may still contain the tensor's last bytes
        std::lock_guard<std::mutex> hold(lock);
        held++, equal += same;
        free_buffers.push_back(host);
        freed.notify_one();
      });
    }
    void finish() {
      for (auto &t : hashers) t.join();
      hashers.clear();
      for (char *b : all_buffers) unlock_host_memory((uint8_t *)b, buffer_bytes); // zeroed there
      all_buffers.clear(), free_buffers.clear();
    }
  } sums;
  bool has_sums() const { return !tensors.empty() && !tensors[0].sum.empty(); }
  // Hashes a unit's tensors (see Sums). Call it from a visit of a pass over whole units.
  void hold_sums(const Piece &piece, cudaStream_t stream) {
    if (!sums.buffer_bytes)
      for (const Tensor &t : tensors) sums.buffer_bytes = std::max(sums.buffer_bytes, t.cells * (t.wide() ? 4 : 2));
    const Tensor &t0 = tensors[of_unit[piece.unit].first];
    sums.take(piece.w0, t0.cells * (t0.wide() ? 4 : 2), t0.sum, stream);
    if (of_unit[piece.unit].second >= 0) {
      const Tensor &t1 = tensors[of_unit[piece.unit].second];
      sums.take(piece.w1, t1.cells * 2, t1.sum, stream);
    }
  }
  // The option -huge-pages (off by default: prover/prover/arguments.h says why). load() sets it
  // before the first call of touched_host_memory(). Static because that function is.
  static inline bool huge_pages = false;
};
} // namespace hf
