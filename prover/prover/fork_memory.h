// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// Sub-proofs (`fork` in identifiers) share GPU scratch, the slab: a working set is dead after its
// run. wipe() zeroes all recorded buffers. Macros at the file's end replace seven CUDA names:
// cudaMalloc, cudaMallocHost, cudaFree, cudaMemset, cudaMemsetAsync, cudaMemcpy, cudaMemcpyAsync.
#pragma once
#include "common/end_name.h"
#include "common/setting.h"
#include <algorithm>
#include <iterator>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <map>
#include <set>
#include <string>
#include <thread>
#include <vector>

// Adds to *count the number of the n bytes at p that differ from value.
// Public source: a count returned through managed memory: NVIDIA, CUDA C++ Programming Guide,
// sections 19.1 "Unified Memory Introduction" and 19.1.2 "Programming Model",
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html.
// The idea is public: memory that CPU and GPU both access: NVIDIA, CUDA Programming Guide,
// "Intro to CUDA C++", "Unified Memory",
// https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/intro-to-cuda-cpp.html.
// Public source: a grid-stride loop: NVIDIA Technical Blog, "CUDA Pro Tip: Write Flexible Kernels
// with Grid-Stride Loops". The post's block count is an integer times the GPU's multiprocessor
// count; ours is the constant 1,024 blocks of 256 threads, also for ranges shorter than the grid.
__global__ void k_fork_memory_count_not(const unsigned char *p, size_t n, unsigned char value,
                                        unsigned long long *count) {
  unsigned long long c = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
    c += p[i] != value;
  atomicAdd(count, c);
}

// Adds to *sum a digest of the n bytes at p: the sum modulo 2^64, over the 8-byte words, of
// SplitMix64's output mix of the word XORed with a multiple of (its index + 1). It serves tests
// that ask whether a buffer has changed, and is no cryptographic hash. p must be a multiple of 8.
// Public source: a sum of hashed locations: [InstantCheck] Adrian Nistor, Darko Marinov, Josep
// Torrellas, "InstantCheck: Checking the Determinism of Parallel Programs Using On-the-fly
// Incremental Hashing", MICRO 2010, sections 2.2 and 3.2; https://doi.org/10.1109/MICRO.2010.55.
// Public source: incremental hashing: [Bellare-Micciancio] Mihir Bellare, Daniele Micciancio, "A
// New Paradigm for Collision-Free Hashing: Incrementality at Reduced Cost", EUROCRYPT 1997,
// sections 1.4 and 3.2; https://link.springer.com/content/pdf/10.1007/3-540-69053-0_13.pdf.
// The digest has the form of the State Hash of [InstantCheck], section 2.2, which uses the
// incremental hashing of [Bellare-Micciancio]; here a word's index stands for the address.
// Public sources for the loop and for the returned sum: see k_fork_memory_count_not().
__global__ void k_fork_memory_digest(const unsigned char *p, size_t n, unsigned long long *sum) {
  unsigned long long s = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (n + 7) / 8; i += (size_t)gridDim.x * blockDim.x) {
    unsigned long long x = 0;
    if (8 * i + 8 <= n)
      x = ((const unsigned long long *)p)[i];
    else
      for (size_t j = 8 * i; j < n; j++) x = x << 8 | p[j];
    x ^= (i + 1) * 0x9E3779B97F4A7C15ull;
    // The output mix of SplitMix64 (THIRD_PARTY.md).
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    s += x ^ (x >> 31);
  }
  atomicAdd(sum, s);
}

// One range ("stretch" in identifiers) of the slab that a sub-proof's run finds as set-up left
// it: the 'bytes' bytes at 'offset' are set to 'value', or copied from 'from' if that is not
// null. 'first' is the number of pieces (fork_memory_piece) of the earlier entries of its table.
struct ForkMemoryStretch {
  size_t offset, bytes;
  const char *from;
  int value;
  unsigned first;
};
// A range is written in pieces of this many bytes.
constexpr size_t fork_memory_piece = (size_t)256 << 10;
// Writes the n ranges of the table, which must not overlap. The kernel requires a
// launch of 256 threads per block, with as many blocks as the n ranges have pieces.
__global__ void __launch_bounds__(256) k_fork_memory_repeat(char *slab, const ForkMemoryStretch *table, int n) {
  int below = 0; // the number of entries e with table[e].first <= blockIdx.x
  for (int e0 = 0; e0 < n; e0 += 256)
    below += __syncthreads_count(e0 + (int)threadIdx.x < n && table[e0 + threadIdx.x].first <= blockIdx.x);
  const ForkMemoryStretch s = table[below - 1];
  const size_t at = (size_t)(blockIdx.x - s.first) * fork_memory_piece,
               len = s.bytes - at < fork_memory_piece ? s.bytes - at : fork_memory_piece;
  char *d = slab + s.offset + at;
  const char *f = s.from ? s.from + at : nullptr;
  // `head` bytes precede d's first multiple of 16 (all if d, f differ mod 16); then 16-byte words.
  const size_t to_a_word = (16 - ((size_t)d & 15)) & 15,
               head = (f && (((size_t)d ^ (size_t)f) & 15)) || to_a_word > len ? len : to_a_word,
               words = (len - head) / 16;
  const unsigned v = (unsigned)(s.value & 255) * 0x01010101u;
  uint4 *dw = (uint4 *)(d + head);
  const uint4 *fw = f ? (const uint4 *)(f + head) : nullptr;
#pragma unroll 4
  for (size_t i = threadIdx.x; i < words; i += 256) dw[i] = f ? fw[i] : make_uint4(v, v, v, v);
  for (size_t i = threadIdx.x; i < head; i += 256) d[i] = f ? f[i] : (char)s.value;
  for (size_t i = head + 16 * words + threadIdx.x; i < len; i += 256) d[i] = f ? f[i] : (char)s.value;
}

// Who shares a buffer and how long its content is valid: fork_alloc()'s third argument. A plain
// cudaMalloc after this header asks for Shape.
enum class Place {
  Own,     // the calling code's for the life of the process
  Shape,   // shared by the sub-proofs of one shape class, whose set-ups must fill it with the same content
  Lasting, // shared by the sub-proofs of one lasting class and parity; still valid in the next layer sub-proof's run
  Fork,    // valid from `run_begin` to `run_end`; a set-up may write it with cudaMemset and
           // cudaMemcpy only (the head's set-up breaks this: it writes with a kernel). In the arena: valid only in its user's step (arena_begin()); an early
           // buffer (early_begin()): up to late_begin()
  Late,    // like Fork, but valid only from late_begin() on. In a layer sub-proof, and in a single
           // object that has early buffers, it shares addresses, if it fits, with memory of the
           // run's first half (a layer sub-proof's: the arena; with late_on_keys also the keys)
  Result   // the calling code's own, written by its sub-proof's run and read after later sub-proofs have run
};

// The allocator's state (one object, fork_memory). Each sub-proof has one set-up and then many
// runs. Other sub-proofs' runs overwrite the slab in between, so what a set-up writes there with
// cudaMemset or cudaMemcpy is logged and repeated in each run; a kernel's write in a set-up is not.
// Sub-proofs are numbered: the layer sub-proofs (one for each layer group) from 0, then the head's,
// then the attention sub-proofs. An exit(n) in this header does not fix cp_prove's exit code: its
// handler end_on_exit() (prover/prover/ends.h) does. Each exit(1) here follows a failed CUDA or
// driver call, and first sets the name that other such failures have, "cuda-call".
struct ForkMemory {
  // Test switches. No program of this repository sets any of these switches; cp_prove sets none.
  struct Test {
#ifdef SERVING_BUILD
    // A serving build (SERVING_BUILD is defined: prover/build.sh's serving targets) has constants
    // instead, all off: no code can turn a switch on, and the compiler can drop what depends on it.
    static constexpr bool own_memory = false, disjoint = false, scribble = false, stale = false, skip_zero_shipped = false, skip_zero_turns = false, left_behind = false, changed_table = false;
    static constexpr int changed_cell_seat = -1, failed_seat = -1;
#else
    bool own_memory = false; // every buffer is a cudaMalloc allocation of its own
    // Each set-up has its own part of the slab. If a run wrote into another's, exit(3) at its end.
    bool disjoint = false;
    // Overwrites with 0xA5: before every run, the slab up to `peak`; in middle_begin() and
    // late_begin(), a layer sub-proof's arena; in late_begin() also a single object's early buffers
    // and, with late_on_keys, the keys; in arena_idle(), the arena. prover/prover/weight_stream.cuh
    // reads it too and fills weight buffers that it releases with 0xEE.
    bool scribble = false;
    // What Prover::wipe_the_working_memory() and wipe_behind_the_wave() zero, and each piece that
    // the slab maps, is first filled with 0xEE. If a byte is not zero afterwards: exit(3).
    bool stale = false;
    // With stale: omits the zeroing, except wipe_behind_the_wave()'s, to show that the count works.
    // Public sources for stale and for a zeroing that is omitted on purpose: see zero_new_piece().
    bool skip_zero_shipped = false;
    // With stale: omits only the zeroing before a turn, to test the count there. (Turn: a request's
    // work on a layer group or the head. Batch pass, or pass, "wave" in messages: the requests
    // proved on one transfer of the weights.)
    bool skip_zero_turns = false;
    // In the attention sub-proofs of the batch slot with this number, after each one's forward
    // pass, the last byte of its matrix qa (q, k, v and o) is changed, as a stray write would. That
    // slot must get no file, and the other slots' files must be those of an honest run. -1: none.
    int changed_cell_seat = -1;
    // Runs look_at_what_stays() at set-up's end and each time a pass's end has restored that state.
    bool left_behind = false;
    // In each batch the slot with this number gets the failure bit UNFIT at its turn in layer group
    // 0. It must get no file, yet take all steps and leave no more in memory than others. -1: none.
    int failed_seat = -1;
    // In the process's first batch, after slot 0's turn in the last layer group, one byte of a long
    // range of set-up's constants changes, as by a stray write (MemoryAsSetUp::change_a_table()).
    // No request of that batch may get a file; the next must be proved as if nothing had happened.
    bool changed_table = false;
#endif
  } test;

  void init(int forks) {
    writes.resize(forks);
    late_writes.resize(forks);
    early_of.assign(forks, {0, 0});
    end_of.assign(forks, 0);
    begin_of.assign(forks, 0);
    setup_end_of.assign(forks, 0);
    has_run.assign(forks, 0);
    results.resize(forks);
    class_of_fork.assign(forks, -1);
    lasting_class_of_fork.assign(forks, -1);
    parity_of_fork.assign(forks, 0);
  }

  using Buffers = std::vector<std::pair<void *, size_t>>;
  // The Result buffers of sub-proof f, and the Lasting buffers that layer sub-proof g writes.
  const Buffers &results_of(int f) const { return results[f]; }
  const Buffers &lasting_of(int g) const { return recorded[lasting_class_of_fork[g]][1 + parity_of_fork[g]]; }
  // While set, every buffer allocated outside any sub-proof's set-up and run is added to this list.
  Buffers *also_list = nullptr;

  // Returns the class of `key`. A layer sub-proof has two. Sub-proofs of one shape (lasting) class
  // allocate the same Shape (Lasting) buffers in the same order. Set-ups fill Shape buffers alike
  // and only zero Lasting ones: runs fill those, so sub-proofs with other constants may share them.
  int shape_class(const std::vector<unsigned> &key) {
    auto it = class_of.find(key);
    if (it != class_of.end()) return it->second;
    const int c = (int)recorded.size();
    recorded.emplace_back();
    class_of[key] = c;
    return c;
  }

  // Set-up of layer sub-proof g: first the buffers and engines (such as the lookups' product tests)
  // that all layer sub-proofs share or g keeps; then, between the next two calls, g's working set.
  void setup_layer_fork(int g) {
    note_end();
    mode = Mode::SharedSetup;
    fork = g;
    pos = begin_of[g] = layer_start(g);
  }
  void setup_working_set(int shape, int lasting, int parity_, int g) {
    if (mode != Mode::SharedSetup || fork != g) pos = begin_of[g] = std::max(start(), shared_scratch_end);
    late_pos = arena_first;
    late_keys_pos = begin_of[0];
    shared_scratch_closed = true;
    mode = Mode::Setup;
    single = false;
    cls = class_of_fork[g] = shape;
    lasting_cls = lasting_class_of_fork[g] = lasting;
    parity = parity_of_fork[g] = parity_;
    fork = g;
    next[0] = next[1] = next[2] = 0;
  }
  // Ends the set-up of a layer sub-proof's working set. The program ends if sub-proof 0 has already
  // run or, unless test.own_memory is set, if the sub-proof requested fewer Shape or Lasting
  // buffers than its class (and parity) has recorded. More are not caught: the list grows.
  void setup_working_set_done() {
    if (has_run[0]) {
      printf("fork memory: internal error: a layer sub-proof is set up after sub-proof 0 has run\n");
      name_the_end("shared-buffer"); // serving: p258 end 1 shared-buffer (M5g)
      exit(4);
    }
    if (!test.own_memory)
      for (int q : {0, 1 + parity})
        if (next[q] != recorded[class_of_list(q)][q].size()) {
          printf(
              "fork memory: internal error: a layer sub-proof of %s class %d allocated %zu shared buffers; the first "
              "sub-proof of that class allocated %zu\n",
              q == 0 ? "shape" : "lasting", class_of_list(q), next[q], recorded[class_of_list(q)][q].size());
          name_the_end("shared-buffer"); // serving: p258 end 1 shared-buffer (M5g)
          exit(4);
        }
    end_of[fork] = setup_end_of[fork] = pos;
    all_setup_end = std::max(all_setup_end, pos);
    layer_setup_end = std::max(layer_setup_end, pos);
    mode = Mode::SharedSetup;
  }

  // Set-up of a single object, one that shares no buffer with another object (the head's sub-proof
  // or an attention object): its Fork and Late buffers are in the slab, everything else is its own.
  void setup_single(int index) {
    note_end();
    mode = Mode::Setup;
    single = true;
    fork = index;
    pos = begin_of[index] = start() + (test.own_memory || test.disjoint ? 0 : single_base);
    early_from = early_to = late_pos = 0;
    early_state = 0;
    late_parts.clear();
  }
  // The two halves of a single object's run. Between these two calls the set-up allocates the
  // Fork buffers that nothing reads after the object's late_begin(), its early buffers. The Late
  // buffers that it allocates afterwards share their addresses, as far as there is room. Each of
  // the two is called only in a single object's set-up, at most once in it, and in this order.
  void early_begin() {
    if (mode != Mode::Setup || !single || early_state != 0) early_misused("early_begin()");
    early_state = 1;
    early_from = early_to = pos;
  }
  void early_end() {
    if (mode != Mode::Setup || !single || early_state != 1) early_misused("early_end()");
    early_state = 2;
    early_to = pos;
    late_pos = early_from;
    early_of[fork] = {early_from, early_to};
  }
  void early_misused(const char *call) const {
    printf("fork memory: internal error: %s: outside the set-up of the head's or an attention sub-proof, twice, or out "
           "of order (sub-proof %d)\n",
           call, fork);
    exit(4);
  }
  void setup_single_done() {
    end_of[fork] = setup_end_of[fork] = pos;
    all_setup_end = std::max(all_setup_end, pos);
    mode = Mode::Off;
    single = false;
  }

  // run_begin() and run_end() enclose one sub-proof's run. run_begin() repeats the set-up's writes
  // in writes[index] and, for a layer sub-proof, in shared_writes; the other lists: see list_of().
  void run_begin(int index) {
    if (test.own_memory) return;
    note_end();
    fork = index;
    if (slab) {
      // Public sources for the fill under test.disjoint: see check_other_parts().
      if (test.scribble || test.disjoint) {
        cudaMemsetAsync(slab, 0xA5, test.disjoint ? all_setup_end : peak, cudaStreamPerThread);
        scribbles++;
      }
      if (launch_own.size() < writes.size()) launch_own.resize(writes.size()), launch_late.resize(late_writes.size());
      if (class_of_fork[fork] >= 0) repeat(shared_writes, launch_shared), look_at_what_is_not_repeated(shared_writes);
      repeat(writes[fork], launch_own[fork]);
      look_at_what_is_not_repeated(writes[fork]);
    }
    if (fork != 0 && !has_run[0]) {
      printf("fork memory: internal error: sub-proof %d runs before sub-proof 0 has run\n", fork);
      exit(4);
    }
    mode = Mode::Run;
    single = false;
    late_begun = middle_begun = keys_and_cells_are_dead = false;
    first_run = !has_run[fork];
    if (first_run && test.disjoint) end_of[fork] = std::max(end_of[fork], fork == 0 ? all_setup_end : end_of[0]);
    if (first_run && fork == 0) shared_begin = end_of[0] = std::max(end_of[0], layer_setup_end);
    has_run[fork] = 1;
    pos = end_of[fork];
  }
  // A layer sub-proof calls it once in its run, when nothing reads the shared engines' scratch any
  // more, and a single object with early buffers when nothing reads those any more. From here on
  // the sub-proof's Late buffers are valid. It repeats what the sub-proof's set-up wrote into them.
  void late_begin() {
    if (test.own_memory || !slab || mode != Mode::Run) return;
    if (late_begun) {
      printf("fork memory: internal error: late_begin() is called twice in the run of sub-proof %d\n", fork);
      exit(4);
    }
    late_begun = true;
    if (class_of_fork[fork] >= 0 && !middle_begun) {
      printf("fork memory: internal error: the run of layer sub-proof %d reaches late_begin() without middle_begin()\n",
             fork);
      exit(4);
    }
    // What the first half of the run leaves: the shared engines'
    // scratch of a layer sub-proof, a single object's early buffers.
    const size_t from = class_of_fork[fork] >= 0 ? arena_first : early_of[fork].first,
                 to = class_of_fork[fork] >= 0 ? shared_scratch_end : early_of[fork].second;
    if (test.scribble && to > from) {
      cudaMemsetAsync(slab + from, 0xA5, to - from, cudaStreamPerThread);
      scribbles++;
    }
    if (test.scribble && late_on_keys && class_of_fork[fork] >= 0 && keys_end > begin_of[0]) {
      cudaMemsetAsync(slab + begin_of[0], 0xA5, keys_end - begin_of[0], cudaStreamPerThread);
      scribbles++;
    }
    repeat(late_writes[fork], launch_late[fork]);
    look_at_what_is_not_repeated(late_writes[fork]);
  }
  void run_end() {
    if (test.own_memory) return;
    if (test.disjoint) check_other_parts();
    end_of[fork] = pos;
    mode = Mode::Off;
  }

  // Prints the sub-proofs' GPU memory use and the slab's division; `when` goes into the first line.
  void report(const char *when) const {
    printf(
        "# GPU memory of the sub-proofs %s, GB: slab reused by every sub-proof %.3f; shared by sub-proofs of one shape "
        "%.3f (sharing saved %.3f);\n#   buffers that belong to a single sub-proof, in total %.3f; first part of the "
        "layer sub-proofs' set-up %.3f. Writes of the set-ups into the slab that the runs of their sub-proofs repeat: "
        "%zu\n#   of the slab: buffers that all layer sub-proofs keep until their link closes %.3f; then scratch that "
        "the steps of a layer sub-proof use in turn %.3f, which its late buffers reuse, at most %.3f, and %.3f on the "
        "forward pass's keys\n",
        when, peak / 1e9, bytes_per_shape / 1e9, bytes_again / 1e9, bytes_own / 1e9, bytes_shared / 1e9, logged,
        arena_first / 1e9, (shared_scratch_end - arena_first) / 1e9, late_peak / 1e9, late_keys_peak / 1e9);
    {
      size_t layer = 0, single_objects = 0;
      for (size_t f = 0; f < end_of.size(); f++)
        (class_of_fork[f] >= 0 ? layer : single_objects) =
            std::max(class_of_fork[f] >= 0 ? layer : single_objects, end_of[f]);
      printf(
          "#   the slab is as long as the sub-proof that needs the most, GB: the largest layer sub-proof %.3f; the "
          "furthest end of the head's and the attention sub-proofs %.3f\n",
          layer / 1e9, single_objects / 1e9);
    }
    // The single objects: one line for each that differs from all earlier ones.
    for (size_t f = 0; f < end_of.size(); f++) {
      bool told = class_of_fork[f] >= 0;
      for (size_t g = 0; g < f && !told; g++)
        told = class_of_fork[g] < 0 && begin_of[g] == begin_of[f] && early_of[g] == early_of[f] &&
               setup_end_of[g] == setup_end_of[f] && end_of[g] == end_of[f];
      if (!told)
        printf(
            "#   sub-proof %zu (the head's or an attention sub-proof), offsets in the slab in GB: it begins at %.3f; "
            "its early buffers, which its late ones reuse, from %.3f to %.3f; its set-up ends at %.3f, its run at "
            "%.3f\n",
            f, begin_of[f] / 1e9, early_of[f].first / 1e9, early_of[f].second / 1e9, setup_end_of[f] / 1e9,
            end_of[f] / 1e9);
    }
    printf(
        "#   in bytes: the slab %zu, the end of layer sub-proof 0 in it %zu; buffers that belong to a single "
        "sub-proof, in total %zu. %zu GPU buffers on record. After its first run layer sub-proof 0 requested %zu "
        "buffers, %zu bytes, from the slab and got them as separate allocations\n",
        peak, end_of.empty() ? (size_t)0 : end_of[0], bytes_own, device_buffers.size(), turned_from_the_slab,
        bytes_turned_from_the_slab);
    if (test.scribble || test.disjoint)
      printf("# fork memory: test mode: the slab, or a part of it in the middle of a run, was overwritten %zu times\n",
             scribbles);
    if (test.scribble && not_scribbled) {
      cudaDeviceSynchronize();
      printf(
          "# fork memory: test mode: %.3f GB, summed over the runs, that the set-up wrote and that are not written "
          "again before a run: %llu bytes of them did not hold the test's fill pattern at the point where the write "
          "would have been repeated\n",
          looked_at / 1e9, *not_scribbled);
    }
  }

  // Off: outside set-ups and runs. SharedSetup: a layer sub-proof's set-up outside its working set.
  enum class Mode { Off, Setup, Run, SharedSetup } mode = Mode::Off;
  bool single = false;
  int cls = 0, parity = 0, fork = 0;
  bool in_setup() const { return mode == Mode::Setup || mode == Mode::SharedSetup; }
  // The slab's first address and mapped bytes, for code that borrows it while no sub-proof runs.
  std::pair<char *, size_t> whole_slab() const { return {slab, mapped}; }
  bool in_slab(const void *p) const { return slab && (const char *)p >= slab && (const char *)p < slab + addresses; }

  // One write of a set-up into the slab: a fill with `value`, or a copy from `copy_of` if not null.
  struct Write {
    size_t offset, bytes;
    int value;
    const void *copy_of;
  };
  std::vector<std::vector<Write>> writes;      // per sub-proof: what its set-up wrote into the slab
  std::vector<std::vector<Write>> late_writes; // per sub-proof: what its set-up wrote into its Late buffers
  // What shared set-ups (Mode::SharedSetup) wrote below shared_scratch_end, except middle_writes.
  std::vector<Write> shared_writes;
  mutable double repeated_set = 0, repeated_copied = 0; // bytes, over the life of the process
  mutable size_t repeated_calls = 0;
  // The table for one launch of k_fork_memory_repeat(), which repeats a list of writes. The table's
  // ranges do not overlap, and writing them leaves what the list's writes, made one after the
  // other, leave. The tables are plain cudaMalloc allocations outside this header's record,
  // because no zeroing or restoring may change them. They hold nothing of a request: per range the
  // offset, the length, the source address of a copy (null for a fill), the fill value and `first`.
  struct Launch {
    size_t writes = 0, cuts = 0; // the list's length and ForkMemory::cuts when the table was made
    ForkMemoryStretch *table = nullptr;
    unsigned stretches = 0, blocks = 0;
  };
  mutable Launch launch_shared, launch_middle;
  mutable std::vector<Launch> launch_own, launch_late; // per sub-proof
  mutable bool planned = false;                        // plan_repeats() has been called
  void plan(const std::vector<Write> &list, Launch &l) const {
    std::vector<ForkMemoryStretch> table;
    std::map<size_t, size_t>
        taken; // first byte and end of what later writes of the list cover; the ranges do not touch
    for (size_t i = list.size(); i-- > 0;) {
      const Write &w = list[i];
      size_t at = w.offset;
      const size_t end = w.offset + w.bytes;
      auto next = taken.upper_bound(at);
      if (next != taken.begin() && std::prev(next)->second > at) at = std::min(end, std::prev(next)->second);
      while (at < end) { // [at, to) is covered by no later write
        const size_t to = next == taken.end() ? end : std::min(end, next->first);
        if (to > at)
          table.push_back({at, to - at, w.copy_of ? (const char *)w.copy_of + (at - w.offset) : nullptr, w.value, 0});
        if (next == taken.end()) break;
        at = std::max(to, std::min(end, next->second));
        ++next;
      }
      if (!w.bytes) continue;
      size_t a = w.offset, b = end; // goes into `taken`, merged with what it overlaps or touches
      auto it = taken.upper_bound(a);
      if (it != taken.begin() && std::prev(it)->second >= a) --it;
      while (it != taken.end() && it->first <= b) {
        a = std::min(a, it->first), b = std::max(b, it->second);
        it = taken.erase(it);
      }
      taken[a] = b;
    }
    l.blocks = 0;
    for (auto &s : table) {
      s.first = l.blocks;
      l.blocks += (unsigned)((s.bytes + fork_memory_piece - 1) / fork_memory_piece);
    }
    if (l.table) cudaFree(l.table);
    l.table = nullptr;
    l.stretches = (unsigned)table.size();
    l.writes = list.size();
    l.cuts = cuts;
    if (table.empty()) return;
    if (cudaMalloc(&l.table, table.size() * sizeof table[0]) != cudaSuccess ||
        cudaMemcpy(l.table, table.data(), table.size() * sizeof table[0], cudaMemcpyHostToDevice) != cudaSuccess) {
      printf("fork memory: a table of %zu address ranges could not be copied to the GPU\n", table.size());
      name_the_end("cuda-call");
      exit(1);
    }
  }
  // Call it after the last set-up: it makes all tables, so that no run allocates or waits for one.
  void plan_repeats() const {
    launch_own.resize(writes.size());
    launch_late.resize(late_writes.size());
    plan(shared_writes, launch_shared);
    plan(middle_writes, launch_middle);
    for (size_t f = 0; f < writes.size(); f++) plan(writes[f], launch_own[f]);
    for (size_t f = 0; f < late_writes.size(); f++) plan(late_writes[f], launch_late[f]);
    cudaDeviceSynchronize(); // the tables are on the GPU, whichever stream runs the sub-proofs
    planned = true;
  }
  void repeat(const std::vector<Write> &list, Launch &l) const {
    for (auto &w : list) (w.copy_of ? repeated_copied : repeated_set) += (double)w.bytes;
    if (l.writes != list.size() || l.cuts != cuts) {
      if (planned) {
        printf("fork memory: internal error: a set-up wrote into the slab after plan_repeats()\n");
        exit(4);
      }
      // A program that does not call plan_repeats() makes the table at the list's first repeat, and
      // again when the list's length or the count of every_run_writes() calls (`cuts`) has changed.
      plan(list, l);
    }
    if (!l.stretches) return;
    repeated_calls++;
    k_fork_memory_repeat<<<l.blocks, 256, 0, cudaStreamPerThread>>>(slab, l.table, (int)l.stretches);
  }
  // The slab starts with the layer sub-proofs' shared Fork buffers, which layer sub-proof 0's
  // shared set-up allocates: up to arena_first the forward pass's keys (lookup keys, not
  // attention's) and cells (committed integers), kept until the sub-proof's segment of the proof
  // (`link` in messages) is closed; then the arena, up to shared_scratch_end. Layer sub-proofs' own
  // buffers follow; single objects start at single_base.
  size_t shared_scratch_end = 0, late_pos = 0, late_peak = 0;
  // The keys are the slab's first buffers, up to keys_end. Their last readers are the lookups'
  // product tests (Prover::g_product_tests()), in the first half of a layer sub-proof's run. With
  // late_on_keys (set in layer sub-proof 0's shared set-up, before any Late buffer is allocated) a
  // layer sub-proof's Late buffer that does not fit in the arena shares the keys' addresses. It
  // stays valid up to run_end(), so the slab's start is then not idle while a segment is closed.
  bool late_on_keys = false;
  size_t keys_end = 0, late_keys_pos = 0, late_keys_peak = 0;
  // The arena: the shared engines' scratch memory, which the steps of a layer sub-proof's run use
  // one after another. Layer sub-proof 0's shared set-up allocates it: between arena_begin() and
  // arena_end() each arena_next() ends one user's buffers, and the next user's start again at the
  // arena's first byte. What a user leaves there is meaningless once the next user has worked.
  size_t arena_first = 0, arena_high = 0;
  bool arena_open = false; // between arena_begin() and arena_end()
  std::vector<size_t> arena_users; // bytes of each user; `g_setup_engines` prints them
  void arena_begin() {
    arena_first = arena_high = pos;
    arena_open = true;
  }
  void arena_next() {
    arena_users.push_back(pos - arena_first);
    arena_high = std::max(arena_high, pos);
    pos = arena_first;
  }
  void arena_end() {
    arena_next();
    pos = arena_high;
    shared_scratch_end = std::max(shared_scratch_end, pos);
    arena_open = arena_middle = false;
  }
  // The bytes at the end of the layer sub-proofs' arena that no buffer uses while a segment is
  // closed: they come after the layer sub-proofs' Late buffers and after the set-up of the head
  // and of each attention object. Call it after all set-ups and before any run.
  size_t idle_at_the_arenas_end() const {
    if (test.own_memory || test.disjoint || !slab) return 0;
    size_t from = arena_first + late_peak;
    for (size_t f = 0; f < end_of.size(); f++)
      if (class_of_fork[f] < 0) from = std::max(from, end_of[f]);
    from = rounded(from);
    return from < arena_high ? arena_high - from : 0;
  }
  size_t single_base = 0; // where single objects begin; set between the layer sub-proofs' set-up and theirs
  // The bytes at the slab's start up to single_base: the keys and cells, which nothing declares
  // dead in a run. 0 if Late buffers share the keys' addresses.
  size_t idle_at_the_start() const {
    if (test.own_memory || test.disjoint || !slab || late_keys_peak) return 0;
    return std::min(arena_first, single_base);
  }
  // Bytes [lent_from, lent_to) of the slab are lent for the life of the process to the code that
  // closes segments ("chain library" in messages). AllAtAClose: the slab's first bytes, whatever
  // is there: the borrower copies aside what is still needed. Nothing lays it or calls lend().
  enum class Lent { Nothing, Start, ArenasEnd, AllAtAClose } lent = Lent::Nothing;
  size_t lent_from = 0, lent_to = 0;
  // Lends `bytes` bytes of one of the two idle ranges: at the arena's end its last bytes, so that a
  // single object whose run takes more of the slab than its set-up did can grow into the rest.
  char *lend(bool at_the_start, size_t bytes) {
    const size_t idle = at_the_start ? idle_at_the_start() : idle_at_the_arenas_end();
    if (lent != Lent::Nothing || has_run[0] || !bytes || bytes > idle) {
      printf(
          "fork memory: internal error: an attempt to lend %zu bytes of the slab where %zu are idle; lent before: %s; "
          "a sub-proof has run: %s\n",
          bytes, idle, lent != Lent::Nothing ? "yes" : "no", has_run[0] ? "yes" : "no");
      exit(4);
    }
    lent = at_the_start ? Lent::Start : Lent::ArenasEnd;
    lent_from = at_the_start ? 0 : (arena_high - bytes) & ~(step - 1);
    lent_to = lent_from + bytes;
    return slab + lent_from;
  }
  // Ends the program if what is lent may be in use. Call it before a segment is closed (`closing`
  // is true), in the run of the segment's sub-proof, and before the borrower begins or finishes a
  // request, outside every run. Under AllAtAClose every point of a single object's run passes.
  // Its one caller, Prover::finish_request(), passes false.
  void check_what_is_lent(bool closing) const {
    if (lent == Lent::Nothing) return;
    bool idle = mode == Mode::Off;
    if (closing && mode == Mode::Run)
      idle = class_of_fork[fork] >= 0 ? late_begun && (keys_and_cells_are_dead || lent == Lent::ArenasEnd)
                                      : lent != Lent::Start || begin_of[fork] >= lent_to;
    else if (closing)
      idle = false;
    if (idle) return;
    printf(
        "fork memory: internal error: the chain library %s while the part of the slab that it was lent may be in use "
        "(sub-proof %d, %s a run)\n",
        closing ? "closes a link" : "begins or finishes a request", fork, mode == Mode::Run ? "in" : "outside");
    exit(4);
  }
  bool keys_and_cells_are_dead = false; // in the current run; nothing sets it to true
  // Call it where no user of the arena is at work: under test.scribble it overwrites the arena.
  void arena_idle() {
    if (!test.scribble || !slab || mode != Mode::Run || arena_high == arena_first) return;
    cudaMemsetAsync(slab + arena_first, 0xA5, arena_high - arena_first, cudaStreamPerThread);
    scribbles++;
  }
  // The arena's last user works in the middle of a run, between the product tests and late_begin().
  // Set-up zeroes some of its buffers, but earlier users write there after run_begin(). So a layer
  // sub-proof calls middle_begin() after its product tests to repeat set-up's writes for that user.
  bool arena_middle = false; // set-up is at the arena's last user
  std::vector<Write> middle_writes;
  bool middle_begun = false; // in the current run
  void middle_begin() {
    if (test.own_memory || !slab || mode != Mode::Run) return;
    if (middle_begun) {
      printf("fork memory: internal error: middle_begin() is called twice in the run of sub-proof %d\n", fork);
      exit(4);
    }
    middle_begun = true;
    arena_idle();
    repeat(middle_writes, launch_middle);
    look_at_what_is_not_repeated(middle_writes);
  }
  size_t early_from = 0, early_to = 0; // of the single object that is being set up
  // 0: no early_begin() yet, 1: between the two calls, 2: after them
  int early_state = 0;
  std::vector<std::pair<size_t, size_t>> early_of; // per sub-proof: a single object's early buffers in the slab
  // the Late buffers of the single object that is being set up
  std::vector<std::pair<size_t, size_t>> late_parts;
  // Set when a layer sub-proof's working set begins, or earlier by fork_alloc_in_slab(); see there.
  bool shared_scratch_closed = false;
  bool late_begun = false; // in the current run
  size_t layer_start(int g) const { return g == 0 ? start() : std::max(start(), shared_scratch_end); }
  std::vector<size_t> end_of; // per sub-proof: where its buffers end in the slab
  std::vector<char> has_run;
  std::vector<Buffers> results;
  std::vector<int> class_of_fork, lasting_class_of_fork, parity_of_fork;
  int lasting_cls = 0;
  // List 0 holds a class's Shape buffers, lists 1 and 2 its Lasting buffers of even and odd parity.
  int class_of_list(int list) const { return list == 0 ? cls : lasting_cls; }
  std::vector<size_t> begin_of, setup_end_of; // per sub-proof: its set-up's part of the slab
  size_t all_setup_end = 0;
  size_t start() const { return test.disjoint ? all_setup_end : 0; }
  // Per class, three lists in the order of allocation: a shape class uses list 0, a lasting class
  // lists 1 and 2. next[q] counts the buffers of list q that the current set-up has requested.
  std::vector<std::array<Buffers, 3>> recorded;
  std::map<std::vector<unsigned>, int> class_of;
  size_t next[3] = {0, 0, 0};

  char *slab = nullptr; // Public source: NVIDIA, CUDA C++ Programming Guide, "Virtual Memory Management".
  // See also: reserved addresses, memory mapped on demand: NVIDIA, CUDA Programming Guide,
  // "Virtual Memory Management",
  // https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/virtual-memory-management.html.
  // take_from_slab() reserves addresses at first use; map_up_to() maps memory; nothing unmaps it.
  static constexpr size_t addresses = (size_t)96 << 30; // reserved addresses; memory is mapped as needed
  // Offsets in the slab: where take_from_slab()'s next buffer starts, the highest `pos` so far, and
  // the end of the mapped memory.
  size_t pos = 0, peak = 0, mapped = 0;
  size_t layer_setup_end = 0; // the largest end of any layer sub-proof's set-up in the slab
  bool first_run = false;
  size_t shared_begin = 0; // where sub-proof 0's run-time buffers start; they end at end_of[0]
  char *store = nullptr;   // copies of what set-ups copied into the slab
  size_t store_pos = 0, store_size = 0;
  // Totals that report() prints. bytes_again: what sharing saved. logged: set-ups' logged writes.
  size_t bytes_own = 0, bytes_per_shape = 0, bytes_again = 0, bytes_shared = 0, logged = 0;
  // Layer sub-proof 0's slab requests after its first run, which fork_alloc() allocates separately.
  size_t turned_from_the_slab = 0, bytes_turned_from_the_slab = 0;

  // A shared set-up has no closing call: the next set-up or run records here how far it reached.
  void note_end() {
    if (mode == Mode::SharedSetup) end_of[fork] = std::max(end_of[fork], pos);
  }

  // Every buffer in the slab starts at a multiple of `step` bytes.
  // rounded() rounds a number of bytes up to such a multiple.
  static constexpr size_t step = 1024;
  static size_t rounded(size_t bytes) { return (bytes + step - 1) & ~(step - 1); }
  // Maps physical memory for the slab's first `end` bytes, in pieces of 256 MiB, and zeroes each
  // new piece. If a driver call or the zeroing fails, it calls exit(1).
  void map_up_to(size_t end) {
    const size_t mapped_now = mapped;
    auto check = [mapped_now](CUresult r, const char *what) {
      if (r != CUDA_SUCCESS) {
        const char *m = nullptr;
        cuGetErrorString(r, &m);
        printf("fork memory: cannot grow the slab beyond %.1f GB: %s failed: %s\n", mapped_now / 1e9, what,
               m ? m : "unknown error");
        name_the_end("cuda-call");
        exit(1);
      }
    };
    const size_t piece = (size_t)256 << 20;
    while (mapped < end) {
      CUmemAllocationProp prop{};
      prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
      prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      int device = 0;
      cudaGetDevice(&device);
      prop.location.id = device;
      CUmemGenericAllocationHandle handle;
      check(cuMemCreate(&handle, piece, &prop, 0), "cuMemCreate");
      check(cuMemMap((CUdeviceptr)slab + mapped, piece, 0, handle, 0), "cuMemMap");
      CUmemAccessDesc access{};
      access.location = prop.location;
      access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      check(cuMemSetAccess((CUdeviceptr)slab + mapped, piece, &access, 1), "cuMemSetAccess");
      zero_new_piece(mapped, piece);
      mapped += piece;
      note_what_is_taken();
    }
  }
  // Maps physical memory for at least the slab's first min(`bytes`, addresses) bytes and allocates
  // none of it; does nothing before the slab's addresses are reserved. For a borrower that needs
  // more of the slab than the sub-proofs do: a batch's test on the weight table (close_memory.cuh).
  void map_at_least(size_t bytes) {
    if (slab) map_up_to(std::min(bytes, addresses));
  }
  // Returns memory from the reserved slab, mapping physical memory as needed.
  void *take_from_slab(size_t bytes) {
    const size_t mapped_now = mapped;
    auto check = [mapped_now](CUresult r, const char *what) {
      if (r != CUDA_SUCCESS) {
        const char *m = nullptr;
        cuGetErrorString(r, &m);
        printf("fork memory: cannot grow the slab beyond %.1f GB: %s failed: %s\n", mapped_now / 1e9, what,
               m ? m : "unknown error");
        name_the_end("cuda-call");
        exit(1);
      }
    };
    if (!slab) {
      cudaFree(0);
      CUdeviceptr base = 0;
      check(cuMemAddressReserve(&base, addresses, 0, 0, 0), "cuMemAddressReserve");
      slab = (char *)base;
    }
    const size_t n = rounded(bytes);
    // Sub-proof 0's run-time buffers serve objects that all layer sub-proofs
    // use, so another layer sub-proof's buffers must not share their addresses.
    if (mode == Mode::Run && fork != 0 && class_of_fork[fork] >= 0 && pos < end_of[0] && pos + n > shared_begin)
      pos = end_of[0];
    if (lent == Lent::ArenasEnd && class_of_fork[fork] < 0 && pos + n > lent_from) {
      printf(
          "fork memory: sub-proof %d (the head's or an attention sub-proof) needs the slab up to %.3f GB in its run, "
          "but from %.3f GB on the slab is lent to the chain library; remove close-on-the-slab from -memory\n",
          fork, (pos + n) / 1e9, lent_from / 1e9);
      exit(2);
    }
    if (pos + n > addresses) {
      printf("fork memory: a sub-proof needs more than the %zu GB of addresses reserved for the slab; raise "
             "'addresses' in prover/prover/fork_memory.h\n",
             addresses >> 30);
      name_the_end("memory-plan"); // serving: p258 end 1 memory-plan (M5g)
      exit(2);
    }
    map_up_to(pos + n);
    void *p = slab + pos;
    pos += n;
    peak = std::max(peak, pos);
    return p;
  }

  // Copies `bytes` bytes from `src` into device memory that is never freed and returns the copy.
  const void *keep_copy(const void *src, size_t bytes, cudaMemcpyKind kind) {
    const size_t n = (bytes + 15) & ~(size_t)15;
    if (store_pos + n > store_size) {
      store_size = std::max((size_t)64 << 20, n);
      store_pos = 0;
      if (device_malloc((void **)&store, store_size) != cudaSuccess) {
        printf("fork memory: out of GPU memory (%zu bytes needed to keep a copy of what the set-up wrote into the "
               "slab)\n",
               store_size);
        name_the_end("cuda-call");
        exit(1);
      }
    }
    void *d = store + store_pos;
    store_pos += n;
    if (cudaMemcpy(d, src, bytes, kind) != cudaSuccess) {
      printf("fork memory: cudaMemcpy of %zu bytes failed: %s\n", bytes, cudaGetErrorString(cudaGetLastError()));
      name_the_end("cuda-call");
      exit(1);
    }
    return d;
  }
  std::map<void *, size_t> device_buffers, host_buffers; // the record: bytes of each live buffer
  // The label of buffers allocated outside any sub-proof's set-up and run. Callers
  // set it around their allocations; ledger() adds the buffers up by label.
  const char *purpose = "other buffers of the prover";
  std::map<void *, const char *> purpose_of;
  // GPU bytes taken by code outside this header, by name, measured with cudaMemGetInfo() before and after
  // its calls.
  std::map<std::string, size_t> taken_by_others;
  static size_t gpu_bytes_in_use() {
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return total_bytes - free_bytes;
  }
  struct LedgerTotals {
    size_t in_use = 0, rooms = 0, in_no_record = 0; // bytes
  };
  // Prints one line that splits the GPU memory in use by owner. Returns all that is in use, the
  // buffers for streamed weights (`rooms`) and, as in_no_record, what neither the record, the slab
  // nor taken_by_others accounts for, such as the CUDA context, loaded kernels, CUDA graphs and
  // cuBLAS's memory.
  LedgerTotals ledger(const char *when) const {
    std::map<std::string, size_t> sum;
    size_t recorded = mapped, named = 0;
    for (auto &b : device_buffers) {
      recorded += b.second;
      auto it = purpose_of.find(b.first);
      if (it != purpose_of.end()) sum[it->second] += b.second, named += b.second;
    }
    size_t free_bytes = 0, total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    const size_t in_use = total_bytes - free_bytes;
    printf("# GPU memory %s, GB: %.3f of %.3f in use:", when, in_use / 1e9, total_bytes / 1e9);
    for (auto &s : sum) printf(" %s %.3f;", s.first.c_str(), s.second / 1e9);
    printf(" the sub-proofs' slab %.3f; the sub-proofs' other buffers %.3f;", mapped / 1e9,
           (recorded - mapped - named) / 1e9);
    size_t others = 0;
    for (auto &s : taken_by_others) {
      printf(" %s %.3f;", s.first.c_str(), s.second / 1e9);
      others += s.second;
    }
    printf(" not accounted for by the prover %.3f\n", ((double)in_use - (double)recorded - (double)others) / 1e9);
    LedgerTotals totals;
    totals.in_use = in_use;
    totals.rooms = sum.count("rooms for streamed weights") ? sum["rooms for streamed weights"] : 0;
    totals.in_no_record = in_use > recorded + others ? in_use - recorded - others : 0;
    return totals;
  }
  std::set<void *> shared_buffers; // Shape and Lasting buffers: other sub-proofs hold them too
  size_t scribbles = 0;
  // device_malloc() returns zeroed memory, so that nothing that an earlier owner left is ever read. The zeroing is
  // complete on return, because the calling code may fill the buffer at once on a stream that is not ordered after the
  // zeroing. The zeroing has a stream of its own, so that a capture of the calling code's stream does not record it.
  cudaStream_t zeroing_stream = nullptr;
  cudaError_t fill_and_wait(void *p, int value, size_t bytes) {
    cudaError_t e = cudaSuccess;
    if (!zeroing_stream) e = cudaStreamCreateWithFlags(&zeroing_stream, cudaStreamNonBlocking);
    if (e == cudaSuccess) e = cudaMemsetAsync(p, value, bytes, zeroing_stream);
    if (e == cudaSuccess) e = cudaStreamSynchronize(zeroing_stream);
    return e;
  }
  // Incremented whenever device_malloc() or fork_free() changes the
  // record: a list built from the record is then stale.
  size_t allocations = 0;
  // The guard on growing GPU memory. No buffer's size is meant to depend on a request's values or
  // on a random value that the prover draws; that none does has not been shown for all allocations.
  // (grouped_blocks()'s host code, prover/weights_blocks/experts.cuh, reads routing tables but uses
  // fixed-size buffers.) At run time this guard is what watches that condition.
  // A service is meant to allocate all its GPU memory in set-up and in the sub-proofs' first runs.
  // hold_to_what_is_taken() is called once, at the end of the process's first batch pass
  // (Prover::wave_has_ended()). The guard, note_what_is_taken(), reports a peak (most_taken) above
  // the one at that call (held_to; 0: no call yet), and nothing at or below it. It only reports:
  // allocating more does not end the process. Its one line is a printf(), which a serving build
  // never shows: ClosedFront::close() (prover/prover/closed_front.h) has redirected standard output
  // to /dev/null before the first pass. What the operator sees there is one constant line, for the
  // batch's alarm End::MemoryGrew (prover/prover/ends.h), which taken_beyond_the_hold() feeds and
  // for which no request loses its files.
  // It counts device_malloc()'s live bytes and the slab's mapped bytes, so it does not notice a
  // slab buffer that fits in memory already mapped (pieces of 256 MiB). It does not count the
  // buffers for streamed weights, Launch's and MemoryAsSetUp::own_memory()'s tables, what cuBLAS
  // allocates itself (prover/common/blas.h), the GPU engine's memory pool (card_pool(),
  // prover/engine/device.cuh), what code compiled without this header allocates
  // (prover/hash_commitment/hf_keeper.cu) or what the CUDA runtime takes, such as a kernel's code
  // at first use.
  // It can go off without any fault: a later pass with more requests than the first can allocate
  // more (concluded from reading the code; not tried). It never resets: most_taken never falls, so
  // taken_beyond_the_hold() then stays true until the process ends.
  size_t recorded_bytes = 0, most_taken = 0, held_to = 0;
  void note_what_is_taken() {
    most_taken = std::max(most_taken, recorded_bytes + mapped);
    if (!taken_beyond_the_hold() || said_what_was_taken) return;
    said_what_was_taken = true; // one line, at the first allocation beyond held_to
    printf("# fork memory: GPU memory was allocated after the limit was fixed: %zu bytes more than the limit of %zu; "
           "the slab's mapped bytes %zu, when the limit was fixed %zu; %s, sub-proof %d; the current label: %s\n",
           most_taken - held_to, held_to, mapped, mapped_at_the_hold,
           mode == Mode::Run ? "in a run" : mode == Mode::Off ? "outside the sub-proofs" : "in a set-up", fork,
           purpose);
  }
  bool said_what_was_taken = false;
  size_t mapped_at_the_hold = 0;
  void hold_to_what_is_taken() { held_to = most_taken, mapped_at_the_hold = mapped; }
  bool taken_beyond_the_hold() const { return held_to && most_taken > held_to; }
  cudaError_t device_malloc(void **p, size_t bytes) {
    cudaError_t e = cudaMalloc(p, bytes);
    if (e != cudaSuccess) return e;
    device_buffers[*p] = bytes;
    allocations++, recorded_bytes += bytes;
    note_what_is_taken();
    if (test.left_behind) born[*p] = {++buffers_born, mode, single ? -1 : fork};
    return fill_and_wait(*p, 0, bytes);
  }
  // Under test.left_behind, for each device buffer: its number in the order of allocation, and the
  // mode and sub-proof (-1: a single object's set-up) in which it was allocated. One that did not
  // enter the record through device_malloc() is numbered at the first look_at_what_stays() that
  // finds it. at_set_ups_end: the digest at the end of set-up; 0: all zeros.
  struct Born {
    int number;
    Mode mode;
    int fork;
  };
  int buffers_born = 0;
  std::map<void *, Born> born;
  std::map<void *, unsigned long long> at_set_ups_end;
  size_t looks_made = 0;
  // Under test.left_behind: shows what the requests proved so far have left on the GPU. Call it
  // while the GPU is idle: once at the end of set-up, before the first request is read, then after
  // the restoring at each pass's end. It writes to none of the prover's buffers. It computes a
  // digest of every device buffer in this header's record, except the weights and the buffers into
  // which they are streamed, and counts the slab's bytes that are not zero. After a pass every
  // buffer must be as it was at the end of set-up, or all zeros if it was allocated since. A line
  // is printed for each that is not, and a last line gives the totals. Buffers in `laid_once` are
  // expected among them, and their lines say so. A difference ends nothing, and no test reads these
  // lines.
  // Not inspected: pinned host memory, and GPU memory outside the record, such as what cuBLAS and
  // prover/hash_commitment/hf_keeper.cu allocate, the GPU engine's memory pool (card_pool(),
  // prover/engine/device.cuh), CUDA graphs and __device__ variables.
  // Public source: one hash per block, compared with a later one, as stated in [libhashckpt]
  // Kurt B. Ferreira, Rolf Riesen, Ron Brightwell, Patrick Bridges, Dorian Arnold, "libhashckpt:
  // Hash-based Incremental Checkpointing Using GPU's", 2011, abstract and section 2.1;
  // U.S. Department of Energy's OSTI, https://www.osti.gov/servlets/purl/1109273,
  // report number SAND2011-3600C. Here the blocks are the allocated buffers, and the kept hashes
  // are those of the end of set-up.
  // Public source for a result returned through managed memory: see k_fork_memory_count_not().
  void look_at_what_stays() {
    static unsigned long long *sum = nullptr; // two words: the digest, and how many bytes are not zero
    bool ok = cudaDeviceSynchronize() == cudaSuccess && (sum || cudaMallocManaged(&sum, 16) == cudaSuccess);
    std::map<void *, unsigned long long> now;
    size_t all = 0;
    for (auto &b : device_buffers) {
      if (!ok) break;
      auto label = purpose_of.find(b.first);
      if (label != purpose_of.end() &&
          (!strcmp(label->second, "rooms for streamed weights") || !strcmp(label->second, "weights kept on the GPU")))
        continue;
      sum[0] = sum[1] = 0;
      k_fork_memory_digest<<<1024, 256>>>((const unsigned char *)b.first, b.second, sum);
      k_fork_memory_count_not<<<1024, 256>>>((const unsigned char *)b.first, b.second, 0, sum + 1);
      ok = cudaDeviceSynchronize() == cudaSuccess;
      if (!ok) break;
      now[b.first] = sum[1] ? (sum[0] ? sum[0] : 1) : 0;
      all += b.second;
      if (!born.count(b.first)) born[b.first] = {++buffers_born, Mode::Off, 0};
    }
    unsigned long long of_the_slab = 0;
    if (ok && slab && mapped) {
      sum[1] = 0;
      k_fork_memory_count_not<<<1024, 256>>>((const unsigned char *)slab, mapped, 0, sum + 1);
      ok = cudaDeviceSynchronize() == cudaSuccess;
      of_the_slab = sum[1];
    }
    if (!ok) {
      printf("test mode: left behind: the GPU's memory could not be read (%s); the program ends\n",
             cudaGetErrorString(cudaGetLastError()));
      name_the_end("cuda-call");
      exit(1);
    }
    const size_t n = ++looks_made;
    if (n == 1) at_set_ups_end = now;
    auto word = [](unsigned long long digest) {
      char text[24];
      snprintf(text, sizeof text, "%016llx", digest);
      return digest ? std::string(text) : std::string("zeros");
    };
    size_t others = 0, once = 0;
    for (auto &b : now) {
      auto then = at_set_ups_end.find(b.first);
      const unsigned long long wanted = then == at_set_ups_end.end() ? 0 : then->second;
      if (b.second == wanted) continue;
      bool laid = false;
      for (auto &l : laid_once) laid = laid || l.first == b.first;
      others++, once += laid;
      const Born &from = born.find(b.first)->second;
      auto label = purpose_of.find(b.first);
      printf("test mode: left behind: look %zu: buffer %d (%s", n, from.number,
             from.mode == Mode::Off           ? "outside the sub-proofs"
             : from.mode == Mode::Setup       ? "a set-up"
             : from.mode == Mode::SharedSetup ? "the shared set-up"
                                              : "a run");
      if (from.mode != Mode::Off) printf(", sub-proof %d", from.fork);
      printf("%s%s): %zu bytes: %s at set-up's end, %s now%s\n", label == purpose_of.end() ? "" : ", ",
             label == purpose_of.end() ? "" : label->second, device_buffers.at(b.first),
             then == at_set_ups_end.end() ? "not there" : word(wanted).c_str(), word(b.second).c_str(),
             laid ? "; laid once" : "");
    }
    printf("test mode: left behind: look %zu, %s: %zu buffers, %zu bytes; not as at set-up's end: %zu, of them laid once: "
           "%zu; bytes of the slab that are not zero: %llu\n",
           n, n == 1 ? "at set-up's end" : "behind a wave", now.size(), all, others, once, of_the_slab);
    fflush(stdout);
  }
  // Zeroes a piece of physical memory that the slab has just mapped: a run can map one, and the
  // zeroing at its request's start did not cover it. Under test.stale (see there) it prints a line.
  // Public source: a marker fill, then a count of what survives: J. Chow, B. Pfaff, T. Garfinkel,
  // M. Rosenblum, "Shredding Your Garbage: Reducing Data Lifetime Through Secure Deallocation",
  // 14th USENIX Security Symposium, 2005,
  // https://static.usenix.org/events/sec05/tech/full_papers/chow/chow_html/index.html.
  // There the count measures how long released data stays in memory and whether a reboot clears it.
  // Public source: a fill with a non-zero byte (glibc's M_PERTURB): Linux man-pages, mallopt(3),
  // M_PERTURB, https://man7.org/linux/man-pages/man3/mallopt.3.html.
  // Here the marker is written just before our own zeroing.
  // Public source: an omitted zeroing: a hand-made case of fault seeding as in mutation testing
  // (Jia and Harman, An Analysis and Survey of the Development of Mutation Testing, IEEE TSE;
  // operator 'statement deletion').
  // Public source for a result returned through managed memory: see k_fork_memory_count_not().
  void zero_new_piece(size_t offset, size_t bytes) {
    cudaError_t e = test.stale ? fill_and_wait(slab + offset, 0xEE, bytes) : cudaSuccess;
    if (e == cudaSuccess && !test.skip_zero_shipped) e = fill_and_wait(slab + offset, 0, bytes);
    static unsigned long long *left = nullptr;
    if (e == cudaSuccess && test.stale && !left) e = cudaMallocManaged(&left, 8);
    if (e == cudaSuccess && test.stale) {
      *left = 0;
      k_fork_memory_count_not<<<1024, 256, 0, zeroing_stream>>>((const unsigned char *)slab + offset, bytes, 0, left);
      e = cudaStreamSynchronize(zeroing_stream);
    }
    if (e != cudaSuccess) {
      printf("fork memory: newly mapped GPU memory could not be zeroed (%s); the program ends\n",
             cudaGetErrorString(e));
      name_the_end("cuda-call");
      exit(1);
    }
    if (!test.stale) return;
    printf("test mode: the slab's bytes from %zu on: a new piece of %zu bytes, mapped %s, was overwritten with 0xEE%s; "
           "bytes that are not zero: %llu%s\n",
           offset, bytes, mode == Mode::Run ? "in a run" : "outside every run",
           test.skip_zero_shipped ? " and, as this mode asks, NOT zeroed" : ", then zeroed", *left,
           !*left                   ? ""
           : test.skip_zero_shipped ? "; this mode goes on"
                                    : "; the program ends, because memory that must be zero is not");
    fflush(stdout);
    if (*left && !test.skip_zero_shipped) exit(3); // skip_zero_shipped: go on, its test needs the first request's count
  }
  // wipe() zeroes every device and pinned host buffer in the record and the mapped slab. It returns
  // false if zeroing GPU memory failed, here or in a fork_free(). It does not reach memory outside
  // the record and the slab, such as the GPU memory that look_at_what_stays() lists as not
  // inspected, and prover/prover/weight_stream.cuh's pinned host memory (HostMemory), which holds
  // the host copy of the weights.
  bool freed_unzeroed = false; // `fork_free` could not zero a buffer before it freed it
  bool wipe() {
    bool ok = cudaDeviceSynchronize() == cudaSuccess && !freed_unzeroed;
    for (auto &b : device_buffers) ok = cudaMemset(b.first, 0, b.second) == cudaSuccess && ok;
    if (slab) ok = cudaMemset(slab, 0, mapped) == cudaSuccess && ok;
    ok = cudaDeviceSynchronize() == cudaSuccess && ok;
    {
      // The pinned host buffers are zeroed in pieces of 256
      // MiB, shared among this call and up to seven workers.
      std::vector<std::pair<char *, size_t>> pieces;
      const size_t piece = (size_t)256 << 20;
      for (auto &b : host_buffers)
        for (size_t o = 0; o < b.second; o += piece)
          pieces.push_back({(char *)b.first + o, std::min(piece, b.second - o)});
      std::atomic<size_t> next{0};
      auto work = [&] {
        for (size_t i; (i = next++) < pieces.size();) explicit_bzero(pieces[i].first, pieces[i].second);
      };
      std::vector<std::thread> workers;
      try {
        for (int w = 0; w < 7; w++) workers.emplace_back(work);
      } catch (...) { // no further worker can be started: this call zeroes what is left
      }
      work();
      for (auto &w : workers) w.join();
    }
    if (!ok)
      fprintf(stderr,
              "fork memory: GPU memory could not be zeroed; weights and masks may stay on the GPU until its memory is "
              "reused or the GPU is reset\n");
    return ok;
  }
  Buffers of_a_request; // Own buffers that, as their owners declare, hold a request's data
  // Memory that the restoring between requests (prover/prover/memory_as_set_up.cuh) skips: as its
  // owners declare, a run writes constants for later runs there, and no request's data. With
  // -overlap also prover/prover/close_memory.cuh's buffers, which are not constants but working
  // memory of a batch's test on the weight table: the borrower zeroes them when it takes them and
  // before it returns them (lent::end_opening(), prover/weights_commitment/lent_memory.h).
  Buffers laid_once;
  // Under test.disjoint: calls exit(3) unless other sub-proofs' parts of the slab still hold 0xA5.
  // Public source: guard bytes as in Microsoft's CRT debug heap: Microsoft, C runtime library
  // documentation, "CRT debug heap details" (parts on _FREE_BLOCK and on _crtDbgFlag),
  // https://learn.microsoft.com/cpp/c-runtime-library/crt-debug-heap-details.
  // Public source: guard bytes of GPU buffers, checked on the GPU after a kernel, as in C. Erb, M.
  // Collins, J. L. Greathouse, "Dynamic Buffer Overflow Detection for GPGPUs", CGO 2017, abstract
  // and the design section;
  // http://www.computermachines.org/joe/publications/pdfs/cgo2017_clarmor.pdf.
  // Here the guarded memory is the other sub-proofs' parts of the slab.
  // Public source for a result returned through managed memory: see k_fork_memory_count_not().
  void check_other_parts() {
    static unsigned long long *count = nullptr;
    if (!count) cudaMallocManaged(&count, 8);
    cudaDeviceSynchronize();
    *count = 0;
    const unsigned char *s = (const unsigned char *)slab;
    // A layer sub-proof may write the shared engines' scratch at the slab's start, so when
    // class_of_fork[fork] >= 0 the check starts at the end of that scratch (or at the
    // sub-proof's own part, if that comes first); otherwise it starts at the slab's first byte.
    const size_t from = class_of_fork[fork] >= 0 ? std::min(shared_scratch_end, begin_of[fork]) : 0;
    if (begin_of[fork] > from) k_fork_memory_count_not<<<1024, 256>>>(s + from, begin_of[fork] - from, 0xA5, count);
    if (all_setup_end > setup_end_of[fork])
      k_fork_memory_count_not<<<1024, 256>>>(s + setup_end_of[fork], all_setup_end - setup_end_of[fork], 0xA5, count);
    cudaDeviceSynchronize();
    if (*count) {
      printf("fork memory: the run of sub-proof %d wrote %llu bytes into other sub-proofs' parts of the slab "
             "(sub-proofs are numbered: layer sub-proofs from 0, then the head's, then the attention sub-proofs)\n",
             fork, *count);
      exit(3);
    }
  }
  // Returns the list that logs the current set-up's write at this offset. Where runs repeat a list:
  //   `late_writes[fork]` late_begin(): Late buffers (layer sub-proofs': below shared_scratch_end)
  //   middle_writes       middle_begin(): the arena's last user
  //   shared_writes       every layer sub-proof's run_begin(): the other shared buffers
  //   `writes[fork]`      this sub-proof's run_begin(): its other buffers
  std::vector<Write> &list_of(size_t offset) {
    bool late_of_single = false;
    if (single)
      for (auto &b : late_parts) late_of_single = late_of_single || (offset >= b.first && offset < b.first + b.second);
    return late_of_single                           ? late_writes[fork]
           : single || offset >= shared_scratch_end ? writes[fork]
           : mode == Mode::SharedSetup              ? (arena_middle ? middle_writes : shared_writes)
                                                    : late_writes[fork];
  }
  // Logs a set-up's write into the slab: a copy from `copy_of` or, if that is null, a fill.
  void log_write(void *p, size_t bytes, int value, const void *copy_of) {
    const size_t offset = (size_t)((char *)p - slab);
    if (arena_open && arena_users.size() == 1) {
      printf(
          "fork memory: internal error: the set-up writes into the scratch of the second user of the layer sub-proofs' "
          "shared scratch area, which the forward pass overwrites before it is read\n");
      exit(4);
    }
    list_of(offset).push_back({offset, bytes, value, copy_of});
    logged++;
  }
  // Call it right after a cudaMemset of the current set-up at p, before plan_repeats(). It declares
  // that every run writes the first `bytes` bytes at p before it reads them, so runs repeat only
  // the rest of that cudaMemset. This is not checked; under test.scribble report() prints how many
  // of these bytes were not 0xA5 at the point where the repeat would have been.
  void every_run_writes(void *p, size_t bytes) {
    if (!in_setup() || !in_slab(p) || !bytes) return;
    const size_t at = (size_t)((char *)p - slab);
    std::vector<Write> &list = list_of(at);
    auto w = std::find_if(list.rbegin(), list.rend(), [&](const Write &x) { return x.offset == at; });
    if (planned || w == list.rend() || w->bytes < bytes || w->copy_of) {
      printf("fork memory: internal error: every_run_writes(): %s\n",
             planned ? "called after plan_repeats()" : "the current set-up has not set these bytes");
      exit(4);
    }
    not_repeated[&list].push_back({at, at + bytes});
    cuts++;
    if (w->bytes == bytes) {
      list.erase(std::next(w).base());
      logged--;
    } else {
      w->offset += bytes;
      w->bytes -= bytes;
    }
  }
  size_t cuts = 0; // calls of every_run_writes() so far: a table made before one of them is stale
  std::map<const void *, std::vector<std::pair<size_t, size_t>>> not_repeated; // by list: first byte and end
  // Under test.scribble, after a list's repeat: counts, in what
  // every_run_writes() removed from the list, the bytes that are not 0xA5.
  unsigned long long *not_scribbled = nullptr;
  double looked_at = 0;
  void look_at_what_is_not_repeated(const std::vector<Write> &list) {
    if (!test.scribble) return;
    auto found = not_repeated.find(&list);
    if (found == not_repeated.end()) return;
    if (!not_scribbled) {
      if (cudaMallocManaged(&not_scribbled, 8) != cudaSuccess) {
        printf("fork memory: test mode: no memory for a counter\n");
        exit(4);
      }
      cudaDeviceSynchronize();
      *not_scribbled = 0;
    }
    for (auto &r : found->second) {
      k_fork_memory_count_not<<<1024, 256, 0, cudaStreamPerThread>>>((const unsigned char *)slab + r.first,
                                                                     r.second - r.first, 0xA5, not_scribbled);
      looked_at += (double)(r.second - r.first);
    }
  }
};
static ForkMemory fork_memory;

// Allocates `bytes` bytes with the given placement, unless one of the cases below replaces it. The
// first sub-proof of a shape class allocates the class's Shape buffers, the first of a lasting
// class and parity its Lasting buffers; later sub-proofs get the same buffers, in the same order.
// Outside set-ups and runs every buffer is a device_malloc() allocation under the label `purpose`.
// Only a new device_malloc() allocation is zeroed: a buffer in the slab may hold what earlier users
// of its addresses left.
static cudaError_t fork_alloc(void **p, size_t bytes, Place place) {
  ForkMemory &m = fork_memory;
  if (place == Place::Result && !m.in_setup()) {
    printf("fork memory: internal error: a Result buffer is allocated outside a sub-proof's set-up\n");
    exit(4);
  }
  if (m.mode == ForkMemory::Mode::Off) {
    cudaError_t e = m.device_malloc(p, bytes);
    if (e == cudaSuccess && m.also_list) m.also_list->push_back({*p, bytes});
    if (e == cudaSuccess) m.purpose_of[*p] = m.purpose;
    return e;
  }
  if (place == Place::Result) {
    cudaError_t e = m.device_malloc(p, bytes);
    if (e == cudaSuccess) {
      m.bytes_own += bytes;
      m.results[m.fork].push_back({*p, bytes});
    }
    return e;
  }
  if (m.test.own_memory) place = Place::Own;
  if ((place == Place::Shape || place == Place::Lasting) && (m.single || m.mode == ForkMemory::Mode::Run))
    place = Place::Own;
  // Only the set-up of a layer sub-proof's working set, or of a single object with early buffers,
  // has addresses for Late buffers to share. Elsewhere, and under test.disjoint, Late becomes Fork.
  if (place == Place::Late && m.single && m.early_state == 1) m.early_misused("a Late buffer before early_end()");
  if (place == Place::Late &&
      (m.mode != ForkMemory::Mode::Setup || m.test.disjoint || (m.single && m.early_to == m.early_from)))
    place = Place::Fork;
  // After its first run sub-proof 0 gets no more of the slab: other
  // layer sub-proofs' run-time buffers may start where its own ended.
  if (m.mode == ForkMemory::Mode::Run && m.fork == 0 && !m.first_run) {
    if (place != Place::Own) m.turned_from_the_slab++, m.bytes_turned_from_the_slab += bytes;
    place = Place::Own;
  }
  if (m.mode == ForkMemory::Mode::SharedSetup) {
    if (place == Place::Fork && m.fork == 0) {
      if (m.shared_scratch_closed) {
        printf("fork memory: internal error: a shared engine requests scratch after layer sub-proof 0 took a buffer of "
               "its own from the slab\n");
        exit(4);
      }
      *p = m.take_from_slab(bytes);
      m.shared_scratch_end = std::max(m.shared_scratch_end, m.pos);
      return cudaSuccess;
    }
    m.bytes_shared += bytes;
    return m.device_malloc(p, bytes);
  }
  if (place == Place::Own) {
    m.bytes_own += bytes;
    return m.device_malloc(p, bytes);
  }
  if (place == Place::Late) {
    const size_t n = ForkMemory::rounded(bytes);
    if (m.late_pos + n <= (m.single ? m.early_to : m.shared_scratch_end)) {
      *p = m.slab + m.late_pos;
      if (m.single) m.late_parts.push_back({m.late_pos, n});
      m.late_pos += n;
      if (!m.single) m.late_peak = std::max(m.late_peak, m.late_pos - m.arena_first);
      return cudaSuccess;
    }
    if (!m.single && m.late_on_keys && m.late_keys_pos + n <= m.keys_end) {
      *p = m.slab + m.late_keys_pos;
      m.late_keys_pos += n;
      m.late_keys_peak = std::max(m.late_keys_peak, m.late_keys_pos - m.begin_of[0]);
      return cudaSuccess;
    }
    place = Place::Fork; // no room left there
  }
  if (place == Place::Fork) {
    *p = m.take_from_slab(bytes);
    return cudaSuccess;
  }
  const int list = place == Place::Lasting ? 1 + m.parity : 0;
  auto &rec = m.recorded[m.class_of_list(list)][list];
  size_t &at = m.next[list];
  if (at < rec.size()) {
    if (rec[at].second != bytes) {
      printf("fork memory: internal error: %s class %d, shared buffer number %zu: this sub-proof requests %zu bytes, "
             "the first sub-proof of the class allocated %zu\n",
             list == 0 ? "shape" : "lasting", m.class_of_list(list), at, bytes, rec[at].second);
      exit(4);
    }
    *p = rec[at++].first;
    m.bytes_again += bytes;
    return cudaSuccess;
  }
  cudaError_t e = m.device_malloc(p, bytes);
  if (e == cudaSuccess) {
    m.shared_buffers.insert(*p);
    rec.push_back({*p, bytes});
    at++;
    m.bytes_per_shape += bytes;
  }
  return e;
}

template <class T_> static cudaError_t fork_alloc(T_ **p, size_t bytes, Place place) {
  return fork_alloc((void **)p, bytes, place);
}
// Allocates a layer sub-proof's own Fork buffer in the shared part of that sub-proof's set-up, straight from
// the slab. (There fork_alloc() would add sub-proof 0's buffer to the shared engines' scratch, and would
// give any other sub-proof a cudaMalloc allocation.) Under test.own_memory the placement is Own instead.
//
// The shared engines' scratch is one range at the slab's start and cannot grow once this buffer follows
// it. So the call sets shared_scratch_closed (not under test.own_memory), and fork_alloc() ends the
// program with an internal error if a shared engine asks for scratch afterwards (placement Fork,
// sub-proof 0, mode SharedSetup).
template <class T_> static cudaError_t fork_alloc_in_slab(T_ **p, size_t bytes) {
  if (fork_memory.test.own_memory) return fork_alloc((void **)p, bytes, Place::Own);
  fork_memory.shared_scratch_closed = true;
  *p = (T_ *)fork_memory.take_from_slab(bytes);
  return cudaSuccess;
}
// Placement Shape: what a plain cudaMalloc gets.
template <class T_> static cudaError_t fork_alloc_default(T_ **p, size_t bytes) {
  return fork_alloc((void **)p, bytes, Place::Shape);
}
// What cudaFree becomes. A buffer in the slab stays as it is (success). Freeing a shared Shape or
// Lasting buffer is an internal error: exit(4). A buffer in the record is zeroed first; if that
// fails, wipe() returns false. A pointer that is not in the record goes to cudaFree unchanged.
static cudaError_t fork_free(void *p) {
  if (fork_memory.in_slab(p)) return cudaSuccess;
  if (fork_memory.shared_buffers.count(p)) {
    printf("fork memory: internal error: a buffer that sub-proofs of one shape share is freed\n");
    exit(4);
  }
  auto it = fork_memory.device_buffers.find(p);
  if (it != fork_memory.device_buffers.end()) {
    cudaDeviceSynchronize();
    if (cudaMemset(p, 0, it->second) != cudaSuccess) fork_memory.freed_unzeroed = true;
    fork_memory.allocations++, fork_memory.recorded_bytes -= it->second;
    fork_memory.device_buffers.erase(it);
    fork_memory.purpose_of.erase(p);
    if (fork_memory.test.left_behind) {
      fork_memory.born.erase(p);
      fork_memory.at_set_ups_end.erase(p); // another buffer may get this address
    }
  }
  return cudaFree(p);
}
// What cudaMallocHost becomes: pinned host memory, zeroed, and recorded so that wipe() zeroes it.
// cudaFreeHost is not replaced: a freed buffer would stay in the record, and wipe() would write it.
template <class T_> static cudaError_t fork_malloc_host(T_ **p, size_t bytes) {
  cudaError_t e = cudaMallocHost(p, bytes);
  if (e == cudaSuccess) fork_memory.host_buffers[(void *)*p] = bytes, memset((void *)*p, 0, bytes);
  return e;
}
// What cudaMemset, cudaMemcpy and their Async forms become. In a set-up, a write into the slab is
// also logged, so that runs repeat it; an Async form then waits for `stream` and calls the plain
// form.
static cudaError_t fork_memset(void *p, int value, size_t bytes) {
  cudaError_t e = cudaMemset(p, value, bytes);
  if (fork_memory.in_setup() && fork_memory.in_slab(p)) fork_memory.log_write(p, bytes, value, nullptr);
  return e;
}
static cudaError_t fork_memcpy(void *d, const void *s, size_t bytes, cudaMemcpyKind kind) {
  cudaError_t e = cudaMemcpy(d, s, bytes, kind);
  if (fork_memory.in_setup() && fork_memory.in_slab(d) && bytes)
    fork_memory.log_write(d, bytes, 0, fork_memory.keep_copy(s, bytes, kind));
  return e;
}
static cudaError_t fork_memset_async(void *p, int value, size_t bytes, cudaStream_t stream = 0) {
  if (fork_memory.in_setup() && fork_memory.in_slab(p)) {
    cudaStreamSynchronize(stream);
    return fork_memset(p, value, bytes);
  }
  return cudaMemsetAsync(p, value, bytes, stream);
}
static cudaError_t fork_memcpy_async(void *d, const void *s, size_t bytes, cudaMemcpyKind kind,
                                     cudaStream_t stream = 0) {
  if (fork_memory.in_setup() && fork_memory.in_slab(d)) {
    cudaStreamSynchronize(stream);
    return fork_memcpy(d, s, bytes, kind);
  }
  return cudaMemcpyAsync(d, s, bytes, kind, stream);
}

// From here on seven CUDA names reach the functions above; no other CUDA call is replaced.
// With --default-stream per-thread the CUDA headers already define four of them (the #undef lines)
// as macros for the per-thread variants. The functions above are compiled with those macros in
// effect, so they still call the per-thread variants.
#undef cudaMemset
#undef cudaMemcpy
#undef cudaMemsetAsync
#undef cudaMemcpyAsync
#define cudaMalloc(p, n) fork_alloc_default((p), (n))
#define cudaMallocHost(p, n) fork_malloc_host((p), (n))
#define cudaFree(p) fork_free(p)
#define cudaMemset(...) fork_memset(__VA_ARGS__)
#define cudaMemcpy(...) fork_memcpy(__VA_ARGS__)
#define cudaMemsetAsync(...) fork_memset_async(__VA_ARGS__)
#define cudaMemcpyAsync(...) fork_memcpy_async(__VA_ARGS__)
