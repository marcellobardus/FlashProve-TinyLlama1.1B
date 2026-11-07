// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// How cp_prove sets up the weight table's keepers, chooses GPU memory to lend them and passes them
// the weights in a batch's opening (the test on that table; `close` in identifiers). A keeper
// (`HfKeeperUnit`, prover/hash_commitment/hf_keeper_unit.h; `Opening::unit`; "opening library")
// holds the mask seed and Merkle tree nodes of a leaf table (one masked encoding of the weights).
#pragma once
#include "common/cp_files.h"
#include "frame/lin_frame.h"
#include "prover/prover.h"
// stop() (ends.h) calls each keeper's forget() (see forget_the_opening_library()), but skips the
// current keeper while `InsideTheCloseLibrary` marks one of its calls as in progress, whichever
// thread called stop(): the keeper's header allows forget() only between two of its other calls.
static hf::HfKeeperUnit *g_opening_library_with_secrets[3] = {nullptr, nullptr, nullptr};
static std::atomic<int> g_inside_the_opening_library{0};
static std::atomic<int> g_opening_at_hand{0}; // index of the current keeper, `closer` (`turn_to`)
static bool (*g_wipe_without_the_opening_library)() = nullptr;
struct InsideTheCloseLibrary {
  InsideTheCloseLibrary() { g_inside_the_opening_library++; }
  ~InsideTheCloseLibrary() { g_inside_the_opening_library--; }
};

// Creates the keepers, which parse the parameter file (`policy` in identifiers). A keeper gets a
// leaf table's mask seed and tree nodes with each batch (isk.cuh) and an opening's sizes in
// set_up_the_shared_chain(). It allocates GPU memory here for the life of the process, so call this
// function early. Each batch in progress has its own keeper. Riders are batches whose opening goes
// on while a later batch is proved: at most one with -overlap, two with -overlap-two.
inline int Prover::setup_opening() {
  const double t0 = epoch();
  openings = 1 + (args.overlap ? 1 : 0) + (args.overlap_two ? 1 : 0);
  static_assert(hf::HF_KEEPER_COPIES == 3, "one unit more than batches can be riders: at most three");
  // Input to the hashes of the weight table's tree and to every hash chain's first hash input.
  const blk::Digest configuration = blk::configuration_digest(configuration_files());
  for (int i = 0; i < openings; i++) {
    hf::HfKeeperSetup setup;
    setup.policy = policy.bytes;
    setup.configuration = configuration.w; // the keeper copies it at set-up
    // (A leaf table is identified by its root, which comes with each batch.)
    setup.generator = args.weights == "generator", setup.test_run = !args.test_coins.empty();
    setup.keep_level = isk_keep;
    setup.at_a_failed_guard = blk::at_a_failed_guard; // a guard that fails inside a keeper ends the program as one of cp_prove's own does (ends.h)
    setup.end_name = &g_end_name;                     // and a name that is set inside a keeper reaches this program's last line
    opening[i].unit = hf::hf_keeper_unit_of_copy(i, setup); // each keeper is a separately linked copy: its code keeps state in statics (prover/build.sh, `keeper_unit`)
    if (!opening[i].unit) {
      printf("the opening: the set-up of the weight table's keeper failed; its own line on standard error gives the reason\n");
      return refused("cp-keeper-set-up");
    }
  }
  at_hand = &opening[0], closer = at_hand->unit.get();
  for (int i = 0; i < openings; i++) g_opening_library_with_secrets[i] = opening[i].unit.get();
  g_wipe_without_the_opening_library = g_end_wipe;
  g_end_wipe = [] {
    // A skipped keeper's mask seed and folded row u stay (host memory); it counts as not zeroed.
    // Calls by the thread that fold_begin() starts are not marked: forget() can run during them.
    bool zeroed = true;
    for (int i = 0; i < 3; i++)
      if (hf::HfKeeperUnit *unit = g_opening_library_with_secrets[i])
        if ((g_inside_the_opening_library && i == g_opening_at_hand) || unit->forget() != 0) zeroed = false;
    if (g_wipe_without_the_opening_library && !g_wipe_without_the_opening_library()) zeroed = false;
    return zeroed;
  };
  printf("# the test on the weight table runs in this process, on %d keeper%s (set up in %.1f s)\n", openings, openings > 1 ? "s" : "", epoch() - t0);
  return 0;
}

// Fills `lin_setup` (LIN: the test on the weight table) for all batches and calls each keeper's
// plan(). Call it after chain_make() makes `frame`, before read_the_counts() reads `lin_setup`.
inline int Prover::set_up_the_shared_chain() {
  // From `frame` (blocks/frame/files.h), where a proof's three hash chains share one list of codes:
  // a request's second chain also reads an oracle message of the shared chain (common to a batch's
  // proofs), so a code number must mean the same in both. hf_lin_setup() would number them from 0.
  lin_setup.set = frame->sized().d.set, lin_setup.codes = frame->sized().proof_codes, lin_setup.sized = frame->sized().batch;
  static_assert(sizeof closer->parameters().configuration == sizeof frame->fixed().configuration.w, "a digest has one length");
  if (memcmp(closer->parameters().configuration, frame->fixed().configuration.w, sizeof frame->fixed().configuration.w)) {
    printf("internal error: the weight table's tree starts from another digest than the one that the chains name as their configuration\n");
    name_the_end("chain-config"); // serving: p258 end 1 chain-config (kept as a return 4)
    return 4;
  }
  if (!hf::hf_lin_fits(closer->parameters(), lin_setup.sized.S)) {
    printf("policy: plan-words: the shared chain: the lines describe a test on another table than the committed one\n");
    return refused("plan-words"); // serving: p258 end 1 plan-words (the printed category)
  }
  lin_setup.shown = hf::hf_shown(closer->parameters(), lin_setup.sized.S.fold.columns);
  lin_setup.plan = frame->proof().chains.at(1).plan;
  // plan() sets a keeper's memory sizes from the opened columns and the folded row; until then, 0.
  const blk::OracleShape &folded = blk::lin_message_u(lin_setup.sized.S).at(0).shape;
  const size_t folded_row_bytes = (size_t)folded.main_rows * lin_setup.codes.base[folded.code].ell * sizeof(blk::E);
  for (int i = 0; i < openings; i++)
    if (opening[i].unit->plan(lin_setup.shown, folded_row_bytes)) {
      printf("the opening: the weight table's keeper refuses the plan's numbers; its own line on standard error gives the reason\n");
      return refused("open-plan"); // serving: p258 end 1 open-plan
    }
  return 0;
}

// Makes o the current opening (`at_hand`). Only slab_for_the_library()'s memory may be lent then.
inline void Prover::turn_to(Opening &o) {
  if (at_hand == &o) return;
  if (!lent_at_the_close.empty()) {
    printf("internal error: the opening: memory is still lent to one keeper when another is turned to\n");
    exit(4);
  }
  at_hand = &o;
  closer = o.unit.get();
  g_opening_at_hand = (int)(&o - opening);
}

// Writes the tensor numbers (`stream` in prover/weights_blocks/block_weights.h) of a non-routed
// kind of weight and returns their count: 2 for kind 0 (q and k), else 1. The kind is not checked.
static inline int tensors_of_kind(int kind, int tensors[2]) {
  int n = 1;
  tensors[1] = 0;
  switch (kind) {
  case 0:
    n = 2, tensors[0] = 2, tensors[1] = 3;
    break;
  case 1:
    tensors[0] = 4;
    break;
  case 2:
    tensors[0] = 5;
    break;
  case 3:
  case 5:
    tensors[0] = 6;
    break;
  case 4:
  case 6:
    tensors[0] = 7;
    break;
  case 9:
    tensors[0] = 8;
    break;
  default:
    tensors[0] = 9;
  }
  return n;
}

// Bytes of one of the two byte planes (low, high) decoded from a tensor stored as quantized blocks.
static inline size_t plane_bytes(const wb::Shape &shape, int tensor, int layer) {
  return (shape.rows(tensor, layer) * (size_t)shape.inputs(tensor, layer) + 255) & ~(size_t)255;
}

// Passes units of layer group g (shape.Lam consecutive layers) to the current keeper (`closer`); a
// unit is one kind of weight of one layer. routed_of: -2: all units, -1: those not routed, else
// that layer's routed units (kind 7, 8 or 10: routed experts' weights). Call lend_the_walk_room()
// first (otherwise the keeper's allocator ends the process) and load the units first (not checked).
// Public source: keeping expert weights group-quantized, unpacked in-kernel, and the attention and
// other layers in bfloat16, is described for gpt-oss in vLLM's blog, "vLLM Now Supports gpt-oss",
// section "MXFP4 MoE", https://vllm.ai/blog/gpt-oss. Here the keeper decodes a routed unit's
// blocks, the model file's, into planes in a work buffer before the product; the other units are
// passed as planes of signed bytes, decoded into scratch memory first if stored only as blocks.
inline int Prover::fold_group_for_opening(int g, const std::vector<int> &unit_kind, const std::vector<int> &unit_layer,
                                          int routed_of) {
  InsideTheCloseLibrary inside;
  cudaStream_t s = cudaStreamPerThread;
  const wb::Shape &shape = block_weights->shape;
  for (size_t i = 0; i < unit_kind.size(); i++) {
    const int kind = unit_kind[i], layer = unit_layer[i];
    // A kind above 11 (embedding, head) is in no layer group: fold_end_for_opening() passes it.
    if (kind > 11 || layer / shape.Lam != g) continue;
    if (routed_of != -2 && (kind == 7 || kind == 8 || kind == 10 ? layer != routed_of : routed_of != -1)) continue;
    if (kind == 7 || kind == 8 || kind == 10) {
      const wb::Blocks *b = block_weights->blocks(kind == 7 ? 10 : kind == 8 ? 11 : 12, layer);
      if (!b) {
        printf("internal error: the opening: a routed tensor of layer %d is not stored as blocks\n", layer);
        return 4;
      }
      const uint8_t *true_bytes = b->bytes;
      int exchanged_a = -1, exchanged_b = -1;
      // Behind `lie::experts()`, a stub of a test hook that returns false: this block never runs.
      if (lie::experts(kind == 7 ? 10 : kind == 8 ? 11 : 12, layer, exchanged_a, exchanged_b, s, true)) {
        const int st = kind == 7 ? 10 : kind == 8 ? 11 : 12;
        const size_t one = shape.rows(st, layer) * (size_t)(shape.inputs(st, layer) / 256) * wb::block_bytes(b->type), all = one * shape.NE;
        static uint8_t *copy = nullptr;
        static size_t copy_bytes = 0;
        if (all > copy_bytes) {
          if (copy) CK(cudaFree(copy));
          CK(cudaMalloc(&copy, all));
          copy_bytes = all;
        }
        CK(cudaMemcpyAsync(copy, b->bytes, all, cudaMemcpyDeviceToDevice, s));
        CK(cudaMemcpyAsync(copy + exchanged_a * one, b->bytes + exchanged_b * one, one, cudaMemcpyDeviceToDevice, s));
        CK(cudaMemcpyAsync(copy + exchanged_b * one, b->bytes + exchanged_a * one, one, cudaMemcpyDeviceToDevice, s));
        true_bytes = copy;
      }
      if (int r = closer->add_blocks(kind, layer, true_bytes, b->type, b->gain, b->scale, s)) {
        printf("the opening: the opening library rejected the routed tensor of kind %d, layer %d (library code %d)\n",
               kind, layer, r);
        return 2;
      }
      continue;
    }
    int streams[2];
    const int n = tensors_of_kind(kind, streams);
    const signed char *w0[2], *w1[2];
    size_t pitch[2];
    // No sub-proof runs now (with -overlap, the_rider_folds() in service.cuh makes this call
    // between two turns of a later batch), so the scratch memory below walk_room() is free.
    char *const scratch_first = slab_without_it().first;
    char *scratch = scratch_first;
    const size_t free_bytes = (size_t)(walk_room().first - scratch_first);
    for (int q = 0; q < n; q++) {
      pitch[q] = (size_t)shape.inputs(streams[q], layer);
      // A test hook (prover/common/lie.cuh) to read another tensor; here it changes nothing.
      int from_stream = streams[q], from_layer = layer;
      lie::place(from_stream, from_layer, s, true);
      if (const wb::Planes *p = block_weights->tensor_planes(from_stream, from_layer)) {
        w0[q] = p->w0, w1[q] = p->w1;
        continue;
      }
      const size_t plane = plane_bytes(shape, streams[q], layer);
      if (scratch + 2 * plane > scratch_first + free_bytes) {
        printf("internal error: the opening: %zu bytes of the sub-proofs' slab are free, too few for the decoded "
               "weights of kind %d, layer %d\n",
               free_bytes, kind, layer);
        return 4;
      }
      // The keeper treats these bytes as weights. A byte that a faulty GPU fails to write must not
      // hold the previous unit or, with -overlap, what a request of a later batch left here.
      CK(cudaMemsetAsync(scratch, 0, 2 * plane, s));
      block_weights->write_planes(from_stream, from_layer, 0, (i8 *)scratch, (i8 *)scratch + plane, s);
      w0[q] = (const signed char *)scratch, w1[q] = (const signed char *)scratch + plane;
      scratch += 2 * plane;
    }
    // The scratch memory holds one unit's planes at a time: write_planes() and this call, which
    // reads them, run in order on s.
    if (int r = closer->add_planes(kind, layer, n, w0, w1, pitch, s)) {
      printf("the opening: the opening library rejected the weights of kind %d, layer %d (library code %d)\n", kind,
             layer, r);
      return 2;
    }
  }
  return 0;
}

// The memory that a keeper keeps throughout an opening (its `slab`): the upper end of the
// sub-proofs' scratch memory (their "slab"), or {nullptr, 0} if that is too small. With -overlap: a
// buffer from hold_the_close_to_its_room(), because the next batch's sub-proofs run meanwhile.
// In identifiers `fork` is a sub-proof.
inline std::pair<char *, size_t> Prover::slab_for_the_library() const {
  if (args.overlap) return at_hand->slab;
  const auto whole = fork_memory.whole_slab();
  InsideTheCloseLibrary inside;
  const size_t need = (closer->slab_bytes() + 511) & ~(size_t)511;
  if (need > whole.second) return {nullptr, 0};
  char *first = (char *)((uintptr_t)(whole.first + whole.second - need) & ~(uintptr_t)511);
  return {first, (size_t)(whole.first + whole.second - first)};
}
// The rest of the sub-proofs' scratch memory; with -overlap, all of it.
inline std::pair<char *, size_t> Prover::slab_without_it() const {
  const auto whole = fork_memory.whole_slab();
  if (args.overlap) return whole;
  const auto top = slab_for_the_library();
  return {whole.first, top.first ? (size_t)(top.first - whole.first) : whole.second};
}

// The walk room: the memory in which a keeper works during fold_begin() and its add_* calls (a walk
// is one pass of an opening over all the weights). Nothing in it lives from one call to the next,
// so it may be taken back and lent again between any two calls: the_rider_folds() relies on that.
// It is the upper end of slab_without_it(); units are decoded below it. {nullptr, 0}: no room.
inline std::pair<char *, size_t> Prover::walk_room() const {
  const auto below = slab_without_it();
  InsideTheCloseLibrary inside;
  const size_t need = (closer->walk_room_bytes() + 511) & ~(size_t)511;
  if (need + 512 > below.second) return {nullptr, 0};
  char *first = (char *)((uintptr_t)(below.first + below.second - need) & ~(uintptr_t)511);
  return {first, (size_t)(below.first + below.second - first) & ~(size_t)511};
}

// The largest number of bytes into which fold_group_for_opening() decodes one unit's weights.
inline size_t Prover::decoding_needs_at_most() {
  if (!args.stream_weights) return 0;
  list_the_units();
  size_t most = 0;
  for (size_t i = 0; i < unit_kind.size(); i++) {
    const int kind = unit_kind[i], layer = unit_layer[i];
    if (kind > 11 || kind == 7 || kind == 8 || kind == 10) continue;
    int streams[2];
    const int n = tensors_of_kind(kind, streams);
    size_t bytes = 0;
    for (int q = 0; q < n; q++)
      if (!block_weights->tensor_planes(streams[q], layer)) bytes += 2 * plane_bytes(block_weights->shape, streams[q], layer);
    most = std::max(most, bytes);
  }
  return most;
}

// Passes the embedding's cells (which == 0) or the head's weights (which == 1) to the current
// keeper. Returns 0, or 2 if the keeper rejected them. Preconditions: see fold_group_for_opening().
inline int Prover::fold_end_for_opening(int which) {
  InsideTheCloseLibrary inside;
  const int r =
      which ? closer->add_head(block_weights->head(), cudaStreamPerThread)
            : closer->add_embedding((const signed char *)block_weights->embedding_in_place(), cudaStreamPerThread);
  if (r)
    printf("the opening: the opening library rejected the %s (library code %d)\n", which ? "head" : "embedding", r);
  return r ? 2 : 0;
}

// Call it after the process's last opening. If a keeper could not zero its part of the GPU memory,
// a run that otherwise succeeds exits with End::NotZeroed. A second call does nothing.
// forget() zeroes the mask seed and its second copy (from which self-checks derive masks, salts and
// pads again), the folded row u and, if CUDA responds, the GPU memory lent to the keeper. Among
// what stays: GPU memory allocated at set-up and the stored tree nodes.
// The nodes are digests of the table's Merkle tree, which hashes each column with a salt; they come
// from a file in the secret key directory (`isk`), and an opening's Merkle paths publish some.
// In stop(), ForkMemory::wipe() then zeroes the memory lent to keepers, also only if CUDA responds.
inline void Prover::forget_the_opening_library() {
  for (hf::HfKeeperUnit *&library : g_opening_library_with_secrets) {
    if (library && library->forget() != 0) opening_library_forgot_all = false;
    library = nullptr;
  }
}
