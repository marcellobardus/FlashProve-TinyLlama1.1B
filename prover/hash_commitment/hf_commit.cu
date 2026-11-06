// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// cp_commit, the committer, on the GPU. From the weights, the configuration (policy.txt, KL.bin,
// layout.txt), the number of leaf tables (-M; `copy` in identifiers) and one seed it generates a
// commitment: `ivk`, the public key directory, and `isk`, the secret one. The command line and its
// usage text are in prover/weights_commitment/arguments.h. Nothing is proved about these programs.
//
// A weight table T is all the weights as a matrix of long rows, each with random masks, encoded,
// under one Merkle tree. The anchor (`first_table` in identifiers) is a commitment's first weight
// table. Leaf table k (not a tree leaf) is a further table of the same weights with its own masks.
// The parameter file is policy.txt. The key file, ivk/key.bin, has the anchor's root and the
// committer's one-time proofs about it. The leaf file ivk/leaf/<k>.bin has leaf table k's root and
// its LINK proof, meant to show that the table holds the anchor's weights. Identifiers and messages
// say "entry" for a leaf file, "link" for a LINK proof, "coins" for private randomness, "draw" for
// sample, "make" for generate, "walk" for a pass over the weights, "judge" for the verifier program
// cp_verify, "refuse" for reject, "asked" for requested, `made_up` for synthetic, `_door` for test
// code or hook, "folder" for directory, "serving build" for production build, `published_id` for
// commitment ID. A tag in square brackets is the name that a check's message prints; only
// [keep-for-the-audit] is printed by none: it marks the code for that option.
// Three phases of work:
//   A  A table's Merkle tree (prover/hash_commitment/hf_commit_table.cuh). In comments M is W, the
//      matrix of all weights, as R long rows. Each row gets its masks from the table's key and is
//      encoded to length 2^fln. A column of T, one symbol of each row, is one tree leaf. T is never
//      stored. The tree's levels from s (-keep) up stay in memory, and level s is written as
//      hcnodes.bin. A root depends only on the weights, the table's key and the configuration.
//      A table has a record: stored answers to samples of its rows. Phase A ends with one more
//      pass over the weights, which computes the answers (make_the_record()): the sample points are
//      derived from the root. The anchor's answers go into the key file, a leaf table's into its
//      leaf file.
//   B  The key file (prover/hash_commitment/hf_commit_once.cuh): the range proof on all cells (M's
//      entries, signed bytes) and weights, and the one-time test on the anchor.
//   L  A leaf file (prover/hash_commitment/hf_entry_make.cuh): the leaf table's LINK proof.
// B and L run the library's code: blk::make_key() (blocks/frame/check_key.h), blk::make_entry()
// (blocks/frame/check_entry.h). Both take a blk::Configured, which the verifier's own code builds
// before a weight is read (verifier/src/cp_configured.h). Formats: docs/format/FORMAT.md.
//
// The first form of the call (with -weights) generates a commitment. It loads the weights once,
// into page-locked host memory, and checks them: the checksums of wsums.txt, the declared ranges,
// the worst case (below). Only then does it sample the seed or, with -public-weights, derive it: a
// run that fails a check has sampled nothing. It works in <isk>.making and <ivk>.making
// (prover/hash_commitment/hf_commit_making.h), renamed at the end; a run that fails or is ended by
// SIGABRT, SIGINT, SIGTERM, SIGHUP or SIGPIPE removes them name by name: only the names that it
// created itself. If somebody else has put a file of another name into one, that file and the
// directory stay, and a message says so. After SIGKILL, any other signal, a crash or a power
// failure both directories stay, and no call removes them: the same call, repeated, is rejected
// and says what is left. In order:
//   the anchor         phases A and B; "cp_verify -key" on the key file's bytes in memory; only
//                      then key.bin, isk.txt and isk/anchor/hcnodes.bin
//   each leaf table k  phases A and L; "cp_verify -leaf" on the leaf file's bytes in memory; only
//                      then isk/leaf/<k>/ and ivk/leaf/<k>.bin
// Each configuration file is read once; the parsed bytes go to the worst-case check, the digest
// h_par (blk::configuration_digest()), isk/config and cp_verify. Later changes under -config have no effect.
//
// What wears out. Every opening of a weight table (the one-time test on the anchor, a LINK proof on
// both its tables, a batch of requests on its leaf table) sends some of the table's columns, and a
// row's masks are to hide only a limited number of opened columns (blocks/frame/configure.h,
// table_columns()). A table's count.bin counts them by its length, from an opening's start;
// cp_commit and the prover program cp_prove begin no opening that would take the count beyond the
// limit. The count is that directory's alone: a copy or a restored backup of isk has its own, which
// no program can detect (docs/operators-duties.md, "Backups and restored copies").
//
// The second form of the call (-leaves, -M or -first on an existing commitment: more leaf tables
// under the same anchor, or the oldest retired) is rejected [cp-commit-first-form-alone]: its code
// below reads a record from key.bin that the key file does not contain. rf::HfRecord, h_0,
// h_card and the check [own-test-hidden] belong to that record and enter no file or key. It is not
// phase A's record (blk::IndexRecord), which the second form leaves empty. Its rules:
//   - No one is to see two LINK proofs for one k. Each has its own private randomness, so none is
//     reused; but each opens its own columns of both tables and raises both counts. An attempt that
//     cp_verify rejects is written to no file; an accepted leaf file is stored in isk/leaf/<k>/
//     before it is published, and from then on that table is never generated again.
//   - A public stop (it depends on hash output alone: a failed nonce search, say) in a LINK proof
//     spends its number k for good: nothing of the attempt is released, and k is not tried again,
//     although a new attempt would have new private randomness. The rule is the operator's: no
//     program marks k. What a LINK proof reveals after stopped attempts has not been examined. The
//     operator sets -M k, and the next leaf tables begin after it (-first k + 1 with a larger -M).
//     In the first form, the only one accepted, a public stop while the key file or a leaf
//     file is generated voids the whole run for good: cp_commit ends with status 4 or by abort(),
//     and its cleanup removes both working directories, unless SIGKILL, any other signal that is
//     not caught, a crash or a power failure comes first. Nothing of the run is kept, retried or
//     published; the operator starts a new commitment, which samples a new seed (not with
//     -public-weights). This rule too is the operator's: beyond removing the files, no program
//     marks a void run.
//   - isk.txt is rewritten only at the end, and only if every leaf table from `first` to `M` - 1
//     (the live ones) exists, so cp_prove finds no isk.txt that names a live leaf table that is
//     missing; it would reject such an isk [cp-isk-leaves].
//   - cp_commit deletes nothing of a retired leaf table; cp_verify accepts its proofs forever. The
//     operator may remove its directory in isk once no cp_prove runs that was started under an
//     older isk.txt.
// The keys (derivations: prover/common/coin_keys.h; calls: prover/hash_commitment/hf_commit_keys.h;
// the rule: docs/format/COIN-KEYS.md). A table's masks, its tree leaves' salts and its rows' pads
// come from its making key; the private randomness in the key file and in each leaf
// file, from a process key. Hiding rests on this: each table's key depends on bytes that are
// sampled for that generation of the table (the seed for the anchor, 32 bytes from the operating
// system for a leaf table). Two tables under one key would differ by codewords of the difference of
// their data, with no mask left. Only Make::a_copy and Make::the_first_table sample these bytes and
// can construct the MakersKey that make_the_table() and make_the_record() take. A process key takes
// 32 system bytes that are stored nowhere, except under -keep-for-the-audit. The public values that
// a key also hashes only guard against a system generator that repeats; no making key reads
// anything of the weights. So under a sampled seed only the anchor and its root are functions of
// (weights, configuration, seed) alone. Under -test-seed the system's bytes are zeros and every
// file repeats: nothing is hidden then. The same holds with -public-weights
// (prover/common/public_weights.h; production builds only), where the seed is derived from the
// option's file: whoever has that file can compute every mask. All keys are derived again by a
// second implementation (prover/hash_commitment/hf_commit_derivation_check.cuh): the process keys are
// compared with its keys before a random value is sampled; every mask (only the first time that a
// table's generation computes it) and the salts of the tree leaves that an opening regenerates,
// with what it derives under its making keys, before a file's bytes leave.
//
// Exit status: 0 done; 1 the verifier's code rejects the configuration ("NO: <name>"); 2 rejected,
// or faulty input; 3 a tensor cannot be converted to integers; 4 an internal error, a stop while a
// file is generated, a leaf table's files gone at the end, or cp_verify's rejection of what this
// run generated; 5 a GPU fault; 6 the integers are not those of wsums.txt; 7 the worst-case check
// (below): not every bounded site fits; 16 a committed integer is out of range; 134 in a shell:
// abort(), which at a failed check in blocks/ or prover/engine/ (BLK_REQUIRE(), ENGINE_CK())
// follows a line "cp_commit: internal error (<name>): ..".
// Without -test-seed the log has times, public numbers, digests of public files and facts about the weights: whether
// the checks passed (if not, why); with wsums.txt, how many tensors have its sums; whether the embedding is wider than
// declared. It also has numbers that follow how the model files store their tensors, such as the host memory taken for them. The worst-case table, computed from the weights, goes only to -worst-case-table's file.
//
// The worst-case check. Whether a request's values leave a range depends on the weights, and such a
// request is answered "failed": on unchecked weights "failed" would be a bit about the weights at a
// prompt of the client's choice. When every bounded site (a range test of the forward pass) fits
// (what that shows and does not: prover/weights_commitment/worst_case.cuh), isk.txt gets the line
// "ranges <h_par>", which cp_prove requires. The check is only this hiding safeguard; no verifier
// check rests on it. A call under which nothing is hidden may leave it out, and then writes no line
// ranges: a call with -test-seed, which no production build has (isk.txt then says "seed test",
// accepted by test builds only); or, in a production build, a call with -public-weights (isk.txt
// then says "seed public", and cp_prove given -public-weights starts without the line:
// Prover::the_ranges_are_guarded()). -worst-case or -worst-case-table forces the check in either
// case, and a pass then writes the line as usual.
#include "hash_commitment/hf_commit_derivation_check.cuh" // first: it defines the macro through which the shared headers call it
#include "common/setting.h"
#include "weights_commitment/lent_memory.h" // before every header that allocates GPU memory
#include "weights_commitment/common.cuh"
#include "weights_commitment/arguments.h"
#include "weights_commitment/fold_source.cuh"
#include "weights_commitment/worst_case.cuh"
#include "weights_commitment/check_commitment.cuh"
#include "weights_commitment/cert_doors.h"
#include "common/cp_files.h"
#include "common/test_words.h"
#include "common/made_up_alone.h" // may test code print or write the weights' values in this run?
#include "common/reduced_env_start.h"
#include "common/parameters_hash.h"
#include "common/public_weights.h"
#include "hash_commitment/hf_format.h"
#include "hash_commitment/hf_seed.h"
#include "hash_commitment/hf_mask.cuh"
#include "hash_commitment/hf_mask_own.cuh"
#include "hash_commitment/hf_shape.cuh"
#include "hash_commitment/hf_record.h"
#include "hash_commitment/hf_commit_api.h"
#include "hash_commitment/hf_commit_walk.h"
#include "hash_commitment/hf_commit_doors.h"
#include "hash_commitment/hf_commit_weights.cuh"
#include "hash_commitment/hf_commit_making.h"
#ifdef SERVING_BUILD
#include "hash_commitment/public_weights_markers.h" // the marker files of -public-weights
#endif
#include "hash_commitment/hf_commit_cells.cuh"
#include "hash_commitment/hf_commit_whole.cuh"
#include "hash_commitment/hf_commit_windows.cuh"
#include "hash_commitment/hf_commit_table.cuh"
#include "hash_commitment/hf_commit_once.cuh"
#include "hash_commitment/hf_entry_make.cuh"
#include "cp_configured.h" // verifier/src: the verifier's code that builds a blk::Configured
#include "frame/heads.h" // blocks/frame: the configuration digest
#include "iop/foldeq.h"  // blocks/iop: a weight table's name
#ifdef HELD_LOOK // a switch for a test of hiding; no target of prover/build.sh sets it
#ifndef BLK_TEST_PROVER_MESSAGES_OUTSIDE
#error "HELD_LOOK without the blocks' switch BLK_TEST_PROVER_MESSAGES_OUTSIDE, whose mark says that the program is not for serving"
#endif
#include "iop/message_dump.h" // blocks/iop
#endif
#include "common/coin_keys.h" // the schedule of the coins' keys (hf_commit_keys.h)
#include "common/cp_keys_own.h" // self-test of the key derivation from a seed
#include "hiding/coins_own.cuh" // and of a generator of private randomness

namespace hf {
using rf::D;
using rf::Opt;
using rf::PolC;
using std::vector;
static std::chrono::steady_clock::time_point now() {
  return std::chrono::steady_clock::now();
}
static double seconds_since(std::chrono::steady_clock::time_point then) {
  return std::chrono::duration<double>(now() - then).count();
}
// Prints a phase's wall time, always in the same format, so that a script can read it from the log.
static void phase_line(const std::string &name, std::chrono::steady_clock::time_point begin) {
  printf("cp_commit: phase %s: %.1f s\n", name.c_str(), seconds_since(begin));
}

// ---- the seed and the keys from it ----
// Keeps a key's memory out of swap space. If mlock() fails, the run continues after a warning.
static void lock_in_memory(const void *at, size_t bytes) {
  if (mlock(at, bytes))
    fprintf(stderr, "cp_commit: warning: a key's memory cannot be locked (%s): the system may write it to its swap space. Raise the limit on locked memory (ulimit -l)\n",
            strerror(errno));
}
// This program's copy of the seed in memory (hf_commit_derivation_check.cuh keeps a second one for its
// comparisons) and the seed file's current path. Never destroyed, so the exit handler can wipe it.
struct Seed : hf_key32 {
  std::string file;
  Seed() : hf_key32{} { lock_in_memory(w, sizeof w); }
  void bytes(uint8_t out[32]) const { memcpy(out, w, 32); }
};
static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "a key's 32 bytes are its eight words, the lowest byte first");
static Seed &the_seed() {
  static Seed &s = *new Seed;
  return s;
}
// The tables' keys with the functions that derive them; the key and leaf files' process keys.
#include "hash_commitment/hf_commit_keys.h"
static TableKey &anchor_key() {
  static TableKey &k = *new TableKey;
  return k;
}
static TableKey &leaf_key() {
  static TableKey &k = *new TableKey;
  return k;
}
static KeptBytes &leaf_fresh() { // fresh bytes of the leaf table being generated (hf_commit_keys.h)
  static KeptBytes &f = *new KeptBytes;
  return f;
}
// Called after the last leaf table, at the program's end and by the exit handler, making::leave().
// Not at a signal that ends the run or at abort(): the process then dies with the keys in memory.
static void wipe_the_keys() {
  seed_wipe(the_seed());
  anchor_key().wipe();
  leaf_key().wipe();
  leaf_fresh().wipe();
  RunKeys::wipe();
  ::hz::key32_wipe();
  ::derivation_check::wipe();
}
// True if the randomness generator has these 32 bytes as its process key and they are not all zero:
// under a zero key anyone could compute every random value. It compares only a 32-bit word derived
// from each key (prover/hiding/trees.cuh, key32_word()): another key passes about once in 2^32.
static bool the_coins_key_is(const uint8_t key[32]) {
  uint8_t any = 0;
  for (int i = 0; i < 32; i++) any |= key[i];
  return any && ::hz::key32_word() == ::hz::key32_word_of(key);
}
// [own-test-key] Exits with status 4 unless the_coins_key_is() accepts the process key that was
// just installed (cp_prove compares the same word: prover/prover/isk.cuh). Only a bug can fail it.
static void hold_the_coins_key(const uint8_t key[32]) {
  if (the_coins_key_is(key)) return;
  fprintf(stderr, "cp_commit: internal error (own-test-key): the coins' state does not hold the process key that was derived, "
                  "or that key is zero. This is a bug in the program: report this line\n");
  exit(4);
}
// The first form's seed: 32 bytes from the operating system's generator (seed_draw()), or derived
// from -public-weights' file, or (test builds) -test-seed's file. Returns 0, or 2 with a message.
static int take_the_seed(const Opt &O) {
  Seed &seed = the_seed();
#ifdef SERVING_BUILD
  if (O.public_seed) {
    uint8_t derived[32];
    if (!::public_weights::seed_of_the_file(O.public_seed, derived))
      return making::refuse("usage", std::string("-public-weights ") + O.public_seed + " is not a readable file of 32 bytes");
    memcpy(seed.w, derived, 32);
    explicit_bzero(derived, sizeof derived);
    return 0;
  }
#else
  if (O.test_seed) {
    std::string bytes;
    if (!::cp::read_file(O.test_seed, bytes) || bytes.size() != 32)
      return making::refuse("usage", std::string("-test-seed ") + O.test_seed + " is not a file of 32 bytes");
    memcpy(seed.w, bytes.data(), 32);
    return 0;
  }
#endif
  (void)O;
  const SeedEnd end = seed_draw(seed_machine_source, seed);
  return end == SeedEnd::TAKEN ? 0 : making::refuse(seed_stop(end), seed_says(end));
}
// The second form reads the seed that the first form stored. Returns 0 or 2.
static int read_the_seed(const Opt &O, const std::string &path) {
  Seed &seed = the_seed();
  seed.file = path;
  if (!seed_from_file(path.c_str(), seed) || seed_is_zero(seed)) { // checked on the open file
    seed_wipe(seed);
    return making::refuse("hc-key-file", path + " is not there, or is not a regular file of 32 bytes, not all zero, this account's own, under one name, with "
                                                "permissions for its owner only");
  }
  // [hc-key-test], as in cp_prove (prover/prover/isk.cuh): a seed in the source is for tests only.
  if (seed_is_of_the_tests(seed) && SERVING_OR(true, !O.test_seed)) {
    seed_wipe(seed);
    return making::refuse(HF_STOP_KEY_TEST, path + " holds a seed that is written in the source, which everybody can read: make a new commitment" SERVING_OR("", "; a test's call takes it with -test-seed"));
  }
#ifndef SERVING_BUILD
  // With -test-seed no fresh bytes enter a leaf table's or a LINK proof's key, and test code is
  // enabled. The file must contain isk's seed, so only a holder of the seed can ask for that, and
  // cp_commit_main() has required isk.txt's line "seed test". Nothing such a call makes is hidden.
  if (O.test_seed) {
    std::string bytes;
    const bool same = ::cp::read_file(O.test_seed, bytes) && bytes.size() == 32 && !memcmp(bytes.data(), seed.w, 32);
    if (!bytes.empty()) explicit_bzero(&bytes[0], bytes.size());
    if (!same) {
      seed_wipe(seed);
      return making::refuse("cp-commit-test-seed", std::string("-test-seed ") + O.test_seed + " does not hold the 32 bytes of " + path);
    }
  }
#endif
  (void)O;
  return 0;
}

// ---- phase A: the parts ----
static bool read_whole_file(const std::string &path, void *bytes, size_t n) { // exactly n bytes, or false
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) return false;
  const bool whole = fread(bytes, 1, n, f) == n && fgetc(f) == EOF;
  fclose(f);
  return whole;
}
struct Plan {
  TablePlan table;       // a part: 2^lp positions of T; np parts at once; s: lowest stored level
  bool whole = false;    // M is in GPU memory; else (at full size) it is read in cuts, runs of rows
  size_t free_bytes = 0; // the GPU's free memory, or -gpu-memory if that is less
};
// Fills plan with what the options leave open. s: the tree's lowest stored level. Call it after
// `weights.load`. Returns 0 or 2.
static int make_plan(const Opt &O, const rf::HfShape &shape, const PinnedWeights &weights, int s, Plan &plan) {
  const HfParams &P = shape.P;
  size_t free_bytes = 0, total = 0;
  CK(cudaMemGetInfo(&free_bytes, &total));
  if (O.gpu_memory) free_bytes = std::min(free_bytes, O.gpu_memory); // -gpu-memory
  plan.free_bytes = free_bytes;
  plan.table.s = s;
  // Without -rows-in-cuts M stays in GPU memory if it takes at most a third of the free memory and
  // 1 GiB and 3 x panel_bytes() also fit. A unit is W's rows for one tensor, or for a layer's query
  // and key; the room is for the largest unit's cells (its panel), its integers, its packed bytes.
  const size_t margin = (size_t)1 << 30;
  const size_t whole = WholeCells::bytes_of(P), making_it = 3 * WholeCells::panel_bytes(shape.Y);
  plan.whole = !O.rows_in_cuts && whole <= free_bytes / 3 && whole + making_it + margin <= free_bytes;
  // Reading in cuts returns slabs of 16 whole rows (hf_commit_windows.cuh), so M needs at least 16.
  if (!plan.whole && P.R < 16) {
    fprintf(stderr, "cp_commit: refused (rows-in-cuts): the committed matrix has %zu rows, and the reading in cuts of the stored "
                    "tensors takes 16 at a time: %s\n",
            P.R, O.rows_in_cuts ? "leave -rows-in-cuts out" : "the matrix does not fit into the free GPU memory whole either");
    return 2;
  }
#ifndef SERVING_BUILD // a production build is to contain no test code, not even its messages
  // A test's injected integer (hf_commit_doors.h) is accepted only where M is whole in GPU memory.
  // With cuts it is rejected, though k_slab_cells() (hf_commit_windows.cuh) would inject it too.
  if (!plan.whole)
    for (size_t unit = 0; unit < shape.Y.U.size(); unit++)
      if (weights.planted.acts_on(unit)) {
        fprintf(stderr, "cp_commit: FORMF_PLANTED_INTEGER acts only where all of the committed matrix lies in GPU memory: this "
                        "shape is read in cuts of the stored tensors\n");
        return 2;
      }
#endif
  const size_t cells = plan.whole ? whole : window_cells_bytes(shape, weights);
  if (cells + margin >= free_bytes) {
    fprintf(stderr, "cp_commit: refused (gpu-memory): %.1f GB are free on a GPU, too little for the rows of the committed "
                    "matrix: see whether another process holds the GPU\n",
            free_bytes / 1e9);
    return 2;
  }
  const int least = std::max(s, 1);
  // With cuts read from host memory, two parts built side by side halve the reading.
  plan.table.np = O.parts_at_once ? O.parts_at_once : plan.whole ? 1 : 2;
  plan.table.lp =
      O.part_bits >= 0 ? O.part_bits : part_bits_that_fit(P, s, plan.table.np, free_bytes - cells - margin);
  if (plan.table.lp < least || plan.table.lp > P.ln || plan.table.lp > 30) {
    fprintf(stderr, "cp_commit: parts of 2^%d positions (%s): a part has at least 2^%d positions (the kept level, and at least 2), "
                    "and at most 2^%d\n",
            plan.table.lp, O.part_bits >= 0 ? "-part-bits" : "what fits the free GPU memory", least, std::min(P.ln, 30));
    return 2;
  }
  if (part_bytes(P, plan.table.lp, plan.table.np) + cells + margin > free_bytes) {
    fprintf(stderr, "cp_commit: refused (gpu-memory): %.1f GB are free on a GPU, too little for parts of 2^%d positions: "
                    "name a smaller -part-bits or -parts-at-once or leave them out, or see whether another process holds "
                    "the GPU\n",
            free_bytes / 1e9, plan.table.lp);
    return 2;
  }
  return 0;
}
// The source of M's rows, for make_table(). It keeps references to shape, plan and weights.
static std::function<RowCells *(int)> cells_maker(const PolC &PO, const rf::HfShape &shape, const Plan &plan,
                                                  PinnedWeights &weights) {
  const long long H = (long long)PO["H"], G = (long long)PO["G"];
  return [&shape, &plan, &weights, H, G](int device) -> RowCells * {
    near_host_pin(device);
    if (plan.whole) return new WholeCells(shape.P, shape.Y, shape.um, H, G, (long long)shape.sh.dh, weights);
    return window_cells(shape, weights, H, G);
  };
}
// Builds a table's tree in heap (HfKept's layout). Exits unless [own-test-masks] has passed.
static void make_the_table(const PolC &PO, const rf::HfShape &shape, Plan &plan, PinnedWeights &weights, const MakersKey &makers,
                           u32 *heap) {
  const TableKey &key = makers.key;
  printf("cp_commit: %llu parts of 2^%d positions; the rows of the committed matrix %s; nodes kept from level %d\n",
         (unsigned long long)1 << (shape.P.ln - plan.table.lp), plan.table.lp,
         plan.whole ? "lie whole in the GPU's memory" : "are read in cuts of the stored tensors from host memory",
         plan.table.s);
  plan.table.key_again = key.again;
  if (!make_table(shape.P, plan.table, &key, cells_maker(PO, shape, plan, weights), heap)) {
    fprintf(stderr, "cp_commit: internal error (%s): a table was made without the masks' own test\n", own_stop(OWN_MASKS));
    exit(4);
  }
  ::derivation_check::the_table_ends();
}
// Computes the record of the table that make_the_table() has just built under this key. The pass
// over the weights comes after the tree because the points at which the rows are sampled are
// derived from the root. One set of pads must serve one record and no more (hf_side_record() in prover/hash_commitment/hf_open_side.cuh).
// So this function takes a MakersKey, which only `Make` can construct
// (prover/hash_commitment/hf_commit_keys.h), and a table that is regenerated from its stored secret
// gets no second record.
// table: 0 for the anchor, k + 1 for leaf table k (for prover/hash_commitment/commit_lie.cuh).
// root: the table's, DW words. Returns 0, 2 (too little GPU memory) or 4, after a line on stderr.
// If the pads fail their self-check or an internal check fails, the process exits with code 4.
// GPU memory comes from the
// driver; buffers with values derived from the weights or the pads are zeroed before they are
// freed.
static int make_the_record(const Opt &O, const PolC &PO, const rf::HfShape &shape, PinnedWeights &weights, const MakersKey &makers, int table, const blk::IndexShape &table_shape,
                           const u32 *root, blk::IndexRecord &record) {
  using namespace rf;
  const HfParams &P = shape.P;
  const TableKey &key = makers.key;
  record = blk::IndexRecord();
  if (P.m0 != shape.Y.m0 || P.k != shape.Y.k) hf_once_internal("the table's record: the parameters and the layout do not fit each other");
  if (live_flags()) hf_once_internal("the table's record: a failure flag is set that belongs to no part of it");
  const Committed committed{O, PO, shape, weights, &key, key.again, HfKept(), nullptr, nullptr, nullptr, nullptr}; // the panels read O, PO, shape and weights
  weights.release();
  hf_once_ready(shape.Y);
  weights.reserve(WeightWalker::WHOLE_UNITS);
  HfOncePanels panels(committed);
  const size_t needs = panels.bytes() + hf_side_record_bytes(P), may = HfOnceSlab::most(O.gpu_memory);
  if (needs > may) {
    fprintf(stderr, "cp_commit: the table's record needs %zu bytes of GPU memory, and %zu may be taken (what is free less 2 GiB, and no more than -gpu-memory). Free memory on the GPU or raise -gpu-memory\n", needs, may);
    weights.release();
    return 2;
  }
  struct ClosePanels {
    HfOncePanels &panels;
    ~ClosePanels() { panels.close(); }
  } close_panels_at_every_return{panels};
  panels.open();
  HfSide side;
  side.P = &P, side.Y = &shape.Y;
  side.held.seed = &key, side.held.key_again = key.again;
  side.short_lived = [](bool) {}; // no memory is lent
  side.walk = [&panels, table](const HfUnits &units) { commit_lie::walking(table, commit_lie::RECORD), panels.walk(units); };
  side.lines = true;
  // The pads are derived again for this use, as for the table's generation and for every opening.
  // If all uses shared one buffer and it were zeroed in host memory, they would agree on zeros,
  // every check would pass, and the published answers would be exact equations on the weights.
  // [own-test-pads]: every pad is compared with a second derivation before the first read and after
  // the last.
  TablePads pads;
  int end = take_table_pads(P, &key, key.again, PADS_FOR_THE_ANSWERS, pads);
  if (end == OWN_GOOD) {
    record = hf_side_record(side, table_shape, blk::index_name(hf_digest(root), table_shape, hf_digest(P.configuration)), pads);
    end = pads_own_test(P, &key, key.again, pads);
  }
  if (end != OWN_GOOD) {
    if (!record.y.empty()) explicit_bzero(record.y.data(), record.y.size() * sizeof(blk::E));
    pads.wipe();
    panels.close();
    fprintf(stderr, "%s: the rows' pads differ from a second derivation under the key's second copy: the program or the machine has a fault. No record is made. The run ends and keeps nothing of "
                    "isk or ivk. Run cp_commit again, on another GPU if there is one\n", own_stop(end));
    exit(4);
  }
  return hf_once_flags("the table's record");
}
// The second form's check of the anchor that it has read [hc-own-node]: regenerates the tree's
// first part (2^16 positions at most, or 2^s) from the weights and the key, and compares its nodes
// at the stored levels with the stored ones. This stops a run with another commitment's seed,
// other weights or another shape before a LINK proof begins. No symbol of T leaves the process.
static bool part_zero_is_the_kept_one(const PolC &PO, const rf::HfShape &shape, const Plan &table_plan,
                                      PinnedWeights &weights, const TableKey &key, const u32 *kept_heap) {
  const HfParams &P = shape.P;
  int device = 0;
  CK(cudaGetDevice(&device));
  Plan plan = table_plan;
  plan.table.lp = std::min(P.ln, std::max({plan.table.s, 1, std::min(plan.table.lp, 16)}));
  const int s = plan.table.s, lp = plan.table.lp;
  // A heap of full length: only the part's nodes are written, and untouched pages take no memory.
  u32 *made = (u32 *)calloc(kept_digests(P.ln, s) * DW, 4);
  if (!made) {
    fprintf(stderr, "cp_commit: no host memory for the test of the anchor's first nodes\n");
    exit(4);
  }
  RowCells *cells = cells_maker(PO, shape, plan, weights)(device);
  {
    PartMaker maker(P, lp, s, &key);
    maker.key_again = key.again;
    ::derivation_check::the_anchors_table_begins();
    maker.parts(0, 1, *cells, made);
    ::derivation_check::the_table_ends();
  }
  cells->release();
  delete cells;
  u32 differ = 0;
  for (int l = s; l <= lp; l++) {
    const size_t at = kept_at(P.ln, l, 0) * DW, words = ((size_t)DW) << (lp - l);
    for (size_t i = 0; i < words; i++) differ |= made[at + i] ^ kept_heap[at + i];
  }
  free(made);
  return !differ;
}

// ---- the check of the weights ----
// One pass over the weights, unit by unit, on this thread's GPU, before anything is committed. It
// compares the integers with wsums.txt's checksums, if any, and feeds the range scan and the
// worst-case check if given. Returns 0, 2 or 6.
static int check_the_weights(const Opt &O, const PolC &PO, const rf::HfShape &shape, PinnedWeights &weights,
                             rf::check_commitment::Scan *scan, rf::worst_case::Check *check) {
  const rf::LayW &Y = shape.Y;
  const auto begin = now();
  size_t blocks = 0;
  for (const rf::UnitW &u : Y.U) blocks = std::max(blocks, (u.off + u.rows + D - 1) / D - u.off / D);
  i8 *pan = nullptr;
  if (scan) CK((cudaMalloc)((void **)&pan, blocks * D * Y.k));
  weights.walk_on(
      cudaStreamPerThread, WeightWalker::WHOLE_UNITS,
      [&](const Piece &piece, cudaStream_t stream) {
        const rf::UnitW &u = *piece.u;
        if (weights.has_sums()) weights.hold_sums(piece, stream);
        if (check) check->take(u, piece.w0, piece.w1);
        if (!scan) return;
        const size_t y0 = u.off / D * D, rows = (u.off + u.rows + D - 1) / D * D - y0;
        ::g5::k_cellsT<<<8192, 256, 0, stream>>>(
            shape.um[piece.unit], (long long)PO["H"], (long long)PO["G"], (long long)shape.sh.dh,
            (const short *)piece.w0, (const short *)piece.w1, (const int *)piece.w0, Y.k, y0, rows, rows, pan);
        CK(cudaGetLastError()); // on the launching thread (prover/docs/cuda-status.md, "The rule")
        scan->unit(PO, u, piece.unit, pan, rows, u.off - y0, Y.k);
      },
      true);
  weights.sums.finish();
  if (pan) {
    CK(cudaMemset(pan, 0, blocks * D * Y.k));
    CK((cudaFree)(pan));
  }
  weights.release();
  CK(cudaGetLastError());
  printf("cp_commit: the check of the weights: %.1f s\n", seconds_since(begin));
  if (!weights.has_sums()) return 0;
  printf("cp_commit: %zu of %zu tensors' integers have the sums of wsums.txt\n", weights.sums.equal, weights.sums.held);
  if (weights.sums.held != weights.source.sums.size()) {
    fprintf(stderr,
            "cp_commit: wsums.txt lists %zu tensors, but the layout that follows from the parameter file has "
            "%zu; nothing is written\n",
            weights.source.sums.size(), weights.sums.held);
    return 2;
  }
  if (weights.sums.equal != weights.sums.held) {
    fprintf(stderr, "cp_commit: the integers computed from the model file do not match the hashes in wsums.txt; "
                    "nothing is written\n");
    return 6;
  }
  return 0;
}
static int read_shape(const Opt &O, const std::string &policy_file, PolC &PO, rf::HfShape &shape) {
  if (!PO.read(policy_file.c_str())) {
    fprintf(stderr, "cp_commit: the parameter file %s cannot be opened, or holds a value that the line above names\n",
            policy_file.c_str());
    return 2;
  }
  // On a model's files the operator is told once what a record publishes, here: no model file is
  // opened yet, no seed taken. A good run's last line points back. Why: docs/operators-duties.md,
  // "A record".
  if (!O.weights_are_made_up() && !O.public_seed)
    printf("cp_commit: note (record): for each stored table (the key file's and each leaf file's) the public files will hold a record: one masked sum of each row's weights, for every client, for good. "
           "Each mask is derived from this run's seed (for a leaf file's table: with that table's kept bytes): whoever gets these later can take the masks off. "
           "Keep the secret folder secret for as long as the weights matter, and never run cp_commit twice from one saved state of a machine (a cloned or restored image): "
           "two commitments with the same masks over different weights give the weights away in part; over the same weights the second is the first again with its counts at zero, so its masks would be used beyond their limit. "
           "Why, and what has been checked: docs/operators-duties.md, \"A record\"\n");
  // s8::set() checks the version word and stores the target and the bits of proof of work
  // (prover/common/state8.h). D, checked below: W's row count is rounded up to a multiple of it.
  if (const std::string why = s8::set(PO.file.frame); !why.empty()) {
    fprintf(stderr, "cp_commit: the parameter file is not admitted: %s\n", why.c_str());
    return 2;
  }
  if (PO["D"] < 1) {
    fprintf(stderr, "cp_commit: the parameter file must have D at least 1\n");
    return 2;
  }
  if (int rc = shape.from_policy(PO)) return rc;
  // A model with experts (NE > 0) uses all four public tables: their digests are required here. A
  // dense model (NE 0) uses three (SILU, RoPE, softmax; not the FS/SIG table) and carries their
  // digests to be the standard parameter form, but is not forced to here: a synthetic dense kit
  // with no digest lines still commits under a test build, and the standard-form check below
  // (cp-commit-not-standard) stops a production commit of a dense kit that would get no YES. Only
  // the first digest line, tsilu0, is looked for here.
  if (!PO.m.count("tsilu0") && PO["NE"] > 0)
    return making::refuse("policy-tables-absent", "the parameter file lacks the lines tsilu0 .. texp" + std::to_string(DW - 1) + ", the digests of the four public tables; no prover "
                                                  "serves such a file (scales/policy_tables.py writes the lines)");
  shape.model.seed = O.generator ? PO["seed_lo"] | ((u64)PO["seed_hi"] << 32) : 0;
  return 0;
}
static void print_digest(const char *name, const u32 *digest) {
  printf("cp_commit: %s =", name);
  for (int i = 0; i < DW; i++) printf(" %08x", digest[i]);
  printf("\n");
}

// ---- the run ----
// What the loop over leaf tables needs of the anchor. RT: its root. h_card: its rf::HfRecord's
// digest.
// The anchor's stored levels and its key (anchor_key()) stay in memory for all LINK proofs, because
// each LINK proof opens the anchor too; wipe_the_keys() wipes the key after the loop.
struct Anchor {
  vector<u32> heap; // the anchor's tree from level s up (HfKept)
  u32 h_card[DW];
  u32 RT[DW];
  // The anchor's record of phase A. The second form of the call, which is rejected (see the top of
  // this file), does not fill it: the ID computed from it would not be the published one.
  blk::IndexRecord record;
};
static int cp_commit_main(int argc, char **argv) {
  static Opt &O = *new Opt; // never destroyed: the parts that are called keep references
  if (!O.parse(argc, argv)) {
    Opt::usage();
    return 2;
  }
  // Called for its exit: a test build without HC_MASK_LOW_LIMIT exits with status 2 if the test
  // variable H_HC_MASK_LIMIT is set, which it cannot obey (hf_mask.cuh). No seed is sampled yet.
  (void)mask_limit(nullptr);
  // The mask generator's self-test against known answers [own-test-generator] (hf_mask_own.cuh):
  // the host half here; the whole test, which adds a kernel, at the first CUDA calls below.
  auto generator_fails = [] {
    fprintf(stderr, "cp_commit: refused (%s): the masks' generator does not give its known answers on this machine\n",
            own_stop(OWN_GENERATOR));
    return 2;
  };
  if (generator_own_test_on_the_host() != OWN_GOOD) return generator_fails();
  // a_test: a run with -test-seed. Nothing that it generates is hidden: the seed comes from a file.
  // -test-bits alone does not set it: the seed is sampled; the log still says "for tests alone".
  const bool a_test = SERVING_OR(false, O.test_seed != nullptr);
#ifdef HELD_LOOK
  blk::prover_messages_outside = false; // normal, except while the key file is generated below
#endif
  made_up_alone::say(O.weights_are_made_up(), a_test); // a model's files never count as synthetic
#ifndef SERVING_BUILD
  cert_doors::nothing_is_hidden = made_up_under_a_tests_seed(); // set just above (prover/common/made_up_alone.h)
#endif
  lost_coins_door::open_if(SERVING_OR(false, O.test_seed != nullptr)); // `open_if` also requires `made_up_under_a_tests_seed`
  // Test hook for a malicious committer (prover/hash_commitment/commit_lie.cuh). In this
  // repository every commit_lie:: function is empty, the identity or a constant in every build, so
  // each commit_lie:: call here is empty, false or 0 at compile time. The build that gives them
  // other bodies is not part of this repository.
  commit_lie::read_the_variable(O.test_seed != nullptr);
#ifdef HELD_LOOK
  if (O.from_weights && !O.generator && !O.made_up_blocks) return making::refuse("usage", "the build for the look at hiding takes made-up weights alone: -weights made-up-blocks or generated");
#endif
  // Every build but the production build (prover/common/setting.h, BUILT_FOR_TESTS_ALONE) accepts
  // only synthetic weights: it knows the test options, and some drop a mask, a salt or a pad, which
  // no verifier sees. So test options never meet a model's files. Checked before such a file or the
  // seed's file is opened. Cost: no byte-for-byte comparison of two versions on *.gguf files.
#if BUILT_FOR_TESTS_ALONE
  const char *const fenced_says = "this program was " BUILT_SAYS ": what it makes may hide nothing, and no verifier sees that. It takes made-up weights alone (made-up-blocks, generated). For a model's files, build the programs with prover/build.sh, without an argument";
  if (O.from_weights && !O.weights_are_made_up()) return making::refuse("tests-build-on-a-model", fenced_says);
#endif
  // -keep-for-the-audit is rejected on a model's files because tools/audit accepts only
  // synthetic weights: no program would ever destroy the bytes that RunKeys::keep() has stored, a
  // long life that the option's design rules out (prover/hash_commitment/hf_commit_keys.h,
  // [keep-for-the-audit]). Neither rejection may be removed without the other. This one is in every
  // build and comes before anything is sampled or written.
  if (O.keep_for_the_audit && !O.weights_are_made_up())
    return making::refuse("cp-commit-kept-on-a-model", "-keep-for-the-audit on a folder of weights: the audit with the key takes commitments to made-up weights alone, so the kept bytes, which with the seed open every "
                                                       "mask in key.bin and the entries, would never be destroyed. Leave the option out");
  // Every run except one on synthetic weights with -test-seed exits if a test variable is set: no
  // run on a model's files reads one. Otherwise it re-executes itself with a reduced environment
  // (prover/common/reduced_env_start.h): some variables make the GPU driver write GPU memory to a file.
  if (!made_up_under_a_tests_seed()) {
    if (const char *word = test_word_in_the_environment()) { // before the seed is drawn or a file is created
      fprintf(stderr, "cp_commit: the environment holds %s, a test variable; test variables are accepted only in a run on made-up weights under "
                      "a test's seed, in the tests' build\n",
              word);
      return 2;
    }
    if (const char *why = reduced_env_start(argv, {"SUDO_UID"})) {
      fprintf(stderr, "cp_commit: refused (no-fresh-start): the program cannot start a second time with a reduced "
                      "environment: %s. Nothing was drawn or written\n",
              why);
      return 2;
    }
  }
  // The account that ran sudo, or 0, for making::reach_by_parts(): "sudo cp_commit ..." then works
  // below that account's own directories, as "sudo cp_prove ..." does. The second start above
  // keeps SUDO_UID; this call removes it from the environment, whoever the caller is, and ignores
  // its value unless the caller is root (prover/common/walked_path.h).
  WalkedPath::typed_sudo = take_who_typed_sudo();
  SERVING_OR(, blk::card_test_byte = ::lent::poison_byte()); // LENT_POISON: a test's fill byte for the GPU engine's buffers (prover/engine/device.cuh); -1: none
  if (SERVING_OR(false, a_test || O.asked < BAR)) // here, so that a re-executed run prints it once
    printf("cp_commit: a run with a test option: what it makes is for tests alone%s\n",
           O.test_seed ? "; the seed is read from a file, so nothing is hidden" : "");
#if BUILT_FOR_TESTS_ALONE
  printf("cp_commit: THIS PROGRAM WAS " BUILT_SAYS ": what it makes is for tests alone\n");
#endif

  if (!O.from_weights) return making::refuse("cp-commit-first-form-alone", "this program makes a commitment and its leaves in one call; it does not add a leaf to an existing commitment");
  const std::string judge = making::find_the_judge();
  if (judge.empty()) return making::refuse("no-verifier-beside", "there is no executable file cp_verify in the same folder as this program");

  // ---- what the call names ----
  ::cp::IskText frame; // isk.txt
  // key.bin's bytes as this process holds them: generated below, or read once in the second form.
  // cp_verify's part-calls (-key, -leaf) get these bytes, so that what is checked is what is used.
  static std::string &key_bytes = *new std::string;
  ::cp::Key published; // second form: key_bytes, cut
  // A run works in <isk>.making and <ivk>.making; making::top_folders_take_their_names() renames
  // them last, isk first. After the checks below O.isk and O.ivk are working names; the final ones:
  const std::string isk_given = O.isk, ivk_given = O.ivk;
  if (O.from_weights) {
    if (O.isk == O.ivk) return making::refuse("usage", "-isk and -ivk name the same folder: the secret folder and the public one must be two");
    // The directories in which the run creates a name are opened here, once, part by part; every
    // later look at a name in one of them starts from its descriptor.
    for (const auto &[name, option] : {std::pair<const std::string *, const char *>{&O.ivk, "-ivk"}, {&O.isk, "-isk"}, {&O.worst_case_table, "-worst-case-table"}})
      if (!name->empty())
        if (int rc = making::reach_by_parts(making::folder_of(*name), *name, option)) return rc;
    for (const std::string &name : {O.ivk, O.isk}) { // final names first: one that exists is rejected plainly; no message says to remove it
      if (name.size() >= 7 && !name.compare(name.size() - 7, 7, ".making")) // else -isk X -ivk X.making would give the secret working directory the public one's name
        return making::refuse("usage", name + ": a name that ends in .making is this program's own, for a folder that is not finished: give another");
      if (making::there(name) && name == O.isk && making::there(making::beside(O.ivk))) // O.ivk does not exist (checked first): the state between a run's last two renames
        return making::refuse(HF_STOP_FILE_THERE, name + " exists already, and so does " + making::beside(O.ivk) + ": a call with these two names is just ending, or was interrupted between its last two steps. Nothing was touched. See docs/operators-duties.md, \"Keys, seeds and machines\"");
      if (making::there(name)) return making::refuse(HF_STOP_FILE_THERE, name + " exists already: a commitment is made in two new folders. Give -isk and -ivk two new names: nothing has to be removed, and a folder that stands under -isk's name may hold a commitment's seed, which nothing brings back");
    }
    // Rejects a working name that exists: a running call's (it holds a lock: "The run's two locks",
    // below), a dead call's, or something that cp_commit has not created. Nothing is changed.
    if (int rc = making::refuse_what_is_left(O.isk, O.ivk)) return rc;
    // Before the long work: can both parents' file systems do what the run needs later?
    for (const std::string &name : {O.ivk, O.isk})
      if (int rc = making::try_the_file_system(name)) return rc;
    O.isk = making::beside(isk_given), O.ivk = making::beside(ivk_given);
    frame.M = O.M, frame.weights = O.weights; // `frame.version` and `frame.ranges` are set below
    frame.test_seed = O.test_seed != nullptr;
    frame.public_seed = O.public_seed != nullptr;
#ifdef SERVING_BUILD
    if (O.public_seed) {
      // Checks the file early, before the long work; take_the_seed() reads it again for the seed.
      uint8_t derived[32];
      const bool read = ::public_weights::seed_of_the_file(O.public_seed, derived);
      explicit_bzero(derived, sizeof derived);
      if (!read) return making::refuse("usage", std::string("-public-weights ") + O.public_seed + " is not a readable file of 32 bytes");
      // A model's files need more than the option: a marker file each. Synthetic weights need none.
      if (!O.weights_are_made_up())
        if (int rc = making::check_the_markers(O.shards)) return rc;
      // Printed only here, so that a call that is rejected above leaves no such line in a log.
      printf("cp_commit: FOR PUBLIC WEIGHTS: with this option the weights are NOT hidden: whoever has the 32-byte file can work out every mask. "
             "The public files carry no mark of it: key, entries, ID and proofs look like any other commitment's\n");
    }
#endif
  } else {
    if (!::cp::read_isk_text(::cp::isk_txt(O.isk), frame) || !hf_version_admitted(frame.version) || !frame.M)
      return making::refuse("cp-isk-frame", ::cp::isk_txt(O.isk) + " is not the text of a commitment of this build (version word " +
                                                std::to_string(HF_VERSION_WORD) + ")");
#ifdef SERVING_BUILD
    if (frame.test_seed)
      return making::refuse("cp-isk-seed-test", ::cp::isk_txt(O.isk) + " says \"seed test\". If cp_commit made this commitment without a test's seed, it did not write that line: somebody else "
                                                                        "writes into isk. Do not strike the line: switch off whatever restarts cp_prove, the model server and whatever calls cp_commit by "
                                                                        "itself, then stop every cp_prove on this isk and wait "
                                                                        "until each has ended, make no further call of cp_commit or cp_prove on it, and go by docs/operators-duties.md, \"If the count "
                                                                        "is lost or in doubt\", before anything else. The line stands on a commitment that was made for tests, under a seed that was "
                                                                        "handed in; a serving build makes no leaf for it");
#else
    if (O.test_seed && !frame.test_seed)
      return making::refuse("cp-commit-test-seed", "-test-seed is given, and " + ::cp::isk_txt(O.isk) + " does not say that the seed is a test's");
#endif
    if (!::cp::read_file(::cp::ivk_key(O.ivk), key_bytes) || !::cp::cut_key(key_bytes, published))
      return making::refuse("cp-key-head", ::cp::ivk_key(O.ivk) + " cannot be read, or does not begin as a key.bin does: the anchor does not stand");
    if (int rc = making::nothing_else_in(O.ivk)) return rc; // here too, so that the call stops before its long work
    if (published.version != frame.version)
      return making::refuse("cp-key-head", ::cp::ivk_key(O.ivk) + " has another version word than isk.txt's, " + std::to_string(frame.version));
    if (!O.take_weights(frame.weights)) return 2;
#if BUILT_FOR_TESTS_ALONE
    if (!O.weights_are_made_up()) return making::refuse("tests-build-on-a-model", fenced_says);
#endif
#ifdef HELD_LOOK
    if (!O.generator && !O.made_up_blocks) return making::refuse("usage", "the build for the look at hiding works on a commitment to made-up weights alone");
#endif
    // The numbers that isk.txt is to have after this run. Neither may decrease: a number below `M`
    // has its LINK proof or was given up; the masks of a leaf table below `first` count as used up.
    if (!O.M_given) O.M = frame.M;
    if (!O.first_given) O.first = frame.first;
    if (O.M < frame.M)
      return making::refuse("cp-commit-M-falls", "-M " + std::to_string(O.M) + " is below isk.txt's M = " + std::to_string(frame.M) + ": the numbers up to M - 1 are spent for good");
    if (O.first < frame.first)
      return making::refuse("cp-commit-first-falls", "-first " + std::to_string(O.first) + " is below isk.txt's first = " + std::to_string(frame.first) + ": a leaf that has left the draw does not come back");
    if (O.first >= O.M)
      return making::refuse("cp-commit-no-live-leaf", "with first = " + std::to_string(O.first) + " and M = " + std::to_string(O.M) + " cp_prove would have no leaf to draw: raise -M in the same call");
    // Without -leaves: all new leaf tables but those below -first (given up after a public stop).
    if (!O.leaves_given) O.leaf_from = std::max(frame.M, O.first), O.leaf_to = O.M;
  }
  const uint32_t M_before = frame.M, first_before = frame.first; // the first form: what it is about to write
  if (O.leaf_to > O.M) return making::refuse("usage", "-leaves names a leaf beyond M - 1 = " + std::to_string(O.M - 1));
  if (O.leaf_from < O.leaf_to && O.leaf_from < O.first)
    return making::refuse("cp-commit-leaf-retired", "-leaves names leaf " + std::to_string(O.leaf_from) + ", below first = " + std::to_string(O.first) + ": no leaf is made that has left the draw");
  const std::string config = O.from_weights ? O.config : ::cp::isk_config(O.isk);
  const std::string policy_file = config + "/policy.txt", kl_file = config + "/KL.bin", layout_file = config + "/layout.txt";

  // The rejections above need no GPU. The first CUDA calls follow.
  near_host_pin();
  // Rejects a GPU that another process already uses, as cp_prove does (prover/prover/inputs.cuh).
  // cp_commit sizes its work by the GPU's free memory, so beside another program either can run out
  // of memory in the middle of its work. A rejection here costs nothing: no configuration or model
  // file has been read, no seed sampled, nothing written. Not detected: a process that comes later
  // or has not yet taken its memory. With -test-seed the run goes on: a small test can share a GPU.
  {
    size_t free_bytes = 0, total_bytes = 0;
    CK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (total_bytes - free_bytes > GPU_IN_USE_AT_THE_START && !a_test)
      return making::refuse("gpu-in-use", std::to_string((total_bytes - free_bytes) >> 20) + " MiB of this GPU's memory are in use, this program's own share included, more than the " +
                                              std::to_string(GPU_IN_USE_AT_THE_START >> 20) + " MiB that this program accepts at its start, so another program is taken to be using the GPU. "
                                              "No seed was drawn and nothing was written. Run the same command on a GPU that nothing else uses (CUDA_VISIBLE_DEVICES chooses one; nvidia-smi shows what runs on each), or when the other program has ended");
  }
  if (generator_own_test(cudaStreamPerThread) != OWN_GOOD) return generator_fails();
  // Self-tests [own-test-coins]: key derivation; a generator from which cp_commit draws nothing
  // (prover/hiding/coins_own.cuh); the one that it uses (prover/engine/stream_coins.cuh). That test
  // runs a kernel of its own: the three that make values are checked only by samples of each draw.
  const bool keys_hold = ::cp::keys_own_test(), coins_hold = ::hz::coins_own_test() == 0;
  ::blk::coins_own_test_card(cudaStreamPerThread); // a known answer on the host; its own kernel's values against the host's; ends the process at a difference
  if (!keys_hold || !coins_hold) {
    fprintf(stderr,
            "cp_commit: refused (%s): the routines that make this program's secret random values and their keys fail "
            "their self-test: the program, its build or the machine is at fault. No seed was drawn or read and nothing was written. "
            "See prover/docs/ends-in-short.md\n",
            ::hz::OWN_TEST_COINS);
    return 2;
  }
  static PolC &PO = *new PolC;
  static rf::HfShape &shape = *new rf::HfShape;
  if (int rc = read_shape(O, policy_file, PO, shape)) return rc;
#ifndef SERVING_BUILD // a production build rejects such a file in `read_shape`
  // A file that misses a condition for hiding (blocks/frame/hiding_conditions.h) is accepted only
  // with -test-seed.
  if (const std::string missed = blk::hiding_conditions_missed_by_the_file(PO.file.frame); !missed.empty() && !frame.test_seed)
    return making::refuse("prover-only-hiding-conditions", "the parameter file misses conditions for hiding (their names and what each asks: blocks/frame/hiding_conditions.h): " + missed +
                                                               ". Make a new commitment, under a parameter file that meets them. A file that misses one is taken for tests alone: by a call that makes "
                                                               "a new commitment, with -test-seed; by a call on a standing one, where isk.txt says \"seed test\", as it does on a commitment made "
                                                               "with -test-seed. No key protects that line, and whoever can write isk.txt can add it: never add it to a commitment that was made "
                                                               "without -test-seed");
#endif
  if (O.from_weights) frame.version = hf_version_word();
  if (frame.version != hf_version_word())
    return making::refuse("cp-isk-frame", ::cp::isk_txt(O.isk) + " has the version word " + std::to_string(frame.version) +
                                              ", and the parameter file in isk " + std::to_string(hf_version_word()));
  const HfParams &P = shape.P;
  const rf::LayW &Y = shape.Y;
  printf("cp_commit: W has 2^%d columns, %zu rows, %zu units; M = %u leaves\n", P.lk, P.m0, Y.U.size(), O.M);
  P.print();
  rf::check_commitment::hold_the_widths(PO, Y);
  // s (-keep): a tree's stored level, at or below the top layer that each kind of opening shows.
  const int highest = P.keep_level_most();
  if (O.from_weights) frame.keep = O.keep_level >= 0 ? O.keep_level : std::min(P.lcl, highest);
  const int s = frame.keep;
  if (s < 0 || s > highest)
    return making::refuse(HF_STOP_KEEP_LEVEL, "the stored level is " + std::to_string(s) + ", above fln - c = " + std::to_string(highest) +
                                                  ", the level of the top layer that an opening shows");
  // The configuration files' bytes. layout.txt may be missing: see cp-commit-no-layout below.
  const std::string &policy_bytes = PO.file.bytes; // the bytes that read_shape() has read and parsed
  std::string kl_bytes, layout_bytes;
  const bool has_layout = !access(layout_file.c_str(), F_OK);
  if (!whole_file(kl_file, kl_bytes) || kl_bytes.size() % 4 || (has_layout && (!whole_file(layout_file, layout_bytes) || layout_bytes.empty()))) {
    fprintf(stderr, "cp_commit: KL.bin (a whole number of 32-bit words) or layout.txt in %s cannot be read\n", config.c_str());
    return 2;
  }
  if (!has_layout && O.asked >= BAR)
    return making::refuse("cp-commit-no-layout", config + " has no layout.txt, and without it no run of cp_verify is a verification: put the layout "
                                                 "file there" SERVING_OR("", " (a test's run, with -test-bits, goes on without it)"));
  // The verifier's own code, compiled in, accepts the configuration files or ends the process as
  // cp_verify does: the reason, then "NO: <name>", status 1. Nothing is sampled or written yet.
  std::unique_ptr<blk::Configured> configured;
  {
    vector<u32> scales(kl_bytes.size() / 4);
    memcpy(scales.data(), kl_bytes.data(), kl_bytes.size());
    if (int rc = cp_with_configured(policy_bytes, layout_bytes, scales, (double)O.asked, true, [&](const blk::Configured &c) {
          configured = std::make_unique<blk::Configured>(c); // a value type: the copy outlives `c`
          return 0;
        }))
      return rc;
  }
  // Accepted files that describe a test on another table than this build generates (for example a
  // salt of another length) are rejected here, before a weight is read and a table made in vain.
  if (const char *why = hf_lin_misfit(P, configured->sized().once.S)) return making::refuse("cp-commit-table-misfit", why);
  // The limits, in columns: a row's masks ("random entries" in messages) allow t.bears opened
  // columns of a table in all. A leaf table needs its LINK proof and one batch; the anchor needs
  // the key file's one-time test and -M LINK proofs: -M counts retired and given-up numbers too.
  {
    const blk::Configured::TableColumns t = configured->table_columns();
    const u64 most = ~(u64)0, links = t.link_on_the_first && (u64)O.M > most / t.link_on_the_first ? most : (u64)O.M * t.link_on_the_first; // saturates where the product would wrap
    const u64 on_the_anchor = links > most - t.one_time ? most : t.one_time + links;
    const u64 batches = t.a_batch && t.bears >= t.link_on_the_further ? (t.bears - t.link_on_the_further) / t.a_batch : 0;
    if (batches != t.batches) // t.batches: the number with which the hiding conditions were checked; the two must agree
      return making::refuse("cp-commit-leaf-budget", "by the columns that the programs count a leaf serves " + std::to_string(batches) + " batches, and the conditions of hiding were asked with " +
                                                         std::to_string(t.batches) + ": a link shows another number of its leaf's columns than a one-time test");
    if (!batches)
      return making::refuse("cp-commit-leaf-budget", "a leaf's link shows " + std::to_string(t.link_on_the_further) + " of its columns and a batch " + std::to_string(t.a_batch) +
                                                         ", and a row's random entries bear " + std::to_string(t.bears) + " shown columns in all: a leaf would serve no batch");
    printf("cp_commit: the limit: a stored table's random entries bear %llu shown columns in all. Of a leaf its own link shows %llu and every batch %llu: each leaf serves at most %llu batches "
           "in all. Of the anchor the key's one-time test shows %llu and every leaf's link %llu: M = %u leaves take %llu. Every opening that is begun is counted, in columns, in its table's %s, in "
           "one folder only, and cp_commit and cp_prove begin no opening that the count does not bear. Never serve a leaf from two folders, and make a new commitment in time\n",
           (unsigned long long)t.bears, (unsigned long long)t.link_on_the_further, (unsigned long long)t.a_batch, (unsigned long long)batches, (unsigned long long)t.one_time,
           (unsigned long long)t.link_on_the_first, O.M, (unsigned long long)on_the_anchor, ::cp::LEAF_COUNT_FILE);
    if (on_the_anchor > t.bears)
      return making::refuse("cp-commit-anchor-budget", std::to_string(t.one_time) + " + M x " + std::to_string(t.link_on_the_first) + " = " + std::to_string(on_the_anchor) +
                                                           " columns of the anchor would be shown, and a row's random entries bear " + std::to_string(t.bears) + ": take fewer leaves");
  }
  // A planning call (cp_verify -plan) before the long work detects a cp_verify of another build, a
  // parameter file that cannot reach the requested bits, and files that let no request pass.
  const making::Plans plans = making::judge_plans(judge, policy_bytes, kl_bytes, has_layout ? &layout_bytes : nullptr, PO["lt"], O.asked);
  if (plans == making::Plans::NONE)
    return making::refuse("cp-commit-judge-plan", judge + " cannot be started, or cannot plan a request under the files of " + config + ", at " +
                                                      std::to_string(O.asked) + " bits: then its own last line, above, says why");
  // Files under which cp_verify never answers YES. Its part-calls test only some conditions of the
  // standard parameter form: without this check a run could leave a commitment that gets no YES.
  if (plans == making::Plans::NO_VERIFICATION && O.asked >= BAR)
    return making::refuse("cp-commit-not-standard", "cp_verify says that no run under the files of " + config + " can be a verification: each of its lines above that begins with \"plan: \" "
                                                    "and holds \"can be a verification\" names what the files miss. Under a commitment to such files no request's run could end "
                                                    "with YES: change the files (verifier/docs/parameter-file.md, \"The standard parameter form\")"
                                                    SERVING_OR("", "; a test's run, with -test-bits, goes on"));
  // The second form reads the anchor's record from the public directory. So cp_verify -key checks
  // key.bin's bytes first: no LINK proof opens a column of either table under another record.
  if (!O.from_weights && !making::judge_passes(judge, config, key_bytes, nullptr, O.asked, false))
    return making::refuse("cp-commit-key-bent", "no link is made under a key.bin that cp_verify -key does not pass. If " + ::cp::ivk_key(O.ivk) +
                                                    " is not as the first call left it, bring that file back: " + ::cp::isk_key(O.isk) + " is a copy of it");
  vector<u32> KL(kl_bytes.size() / 4); // KL.bin as words, for the worst-case check
  memcpy(KL.data(), kl_bytes.data(), kl_bytes.size());
  // The configuration digest (docs/format/FORMAT.md, R4.9), from which the hashes of the
  // weight table's tree start: over the files' bytes in the order policy.txt, layout.txt, KL.bin.
  const blk::Digest configuration = blk::configuration_digest({policy_bytes, layout_bytes, kl_bytes});
  memcpy(shape.P.configuration, configuration.w, 4 * DW);
  // h_par, which isk.txt's line `ranges` holds, is that digest.
  const vector<u32> h_par(configuration.w, configuration.w + DW);
  // The public values that enter the keys of the private random values (prover/common/coin_keys.h):
  // the configuration digest, a table's shape and name, and the commitment ID.
  const blk::IndexShape table_shape = hf_index_shape(shape.P);
  const TableFacts facts(table_shape, configuration);
  auto the_name_under = [&](const u32 root[DW]) { return blk::index_name(hf_digest(root), table_shape, configuration); };
  auto the_id_under = [&](const u32 first_root[DW], const blk::IndexRecord &first_record) {
    return blk::published_id(configuration, the_name_under(first_root), blk::record_seeds(table_shape, the_name_under(first_root)), first_record.y);
  };
  blk::Digest id{}; // set once the anchor's root is known
  if (!(configuration == configured->configuration())) {
    fprintf(stderr, "cp_commit: internal error: the weight table's tree would start from another digest than the one that the chains name as their configuration\n");
    return 4;
  }
  const vector<u32> units_digest = rf::TH(rf::hf_unit_list_words(Y));
  const u32 *const units = units_digest.data();
  // log2 of a table's rows, and its masks per row (first and fourth numbers of the line `table`):
  // for the second implementation's comparison of masks.
  ::derivation_check::a_table_has(PO.file.frame.table_rows_log, PO.file.frame.table_random);

  // ---- the weights, and their check ----
  rf::worst_case::Check check;
  // Zeroes the check's GPU buffers that hold weights or values computed from them, all but the flag
  // word `fault` (Check::end()), at each return; not at exit(), abort() or an uncaught exception.
  struct EndsTheCheck {
    rf::worst_case::Check *begun = nullptr;
    void here() {
      if (begun) begun->end();
      begun = nullptr;
    }
    ~EndsTheCheck() { here(); }
  } ends_the_check;
  FILE *table_file = nullptr; // -worst-case-table
  // Removes -worst-case-table's file if the call fails: its numbers are computed from the weights.
  // The destructor removes it at each return from here on. Once making::begin() has armed the run's
  // cleanup, the file is on the cleanup list instead and goes with all else that the call has
  // created, also at a signal that the cleanup catches. Two endings keep it: success, and status 7,
  // where the table is the answer that the operator asked for. It is created and removed through a
  // descriptor of its directory (prover/hash_commitment/hf_commit_making.h, via()), and overwritten
  // with zeros first, like every file that the run creates (zeros_first()). Not covered: exit(),
  // abort() or a signal before the cleanup is armed. Until the check writes its table, that leaves
  // an empty file; from then to making::begin(), a short stretch, part or all of the table.
  struct TableGoes {
    std::string name;
    ~TableGoes() {
      if (name.empty()) return;
      if (const int failed = making::remove_name({name, false}); failed && errno != ENOENT) making::say_why_it_stays(name, failed);
    }
  } table_goes;
  if (O.worst_case) {
    std::string why;
    if (!check.read(PO, shape.sh, KL, why)) {
      fprintf(stderr, "cp_commit: the worst-case check: %s" SERVING_OR("", " (a test's run, under -test-seed, leaves the check out)") "\n",
              why.c_str());
      return 2;
    }
    if (!O.worst_case_table.empty()) { // for its owner only: its numbers are computed from the weights
      const bool reached = making::reach(making::folder_of(O.worst_case_table));
      const making::Via at = making::via(O.worst_case_table);
      const int fd = reached ? openat(at.fd, at.name, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600) : -1;
      if (fd >= 0) table_goes.name = O.worst_case_table;
      if (!(table_file = fd < 0 || !making::note_created(fd) ? nullptr : fdopen(fd, "w")))
        return making::refuse("no-table-file", std::string("cannot create ") + O.worst_case_table + " as a new file: " + strerror(errno) + ". If that name is taken, give one that is not, in a folder of your own: -worst-case-table never writes over a file that is already there");
    }
    check.begin();
    ends_the_check.begun = &check;
  }
#ifndef SERVING_BUILD
  if (O.from_weights && !O.worst_case) printf("cp_commit: a test's run: the worst-case check is left out, and isk.txt gets no line ranges\n");
#else
  if (O.from_weights && !O.worst_case) printf("cp_commit: -public-weights: the worst-case check is left out, and isk.txt gets no line ranges\n");
#endif
  static PinnedWeights &weights = *new PinnedWeights;
  weights.joined_cut_bytes = O.piece_bytes;
  auto begin = now();
  weights.load(O, PO, shape);
  phase_line("weights", begin);
  begin = now();
  rf::check_commitment::Scan scan;
  scan.as_declared = false, scan.say = made_up_under_a_tests_seed(); // derived from the weights: an out-of-range integer's count, size and position are printed only on synthetic ones with -test-seed;
  scan.begin(); // whether the embedding is wider than its width line is printed below in every run that gets that far
  if (int rc = check_the_weights(O, PO, shape, weights, &scan, O.worst_case ? &check : nullptr)) return rc;
  CK(cudaGetLastError()); // exits if a launch of this host thread failed, before any result or write (prover/docs/cuda-status.md, "The rule")
  // Check before you write: a run that finds an integer out of range ends here, before the seed is
  // sampled and before anything of ivk or isk is written. The ideas are public: Saltzer, Reed,
  // Clark, 'End-to-End Arguments in System Design', ACM Transactions on Computer Systems 2(4),
  // 1984, 277-288, section 'End-to-end caretaking'; Apache Iceberg documentation, 'Branching and
  // Tagging', section 'Audit Branch' (neither section says "check before you write"; the tie to
  // this sentence is loose).
  if (scan.outside) {
    fprintf(stderr, "cp_commit: refused (integer-out-of-range): a committed integer lies outside the range that the policy "
                    "declares for its kind (wb_..; for the embedding a making admits up to 2^30 - 1); nothing is written\n");
    return 16;
  }
  printf("cp_commit: every committed integer lies in its declared range%s\n",
         scan.embedding_wider ? ", except for the embedding, which is wider than the policy's wb_emb" : "");
  if (O.worst_case) {
    // The table's numbers are computed from the weights: they go only to -worst-case-table's file.
    const bool fits = check.table(table_file, nullptr, Y.U.size(), {});
    if (table_file && (ferror(table_file) || fflush(table_file) || fsync(fileno(table_file)) || fclose(table_file))) {
      fprintf(stderr, "cp_commit: %s (-worst-case-table) could not be written completely: %s; no commitment is made, and the incomplete file is removed\n",
              O.worst_case_table.c_str(), strerror(errno));
      return 2;
    }
    if (check.why_no_bound) printf("cp_commit: the worst-case check: %s: no bound can be given\n", check.why_no_bound);
    // The result covers the check's sites. That no request fails for a range follows only if the
    // forward pass tests no other range than those that the check bounds
    // (prover/weights_commitment/worst_case.cuh, at its top). This program does not check that, and
    // no test of this repository compares the two lists.
    printf("cp_commit: the worst-case check: %s\n", fits ? "every bounded site fits: at none of the sites that the check bounds can a prompt drive the value out of its range"
                                                         : "NOT every bounded site fits: at some site a prompt may drive the value out of its range; no commitment is made");
    ends_the_check.here();
    if (!fits) {
      table_goes.name.clear(); // the table stays: it says which sites do not fit
      return 7;
    }
    // The digest of the configuration files as this process has read them, written only when the
    // check has passed; cp_prove compares it with its own [ranges-not-guarded].
    frame.ranges = ::cp::digest_hex(h_par.data(), h_par.size()); // for isk.txt's line `ranges`
  }
  phase_line("check", begin);

  // A table's `Plan` is computed when the table starts, from the GPU memory free then: generating
  // the key file or a leaf file keeps some (hf_once_ready()). No file's bytes depend on it.
  Plan plan;
  auto plan_a_table = [&]() {
    plan = Plan();
    return make_plan(O, shape, weights, s, plan);
  };
  if (int rc = plan_a_table()) return rc; // here for the first time, so that too small a GPU ends the run before the seed is drawn
  const size_t heap_words = kept_digests(P.ln, s) * DW, nodes_bytes = heap_words / 2 * 4; // a heap's second half is hcnodes.bin
  static Anchor &anchor = *new Anchor;
  anchor.heap.assign(heap_words, 0);
  if (int rc = making::begin(wipe_the_keys, O.from_weights, O.leaf_from < O.leaf_to ? O.leaf_to - O.leaf_from : 0)) return rc; // first form: all or nothing (hf_commit_making.h, `Road`)
  if (!table_goes.name.empty()) making::made(table_goes.name), table_goes.name.clear();  // from here on it is on the cleanup's list; the run's good end forgets the list, and the file stays
  MadeFiles files;
  u64 attempts = 0;
  auto kept_of = [s](const vector<u32> &heap) {
    HfKept kept;
    kept.s = s, kept.heap = heap.data();
    return kept;
  };
  // [own-test-hidden] Exits if the record's tier is "hidden" and the key is zero, or the reverse.
  auto hold_the_tier = [](const rf::HfRecord &record, const TableKey &key) {
    if (own_hidden(record.word(HF_REC_TIER) == HF_TIER_HIDDEN, !seed_is_zero(key)) == OWN_GOOD) return;
    fprintf(stderr, "cp_commit: internal error (%s): the record's tier is not the run's\n", own_stop(OWN_HIDDEN));
    exit(4);
  };

  // [keep-for-the-audit] The path at which RunKeys::keep() stores 32 fresh bytes; empty: none.
  auto kept_for_the_audit = [&](const std::string &file) { return O.keep_for_the_audit ? ::cp::isk_audit_dir(O.isk) + "/" + file : std::string(); };

  // ---- the anchor ----
  if (O.from_weights) {
    for (const std::string &name : {ivk_given, isk_given}) // again: loading and checking the weights took time, and nothing reserves the final names
      if (making::there(name)) return making::refuse(HF_STOP_FILE_THERE, name + " exists already: a commitment is made in two new folders. Give -isk and -ivk two new names: nothing has to be removed, and a folder that stands under -isk's name may hold a commitment's seed, which nothing brings back");
    // The run's two locks, one on each working directory, taken right after it is created and never
    // released: the system drops them when the process ends, however it ends. A lock is on the
    // directory, not on its name, so the last two renames keep it. After SIGKILL, any other
    // uncaught signal, a crash or a power failure both directories stay; the locks tell a later
    // call, and whoever reads its rejection, whether a call still runs in either, even when -isk
    // names one earlier call's directory and -ivk another's. make_folder() syncs the parent
    // directory between mkdir() and the lock. The later call (making::refuse_what_is_left()) tries
    // twice, 0.3 s apart: if that sync takes longer, a running call passes for a dead one.
    for (const std::string &name : {O.isk, O.ivk}) {
      if (!making::make_folder(name)) return 4;
      int the_runs_lock = -1;
      for (int i = 0; i < 100 && (the_runs_lock = making::hold_leaf(name)) < 0; i++) usleep(10 * 1000); // another call's try (`refuse_what_is_left`, above) holds it briefly
      if (the_runs_lock < 0) {
        fprintf(stderr, "cp_commit: %s, which this call has just made, cannot be opened and locked. A network file system may not support flock() on a directory: give -isk and -ivk names on a local one\n", name.c_str());
        return 4;
      }
    }
    printf("cp_commit: the two folders are made as %s and %s and get their names, %s and %s, at the run's good end. After an end by Ctrl-C, SIGTERM or a fault of its own the program has removed both. After SIGKILL, a crash or a power failure they stay: the same call, made again, then says what is left\n", O.isk.c_str(),
           O.ivk.c_str(), isk_given.c_str(), ivk_given.c_str());
    if (O.keep_for_the_audit && !making::make_folder(::cp::isk_audit_dir(O.isk))) return 4; // only with the option: a missing directory means that it was never given
    if (O.keep_for_the_audit) // after every rejection of arguments and names: the directory printed is this run's
      printf("cp_commit: -keep-for-the-audit: the 32 bytes of the system of the key file's making and of each entry's are kept in %s (until the run's good end: in %s). Together with the seed they open every mask in key.bin and the entries. UNTIL THE AUDIT "
             "WITH THE KEY HAS SAID YES, NOTHING OF THIS COMMITMENT LEAVES. If this run is killed or the machine fails, they stay where they are then, beside the seed: the same call, made again, then says what is left\n",
             ::cp::isk_audit_dir(isk_given).c_str(), ::cp::isk_audit_dir(O.isk).c_str());
    // isk's copy of the configuration files, which are public. It is written first
    // because cp_verify's part-calls read the configuration from this directory.
    const std::string isk_config = ::cp::isk_config(O.isk);
    if (!making::make_folder(making::beside(isk_config)) ||
        !making::write_new(making::beside(isk_config) + "/policy.txt", policy_bytes.data(), policy_bytes.size()) ||
        !making::write_new(making::beside(isk_config) + "/KL.bin", kl_bytes.data(), kl_bytes.size()) ||
        (has_layout && !making::write_new(making::beside(isk_config) + "/layout.txt", layout_bytes.data(), layout_bytes.size())))
      return 4;
    int record_status = 0;
    if (int rc = Make::the_first_table(
            O, facts, anchor_key(),
            [&] { // the seed's file is written before the first key is derived and the first mask generated: each key's second copy is derived from the file
              the_seed().file = making::beside(::cp::isk_seed(O.isk));
              return making::write_new(the_seed().file, the_seed().w, 32) ? 0 : 4;
            },
            [&](const MakersKey &key) {
              begin = now();
              commit_lie::walking(0, commit_lie::TABLE);
              make_the_table(PO, shape, plan, weights, key, anchor.heap.data());
              record_status = make_the_record(O, PO, shape, weights, key, 0, table_shape, anchor.heap.data() + DW, anchor.record);
            }))
      return rc;
    if (record_status) return record_status;
    memcpy(anchor.RT, anchor.heap.data() + DW, 4 * DW);
    phase_line("anchor-table", begin);
    begin = now();
    // `record` is an rf::HfRecord, not anchor.record. It is written to no file and enters no key.
    rf::HfRecord record(HF_TIER_HIDDEN, P, units, h_par.data(), anchor.RT);
    u32 h0[DW];
    record.h_0(h0);
    hold_the_tier(record, anchor_key());
    // The anchor's count of opened columns (cp::count_first_table_columns()): created empty,
    // under its final name, before the first opening. once_make() and entry_make() raise it.
    if (!making::make_folder(::cp::isk_anchor_dir(O.isk)) || !making::write_new(::cp::isk_anchor_count(O.isk), "", 0)) return 4;
    id = the_id_under(anchor.RT, anchor.record);
    RunKeys::take_the_key_files(id, SERVING_OR(O.public_seed != nullptr, O.test_seed != nullptr), kept_for_the_audit(::cp::AUDIT_KEY_FILE));
    RunKeys::hold_the_key_files(id);
    blk::Writer key_file;
    {
      const Committed committed{O,  PO,           shape,  weights,  &anchor_key(), anchor_key().again, kept_of(anchor.heap),
                                h0, h_par.data(), &files, &attempts};
      const std::vector<uint8_t> root((const uint8_t *)anchor.RT, (const uint8_t *)anchor.RT + 4 * DW); // a table's commitment in first queries
#ifdef HELD_LOOK
      // A test of hiding (synthetic weights, -test-seed, variable CHAIN_HELD_STEM): the key file is
      // made with the verifier's messages fixed, and dumped at that stem. cp_verify rejects it.
      const char *held = made_up_under_a_tests_seed() ? TESTENV("CHAIN_HELD_STEM") : nullptr;
      if (held) key_file.keep_list(), blk::prover_messages_outside = blk::first_query_stands_still = true;
#endif
      if (int rc = once_make(committed, *configured, root, anchor.record, key_file)) return rc;
#ifdef HELD_LOOK
      if (held) blk::message_dump(key_file, held), blk::prover_messages_outside = blk::first_query_stands_still = false;
#endif
      if (int rc = commit_lie::closing(0)) return rc; // test hook (commit_lie.cuh): returns 0 in this repository
    }
    RunKeys::hold_the_key_files(id);
    RunKeys::the_key_file_is_made();
    ::hz::key32_wipe();
    weights.release();
    phase_line("key-file", begin);
    record.h_card(anchor.h_card);
    begin = now();
    CK(cudaGetLastError()); // exits if a launch of this host thread failed, before any result or write (prover/docs/cuda-status.md, "The rule")
    // The key file, key.bin.
    std::string &key_bin = key_bytes;
    key_bin.assign((const char *)key_file.data(), key_file.size()); // make_key()'s bytes and nothing else
    const std::string ivk_key = ::cp::ivk_key(O.ivk), isk_anchor = ::cp::isk_anchor_dir(O.isk);
    // key.bin and the anchor's nodes are written only after cp_verify -key accepts the bytes.
    const bool passed = making::judge_passes(judge, making::beside(isk_config), key_bin, nullptr, O.asked);
    phase_line("anchor-judge", begin);
    if (!passed && !commit_lie::passes("own-judge")) { // cleanup removes both working directories
      fprintf(stderr, "cp_commit: the anchor is not made. The refused key.bin, %zu bytes, is written nowhere%s\n",
              key_bin.size(), O.public_seed ? "" : ". A new call draws a new seed");
      return 4;
    }
    if (!making::write_new(making::beside(ivk_key), key_bin.data(), key_bin.size())) return 4;
    begin = now();
    const std::string text = ::cp::isk_text(frame);
    if (!making::write_new(isk_anchor + "/" + HF_NODES_FILE, anchor.heap.data() + heap_words / 2, nodes_bytes) || // the directory has had its final name since the count was created
        // key.bin again, in isk.
        !making::write_new(making::beside(::cp::isk_key(O.isk)), key_bin.data(), key_bin.size()) ||
        !making::write_new(making::beside(::cp::isk_txt(O.isk)), text.data(), text.size()))
      return 4;
    if (int rc = making::nothing_else_in(O.ivk)) return rc;
    for (const std::string &goal : {::cp::isk_seed(O.isk), isk_config, ::cp::isk_key(O.isk), ::cp::isk_txt(O.isk)}) // isk/anchor already has its final name
      if (!making::takes_its_name(goal)) return 4;
    if (!making::public_file_takes_its_name(ivk_key)) return 4;
    the_seed().file = ::cp::isk_seed(O.isk);
    making::step_done();
    phase_line("anchor-files", begin);
  } else {
    begin = now();
    if (int rc = read_the_seed(O, ::cp::isk_seed(O.isk))) return rc;
    make_again_the_first_table(anchor_key(), facts);
    rf::HfRecord record;
    const char *fault = published.rest.size() < (size_t)HF_RECORD_FILE_WORDS * 4 ||
                                !record.read_bytes((const uint8_t *)published.rest.data(), (size_t)HF_RECORD_FILE_WORDS * 4)
                            ? "hc-record-length"
                            : record.fails(P, units, h_par.data());
    if (!fault && record.level() != 0) fault = "hc-link-anchor-head";
    if (!fault && record.word(HF_REC_TIER) != HF_TIER_HIDDEN) fault = "hc-link-tier";
    if (fault) return making::refuse(fault, "the record in " + ::cp::ivk_key(O.ivk) + " is not an anchor's under this isk's configuration files");
    const std::string nodes_file = ::cp::isk_anchor_nodes(O.isk);
    if (!read_whole_file(nodes_file, anchor.heap.data() + heap_words / 2, nodes_bytes))
      return making::refuse(HF_STOP_NODES_ROOT, nodes_file + " does not hold the 2^(fln - keep) digests of the stored level");
    kept_rebuild(anchor.heap.data(), P.ln, s, TreeTags(P.configuration, HF_NUM_F, (u32)P.R));
    if (memcmp(anchor.heap.data() + DW, record.RT(), 4 * DW))
      return making::refuse(HF_STOP_NODES_ROOT, nodes_file + "'s nodes do not hash up to the root of the anchor's record");
    if (!part_zero_is_the_kept_one(PO, shape, plan, weights, anchor_key(), anchor.heap.data()))
      return making::refuse(HF_OWN_TEST_NODE, "the first nodes of the anchor's tree, made again from these weights and this seed, are not "
                                              "the stored ones: the anchor was made under another seed or from other weights");
    memcpy(anchor.RT, record.RT(), 4 * DW);
    record.h_card(anchor.h_card);
    phase_line("anchor-read", begin);
  }
  print_digest("the anchor's RT", anchor.RT);
  id = the_id_under(anchor.RT, anchor.record); // in the second form, which is rejected above: from the root that the kept nodes hash up to, and an empty record
  print_digest("the anchor's h_card", anchor.h_card);
  {
    // The commitment ID (hash of configuration digest, anchor's name, record's seeds and answers),
    // recomputed from key.bin.
    blk::Reader head((const blk::u8 *)key_bytes.data(), key_bytes.size());
    blk::UnprovedTable first_table;
    const char *why = nullptr;
    if (!blk::read_key_table_unproved(*configured, head, first_table, why)) {
      fprintf(stderr, "cp_commit: internal error: the key's first bytes cannot be read back: %s\n", why ? why : "no reason given");
      return 4;
    }
    if (!(blk::published_id(*configured, first_table.reader()) == id)) { // the process keys are derived from `id`, computed from the root in memory
      fprintf(stderr, "cp_commit: internal error: the ID that follows from the key's first bytes is not the one that follows from the first table's root and record in hand\n");
      return 4;
    }
    // As cp_verify spells it (-id): the digest's 32 bytes in order, as 64 lower-case hex digits
    // without blanks, on one line.
    blk::u8 id_bytes[blk::DIGEST_BYTES];
    blk::digest_to_bytes(id, id_bytes);
    printf("commitment ID: ");
    for (int i = 0; i < (int)blk::DIGEST_BYTES; i++) printf("%02x", id_bytes[i]);
    printf("\n");
  }

  // ---- the leaves ----
  if (O.sweep && !making::sweep_texts(O.isk)) return 4; // after the rejections above: a call that they reject has changed nothing
  if (O.leaf_from < O.leaf_to && // first form: the two directories are this run's and go on its cleanup list
      (O.from_weights ? !making::make_folder(::cp::isk_leaves(O.isk)) || !making::make_folder(::cp::ivk_leaves(O.ivk))
                    : !making::shared_folder(::cp::isk_leaves(O.isk)) || !making::shared_folder(::cp::ivk_leaves(O.ivk))))
    return 4;
  static vector<u32> &heap = *new vector<u32>;
  const LinkAnchor link_anchor{&anchor_key(), anchor_key().again, kept_of(anchor.heap), anchor.h_card};
  const std::string isk_config = ::cp::isk_config(O.isk); // it exists by now: the part-calls on the leaf files read it
  // The end that the two refusals below share. Both belong to the second form of the call (a leaf is
  // added to a commitment that stands), which cp_commit rejects above
  // [cp-commit-first-form-alone]: no run prints them. The steps are on the page that it names.
  const std::string what_now = "Switch off whatever restarts cp_prove, the model server and whatever calls cp_commit by itself, stop every cp_prove on this isk and wait until each has ended. "
                               "Then start nothing and make no call on this isk, and change nothing in it, before you have gone by docs/operators-duties.md, \"If the count is lost or in doubt\"";
  for (uint32_t k = O.leaf_from; k < O.leaf_to; k++) {
    const std::string name = "leaf-" + std::to_string(k), entry = ::cp::ivk_entry(O.ivk, k), folder = ::cp::isk_leaf_dir(O.isk, k);
    if (making::there(entry)) { // the leaf file is published last, so isk/leaf/<k>/ must exist too
      if (!making::there(folder))
        return making::refuse("cp-commit-entry-alone", entry + " is there, and " + folder + " is not: this isk has not made that entry, or was set back to a state before it did. "
                                                       "cp_prove cannot serve the leaf from this isk, and no call removes or replaces the entry. Do NOT remove the entry to get on: the leaf "
                                                       "would be made a second time, and a published number would have two links. Never put a copy or a backup of the leaf's folder in its place: its " +
                                                       ::cp::LEAF_COUNT_FILE + " may be older. " + what_now);
      printf("cp_commit: leaf %u stands and is not made again\n", k);
      continue;
    }
    // isk has leaf table k but ivk no leaf file: a run died before publishing it, or ivk lost it.
    // The stored copy is published once cp_verify -leaf accepts it. No second LINK proof is made
    // for this number: it would open other columns.
    if (making::there(folder)) {
      const int held = making::hold_leaf(folder); // the run that made it may be about to publish
      if (held < 0) return making::leaf_is_held(folder, k);
      std::string kept_bytes, kept_record;
      if (!making::there(::cp::isk_leaf_fresh(O.isk, k)) || !making::there(::cp::isk_leaf_count(O.isk, k)) || // without them cp_prove cannot serve the leaf, and nothing makes them again
          !::cp::read_file(making::kept_entry(folder), kept_bytes) || !::cp::read_file(::cp::isk_leaf_record(O.isk, k), kept_record) ||
          kept_record.empty() || kept_bytes.compare(0, kept_record.size(), kept_record) || // the entry begins with this leaf's record
          !making::judge_passes(judge, isk_config, key_bytes, &kept_bytes, O.asked, false))
        return making::refuse("cp-commit-kept-entry", folder + " stands, ivk has no entry for the leaf, and " + making::kept_entry(folder) +
                                                          " cannot be read, does not begin with the leaf's record or does not pass cp_verify "
                                                          "-leaf, or the folder has no " + ::cp::LEAF_FRESH_FILE + " or no " + ::cp::LEAF_COUNT_FILE + ". No call makes this leaf again or "
                                                          "removes it. Never make a " + ::cp::LEAF_COUNT_FILE + " and never bring one back: where it is missing, nobody can say how many batches "
                                                          "the leaf has served. Never put a copy or a backup of the folder in its place either. " + what_now);
      if (int rc = making::nothing_else_in(O.ivk)) return rc;
      if (!making::zero_times(::cp::ivk_key(O.ivk)) || !making::publish_entry(entry, kept_bytes)) return 4;
      making::step_done();
      printf("cp_commit: leaf %u stands in isk and is not made again: its kept entry is published\n", k);
      close(held);
      continue;
    }
    if (O.sweep)
      if (int rc = making::sweep_leaf(O.isk, k)) return rc;
    if (making::there(making::beside(folder)))
      return making::refuse("cp-commit-leaf-begun", making::beside(folder) + " is there: another call is at work on leaf " + std::to_string(k) +
                                                        ", or a dead call has left this. Only if no other call is at work on it: -sweep removes it");
    if (!making::make_folder(making::beside(folder))) return 4;
    const int held = making::hold_leaf(making::beside(folder)); // until the entry is published
    if (held < 0) return making::leaf_is_held(making::beside(folder), k);
    printf("cp_commit: leaf %u begins\n", k); // the log lines from here to the leaf's last phase line belong to this leaf
    begin = now();
    if (int rc = plan_a_table()) return rc;
    const bool repeatable = SERVING_OR(O.public_seed != nullptr, O.test_seed != nullptr);
    heap.assign(heap_words, 0);
    blk::IndexRecord copy_record;
    int record_status = 0;
    Make::a_copy(k, repeatable, facts, leaf_fresh(), leaf_key(), [&](const MakersKey &key) {
      commit_lie::walking((int)k + 1, commit_lie::TABLE);
      make_the_table(PO, shape, plan, weights, key, heap.data());
      record_status = make_the_record(O, PO, shape, weights, key, (int)k + 1, table_shape, heap.data() + DW, copy_record);
    });
    if (record_status) return record_status;
    u32 RT[DW];
    memcpy(RT, heap.data() + DW, 4 * DW);
    phase_line(name + "-table", begin);
    // [own-test-link], the part checked before the LINK proof begins:
    // two tables on the same weights with one root are one table.
    if (!memcmp(anchor.RT, RT, 4 * DW)) return making::refuse(HF_OWN_TEST_LINK, "the leaf's table has the anchor's root");
    begin = now();
    rf::HfRecord record(HF_TIER_HIDDEN, P, units, h_par.data(), RT);
    record.set_copy();
    u32 h0[DW];
    record.h_0(h0);
    hold_the_tier(record, leaf_key());
    const std::vector<blk::u8> answers_in_the_key = blk::record_bytes(copy_record);
    RunKeys::take_an_entrys(k, id, the_name_under(RT), answers_in_the_key, repeatable, kept_for_the_audit(::cp::audit_entry_file(k)));
    attempts = 0;
    blk::Writer entry_file;
    {
      const Committed committed{O,  PO,           shape,  weights,  &leaf_key(), leaf_key().again, kept_of(heap),
                                h0, h_par.data(), &files, &attempts};
      const std::vector<uint8_t> first_root((const uint8_t *)anchor.RT, (const uint8_t *)anchor.RT + 4 * DW), copy_root((const uint8_t *)RT, (const uint8_t *)RT + 4 * DW);
      // The leaf table's count.bin is created empty here, before anything of the table is opened.
      // It counts the LINK proof's columns now and each batch's later (cp_prove), up to `bears`.
      // It is raised through `held`, which locks the directory: count_table_columns() would wait
      // for that lock for ever. entry_make() counts the anchor's columns.
      if (!making::write_new(making::beside(folder) + "/" + ::cp::LEAF_COUNT_FILE, "", 0)) return 4;
      const uint64_t bears = configured->table_columns().bears;
      auto count_the_copy = [held, bears, k](uint64_t columns) {
        const ::cp::Counts c = ::cp::count_table_columns_held(held, columns, bears);
        if (c.say == ::cp::Counts::ROOM) return true;
        fprintf(stderr, "cp_commit: leaf %u: the count of the copy's opened columns %s: %llu columns are counted, %llu more are asked for, a row's random entries bear %llu\n", k,
                c.say == ::cp::Counts::SPENT ? "leaves no room" : "could not be raised and made lasting", (unsigned long long)c.begun, (unsigned long long)columns, (unsigned long long)bears);
        return false;
      };
      if (int rc = entry_make(committed, link_anchor, *configured, first_root, anchor.record, copy_root, copy_record, count_the_copy, entry_file)) return rc;
      if (int rc = commit_lie::closing((int)k + 1)) return rc;
    }
    RunKeys::an_entry_is_made();
    ::hz::key32_wipe();
    leaf_key().wipe();
    weights.release();
    phase_line(name + "-entry", begin);
    const std::string entry_bytes((const char *)entry_file.data(), entry_file.size()); // make_entry()'s bytes and nothing else: what is read back here, checked by cp_verify and then written
    // Checks that the leaf table's answers in the leaf file, from which cp_prove will derive its
    // keys, are the bytes from which take_an_entrys() above derived the leaf file's process key.
    {
      blk::Reader head((const blk::u8 *)entry_bytes.data(), entry_bytes.size());
      blk::UnprovedTable in_the_entry;
      const char *why = nullptr;
      if (!blk::read_entry_table_unproved(*configured, head, in_the_entry, why) || blk::record_bytes(in_the_entry.reader().record()) != answers_in_the_key) {
        fprintf(stderr, "cp_commit: internal error: leaf %u: the entry's first part cannot be read back, or its record's answers are not those that the entry's key was derived from (%s). The entry is written nowhere\n", k,
                why ? why : "the answers differ");
        return 4;
      }
    }
    begin = now();
    CK(cudaGetLastError()); // exits if a launch of this host thread failed, before any result or write (prover/docs/cuda-status.md, "The rule")
    // Until cp_verify -leaf accepts the leaf file's bytes, which are still only in memory, nothing
    // of the leaf table but its count is written. A rejected leaf file is written nowhere: nobody
    // sees its LINK proof. Its columns are counted all the same; the leaf table's count is removed
    // with its directory, leaf/<k>.making, and a later table has new masks (not with -test-seed or
    // -public-weights). The first form's cleanup removes both working directories; in the second
    // the anchor's count stays raised.
    // Once cp_verify accepts, the table's 32 system bytes are stored too (isk/leaf/<k>/fresh.bin):
    // cp_prove derives the table's making key from them and the seed, to regenerate the masks.
    const bool passed = making::judge_passes(judge, isk_config, key_bytes, &entry_bytes, O.asked);
    phase_line(name + "-judge", begin);
    if (!passed && !commit_lie::passes("own-judge")) { // cleanup removes leaf/<k>.making, count too
      fprintf(stderr, "cp_commit: leaf %u is not made. The refused entry, %zu bytes, is written nowhere\n", k, entry_bytes.size()); // no advice to call again: no call regenerates one leaf table
      return 4;
    }
    begin = now();
    if (!making::write_new(making::beside(folder) + "/" + HF_NODES_FILE, heap.data() + heap_words / 2, nodes_bytes) ||
        !making::write_new(making::beside(folder) + "/" + ::cp::LEAF_FRESH_FILE, leaf_fresh().to_be_written(), KeptBytes::BYTES) ||
        !making::write_new(making::kept_entry(making::beside(folder)), entry_bytes.data(), entry_bytes.size())) // count.bin has existed since the table was made
      return 4;
    leaf_fresh().wipe();
    // After this rename a run that finds the directory does not regenerate the leaf table. So the
    // rename and the names in the directory must be on disk before anyone can see the leaf file.
    if (!making::sync_dir(making::beside(folder)) || !making::takes_its_name(folder)) return 4;
    making::step_done(); // takes no name off the cleanup list in a run that creates a commitment: it keeps nothing before `run_done`
    if (!making::sync_dir(::cp::isk_leaves(O.isk))) {
      fprintf(stderr, "cp_commit: leaf %u is made, but the folder %s could not be brought to the disk, so the entry is not published%s\n", k, ::cp::isk_leaves(O.isk).c_str(),
              O.from_weights ? "" : ". The leaf is kept: a new call publishes it");
      return 4;
    }
    if (int rc = making::nothing_else_in(O.ivk)) return rc; // second form: a new call publishes it
    // Resets key.bin's timestamps again: a run of the second form has read the file.
    if (!making::zero_times(::cp::ivk_key(O.ivk)) || !making::publish_entry(entry, entry_bytes)) return 4;
    phase_line(name + "-files", begin);
    making::step_done();
    close(held);
    u32 h_card[DW];
    record.h_card(h_card);
    print_digest((name + ": RT").c_str(), RT);
    print_digest((name + ": h_card").c_str(), h_card);
  }
  weights.release();
  weights.end(); // after the weights' last use on a GPU
  wipe_the_keys();
  auto standing = [&](uint32_t from, uint32_t to) {
    uint32_t stand = 0;
    for (uint32_t k = from; k < to; k++) stand += making::there(::cp::ivk_entry(O.ivk, k)) && making::there(::cp::isk_leaf_dir(O.isk, k));
    return stand;
  };

  // ---- isk.txt's new numbers (the second form with -M or -first) ----
  // Concurrent runs with equal -M and -first that find every live leaf table present write the same
  // bytes. making::text_moves_on() rewrites isk.txt under a lock and rejects a text that lowers `M`
  // or `first`. A shared file system may grant the lock to each machine: nothing guards isk.txt
  // then, and of two calls with different numbers that end together one can lower `M` or `first`.
  // Such concurrent calls, like overlapping -leaves, are the operator's error.
  if (O.M != M_before || O.first != first_before) {
    const uint32_t live = O.M - O.first, stand = standing(O.first, O.M);
    if (stand == live) {
      frame.M = O.M, frame.first = O.first;
      if (int rc = making::text_moves_on(O.isk, frame)) return rc;
      printf("cp_commit: isk.txt now says M = %u, first = %u\n", frame.M, frame.first);
    } else
      printf("cp_commit: isk.txt is as it was (M = %u, first = %u): %u of the %u leaves %u .. %u stand\n", M_before, first_before, stand, live,
             O.first, O.M - 1);
  }
  if (O.from_weights) { // everything is written, checked, renamed inside and synced; the keys are wiped
    // An incomplete commitment is not renamed and gets no "done" with status 0, unless -leaves (for
    // tests) asked for a part. Every leaf table was generated above or the run has ended, so this
    // check fails only if a file has disappeared during the run.
    if (const uint32_t stand = standing(frame.first, frame.M); !O.leaves_given && stand != frame.M - frame.first) {
      fprintf(stderr, "cp_commit: NOT done: only %u of the M = %u leaves exist at the end of the run: files have gone while it worked. Nothing is published, and what the run has made is removed\n", stand, frame.M);
      return 4;
    }
    if (int rc = making::nothing_else_in(O.ivk)) return rc; // [cp-commit-ivk-holds], before the act that publishes
    if (!making::top_folders_take_their_names(isk_given, ivk_given)) return 4;
  }
  making::run_done(); // from here on no exit removes anything; before the line that says that ivk can be handed out
  if (O.from_weights) O.isk = isk_given, O.ivk = ivk_given; // what follows uses and prints the final names
  // The last line counts the live leaf tables of isk.txt as this run leaves it.
  const uint32_t live = frame.M - frame.first, stand = standing(frame.first, frame.M);
  // The last line says whether ivk is complete, whether it is for tests only (-test-seed,
  // -test-bits or any build but the production build), whether an audit must pass first and
  // whether it hides nothing (-public-weights).
  const char *const a_tests = SERVING_OR(nullptr, frame.test_seed ? "WAS MADE UNDER A TEST'S SEED: nothing in it is hidden from whoever holds the seed's file"
                                                  : O.asked < BAR ? "cp_verify WAS ASKED FOR FEWER BITS THAN A VERIFICATION NEEDS (a test's option)"
                                                                  : "WAS MADE BY A PROGRAM THAT WAS BUILT FOR TESTS ALONE: it may hide nothing, and no verifier sees that");
  const std::string whole = std::string(stand == live ? "is complete" : "is NOT complete") +
                            (a_tests         ? std::string(stand == live ? ", BUT " : ", AND ") + a_tests + "; it is for tests alone and is NOT to be handed out"
                             : stand != live        ? ": it is handed out only when all of them stand, and no call of this program can bring that about: for a commitment to hand out, make a new one, whole"
                             : O.keep_for_the_audit ? ", AND IS HANDED OUT ONLY WHEN THE AUDIT WITH THE KEY HAS SAID YES: until then nothing of this commitment leaves, not its ID either (this run's own output holds it: keep that, and every log of it, on this machine), and no cp_prove is started on the secret folder "
                                                      "(docs/operators-duties.md, \"The audit with the key\")"
                             : frame.public_seed    ? ", and hides nothing: it was made with -public-weights (isk.txt says \"seed public\"), and with this option the weights are NOT hidden: whoever has the 32-byte file can work out every mask. "
                                                      "The public files carry no mark of it: key, entries, ID and proofs look like any other commitment's. Never use this option on weights that are to stay hidden"
                                                    : " and can be handed out");
  // Behind "can be handed out", if read_shape() has printed its note on a record.
  const char *const noted = !a_tests && stand == live && !O.weights_are_made_up() && !O.public_seed ? ". It holds a record (see the note above)" : "";
  const std::string secret = frame.public_seed ? std::string() : ". " + O.isk + " is secret"; // under -public-weights nothing in that folder is secret
  // Printed in a test run, so that the test can tell that the fill byte was used.
  SERVING_OR(, if (blk::card_test_byte >= 0) printf("cp_commit: a test's byte %02x (LENT_POISON) was laid over the GPU engine's buffers %llu times, %llu bytes in all\n", (unsigned)blk::card_test_byte, blk::card_fills.load(), blk::card_bytes_filled.load()));
  if (!frame.first)
    printf("cp_commit: done. %u of the M = %u leaves stand, each with its entry in %s, which %s%s%s\n", stand, frame.M, O.ivk.c_str(), whole.c_str(),
           noted, secret.c_str());
  else
    printf("cp_commit: done. %u of the %u live leaves %u .. %u stand, each with its entry in %s, which %s%s%s\n", stand, live, frame.first,
           frame.M - 1, O.ivk.c_str(), whole.c_str(), noted, secret.c_str());
  return 0;
}
} // namespace hf

// The whole program: process settings, then the run.
// Returns the exit status; the caller calls _exit().
static int cp_commit_program(int argc, char **argv) {
  // Disables core dumps: a core file would contain the weights' integers and the keys.
  const struct rlimit none = {0, 0};
  if (setrlimit(RLIMIT_CORE, &none) || prctl(PR_SET_DUMPABLE, 0)) return 2;
  umask(077); // whatever this process, a library or cp_verify, which it starts, creates is for its owner alone. The program's own files and directories get their modes at their creation
  if (argc == 2 && !strcmp(argv[1], "--setting")) {
    // dw: a digest's length in 32-bit words, printed so that a script can read it.
    const bool said = printf("setting: cp_commit" SERVING_OR("; a serving build", "") BUILT_SETTING "; dw=%d\n", DW) > 0 && !fflush(stdout);
    _exit(said ? 0 : 2);
  }
  setvbuf(stdout, nullptr, _IOLBF, 0); // a run takes hours, and its log is read while it runs: every line goes out at once
  // A failed check (`guard` in identifiers) in the proof library (blocks/) or the GPU engine
  // (prover/engine/) prints its own line (file, line number, the check's text or CUDA's error) and
  // ends the process by abort(). In between, this callback prints a line with a name that a script
  // can read: the check's own or, if it has none, "failed-check". It returns, and abort() follows
  // (blocks/base.h, end_at_a_failed_guard()). Once making::begin() has installed the SIGABRT
  // handler, that handler removes what the call has created. CardArena::die()
  // (prover/engine/arena.h) calls abort() directly, without this line.
  blk::at_a_failed_guard = [](const char *name) noexcept { fprintf(stderr, "cp_commit: internal error (%s): a check inside the program has failed. The program or the machine has a fault; nothing of this run may be used\n", name ? name : "failed-check"); };
  g_st = cudaStreamPerThread;
  const int rc = hf::cp_commit_main(argc, argv);
  hf::making::leave(); // the run's cleanup: the calling code's _exit() runs no exit handler
  hf::wipe_the_keys();
  return rc;
}

#ifndef CP_COMMIT_NO_MAIN // a test program includes this file and has a main of its own
#ifdef TEST_HOOKS
#error "TEST_HOOKS is only for test programs with a main of their own: never build cp_commit's own main with it"
#endif
int main(int argc, char **argv) {
  // A process started as the supervisor of one run of cp_verify (prover/common/judge.h) ends here,
  // before main() reads anything or calls a CUDA function.
  if (const int status = judge::supervise_if_asked(argc, argv); status >= 0) return status;
  const int rc = cp_commit_program(argc, argv);
  fflush(stdout);
  fflush(stderr);
  _exit(rc); // the files are written and closed, so no destructor or exit handler needs to run
}
#endif
