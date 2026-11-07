// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The command line of cp_commit, the program that commits to a model's weights (struct Opt, with
// the usage text), and the values of the parameter file, policy.txt (struct PolC). A command line
// that Opt::parse() accepts can still be rejected by its caller, cp_commit_main()
// (prover/hash_commitment/hf_commit.cu). The includer supplies u32 and vector.
//
// Terms: `ivk` and `isk` are the public and the secret key directory ("folder" in messages);
// isk/isk.txt records the commitment's settings, and isk/seed.bin holds its seed, 32 secret bytes.
// A weight table is all the weights and random masks, encoded, under a Merkle tree. A commitment
// has a first one, the anchor, and M leaf tables ("leaf" in options; not tree leaves): further
// tables of the same weights, each with its own masks. The key file is ivk/key.bin; a leaf file
// ("entry" in messages) is ivk/leaf/<k>.bin.
#pragma once
#include "common/parameter_file.h"
#include "frame/hiding_conditions.h" // ../blocks
#include <algorithm>
#include <dirent.h>
#include <set>
#include <string>
#include <unistd.h>
namespace rf {
// The parameter file's values by name ("Pol" for policy.txt). operator[] exits with status 2 on a
// missing name. The prover program cp_prove uses PolC too (prover/hash_commitment/hf_keeper.cu).
struct PolC {
  std::map<std::string, u32> m;
  parameter_file::Parameters file; // the file as it was read
  std::string why;                 // why the file was rejected; empty if it was accepted
  u32 operator[](const char *n) const {
    auto it = m.find(n);
    if (it == m.end()) {
      fprintf(stderr, "the parameter file has no line %s\n", n);
      exit(2);
    }
    return it->second;
  }
  // Reads the file and checks its syntax and limits with load() (prover/common/parameter_file.h),
  // as cp_prove does. Returns false, after a line on stderr, if the file is rejected; m and file
  // are filled even if a limit fails. A production build (SERVING_BUILD; "serving build" in
  // messages) also rejects a file that misses one of the conditions that the arguments for hiding
  // put on it; only the part of them that can be tested on the file alone is tested here
  // (blocks/frame/hiding_conditions.h). In any other build PolC does not test them:
  // cp_commit_main() does, and accepts such a file only with -test-seed.
  bool read(const char *p) { return judged(parameter_file::load(p, file)); }
  // The same for bytes already in memory, so that a caller can parse exactly the bytes that it
  // hashes.
  bool take(const std::string &bytes) {
    const std::string why = parameter_file::parse(bytes, file);
    return judged(why.empty() ? parameter_file::limit_missed(file) : why);
  }
  // Checks the syntax only, not the limits or the conditions for hiding. For unit tests at toy
  // sizes, which the limits exclude; no program that commits or proves calls it.
  bool read_form(const char *p) { return taken(parameter_file::load_form(p, file)); }

private:
  // Adds a production build's check of the conditions for hiding.
  bool judged(std::string refusal) {
#ifdef SERVING_BUILD
    const std::string missed = refusal.empty() ? blk::hiding_conditions_missed_by_the_file(file.frame) : "";
    if (!missed.empty())
      refusal = parameter_file::refusal("prover-only-hiding-conditions", file.line_of("mbln"),
                                        "the file misses conditions for hiding (their names and what each asks: blocks/frame/hiding_conditions.h), which a "
                                        "serving build requires: " +
                                            missed);
#endif
    return taken(refusal);
  }
  bool taken(const std::string &refusal) {
    why = refusal;
    m = file.m;
    if (!why.empty()) fprintf(stderr, "the parameter file is not admitted: %s\n", why.c_str());
    return why.empty();
  }
};
static void app(vector<u32> &w, const u32 *p, size_t n) {
  w.insert(w.end(), p, p + n);
}
// cp_commit's command line.
struct Opt {
  // The first form (-weights is given) creates a commitment, in two new directories. The second,
  // for work on an existing commitment, needs at least one of -leaves, -M and -first: parse()
  // accepts it, but cp_commit_main() rejects it (`cp-commit-first-form-alone`); usage() omits it.
  bool from_weights = false;
  std::string ivk, isk, config;
  // -M: the number of leaf tables. -first (second form only): the first one that cp_prove still
  // uses; those below it are retired for good. Where the second form omits either, the caller
  // fills in isk.txt's number.
  uint32_t M = 0, first = 0;
  bool M_given = false, first_given = false;
  // -leaves a..b: create only the leaf tables a to b - 1; "none" (first form only): create none.
  // Without -leaves the first form creates all M; the second form's range is set by the caller.
  uint32_t leaf_from = 0, leaf_to = 0;
  bool leaves_given = false;
  // -sweep (second form only): delete what an interrupted run left: the unfinished leaf tables
  // (isk/leaf/<k>.making) in this run's range, and every file isk/isk.txt.making.<digits>. A leaf
  // table that a running cp_commit has locked is not deleted, and the run stops there (hold_leaf(),
  // prover/hash_commitment/hf_commit_making.h); for the isk.txt.making files no lock is tested.
  // Without -sweep the run stops at any unfinished leaf table in its range.
  bool sweep = false;
  // -weights: "generated" (the tests' generator), "made-up-blocks" (synthetic weights stored as
  // quantized blocks, for any size) or a directory with wsums.txt, fold.txt and the model's GGUF
  // files, here as an absolute path. cp_commit_main() stores the value in isk.txt.
  std::string weights;
  bool generator = false, made_up_blocks = false;
  const char *fold = nullptr; // the directory; its wsums.txt and fold.txt govern the conversion
  vector<std::string> shards; // its *.gguf files, as full paths, sorted by name
  // Whether the weights arrive tensor by tensor and are converted to integers.
  bool by_tensors() const { return made_up_blocks || !shards.empty(); }
  // Whether to run the worst-case check (prover/weights_commitment/worst_case.cuh): at the sites of
  // the forward pass that the check bounds, can any prompt drive a value out of its range? If the
  // check fails, cp_commit exits with status 7, before the seed is drawn. Every run of the first
  // form runs it, except a run under which nothing is hidden anyway -- a test run with -test-seed,
  // or a production run with -public-weights -- that gives neither -worst-case nor
  // -worst-case-table. The check is only a hiding safeguard (it stops a "failed" that would be a bit
  // about the weights at a prompt of the client's choice); no verifier check depends on it, so a run
  // whose weights are published may leave it out. When it is left out, isk.txt gets no line ranges,
  // and cp_prove under -public-weights starts without one (Prover::the_ranges_are_guarded()).
  bool worst_case = false;
  // -worst-case-table (first form only): a new file for the check's full table, readable by its
  // owner only, because its numbers are computed from the weights. Empty: not given.
  std::string worst_case_table;
  // How the work uses GPU memory. None of these options changes a byte of an output file.
  // A table's Merkle tree is built in parts of 2^lp positions (-part-bits <lp>), n of them side by
  // side in one pass over the rows (-parts-at-once <n>). -rows-in-cuts reads the stored tensors in
  // chunks ("cuts") even where the whole committed matrix would fit in GPU memory.
  // -gpu-memory caps three things, each separately: the buffers for building a weight table, the
  // range proof's pool of packed weights, and the arena (`HfOnceSlab`) of the code that writes the
  // key file or a leaf file. It does not cap the proof engine's memory beside that arena
  // (prover/hash_commitment/hf_commit_once.cuh).
  size_t piece_bytes = (size_t)1 << 30; // -piece-bytes: GPU memory for one tensor chunk's integers
  int part_bits = -1;    // -part-bits; -1: chosen by the program
  int parts_at_once = 0; // -parts-at-once; 0: chosen by the program
  size_t gpu_memory = 0; // -gpu-memory <GiB>, here in bytes; 0: no cap
  bool rows_in_cuts = false; // -rows-in-cuts
  // -huge-pages: ask Linux for huge pages for the host copy of the weights and for the buffers of
  // the check against wsums.txt (touched_host_memory(),
  // prover/hash_commitment/hf_commit_weights.cuh). Off by default: prover/prover/arguments.h says
  // why. It too changes no byte of an output file.
  bool huge_pages = false;
  // -keep (first form only): the level s of each Merkle tree, counted up from the tree leaves,
  // whose nodes are stored in hcnodes.bin; recorded in isk.txt. -1: the default, the lower of
  // `lcl` (the parameter file's lkW - fb) and keep_level_most(), the highest level allowed: no
  // opening's Merkle paths end below it (both in prover/hash_commitment/hf_table_params.h).
  int keep_level = -1;
  // -keep-for-the-audit (first form, every build): RunKeys::keep()
  // (prover/hash_commitment/hf_commit_keys.h) writes into <isk>/audit the 32 fresh system bytes
  // from which, with the seed, the key file's private randomness is derived, and those of each
  // leaf file, for one audit, described in docs/operators-duties.md, "The audit with the key".
  // Not with -leaves: the audit covers the key file and all leaf files, and a leaf file created
  // later could not be audited before the commitment is released. cp_commit_main() rejects the
  // option on a model's files. No byte of any other file depends on it.
  bool keep_for_the_audit = false;
  // -public-weights <file of 32 bytes> (first form only), for weights that are public
  // (prover/common/public_weights.h). The seed is derived from the file's bytes, and no fresh
  // system bytes go into any file. So the same command on the same files gives the same commitment,
  // and anyone who has the file can compute every mask: nothing is hidden, and what other comments
  // here say of a secret seed or directory does not hold. isk.txt then has the line "seed public".
  // On a model's files the run also needs a marker file next to each of them
  // (prover/hash_commitment/public_weights_markers.h). Not with -keep-for-the-audit: no fresh bytes
  // are drawn, so there is nothing to keep or to audit. Everything else is as in any other run.
  // Only a production build reads the option; in every other build `public_seed` stays null.
  const char *public_seed = nullptr;
  // Test options; a production build has none of them.
  //   -test-seed: a file of 32 bytes. The first form takes them as its seed; the second form
  //     compares them with isk/seed.bin. No fresh system bytes are then mixed in, so that the same
  //     command writes the same files. Nothing is hidden in such a run.
  //   -test-bits <n> (`asked`): cp_commit's runs of the verifier program cp_verify request n bits,
  //     fewer than the target, BAR (prover/common/setting.h). A part-call of cp_verify (-key,
  //     -leaf) then counts as passed at exit status 14, `NOT-A-VERIFICATION`, instead of 3 or 11.
  //     For toy configurations.
  const char *test_seed = nullptr;
  int asked = BAR; // -test-bits
  // Whether the weights are synthetic ("made up" in names and messages): -weights was "generated"
  // or "made-up-blocks", and no *.gguf file is listed. cp_commit_main() goes by this answer where
  // it rejects a model's files, and the loader, PinnedWeights::load()
  // (prover/hash_commitment/hf_commit_weights.cuh), opens files only if `shards` is not empty:
  // testing both here keeps the two from disagreeing. Any directory counts as a model's files,
  // whoever wrote them. In the second form the answer is valid only after the caller's
  // take_weights().
  bool weights_are_made_up() const { return (generator || made_up_blocks) && shards.empty(); }

  static void usage() {
    fprintf(
        stderr,
        "usage: cp_commit -weights <generated | made-up-blocks | folder> -config <folder with policy.txt, KL.bin, layout.txt> "
        "-M <leaves> -ivk <new folder> -isk <new folder> " SERVING_OR("", "[-leaves <a..b | none>] ") "[-keep <s>] [-worst-case-table <new file>] [-keep-for-the-audit]" SERVING_OR(" [-public-weights <file>]", "") "\n"
        "The call draws the seed and makes, in two new folders, the anchor with its one-time files and then the M leaves. No call changes a commitment that stands.\n"
        "ivk is public, and complete when key.bin and all M entries stand; isk is secret. cp_verify must lie in the same folder "
        "as this program.\n"
        "A folder of weights holds wsums.txt, fold.txt and the model's *.gguf files.\n"
        "The worst-case check: the call ends with status 7, before the seed is drawn, unless at none of the sites of the forward "
        "pass that the check bounds a prompt can drive the value out of its range. When the check has passed, isk.txt gets the line ranges, without which "
        "cp_prove does not start. "
        SERVING_OR("Under -public-weights the check is left out (nothing is hidden, so it only stops a \"failed\" that would reveal nothing new): isk.txt then gets no "
                   "line ranges, and cp_prove, given -public-weights, starts without one; -worst-case or -worst-case-table still forces the check. ", "")
        "-worst-case-table: the check's table, a line for each site and layer, goes into a new file for "
        "its owner only; its numbers are computed from the weights, so keep it as secret as isk. -worst-case: "
        SERVING_OR("makes the check under -public-weights too; elsewhere it changes nothing", "makes the check under -test-seed too; elsewhere it changes nothing") ".\n"
        "-keep-for-the-audit (not with -leaves): the 32 fresh bytes of the key file's making and of each entry's are kept in <isk>/audit, for the audit with the key, which destroys them before it says yes. Together with the seed they "
        "open every mask in key.bin and the entries: until that yes nothing of the commitment leaves, and no cp_prove is started on isk (docs/operators-duties.md, \"The audit with the key\").\n"
        SERVING_OR("-public-weights <file of 32 bytes>: for public weights only: with this option the weights are NOT hidden: whoever has the 32-byte file can work out every mask. "
                   "The seed is derived from the file and no fresh randomness goes into any file, so the same call on the same files gives the same commitment. "
                   "The public files carry no mark of it: key, entries, ID and proofs look like any other commitment's. isk.txt then ends in \"seed public\", and every call of cp_prove on the "
                   "commitment needs the option with the same file. Nothing has been argued on whether the requests of one batch stay hidden from one another under such a commitment; "
                   "serve other people's requests under it one request a batch, or not at all.\n"
                   "    On a model's files the call also needs, next to each file *.gguf, a file of the same name with .weights-are-public appended. It contains the model file's SHA-256 as "
                   "sha256sum prints it (the command that creates it: docs/operators-duties.md, \"Keys, seeds and machines\", \"Declaring a model file's weights public\"). Creating it declares that this file's weights are public.\n"
                   "    The call computes each model file's SHA-256 and exits with an error if it differs from the declared one. It does not detect a file that is modified in place while it "
                   "runs. cp_prove does not read the .weights-are-public files.\n"
                   "    The worst-case check is left out under this option, because nothing is hidden: it only guards against a \"failed\" that would reveal a bit about hidden weights, and here the "
                   "weights are published. isk.txt then gets no line ranges, and cp_prove, given -public-weights, starts without one; a request may still end \"failed\" if a value leaves its range. "
                   "Give -worst-case (or -worst-case-table) with -public-weights to run the check anyway: then a pass writes the line ranges as usual.\n"
                   "    What this text says of a drawn seed and of a secret isk does not hold then. The option does not go with -keep-for-the-audit. Never use this option on weights that are to "
                   "stay hidden.\n",
                   "")
        "A call works on one GPU, the first that it sees (choose it by CUDA_VISIBLE_DEVICES), and uses nearly all of its memory: run nothing else on that GPU.\n"
        "optionally, without effect on any file: [-part-bits <lp>]; [-parts-at-once <n>]; [-rows-in-cuts]; [-gpu-memory <GiB>]; "
        "[-piece-bytes <n>]\n"
        "-huge-pages, also without effect on any file: ask Linux for huge pages for the host copy of the weights. Without it the machine's own setting (/sys/kernel/mm/transparent_hugepage/enabled) decides. Off by default, because on a "
        "machine that is short of free 2 MiB blocks the request can make the start take far longer.\n"
#ifndef SERVING_BUILD
        "-leaves a..b, for tests: of the M leaves only a, a + 1, .., b - 1 are made (b itself is not among them), with none the anchor alone; no call can finish such a commitment\n"
        "for tests: -test-seed <file of 32 bytes>: the seed is read and not drawn, a link's coins take in no fresh bytes, and nothing is hidden; isk.txt then says "
        "\"seed test\", and the worst-case check is left out unless -worst-case or -worst-case-table is given; "
        "-test-bits <n>: cp_verify's part-calls ask for n bits, below the bar, and are no verification. "
        "On a folder of weights no option for tests is taken: with -weights both go with generated or made-up-blocks alone\n"
#endif
        "or:    cp_commit --setting: print whether this is a serving build and the words of a digest, and exit\n");
  }
  // A decimal whole number from 0 to `most`, or -1 for any other text.
  static long long number(const char *text, long long most) {
    char *end = nullptr;
    if (*text < '0' || *text > '9') return -1;
    const long long v = strtoll(text, &end, 10);
    return *end || v > most ? -1 : v;
  }
  // Parses the value of -weights, from the command line or from isk.txt. A directory's path is made
  // absolute, so that isk.txt does not depend on the working directory. Returns false, after a line
  // on stderr, if the directory cannot be opened, has no *.gguf file, or has a line feed in its
  // path.
  bool take_weights(const std::string &word) {
    weights = word;
    generator = word == "generated", made_up_blocks = word == "made-up-blocks";
    if (generator || made_up_blocks) return true;
    if (char *const whole = realpath(word.c_str(), nullptr)) weights = whole, free(whole);
    DIR *const list = weights.find('\n') == std::string::npos ? opendir(weights.c_str()) : nullptr;
    for (const struct dirent *entry; list && (entry = readdir(list));) {
      const std::string name = entry->d_name;
      if (name.size() > 5 && !name.compare(name.size() - 5, 5, ".gguf")) shards.push_back(weights + "/" + name);
    }
    if (list) closedir(list);
    std::sort(shards.begin(), shards.end());
    fold = weights.c_str();
    if (shards.empty()) fprintf(stderr, "cp_commit: -weights %s: not a folder with *.gguf files\n", weights.c_str());
    return !shards.empty();
  }
  // Parses the command line. Returns false, or exits with status 2 after a message, if it is
  // rejected. The caller still checks -leaves against M and `first`.
  bool parse(int argc, char **argv) {
    const char *weights_word = nullptr, *leaves = nullptr;
    long long leaves_M = -1, first_leaf = -1;
    std::set<std::string> seen;
    for (int i = 1; i < argc; i++) {
      const std::string a = argv[i];
      if (!seen.insert(a).second) {
        fprintf(stderr, "cp_commit: %s is given twice\n", a.c_str());
        return false;
      }
      auto val = [&]() -> const char * {
        if (i + 1 >= argc || !*argv[i + 1]) {
          fprintf(stderr, "cp_commit: %s requires a value\n", a.c_str());
          exit(2);
        }
        return argv[++i];
      };
      auto count = [&](long long most) {
        const long long v = number(val(), most);
        if (v < 0) {
          fprintf(stderr, "cp_commit: %s takes a whole number from 0 to %lld\n", a.c_str(), most);
          exit(2);
        }
        return v;
      };
      if (a == "-weights")
        weights_word = val();
      else if (a == "-config")
        config = val();
      else if (a == "-M")
        leaves_M = count(0xFFFFFFFFll);
      else if (a == "-first")
        first_leaf = count(0xFFFFFFFFll);
      else if (a == "-ivk")
        ivk = val();
      else if (a == "-isk")
        isk = val();
      else if (a == "-leaves")
        leaves = val();
      else if (a == "-sweep")
        sweep = true;
      else if (a == "-keep")
        keep_level = (int)count(33);
      else if (a == "-worst-case")
        worst_case = true; // the option was given; the final value is set below
      else if (a == "-worst-case-table")
        worst_case_table = val();
      else if (a == "-piece-bytes")
        piece_bytes = (size_t)count((long long)1 << 40);
      else if (a == "-part-bits")
        part_bits = (int)count(30);
      else if (a == "-parts-at-once")
        parts_at_once = (int)count(8);
      else if (a == "-gpu-memory")
        gpu_memory = (size_t)count(1024) << 30;
      else if (a == "-huge-pages")
        huge_pages = true;
      else if (a == "-rows-in-cuts")
        rows_in_cuts = true;
      else if (a == "-keep-for-the-audit")
        keep_for_the_audit = true;
#ifdef SERVING_BUILD
      else if (a == "-public-weights")
        public_seed = val();
#endif
#ifndef SERVING_BUILD
      else if (a == "-test-seed")
        test_seed = val();
      else if (a == "-test-bits")
        asked = (int)count(BAR - 1); // below the bar: so "asked < BAR" says that the option was given
#endif
      else
        return false;
    }
    for (std::string *folder : {&ivk, &isk, &config}) // without trailing slashes
      while (folder->size() > 1 && folder->back() == '/') folder->pop_back();
    if (ivk.empty() || isk.empty()) return false;
    // Makes -ivk, -isk and -worst-case-table absolute: the program prints these names, for example
    // in the rejection that names what an interrupted run left and says that it can be removed, and
    // such a line may be read later, in another working directory, where a relative name means
    // something else. An absolute name stays as given: no symbolic link in it is resolved.
    for (std::string *name : {&ivk, &isk, &worst_case_table})
      if (!name->empty() && name->front() != '/') {
        char *const here = getcwd(nullptr, 0); // with these arguments glibc allocates the buffer
        if (!here) return fprintf(stderr, "cp_commit: %s is a relative name, and the working folder's own name cannot be read\n", name->c_str()), false;
        *name = std::string(here) + (here[1] ? "/" : "") + *name, free(here);
      }
    from_weights = weights_word != nullptr;
    M_given = leaves_M >= 0, first_given = first_leaf >= 0, leaves_given = leaves != nullptr;
    if (from_weights ? config.empty() || leaves_M < 1 || first_given || sweep
                   : !config.empty() || !(leaves_given || M_given || first_given) || keep_level >= 0 || worst_case || !worst_case_table.empty() || keep_for_the_audit || public_seed)
      return false;
    worst_case = from_weights && ((!test_seed && !public_seed) || worst_case || !worst_case_table.empty());
    M = M_given ? (uint32_t)leaves_M : 0, first = first_given ? (uint32_t)first_leaf : 0;
    if (leaves && strcmp(leaves, "none")) {
      const char *const dots = strstr(leaves, "..");
      const long long from = dots ? number(std::string(leaves, dots).c_str(), 0xFFFFFFFFll) : -1,
                      to = dots ? number(dots + 2, 0xFFFFFFFFll) : -1;
      if (from < 0 || to < from) return false;
      leaf_from = (uint32_t)from, leaf_to = (uint32_t)to;
    } else if (leaves && !from_weights)
      return false;
    if (!leaves && from_weights) leaf_from = 0, leaf_to = M;
    // A production build creates all M leaf tables: the second form is rejected, so no run could
    // finish a commitment that has only some of them.
    if (SERVING_OR(from_weights && leaves_given, false)) return false;
    if (keep_for_the_audit && public_seed) return false; // see `public_seed`
    if (keep_for_the_audit && leaves_given) return false; // see `keep_for_the_audit`
    return !from_weights || take_weights(weights_word);
  }
};
} // namespace rf
