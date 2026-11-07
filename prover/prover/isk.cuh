// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// How the prover program cp_prove uses isk, a commitment's secret key directory: start-up checks
// and a batch's start, which picks a leaf table and derives the batch's keys. A leaf table (`leaf`,
// `copy` in names and output) is a weight table (`Index` in names) with its own masks; isk.txt's
// `first` to `M` - 1 are live. cp_prove writes no file of isk but a leaf table's count.bin, whose
// length counts opened columns, and, with -keep-for-the-audit, audit/batch.fresh. [x]: a message's
// printed name. int results: 0, or nonzero after a message (read_the_seed(): one case without).
#pragma once
#include "common/coin_keys.h"          // the key schedule of the private randomness (`coins`)
#include "common/cp_files.h"
#include "common/cp_keys_own.h"        // [own-test-coins], in three parts
#include "common/public_weights.h"     // the option -public-weights
#include "hiding/coins_own.cuh"
#include "engine/stream_coins.cuh"    // `coins_own_test_card`: one generator's self-check
#include "hash_commitment/hf_format.h" // the version word; `HF_STOP_KEY_FILE`, `HF_STOP_KEY_TEST`
#include "hash_commitment/hc_format.h" // the seeds that are written in the source
#include "hash/hash.h"                // ../blocks: the hash calls
#include "frame/check_entry.h"        // ../blocks: read_entry_table_unproved()
#include "prover/batch_keys_held.h"   // [own-test-batch-keys]: both batch keys derived a second way
#include "prover/chain_coins.cuh"
#include "prover/opening_in_process.cuh"
#include "prover/prover.h"
#include <algorithm>
#include <cerrno>
#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

// Opens isk (isk_held), locks it until the process ends and reads isk.txt [cp-isk], checks that
// three files of each live leaf table are readable [cp-isk-leaves] and that isk/audit holds nothing
// [cp-kept-for-the-audit], rejects options and test environment variables that isk.txt does not
// allow, and fills in the fields of args that come from isk. Leaf tables below `first` are not
// looked at. isk.txt is not read again: a later change counts from the next start. Call it after
// args.parse().
inline int Prover::read_isk() {
  // isk is opened once; isk_held (cp::SecretFolder) keeps the descriptor until the process ends.
  // What cp_prove writes in isk, a leaf table's count.bin and audit/batch.fresh, is reached from it
  // and so goes into the directory checked here, even if another directory or a symbolic link is
  // later put at its path. Also read from it: isk.txt, seed.bin, the names in isk/audit and, at a
  // batch's start, fresh.bin and entry.bin. Still reached by path: the configuration files, the
  // three files checked below and hcnodes.bin at a batch's start.
  // The path is walked as a spool directory's path is (prover/prover/spool.h,
  // Spool::open_by_parts()): part by part, and no part may be a symbolic link. Each directory in
  // which a part is looked up must belong to root or to cp_prove's own account (for root under
  // sudo, also to the account that typed sudo: Spool::typed_sudo). Group and others must have no
  // write permission in it, unless it is sticky, as /tmp is, and the part found in it also has one
  // of these owners. So no other account can have put another directory at a name on the way. What
  // an account that the walk trusts has put there is accepted, even an older copy of isk, with its
  // older count (docs/operators-duties.md, "Backups and restored copies"): no program can tell a
  // copy from the original. A relative path is rejected, as a spool directory's is. It would start
  // at the working directory, and how the user got there cannot be checked: a shell's cd follows
  // links, so after "cd <another account's link>" the working directory is one of that account's
  // choice. That can be another directory of the user's own, closed to others, which holds another
  // secret key directory or an older copy of this one.
  if (args.isk[0] != '/') {
    printf("refused (cp-isk): the name given with -isk does not begin with /. Name the secret folder by its full path from /, for instance \"$PWD/isk\": a relative name starts at the working directory, and cp_prove cannot tell whether a link led you there (cd follows links)\n");
    return refused("cp-isk");
  }
  std::string on_the_way;
  const int opened = Spool::open_by_parts(args.isk, on_the_way);
  if (opened < 0 && !on_the_way.empty()) {
    for (const auto &[theirs, here] : {std::pair<std::string, std::string>{"-spool or -out", "-isk"}, {"run this as the user who ran cp_prove", "run cp_prove as the account that ran cp_commit"}})
      for (size_t at; (at = on_the_way.find(theirs)) != std::string::npos;) on_the_way.replace(at, theirs.size(), here); // the routine's advice is worded for other callers
    printf("refused (cp-isk): the path named with -isk is refused, and nothing was touched: %s. Keep the secret folder below directories in which only you or root can write, and name it without symbolic links\n", on_the_way.c_str());
    return refused("cp-isk");
  }
  if (!isk_held.take(opened)) {
    const int why = errno;
    printf("refused (cp-isk): the folder named with -isk %s%s. cp_commit makes it with mode 0700 for the account that runs it; nobody else may write into it or put another folder in its place\n",
           why ? "cannot be opened as a folder that is no symbolic link (if it is a link, name the folder itself): " : "is not this account's own, or has a permission for group or others (ls -ld shows both; run cp_prove as the account that ran cp_commit; if the mode was changed, find out by whom before you set it back)", why ? strerror(why) : "");
    return refused("cp-isk");
  }
  // One cp_prove at a time runs on a secret key directory, as a service or for one batch: an
  // exclusive flock() on the descriptor just accepted, taken without waiting. This function has
  // read and written nothing in isk before it. (A build without the restricted output mode has read
  // isk.txt once, by its path: second_start_unless_nothing_is_hidden(), prover/prover/cp_prove.cu.)
  // Nothing unlocks or closes the descriptor, so the lock lasts until the process ends. The system
  // then drops it, however the process ends, also after SIGKILL: there is no lock file, and nothing
  // to clean up. The descriptor is close-on-exec (Spool::open_by_parts()).
  // The lock is on the directory, not on its name. The call of cp_commit that creates a secret key
  // directory holds the same lock on it until that call ends (prover/hash_commitment/hf_commit.cu,
  // "The run's two locks"), so until then cp_prove does not start on it either.
  // Who does not take this lock, and so runs beside a running cp_prove: cp_prove -clear, which
  // opens no isk; the audits (tools/audit). cp_commit rejects a call on a commitment that exists.
  // Not detected: a process on a copy of the directory, or one on another machine, unless the file
  // system applies flock() across machines.
  // If -spool names this same directory by mistake, the spool directory's own lock (Spool::open(),
  // prover/prover/spool.h) then fails against this one, and its message speaks of another cp_prove.
  if (flock(isk_held.fd, LOCK_EX | LOCK_NB)) {
    const int why = errno;
    if (why == EWOULDBLOCK)
      printf("refused (cp-isk): the folder named with -isk is in use: another cp_prove runs on it, the call of cp_commit that creates it has not ended, or another program of yours holds a lock on the folder. Only one cp_prove at a time runs on a secret folder, "
             "whether it serves a spool directory or proves one batch (-in, -out). This call has changed nothing in the folder. Wait until the other process has ended, or stop that service "
             "(docs/service.md, \"How the service ends\"), and start again. There is no lock file: remove nothing, and do not copy the folder to get past this message: a copy counts for itself, and "
             "this lock does not see it (docs/operators-duties.md, \"Backups and restored copies\")\n");
    else
      printf("refused (cp-isk): the folder named with -isk cannot be locked: %s. This call has changed nothing in the folder. cp_prove needs flock() on this folder and on its subfolder leaf, which some "
             "shared and network file systems do not support for a directory (docs/operators-duties.md, \"Keys, seeds and machines\"). Do not copy or move the folder to get past this message. If it cannot be "
             "locked where it is, make a new commitment on a local disk (docs/operators-duties.md, \"A new commitment\")\n",
             strerror(why));
    return refused("cp-isk");
  }
  cp::IskText text;
  if (!cp::read_isk_text(isk_held, text) || !text.M) {
    printf("refused (cp-isk): isk.txt in the folder named with -isk cannot be read or is not what cp_commit writes. First check that -isk names the right folder. Do not repair the folder from a backup or a copy: a restored folder has an older count (docs/operators-duties.md, \"Backups and restored copies\"). If it cannot be used as it is, make a new commitment\n");
    return refused("cp-isk");
  }
  if (!hf_version_admitted(text.version)) {
    printf("refused (cp-isk): isk is of another build or format: its version word is %u, this program's is %u\n", text.version,
           HF_VERSION_WORD);
    return refused("cp-isk");
  }
  isk_version_word = text.version; // `read_inputs` compares it with policy.txt's version word
  isk_leaves = text.M, isk_first = text.first, isk_keep = text.keep, isk_ranges = text.ranges, isk_test_seed = text.test_seed;
#ifdef SERVING_BUILD
  // With a test seed's file anyone can compute every mask; the worst-case check may be missing.
  if (text.test_seed) {
    printf("refused (cp-isk-seed-test): isk.txt says \"seed test\". If cp_commit made this commitment without a test's seed, it did not write that line: "
           "somebody else writes into isk. Do not strike the line: switch off whatever restarts cp_prove, the model server and whatever calls cp_commit by itself, "
           "then stop every cp_prove on this isk and wait until "
           "each has ended, make no further call of cp_commit or cp_prove on it, and go by docs/operators-duties.md, \"If the count is lost or in doubt\", "
           "before anything else. The line stands on a commitment that was made for tests, under a seed that was handed in; a serving build does not serve it\n");
    isk_fault = "cp-isk-seed-test";
    name_the_end(isk_fault); // for the last line
    return 2;
  }
#else
  if (!args.test_coins.empty() && !text.test_seed) {
    printf("refused (cp-test-coins): -test-coins is taken only where isk.txt says \"seed test\", as it does on a commitment that cp_commit made with "
           "-test-seed: with -test-coins nothing is hidden from whoever has the coins' file, and one file used twice repeats the coins. For a test, "
           "make a commitment with -test-seed. No key protects that line, and whoever can write isk.txt can add it: never add it to a commitment "
           "that was made without -test-seed, where it would also take away other checks, among them that of isk.txt's line ranges\n");
    return refused("cp-test-coins");
  }
  if (args.test_leaf >= 0 && (args.test_leaf < (long long)isk_first || args.test_leaf >= (long long)isk_leaves)) {
    printf("-test-leaf: the commitment's live leaves are %u to %u\n", isk_first, isk_leaves - 1);
    return 2;
  }
#endif
  // A commitment to public weights (isk.txt's line "seed public") hides nothing. It is served only
  // if that line and the option -public-weights are both present, so that no single mistake serves
  // one. Only a production build ("serving" in names and output) has the option; every other build
  // serves no such commitment. No key protects the line.
  if (text.public_seed && args.public_weights.empty()) {
    printf("refused (cp-isk-seed-public): isk.txt says \"seed public\". cp_commit writes that line only "
           SERVING_OR("with -public-weights, and such a commitment hides nothing. To serve it, give -public-weights with the file that it was created with. If you created this commitment without that option, ",
                      "for a commitment to public weights, which hides nothing. This build does not serve such a commitment; the serving build (./build.sh without an argument) does, with -public-weights. If you did not create it so, ")
           "someone else writes into isk. Then do not remove the line: switch off whatever restarts cp_prove, the model server and whatever calls cp_commit by itself, then stop every cp_prove on this isk "
           "and wait until each has ended, make no further call of cp_commit or cp_prove on it, and go by docs/operators-duties.md, \"If the count is lost or in doubt\", before anything else\n");
    return refused("cp-isk-seed-public");
  }
  if (!text.public_seed && !args.public_weights.empty()) {
    printf("refused (cp-isk-seed-public): -public-weights is given, but isk.txt does not say \"seed public\": the option is accepted only for a commitment that cp_commit created with it. Check -isk, or "
           "leave the option out\n");
    return refused("cp-isk-seed-public");
  }
  if (!args.public_weights.empty()) {
    // An early check of the option's file, while a message can still be printed. read_the_seed()
    // reads it again and compares the derived seed with seed.bin, but after closed_front.close()
    // (in Prover::run()), when only the rejection's name is printed.
    uint8_t derived[32];
    const bool read = public_weights::seed_of_the_file(args.public_weights.c_str(), derived);
    explicit_bzero(derived, sizeof derived);
    if (!read) {
      printf("refused (cp-isk-seed-public): -public-weights %s is not a readable file of 32 bytes\n", args.public_weights.c_str());
      return refused("cp-isk-seed-public");
    }
  }
  // hcnodes.bin: stored tree nodes. fresh.bin: 32 bytes drawn with the table. entry.bin, the leaf
  // file: the table's root, read by every batch, and the LINK proof that ties it to the anchor (the
  // first weight table).
  for (uint32_t k = isk_first; k < isk_leaves; k++)
    for (const std::string &path : {cp::isk_leaf_nodes(args.isk, k), cp::isk_leaf_fresh(args.isk, k), cp::isk_leaf_entry(args.isk, k)}) {
      struct stat is;
      if (stat(path.c_str(), &is) || !S_ISREG(is.st_mode) || access(path.c_str(), R_OK)) {
        printf("refused (cp-isk-leaves): leaf %u of the live leaves %u to %u has no readable %s: the program starts only on a whole isk. "
               "Do not repair it from a backup or a copy: a restored folder has an older count (docs/operators-duties.md, \"Backups and restored copies\"). Make a new commitment\n",
               k, isk_first, isk_leaves - 1, path.c_str());
        return refused("cp-isk-leaves");
      }
    }
  // Checked in every build. With -keep-for-the-audit, cp_commit stores in isk/audit the 32 bytes
  // from the operating system's generator that went into the key file (key.bin) and into each leaf
  // file. With the seed they reveal every mask in those files. The audit (tools/audit) destroys
  // them before it answers YES in a run with --erase-kept-files; no other program enforces that no
  // prover starts before that YES. cp_prove's option of the same name stores one batch's 32 bytes
  // there; with the seed they reveal all private randomness of that batch's proofs. The batch's
  // audit erases them at every verdict, not only at a YES (when it does not erase them: tools/audit/audit_batch.sh). This check shows only that nothing is
  // left there, not that an audit answered YES. The message names no file to remove: removing
  // them is the audit's task.
  if (cp::audit_is_owed(isk_held)) {
    printf("refused (cp-kept-for-the-audit): something lies in the folder audit of the secret folder (or that folder cannot be listed). If this commitment was made with cp_commit's -keep-for-the-audit, the audit with the key "
           "has not yet said YES in a run with --erase-kept-files. Until it has, nothing of the commitment may leave and no proof is made on it. If an earlier call of cp_prove ran with -keep-for-the-audit, that batch's kept bytes may still be there: its audit (tools/audit/audit_batch.sh with --erase-kept-files) erases them "
           "with its verdict, if its last line says so; with --rehearsal it erases nothing. None of that batch's responses may leave unless it has said YES. If audit is a file or a link, or cannot be listed, no audit changes that: the secret folder is then not as the programs "
           "left it: find out who changed it before anything else is done with it. Otherwise run the audit that applies (tools/audit/README.md): at its YES it destroys what is kept there; at any other answer see the table in that page\n");
    return refused("cp-kept-for-the-audit");
  }
  // The configuration files. layout.txt is read from the same directory as policy.txt.
  args.policy = cp::isk_config(args.isk) + "/policy.txt", args.layer_words = cp::isk_config(args.isk) + "/KL.bin";
  // isk.txt's line `weights`: "generated" or "made-up-blocks" (synthetic weights; printed:
  // `made-up`), or the absolute path of the directory given to the committer program cp_commit,
  // with wsums.txt, fold.txt and the model's *.gguf files, sorted by name as in its take_weights().
  args.weights = text.weights == "generated" ? "generator" : text.weights;
  if (!cp::made_up(text)) { // a directory, or rejected (the internal keyword "generator" too)
    if (DIR *const list = text.weights[0] == '/' ? opendir(text.weights.c_str()) : nullptr) {
      while (const struct dirent *const e = readdir(list)) {
        const std::string name = e->d_name;
        if (name.size() > 5 && !name.compare(name.size() - 5, 5, ".gguf")) args.shards.push_back(text.weights + "/" + name);
      }
      closedir(list);
    }
    std::sort(args.shards.begin(), args.shards.end());
    if (args.shards.empty()) {
      printf("refused (cp-weights): isk.txt's line weights is neither generated nor made-up-blocks nor, from the root, a "
             "folder that can be read and holds files *.gguf\n");
      return refused("cp-weights");
    }
    args.fold = text.weights, args.weights = "files";
  }
  // Three test variables output values derived from the weights (prover/common/made_up_alone.h):
  // they need synthetic weights and a test seed. Rejected here, before a model file, the seed or a
  // GPU is touched; read_inputs() requires only -test-coins. The code that reads them checks again.
  const bool weights_are_made_up = cp::made_up(text); // false: treated as a model's files
  made_up_alone::say(weights_are_made_up, isk_test_seed);
  const char *const this_run_has = isk_test_seed          ? "weights that are not made up"
                                   : weights_are_made_up ? "a seed that is no test's"
                                                         : "weights that are not made up and a seed that is no test's";
  if (TESTENV("CHAIN_PUT_VECTORS") && !made_up_under_a_tests_seed()) {
    printf("refused (test-put-vectors): CHAIN_PUT_VECTORS is for runs on made-up weights under isk.txt's \"seed test\" alone: this run has %s\n", this_run_has);
    return refused("test-put-vectors");
  }
#ifndef SERVING_BUILD // a serving build reads no test's variable and holds none's name
  // They output a request's tokens, top choices and logit gaps (file), and a layer group's maxima.
  for (const char *name : {"CHAIN_ROWS_OUT", "FORMF_STREAMS_REACH"})
    if (getenv(name) && !made_up_under_a_tests_seed()) {
      printf("refused (test-made-up-alone): %s is for runs on made-up weights under isk.txt's \"seed test\" alone: this run has %s\n", name, this_run_has);
      return refused("test-made-up-alone");
    }
#endif
  // On a model's files any test variable test_word_in_the_environment() finds is refused, whatever the seed: the three
  // above are known to reveal weight-derived values, another could, and such a run keeps its caller's whole environment
  // (cp_prove.cu). Cost: timing prints (use -measure), tuning by a variable, tools that enter through the environment.
  if (!weights_are_made_up)
    if (const char *word = test_word_in_the_environment()) {
      printf("refused (test-words-on-a-model): the environment holds %.*s, a test's variable, and isk.txt's weights are a model's files: test variables (timing prints and tuning among them) are for "
             "made-up weights alone; times have the option -measure\n",
             (int)strcspn(word, "="), word);
      return refused("test-words-on-a-model");
    }
  // -keep-for-the-audit is for a run made for an audit, on a directory of requests (-in, -out): a
  // service (-spool), which takes its batches from outside, never stores a batch's 32 bytes. On a
  // model's files the option is rejected for the same reason as cp_commit's option of that name:
  // the audit accepts only commitments to synthetic weights, so nothing would erase the bytes.
  if (args.keep_for_the_audit && !args.one_batch()) {
    printf("refused (cp-prove-kept-in-a-service): -keep-for-the-audit is only allowed with -in and -out: a service keeps no batch's random bytes\n");
    return refused("cp-prove-kept-in-a-service");
  }
  if (args.keep_for_the_audit && !weights_are_made_up) {
    printf("refused (cp-prove-kept-on-a-model): -keep-for-the-audit on a model's files: the audit with the key takes commitments to made-up weights alone, so nothing would ever erase the kept bytes. Use a commitment to made-up weights\n");
    return refused("cp-prove-kept-on-a-model");
  }
  // Overlapping batches are not served on a model's files, whatever the seed. The written
  // hiding argument does not cover them: while an earlier batch's opening (its test on
  // the weight table) is in progress, a failure that no request caused or an exit costs one or two
  // more batches, and a response's time shows more of a neighbour batch's size and arrival time. On
  // synthetic weights none of that can harm anybody, so there the options run under any seed, for tests and
  // measurements. Nothing in the theory speaks against overlap. The theory requires that
  // whether batches overlap depends on the schedule alone; of its nine conditions for that, three
  // rest on reading the sources alone and six are not established for these programs. So the
  // code below rejects overlapping batches on a model's files.
  if (!weights_are_made_up && args.overlap) {
    printf("refused (overlap-on-a-model): -overlap and -overlap-two run on made-up weights alone: no argument for weight hiding covers them. Leave the option out, or use a commitment to made-up weights\n");
    return refused("overlap-on-a-model");
  }
  // Only a production build takes a model's files: prover/common/setting.h.
#if BUILT_FOR_TESTS_ALONE
  if (!weights_are_made_up) {
    printf("refused (tests-build-on-a-model): this program was " BUILT_SAYS ": what it makes may hide nothing, and no verifier sees that. It serves commitments to made-up weights alone. For a model's files, build the programs with prover/build.sh, without an argument\n");
    return refused("tests-build-on-a-model");
  }
#endif
  // From here, isk.txt's "seed test" implies synthetic weights: the production build has rejected
  // the line, every other build a model's files. With the line, -test-coins (with it -test-leaf,
  // -test-no-requests, test variables, a seed from the source) is accepted; the line `ranges` is not
  // required; fresh.bin may be zeros; read_inputs() only prints unmet hiding conditions
  // (blk::configure() still rejects them).
  args.stream_weights = args.weights != "generator";
  if ((args.one_room_for_experts || args.experts_by_layer) && !args.stream_weights) {
    printf("-memory: one-room-for-experts, experts-by-layer and tensor-rooms require weights stored as blocks "
           "(made-up-blocks or a model's files)\n");
    return 2;
  }
  // One batch (-in): its requests are counted now; the set-up sizes memory by their number.
  if (args.one_batch()) {
    std::vector<std::string> names;
    std::string odd;
    if (!Spool::requests_in(args.in, names, odd) && !odd.empty()) {
      printf("refused (cp-in-name): the folder named with -in holds \"%s\": it may hold nothing but regular files <name>.req, "
             "<name> of 1 to 64 characters out of A-Z a-z 0-9 _ and -. Remove nothing because of this message: first check that -in names the folder that you mean.\n",
             odd.c_str());
      return refused("cp-in-name");
    }
    if (names.empty() || names.size() > (size_t)SLOTS_MOST) {
      printf("refused (cp-batch): the folder named with -in cannot be read, or does not hold 1 to %d files <name>.req\n",
             SLOTS_MOST);
      return refused("cp-batch");
    }
    args.batch = (int)names.size(), args.in_step = std::min(args.in_step, args.batch);
  }
#ifndef SERVING_BUILD
  // The file given with -test-coins.
  if (!args.test_coins.empty()) {
    struct stat is;
    if (stat(args.test_coins.c_str(), &is) || !S_ISREG(is.st_mode) || is.st_size < 32 || is.st_size % 32) {
      printf("-test-coins: the file must hold 32 bytes for each batch\n");
      return 2;
    }
    test_coins_batches = (long long)(is.st_size / 32);
  }
#endif
  return 0;
}

// [ranges-not-guarded] Checks isk.txt's line `ranges`. Whether a request's values leave a range
// depends on the weights, so no request may be able to cause that. cp_commit's worst-case check
// (prover/weights_commitment/worst_case.cuh) is meant to rule it out at the sites that it bounds
// (that the forward pass tests no other range is checked only on the sources' text, by a script
// outside this repository). When it passes, cp_commit writes the line with h_par, the digest of the
// three configuration files it ran with; under other files the line means nothing. No key protects
// isk.txt (docs/limits-all-rows.tsv, row H7.1). Under "seed test" (test builds only) the line is
// not required. A production serve of a commitment to public weights (-public-weights, which
// read_isk() has already paired with isk.txt's "seed public") also does without the line when
// cp_commit left the check out: nothing is hidden there, so a "failed" that depends on the weights
// reveals nothing a published-weights client could not compute itself; the line is still honoured
// when cp_commit ran -worst-case and wrote it. This safeguard is never a verifier check, and a
// hidden-weight commitment must not become provable without it. Call it after read_inputs(), before
// the spool directory is opened: a start under changed files then changes nothing there.
inline int Prover::the_ranges_are_guarded() {
#ifndef SERVING_BUILD
  if (isk_test_seed) {
    printf("# isk.txt's line ranges is not asked for (isk.txt says \"seed test\"): in this run \"failed\" may depend on the weights\n");
    return 0;
  }
#endif
#ifdef SERVING_BUILD
  // -public-weights has already been paired with isk.txt's "seed public" [cp-isk-seed-public]. If
  // cp_commit left the worst-case check out (no line ranges), serve without one: the weights are
  // published, so a weight-dependent "failed" hides nothing. A missing line without -public-weights
  // is not accepted here, below, so a hand-edited isk.txt does not slip a hidden-weight commitment
  // past the check. A commitment that also ran -worst-case still carries a real line, matched below.
  if (!args.public_weights.empty() && isk_ranges.empty()) {
    printf("# isk.txt's line ranges is not asked for (isk.txt says \"seed public\" and -public-weights is given): cp_commit left the worst-case check "
           "out under public weights, so a request may end \"failed\" if a value leaves its range; nothing is hidden under this commitment\n");
    return 0;
  }
#endif
  if (!isk_ranges.empty() && isk_ranges == cp::digest_hex(parameters_hash_words.data(), parameters_hash_words.size())) return 0;
  printf("refused (ranges-not-guarded): isk.txt has no line ranges for these configuration files. cp_commit's first form writes the "
         "line when its worst-case check has passed; its second form never adds it: make the commitment again\n");
  isk_fault = "ranges-not-guarded";
  name_the_end(isk_fault); // for the last line
  return 2;
}

// [own-test-coins] Three self-checks before the seed is read: the functions that derive a batch's
// keys from the seed, and two generators of private randomness, must give known answers under keys
// written in the source. The third exits by itself and runs its own kernel. Not detected, among
// others: a wrong key or counter at a request; a value unused or reused; another kernel's fault; a
// fault that arises after start-up. No verifier can see whether a private random value is what its
// key gives: the self-checks narrow that gap and do not show that anything is hidden.
inline int Prover::test_the_coins() {
  const bool keys_hold = cp::keys_own_test(), coins_hold = hz::coins_own_test() == 0;
  blk::coins_own_test_card(cudaStreamPerThread); // host and GPU; exits if an answer differs
  if (keys_hold && coins_hold) return 0;
  printf("refused (%s): the routines that make this program's secret random values and their keys fail their self-test: the "
         "program, its build or the machine is at fault. No seed was read and no request was taken. See prover/docs/ends-in-short.md\n",
         hz::OWN_TEST_COINS);
  return refused(hz::OWN_TEST_COINS);
}

// Reads seed.bin into isk_seed, where it stays until the process ends (wipe_the_seed()).
// [hc-key-file]: no symbolic link, 32 bytes, not all zero, and cp::own_and_closed(): a regular
// file, this account's own, under one name (no second hard link), no access for group or others.
// [hc-key-test]: without -test-coins, no seed written in the source. [cp-isk-seed-public]: with
// -public-weights, the seed derived from the option's file; that rejection has a name, no message.
// The checks of seed.bin use its descriptor, and its bytes pass through no C library buffer, which
// would be freed without being zeroed. Call after setup_opening(), which runs [own-test-generator].
inline int Prover::read_the_seed() {
  const int fd = openat(isk_held.fd, cp::ISK_SEED_FILE, O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK); // relative to the descriptor that `read_isk` opened
  struct stat is;
  uint8_t bytes[sizeof isk_seed + 1] = {0};
  bool good = fd >= 0 && !fstat(fd, &is) && cp::own_and_closed(is) && is.st_size == (off_t)sizeof isk_seed;
  size_t have = 0;
  while (good && have < sizeof bytes) {
    const ssize_t got = read(fd, bytes + have, sizeof bytes - have);
    if (got < 0 && errno == EINTR) continue;
    if (got <= 0) {
      good = got == 0;
      break;
    }
    have += (size_t)got;
  }
  if (fd >= 0) close(fd);
  uint8_t any = 0;
  for (size_t i = 0; i < sizeof isk_seed; i++) any |= bytes[i];
  good = good && have == sizeof isk_seed && any;
  if (good && !args.public_weights.empty()) {
    uint8_t derived[32] = {0}, differ = 0;
    const bool read = public_weights::seed_of_the_file(args.public_weights.c_str(), derived);
    for (int i = 0; i < 32; i++) differ |= derived[i] ^ bytes[i]; // every byte is compared, whatever the first ones say
    explicit_bzero(derived, sizeof derived);
    if (!read || differ) {
      explicit_bzero(bytes, sizeof bytes);
      return refused("cp-isk-seed-public"); // a serving build only: its output is closed by now
    }
  }
  // A seed written in the source is public [hc-key-test]. The comparison has no early exit.
  bool of_the_tests = false;
  for (int n = 0; n < HC_TEST_SEEDS; n++) {
    uint32_t differ = 0;
    for (int i = 0; i < 32; i++) differ |= (uint32_t)bytes[i] ^ (hc_test_seed_word(n, i / 4) >> 8 * (i % 4) & 255);
    of_the_tests |= !differ;
  }
  if (good && of_the_tests && args.test_coins.empty()) {
    explicit_bzero(bytes, sizeof bytes);
    printf("refused (%s): seed.bin holds a seed that is written in the source" SERVING_OR("", "; it is taken with -test-coins alone") "\n", HF_STOP_KEY_TEST);
    return refused(HF_STOP_KEY_TEST);
  }
  mlock(isk_seed, sizeof isk_seed); // to keep the page out of swap; a failed call goes unnoticed
  if (good) memcpy(isk_seed, bytes, sizeof isk_seed);
  explicit_bzero(bytes, sizeof bytes);
  if (!good)
    printf("refused (%s): seed.bin must be a plain file, no symbolic link, of 32 bytes that are not all zero, this account's own, with no second name (hard link), without "
           "permissions for group and others\n",
           HF_STOP_KEY_FILE);
  return good ? 0 : refused(HF_STOP_KEY_FILE);
}

// Part of the zeroing at an exit (g_end_wipe, set in read_inputs()). Exits that skip it:
// prover/docs/ends-in-short.md, "Exits that skip the zeroing".
inline void Prover::wipe_the_seed() {
  explicit_bzero(isk_seed, sizeof isk_seed);
}

// Prints why the column counts stop the service [cp-copy-spent] [cp-copy-count]; sets isk_fault.
inline void Prover::the_counts_refuse(const cp::Counts &c) {
  if (c.say == cp::Counts::SPENT) {
    isk_fault = "cp-copy-spent";
    // In columns, not batches: the count includes those of the table's own LINK proof (`link`).
    printf("refused (cp-copy-spent): the openings begun on leaf %u's stored copy (its own link and every batch so far) show %llu of the %llu columns that its masks bear, and a batch "
           "shows %llu more, so cp_prove takes no further batch. "
           "Requests that wait stay where they lie; those of a batch that was taken before the leaf was full get \"failed\": send them again. If "
           "the leaf is full earlier or later than your own record says, or you keep no such record (a sign that somebody else writes into this "
           "folder or steers the draw): switch off whatever restarts cp_prove, the model server and whatever calls cp_commit by itself, then stop every cp_prove on this folder and wait until each has "
           "ended, make no further call of cp_commit or cp_prove on it, and go by docs/operators-duties.md, \"If the "
           "count is lost or in doubt\", before anything else. Otherwise: switch off whatever restarts cp_prove, the model server and whatever calls cp_commit by itself, stop every cp_prove on this -isk folder and wait "
           "until each has ended. These programs add no leaf to a commitment that stands and retire none: make and publish a new commitment (that page, \"Renewing before the number\")\n",
           c.leaf, (unsigned long long)c.begun, (unsigned long long)a_leaf_bears, (unsigned long long)a_batch_counts);
  } else {
    isk_fault = "cp-copy-count";
    printf("refused (cp-copy-count): the count of leaf %u, %s, cannot be read, locked or raised: %s. If all three of these hold: the file is there as cp_prove left it; "
           "this folder was never copied, restored or set back; and the disk, the "
           "mount, permissions that keep cp_prove itself out (of a file that is still its own user's, never had a bit for group or others, and "
           "whose mode you know who of yours set, and when) or the file system's locks are at fault: then mend those and start again: cp_prove must "
           "be able to lock the folder leaf/ and to append to this one file. A count file or a leaf's folder that had permissions for group or others (if that let anybody who is not trusted with the weights read seed.bin or the folder whole: docs/operators-duties.md, \"If the count is lost or in doubt\", before anything else), or another owner, counts as replaced: do not only set the mode right, and look at every other live leaf's folder and count "
           "file in the same way before you change any mode: each that was open goes too. If the file is missing or was replaced, do NOT create "
           "it, and never put a copy or a backup of it in its place: nobody knows then how many batches the leaf has served, and it is not "
           "served again. Unless you know who of yours removed or changed that file, and when (a count file that is missing, replaced, shorter "
           "than at your last look, or that has another owner or other modes is a sign that somebody else writes into this folder): switch off whatever restarts cp_prove, the model server and whatever calls cp_commit by itself, then stop every cp_prove on this folder and wait until each has ended, make no further call of cp_commit "
           "or cp_prove on it, and go by docs/operators-duties.md, \"If the count is lost or in doubt\", before "
           "anything else. Otherwise: switch off whatever restarts cp_prove, the model server and whatever calls cp_commit by itself, then stop every cp_prove on this -isk folder and wait until each "
           "has ended. If this folder was copied, restored or set back, every leaf in it may have an older count: start nothing and make no call on it "
           "before you have gone by docs/operators-duties.md, \"Backups and restored copies\". If not: these programs retire no leaf of a commitment that stands, and cp_prove serves none of its leaves while "
           "one has no good count: make and publish a new commitment (that page, \"Renewing before the number\")\n",
           c.leaf, cp::isk_leaf_count(args.isk, c.leaf).c_str(), c.error ? strerror(c.error) : "it is no plain file of this account's own, under one name, for its owner alone; or its folder is not this account's or root's, or group or others may write in it; or the write wrote nothing");
  }
}

// True, with isk_fault set, if a live leaf table has no room for a batch or no readable count.
// ServiceWork::must_end() asks before requests are taken, to lose none to a rejected batch.
// count_a_batch() looks again, under its own lock, when it counts the batch.
inline bool Prover::a_live_leaf_is_spent() {
  const cp::Counts c = cp::look_at_the_counts(isk_held, isk_first, isk_leaves, a_leaf_bears, a_batch_counts);
  if (c.say != cp::Counts::ROOM) the_counts_refuse(c);
  return c.say != cp::Counts::ROOM;
}

// Takes a_leaf_bears, the limit on the columns that one table may show, from the configuration, and
// checks that it can lock isk/leaf and open each live leaf table's count.bin as a batch's start
// will. If it cannot, or a live table has no room for a batch or no readable count, it rejects the
// start-up (the_counts_refuse() prints why). Otherwise it prints each live table's count. A missing
// count is never taken for 0: what the table has shown is then unknown. Call after
// set_up_the_shared_chain(). A program that runs the set-up but begins no batch need not call it.
inline int Prover::read_the_counts() {
  {
    // The length of count.bin is the number of the table's columns shown by openings begun so far:
    // its LINK proof's (twice, if that proof is repeated), then a_batch_counts for each batch.
    const blk::Configured::TableColumns t = frame->table_columns();
    a_leaf_bears = t.bears, a_batch_counts = t.a_batch;
    if (!a_batch_counts || a_batch_counts != lin_setup.sized.S.fold.columns) {
      printf("internal error: the frame's number of columns that a batch shows is not the shared chain's\n");
      name_the_end("batch-columns"); // serving: p258 end 1 batch-columns (kept as a return 4)
      return 4;
    }
  }
  cp::Counts c = cp::the_counts_can_be_raised(isk_held, isk_first, isk_leaves);
  if (c.say == cp::Counts::ROOM) c = cp::look_at_the_counts(isk_held, isk_first, isk_leaves, a_leaf_bears, a_batch_counts);
  if (c.say != cp::Counts::ROOM) {
    the_counts_refuse(c);
    name_the_end(isk_fault); // for the last line
    return 2;
  }
  const int leaves = cp::open_the_leaves(isk_held);
  for (uint32_t k = isk_first; k < isk_leaves; k++) {
    uint64_t begun = 0;
    if (cp::batches_begun(leaves, k, begun) && begun <= a_leaf_bears) {
      if (a_batch_counts == 1)
        printf("# leaf %u: %llu of %llu batches begun, %llu left\n", k, (unsigned long long)begun, (unsigned long long)a_leaf_bears,
               (unsigned long long)(a_leaf_bears - begun));
      else
        printf("# leaf table %u: %llu of %llu columns revealed; a batch reveals %llu: room for %llu batches\n", k, (unsigned long long)begun,
               (unsigned long long)a_leaf_bears, (unsigned long long)a_batch_counts, (unsigned long long)((a_leaf_bears - begun) / a_batch_counts));
    }
  }
  if (leaves >= 0) close(leaves);
  return 0;
}

// Starts a batch, once its requests and their order are fixed, before any proof begins. In order:
//   rho      32 bytes from one call of the operating system's generator; cp_prove draws no others
//            for a batch's proofs. A test passes them in. They are stored only with
//            -keep-for-the-audit, for the one batch of a call with -in and -out and for one audit,
//            which erases them with its verdict (when it does not: tools/audit/audit_batch.sh): with the seed they reveal all private randomness
//            of that batch's proofs. A proof made without the option under drawn bytes cannot be
//            audited: tools/audit/README.md, "What a YES does not cover", point 1.
//   Kbat     the batch's first key: from rho, a key derived from the seed, and a digest of its
//            other inputs, the requests in their slots among them (coin_keys::batch_first_key())
//   k        the leaf table's number, from Kbat; count_a_batch() counts the batch or rejects it
//   kM_k     the leaf table's mask seed (hand.mask_seed), from the seed and fresh.bin
//   entry.bin's first part: the table's root and its record (stored sample answers), in the first
//            query (first hash input) of requests' first hash chains and the shared chain
//   Kproc_b  the process key of all the batch's private randomness, from Kbat, the table's name
//            and its record's answers (coin_keys::batch_process_key())
// Kproc_b goes to three places, only here: the generator for the sub-proofs, each for one part of
// the forward pass (hz::key32_renew(), which also resets its counters), slot_pads_second (a
// self-check of their mask entries, `pads`), and new generators for the batch and each occupied
// slot (prover/prover/chain_coins.cuh). These get the key as an argument: one made later, under the
// key that the sub-proofs' generator then holds, could get a neighbour batch's key with -overlap;
// in a test, such a fault passed every self-check and a run of cp_verify.
// prover/tests/coin_contexts_listed.py, mode "services", lists every such generator that its own
// runs make and requires each to be under its batch's key. The keeper unit
// (`closer`; `library` in names, output), which owns the weight table, gets no batch key. rho, Kbat
// and the digests are zeroed before the return, Kproc_b's copies in the first two places after the
// batch's own proofs (prove_the_batch()). Each generator keeps a key derived from Kproc_b until it
// is destroyed, and the zeroing at an exit does not cover it (prover/hiding/engine_coins.cuh).
// Returns End::None, or why the batch gets no proof: the keeper unit holds no batch, the first two
// places hold no key, and isk_fault names the cause and ends the service. If generators were made,
// the_batch_gets_no_proof() destroys the slots' but not the batch's (chain_coins_of_the_batch()).
inline End Prover::begin_the_batch_on_a_leaf() {
  if (makes_no_response) {
    printf("internal error: a program that has said that it makes no response begins a batch\n");
    exit(4);
  }
  opening_of_the_batch = nullptr;
  const double head_begins = epoch();
  uint8_t rho[32];
  HeadLaps &laps = head_laps;
  laps = HeadLaps{};
#ifdef SERVING_BUILD
  cp_fresh32(rho);
#else
  if (args.test_coins.empty()) {
    cp_fresh32(rho);
  } else {
    FILE *f = fopen(args.test_coins.c_str(), "rb");
    const bool read = f && !fseek(f, (long)(32 * test_coins_used++), SEEK_SET) && fread(rho, 1, 32, f) == 32;
    if (f) fclose(f);
    if (!read) {
      printf("-test-coins: the file holds no coins for a further batch\n");
      exit(2);
    }
  }
#endif
  // [keep-for-the-audit] rho is stored right after it is drawn: before any key is derived from it,
  // and so before the leaf table is picked and the batch is counted. If it cannot be stored, the
  // batch is rejected with nothing derived, proved or counted, and the run ends. cp_prove never
  // reads the file: under the same rho every other file has the same bytes with the option as
  // without it. If the operator's warning cannot be printed, the run stops; the file stays.
  if (args.keep_for_the_audit) {
    if (!cp::keep_a_batchs_bytes(isk_held, rho)) {
      explicit_bzero(rho, 32);
      printf("refused (cp-prove-kept-not-written): the file batch.fresh in the folder audit of the secret folder could not be created, written or synced. Nothing was derived from the batch's random bytes, nothing is proved or "
             "counted, and each request gets .failed. If that file exists now, from this call or an earlier one, run tools/audit/audit_batch.sh with --erase-kept-files, which erases it. If audit exists and is not a folder of yours with mode 0700, do not change its owner or mode: change nothing in the secret folder, and make a new commitment (docs/faults.md)\n");
      isk_fault = "cp-prove-kept-not-written";
      return End::Refused;
    }
    if (!closed_front.say_that_bytes_are_kept()) {
      explicit_bzero(rho, 32);
      stop(End::WriteFailed);
    }
  }
  laps.system_bytes = epoch() - head_begins;
  uint8_t Kbat[32];
  // Kbat: the bytes of first_key, the batch's first key. Its inputs besides rho and the seed: the
  // configuration digest, coin_keys::PROVER_REVISION, the number of leaf tables, the first live
  // one, the number of slots, each slot's request (`tokens`: T tokens, r, e) or that it is empty.
  blk::CoinKey first_key;
  {
    const int slots = args.test_no_requests ? 0 : batch.B; // 0, which no batch has: a test option
    std::vector<coin_keys::Slot> list;
    for (int j = 0; j < slots; j++) list.push_back(coin_keys::Slot{!batch.slot[j].tokens.empty(), batch.slot[j].tokens}); // an empty slot has no tokens
    blk::CoinKey seed = blk::coin_key_from_bytes(isk_seed);
    first_key = coin_keys::batch_first_key(seed, rho, frame->fixed().configuration, isk_leaves, isk_first, list);
    coin_keys::key_bytes(Kbat, first_key); // for the choice of the leaf table, below
    batch_keys::the_first_key_is(Kbat, isk_seed, rho, frame->fixed().configuration, isk_leaves, isk_first, list);
    blk::wipe(seed);
    for (coin_keys::Slot &s : list) explicit_bzero(s.words.data(), 4 * s.words.size()); // clients' data
  }
  Batch::ForTheLibrary &hand = batch.for_the_library;
  const uint32_t k = args.test_leaf >= 0 ? (uint32_t)args.test_leaf : isk_first + cp::draw_leaf(Kbat, isk_leaves - isk_first);
  batch.leaf = k;
  explicit_bzero(rho, sizeof rho);

  // The batch is counted before any of the leaf table is read; k and the keys do not depend on the
  // counts. A thread syncs the count: the_count_is_lasting() or ~CountSync() waits for it.
  const double head_counted_from = epoch();
  int folder;
  cp::CountSync sync;
  if (const cp::Counts c = cp::count_a_batch(isk_held, isk_first, isk_leaves, k, a_leaf_bears, folder, a_batch_counts, sync, args.measure ? &laps.count : nullptr); c.say != cp::Counts::ROOM) {
    explicit_bzero(Kbat, sizeof Kbat);
    blk::wipe(first_key);
    if (c.say == cp::Counts::FAULT && c.error == EAGAIN) // `CountSync::start` sets it: no thread
      printf("cp_prove: the count of leaf %u was raised for this batch, but no thread could be started to bring it to the disk: the machine is short of threads or memory, whatever the lines below say of the file\n", k);
    the_counts_refuse(c);
    return End::Refused;
  }
  const double head_counted = epoch(); // the batch is counted; `Opening::counted` is set below

  // fresh.bin: the 32 bytes that cp_commit drew when it generated the leaf table. Other bytes give
  // other masks, and the opening then fails its check of the nodes [hc-own-node]. Under zeros a
  // table generated twice has equal masks. Zeros pass only with "seed test", which no key protects,
  // and with -public-weights, where cp_commit draws no bytes so that anyone can generate the same
  // commitment.
  uint8_t fresh[32], any = 0;
  const bool read = cp::read_32_secret_bytes(folder, cp::LEAF_FRESH_FILE, fresh);
  for (uint8_t byte : fresh) any |= byte;
  const bool fresh_good = read && (any || isk_test_seed || !args.public_weights.empty());
  bool good = fresh_good;
  {
    // The table's making key (copy_table_key()), as the committer derives it: from the seed, k,
    // fresh.bin, the configuration digest and the table's shape. The shape words are cut out of a
    // table name: root | configuration digest | shape words (blocks/iop/foldeq.h, index_name()).
    const blk::Digest &c = frame->fixed().configuration;
    const std::vector<blk::u8> name = blk::index_name(blk::Digest{}, frame->fixed().lin.fold.index, c);
    blk::Reader words(name.data(), name.size());
    blk::Digest passed_over;
    coin_keys::Shape S;
    bool read_off = words.get(passed_over) && words.get(passed_over);
    for (blk::u64 x; read_off && !words.at_end();) read_off = words.get_u64(x), S.push_back(x);
    if (!read_off || S.empty()) {
      printf("internal error: a stored table's name is not two digests and whole words of shape\n");
      name_the_end("table-name"); // serving: p258 end 1 table-name (kept as an exit(4), at the batch's start)
      exit(4);
    }
    blk::CoinKey seed = blk::coin_key_from_bytes(isk_seed), making = coin_keys::copy_table_key(seed, k, fresh, c, S);
    coin_keys::key_bytes(hand.mask_seed, making);
    blk::wipe(seed), blk::wipe(making);
  }
  explicit_bzero(fresh, sizeof fresh);
  std::string file;
  // entry.bin is read through the directory descriptor that count_a_batch() opened. Only its
  // first part is used (header | root | the record's answers), and only its form is checked.
  // The LINK proof that ties the table to the anchor is checked by the verifier, not here.
  blk::UnprovedTable unproved;
  const char *entry_says = nullptr;
  good = good && cp::read_file(folder, cp::LEAF_ENTRY_FILE, file) && file.size() <= 8 + frame->entry().chains.at(0).plan->size_of(0).bytes;
  close(folder);
  if (good) {
    blk::Reader reader((const blk::u8 *)file.data(), file.size());
    good = blk::read_entry_table_unproved(*frame, reader, unproved, entry_says);
    if (!good) printf("# the entry of leaf %u: %s\n", k, entry_says ? entry_says : "refused without a reason");
  }
  if (!good) {
    if (fresh_good)
      printf("refused (cp-leaf-record): the entry file of leaf %u cannot be read, is too long, or is not what cp_commit writes under this configuration folder; "
             "a line above that begins \"# the entry of leaf\", if there is one, says what is wrong with it\n",
             k);
    else
      printf("refused (cp-leaf-record): leaf %u cannot be served without its %s, the 32 bytes that cp_commit drew when it made the leaf. The file "
             "is missing, is no plain file of 32 bytes, this account's own, under one name, with permissions for its owner only, or holds 32 zero bytes, which cp_commit writes only for "
             "a commitment that hides nothing" SERVING_OR(" (-public-weights; isk.txt then says \"seed public\")", "") ". Unless you know who of yours removed or changed that file, and when (a leaf's file that is missing, replaced, or that "
             "has another owner or other modes is a sign that somebody else writes into this folder): switch off whatever restarts cp_prove, the model server and whatever calls cp_commit by itself, then "
             "stop every cp_prove on this folder and wait until each has ended, make no further call of cp_commit or cp_prove on it, "
             "and go by docs/operators-duties.md, \"If the count is lost or in doubt\", before anything else. Otherwise: switch off whatever restarts cp_prove, the model server and whatever calls cp_commit by itself, then stop every cp_prove on this -isk folder and wait until each has ended. A file that had permissions for group or others (if that let anybody who is not trusted with the weights read seed.bin or the folder whole: docs/operators-duties.md, \"If the count is lost or in doubt\", before anything else), or another owner, counts as replaced: do not only set the mode right, and look at every "
             "live leaf's folder and files in the same way before you change any mode: none that was open gets a file back or is served again, this one too; and if this folder "
             "was copied, restored or set back, every leaf in it may have an older count: start nothing and make no call on it "
             "before you have gone by docs/operators-duties.md, \"Backups and restored copies\". If neither, bring that one file alone back as cp_commit wrote it, "
             "never the folder and never its count file; or, since these programs retire no leaf of a commitment that stands, make and publish a new commitment (that page, \"Renewing before the number\")\n",
             k, cp::LEAF_FRESH_FILE);
    explicit_bzero(Kbat, sizeof Kbat);
    blk::wipe(first_key);
    hand.wipe();
    isk_fault = "cp-leaf-record";
    return End::Refused;
  }
  const blk::IndexReader &copy = unproved.reader();
  hand.root = copy.commitment();
  hand.table = blk::IndexName{copy.commitment(), copy.record()};
  hand.waits = true;
  {
    // With the table's name in the key, two leaf tables do not share a batch's randomness.
    blk::Writer answers;
    for (const blk::E &y : copy.record().y) answers.put(y);
    blk::CoinKey process = coin_keys::batch_process_key(first_key, copy.name(), answers.bytes());
    coin_keys::key_bytes(hand.process_key, process);
    batch_keys::the_process_key_is(hand.process_key, Kbat, copy.name(), answers.bytes());
    // All the batch's generators are made here, while `process` exists; its bytes, copied above,
    // go to three more calls below (key32_renew(), take_the_key(), key32_word_of()) and are then
    // zeroed. That no other code makes a generator or reads the key is checked only on the
    // sources' text, by a script outside this repository, and not for slot_pads_second's copy.
    batch.coins = chain_coins_of_the_batch(process);
    for (int j = 0; j < batch.B; j++) {
      batch.slot[j].attempt = (uint64_t)j;
      if (batch.slot[j].state != SlotState::Empty) batch.slot[j].coins = chain_coins_of_a_seat(process, batch.slot[j]);
    }
    blk::wipe(process), blk::wipe(first_key);
  }
  explicit_bzero(Kbat, sizeof Kbat);
  const double head_read = epoch();
  hz::key32_renew(hand.process_key);
  slot_pads_second.take_the_key(hand.process_key);
  batch.key_word = hz::key32_word_of(hand.process_key);
  end_if_the_keys_differ(hz::key32_word(), batch.key_word, "its own copy, at the start of the batch");
  explicit_bzero(hand.process_key, sizeof hand.process_key); // nothing reads it after this line
  for (int i = openings - 1; i >= 0; i--)
    if (std::find(riders.begin(), riders.end(), &opening[i]) == riders.end()) opening_of_the_batch = &opening[i];
  if (!opening_of_the_batch) { // riders: earlier batches still being opened; one `Opening` is spare
    printf("internal error: a batch begins and every opening holds a rider\n");
    exit(4);
  }
  opening_of_the_batch->count = std::move(sync), opening_of_the_batch->counted = true; // into the batch's own object; whatever an earlier batch has left there is gone
  const double head_keyed = epoch();
  const End why = begin_the_batch_in_the_library();
  // -measure: one line per batch on where this function ("head" in output) spends its time:
  // normally milliseconds, but 42 s and 172 s were seen on busy machines, with the GPU idle. It can
  // wait for the file system (the count, the leaf table's two files; batch.fresh, timed with "the
  // system's 32 bytes"), for memory, and for one query of the CUDA runtime. The clock is read in
  // every run, and no wait for the GPU is added for it.
  if (args.measure &&
      !closed_front.say_closed("measure: head parts leaf %u; the system's 32 bytes %.3f; keys and draw (with them) %.3f; the count: lock %.3f look %.3f open %.3f write %.3f fstat, the sync's start, close %.3f; "
                               "the leaf's files, the process key %.3f; coins, key's copies %.3f; the library: engine %.3f leaf_begin %.3f names, order %.3f; in all %.3f s\n",
                               batch.leaf, laps.system_bytes, head_counted_from - head_begins, laps.count.lock, laps.count.look, laps.count.open, laps.count.write, laps.count.rest,
                               head_read - head_counted, head_keyed - head_read, laps.engine, laps.leaf_begin, laps.names_and_order, epoch() - head_begins))
    stop(End::WriteFailed);
  if (TESTENV("CHAIN_TIMES"))  // where a batch's head spends its time
    printf("# head: keys and draw %.3f | the count (lock, append) %.3f | the leaf's files, the process key %.3f | coins, key's copies %.3f | the keeper's leaf_begin, engine, opening %.3f s\n",
           head_counted_from - head_begins, head_counted - head_counted_from, head_read - head_counted, head_keyed - head_read, epoch() - head_keyed);
  if (why != End::None) hz::key32_wipe(), slot_pads_second.wipe_the_key();
  return why;
}

// Returns only if what o's batch appended to count.bin has reached the disk; otherwise, also if no
// sync was ever started for the batch, exits [cp-copy-count]. fold_begin() and stage_the_proofs()
// call it first, so before that no opening starts and none of the batch's token files or proof
// files is written. At the exit nothing is taken back from count.bin, and riders (earlier batches
// still being opened) are dropped too: failing only this batch would hide as well, but on a path
// that no test runs. The outcome depends on the machine, on no secret.
inline void Prover::the_count_is_lasting(Opening &o) {
  const bool first = o.count.under_way();
  const double from = epoch();
  const cp::CountSync::Said &said = o.count.lasting();
  if (!said.yes) {
    printf("cp_prove: the count of leaf %u was raised for this batch, but it could not be brought to the disk (the sync, or the close behind it, has failed). The batch's opening is not begun, none of its replies or "
           "proofs is written, the work on it is lost, and the program ends. With -spool the requests of this batch and of batches still in work get \"failed\" at the next start: send them again. What to look at:\n",
           o.entry);
    cp::Counts c;
    c.leaf = o.entry, c.error = said.error;
    the_counts_refuse(c);
    name_the_end(isk_fault); // for the last line
    stop(End::WriteFailed);
  }
  // A disk stall that outlasts the batch's own proofs shows in this line.
  if (first && args.measure && !closed_front.say_closed("measure: the sync of leaf %u's count took %.3f s beside the passes; the batch waited %.3f s for it\n", o.entry, said.seconds, epoch() - from)) stop(End::WriteFailed);
}

// [own-test-key] Exits if the sub-proofs' generator holds another key than the batch's: it would
// repeat one-time values in proofs that still verify. `holds`, `owed`: the 32-bit check words of
// its key and of the batch's (prover/hiding/trees.cuh, key32_word()). `whose`: which copy and when,
// for the message. Called once, at a batch's start. Not detected, among others: a key derived
// wrongly, changed later or used in two batches; counters that repeat under the right key; two keys
// with equal check words (about 1 in 2^32).
inline void Prover::end_if_the_keys_differ(uint32_t holds, uint32_t owed, const char *whose) {
  if (holds == owed) return;
  printf("own-test-key: internal error: cp_prove found the wrong batch's secret key in one of its copies of that key (%s). This is a "
         "bug in the program, not in a request. It ends now and releases no proof of any batch in progress. At the next start "
         "those requests are answered \"failed\"; send them again, and report this line\n",
         whose);
  name_the_end("own-test-key");
  exit(4);
}

// Begins the batch in opening_of_the_batch, which holds no batch, from batch.for_the_library, which
// it zeroes: the keeper unit takes the leaf table (leaf_begin()), the batch's engine is made, and
// BatchOpening::begin() claims its columns, all before any request's first chain, whose first query
// names the table. Returns End::None, or End::Refused, isk_fault set, opening_of_the_batch null.
inline End Prover::begin_the_batch_in_the_library() {
  Batch::ForTheLibrary &hand = batch.for_the_library;
  if (!hand.waits) return End::None;
  double mark = epoch();
  auto lap = [&mark](double &seconds) {
    const double t = epoch();
    seconds += t - mark, mark = t;
  };
  turn_to(*opening_of_the_batch);
  Opening &mine = *opening_of_the_batch;
  mine.engine = std::make_unique<blk::EngineGpu>(cudaStreamPerThread);
  lap(head_laps.engine);
  // To save GPU memory, lower tree nodes and salts are recomputed at an opening (the salts from
  // Slot::coins, which outlives the trees); the encoder's work space is returned after each call.
  mine.engine->keep(blk::TreeKeep{false, false, false});
  hf::HfKeeperHooks hooks;
  // begin_the_batch_on_a_leaf() has added the batch's columns to count.bin, so this
  // callback only confirms that: once, and only for that number of columns. True
  // means written, not yet on disk: the_count_is_lasting() waits for the disk.
  hooks.count = [this, &mine](uint64_t columns) { // `mine` lives as long as the Prover
    const bool yes = mine.counted && columns == lin_setup.sized.S.fold.columns;
    mine.counted = false;
    return yes;
  };
  blk::EngineGpu *engine = mine.engine.get();
  hooks.adopt = [engine](const void *card, size_t rows, size_t len) { return engine->adopt(card, blk::Cell::WORD, blk::E_DEGREE, rows, len); };
  hooks.moved = [engine](blk::TableId t, const void *card) { engine->moved(t, card); };
  // The engine keeps its GPU memory during a pass over the weights: these two do nothing.
  hooks.leave = [] {};
  hooks.back = [] {};
  int r;
  {
    InsideTheCloseLibrary inside;
    r = closer->leaf_begin(hand.root.data(), hand.root.size(), hand.table.record, hand.mask_seed, cp::isk_leaf_nodes(args.isk, batch.leaf).c_str(), std::move(hooks));
  }
  lap(head_laps.leaf_begin);
  const blk::IndexName named_by_the_entry = hand.table; // public; saved from the wipe below
  hand.wipe();
  if (!r) {
    // The first queries of the batch's shared chain and of its requests' second chains
    // are made by the functions that the verifier calls (blocks/frame/check.h).
    mine.table = blk::IndexName{closer->holder().commitment(), closer->holder().record()};
    mine.entry = (blk::u32)batch.leaf;
    {
      // entry.bin and the keeper unit must agree on the table's root and record.
      blk::Writer by_the_file, by_the_keeper;
      blk::public_bytes(by_the_file, named_by_the_entry), blk::public_bytes(by_the_keeper, mine.table);
      if (by_the_file.bytes() != by_the_keeper.bytes()) {
        printf("internal error: the keeper names the stored copy of leaf %u otherwise than the copy's entry does\n", batch.leaf);
        name_the_end("leaf-name"); // serving: p258 end 1 leaf-name (kept as an exit(4), at the batch's start)
        exit(4);
      }
    }
    const blk::Configured *configured = frame.get();
    const blk::IndexName *table = &mine.table;
    BatchBinds binds;
    binds.shared = [configured, table](blk::Binder &b, const blk::LinBindingRoot &root) { return blk::bind_shared_chain(b, *configured, *table, root); };
    binds.second = [configured](blk::Binder &b, const blk::Sealed &own, const blk::Sealed &shared) { blk::bind_second_chain(b, *configured, own, shared); };
    mine.coins = batch.coins.get();
    mine.order = std::make_unique<BatchOpening>(lin_setup, frame->fixed().binding, frame->fixed().configuration, frame->proof().chains.at(2).plan, std::move(binds), closer->holder(), *mine.engine, *batch.coins,
                                                [engine](const blk::ChainPrefix &p, unsigned bits) { return engine->search_nonce(p, bits); });
    if (!mine.order->begin()) {
      printf("refused (cp-leaf-count): the stored copy of leaf %u covers no further batch, or the batch was not counted\n", batch.leaf);
      mine.order.reset();
      closer->leaf_end();
      r = 2;
    }
  }
  lap(head_laps.names_and_order);
  if (r) mine.engine.reset(), mine.count = cp::CountSync(), mine.counted = false; // the batch ends here: its sync is waited for
  if (r == 2) {
    printf("refused (cp-leaf-record): the opening library refuses the record of leaf %u; its own line on standard error "
           "gives the reason\n",
           batch.leaf);
    isk_fault = "cp-leaf-record";
    opening_of_the_batch = nullptr;
    explicit_bzero(&batch.key_word, sizeof batch.key_word);
    return End::Refused;
  }
  if (r) {
    printf("internal error: the opening library's batch_begin returned %d\n", r);
    name_the_end("leaf-begin"); // serving: p258 end 1 leaf-begin (kept as an exit(4), at the batch's start)
    exit(4);
  }
  return End::None;
}
