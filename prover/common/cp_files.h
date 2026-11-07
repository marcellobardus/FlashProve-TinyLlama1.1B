// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The names and layouts of the files in a commitment's two key directories, and each weight table's
// count of opened columns. `ivk` is the public key directory and `isk` the secret one: the
// committer program cp_commit creates both, and the prover program cp_prove reads `isk` and appends
// to its counts (with -keep-for-the-audit it also writes one file there). Host code without CUDA;
// g++ compiles it.
// A caller must count every opening of a weight table before the opening begins: see "the counts of
// opened columns" below.
//
// Terms. A weight table is all the weights as a matrix of long rows, each with random masks,
// encoded, under a Merkle tree. The anchor (`first_table` in count_first_table_columns()) is a
// commitment's first weight table; no request is proved against it. Leaf table k (`leaf` in
// identifiers and file names) is a further weight table of the same weights with its own masks; it
// is not a leaf of a Merkle tree. The key file (key.bin) has the anchor's root and cp_commit's
// one-time proofs about the anchor. Leaf file k (`entry` in identifiers) has leaf table k's root
// and its LINK proof, which is meant to show that the table contains the anchor's weights.
#pragma once
#include "common/coin_chacha.h" // cp_key(), CP_LF
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <fcntl.h>
#include <future>
#include <string>
#include <string_view>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

// Namespace `cp`, shared by cp_commit and cp_prove: this file and the self-test in cp_keys_own.h.
namespace cp {
// The headers (`head` in identifiers) of the key file and of a proof file are defined in
// docs/format/FORMAT.md, R13.1 and R14.2.
constexpr uint32_t KEY_TAG = 0x4B495043u;   // "CPIK", key.bin's first word
constexpr uint32_t PROOF_TAG = 0x46505043u; // "CPPF", a proof file's first word
constexpr size_t KEY_HEAD_BYTES = 8, PROOF_HEAD_BYTES = 12;

// ---- names ----
constexpr const char *ISK_SEED_FILE = "seed.bin", *ISK_TEXT_FILE = "isk.txt"; // see SecretFolder
static inline std::string isk_seed(const std::string &isk) { return isk + "/" + ISK_SEED_FILE; }
static inline std::string isk_config(const std::string &isk) { return isk + "/config"; } // policy.txt, KL.bin, layout.txt
static inline std::string isk_txt(const std::string &isk) { return isk + "/" + ISK_TEXT_FILE; }
// cp_commit's copy of ivk's key.bin.
static inline std::string isk_key(const std::string &isk) { return isk + "/key.bin"; }
static inline std::string isk_anchor_dir(const std::string &isk) { return isk + "/anchor"; }
static inline std::string isk_anchor_nodes(const std::string &isk) { return isk + "/anchor/hcnodes.bin"; }
// The anchor's count of opened columns (see Counts), raised by count_first_table_columns().
constexpr const char *ANCHOR_COUNT_FILE = "count.bin";
static inline std::string isk_anchor_count(const std::string &isk) { return isk + "/anchor/" + ANCHOR_COUNT_FILE; }
// The secret directory as cp_prove uses it: opened once, at the start, and kept open until the
// process ends. cp_prove reaches everything that it writes below isk (a leaf table's count.bin;
// with -keep-for-the-audit, batch.fresh) from this descriptor, name by name, without following a
// symbolic link. So whoever can write in a directory above isk, and puts another directory or a
// link under isk's name after the start, redirects no write of this process: the writes still go
// into the directory that take() checked.
// Prover::read_isk() and Prover::read_the_seed() (prover/prover/isk.cuh) also read isk.txt,
// seed.bin and the names in isk/audit through this descriptor, so they and the counts belong to one
// directory. read_file() opens isk.txt without O_NOFOLLOW, in a directory in which only this user
// or root can write. cp_prove still reaches by path:
//   - the configuration files;
//   - the leaf tables' files when read_isk() checks that they exist, and hcnodes.bin for a batch;
//   - isk.txt in second_start_unless_nothing_is_hidden() (prover/prover/cp_prove.cu), only if
//     `closed_front.on` is false (a test build).
// open(): the path's last name must be a directory, not a symbolic link. Links earlier in the path
// are followed, as in any path, so read_isk() walks the path itself and calls take(). Call open()
// or take() once on an object: a second call overwrites the first descriptor without closing it.
// What neither the descriptor nor that walk can detect: an older copy of the same secret directory
// that this user or root has put under isk's name between two starts of cp_prove. cp_prove accepts
// the copy. Its counts are the older ones, so the columns opened since it was made go uncounted,
// and the part of a table's limit that they used is spent a second time: see
// "the counts of opened columns" below and docs/operators-duties.md, "Backups and restored copies".
struct SecretFolder {
  int fd = -1;
  bool open(std::string path) {
    for (;;) { // with a final "/" or "/." the system would follow a link despite O_NOFOLLOW
      if (path.size() > 1 && path.back() == '/') path.pop_back();
      else if (path.size() > 2 && !path.compare(path.size() - 2, 2, "/.")) path.resize(path.size() - 2);
      else break;
    }
    errno = 0;
    return take(::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
  }
  // Checks and keeps a descriptor that the caller has opened: read_isk() opens isk with
  // Spool::open_by_parts() (prover/prover/spool.h), part by part from /, following no symbolic
  // link. opened: that descriptor, which this object owns from here on, or -1 with the failed
  // call's errno, which is kept. It must refer to a directory that belongs to this process's
  // effective user and has no permission for group or others, as cp_commit creates isk (mode
  // 0700). Otherwise take() closes it and returns false with `fd` -1; errno is then 0 if every
  // call succeeded but the type, owner or mode is wrong.
  bool take(int opened) {
    fd = opened;
    if (fd >= 0) errno = 0;
    struct stat is;
    if (fd >= 0 && !fstat(fd, &is) && S_ISDIR(is.st_mode) && is.st_uid == geteuid() && !(is.st_mode & 077)) return true;
    const int why = errno;
    if (fd >= 0) close(fd);
    fd = -1, errno = why;
    return false;
  }
};
// [keep-for-the-audit] cp_commit and cp_prove each have an option -keep-for-the-audit, and create
// isk/audit only with it. cp_prove's file is batch.fresh, below. cp_commit writes one file for the
// key file and one for each leaf file. Each has, raw, the 32 bytes from the operating system (zeros
// under the test option -test-seed) from which, with the seed (seed.bin, 32 secret bytes), the
// private randomness of that file's proofs was derived. cp_commit writes each file once
// (RunKeys::keep() in prover/hash_commitment/hf_commit_keys.h), and neither it nor cp_prove reads
// one. They are for the audit in tools/audit/, which destroys them before it answers YES.
constexpr const char *AUDIT_FOLDER = "audit";
static inline std::string isk_audit_dir(const std::string &isk) { return isk + "/" + AUDIT_FOLDER; }
constexpr const char *AUDIT_KEY_FILE = "key.fresh";
static inline std::string audit_entry_file(uint32_t k) { return "entry." + std::to_string(k) + ".fresh"; }
// Returns true if isk/audit contains anything, whatever its name, is not a directory (a symbolic
// link is not followed) or cannot be opened and listed to its end. Returns false if nothing has
// that name (as without the option) or the directory is empty, as the audit leaves it after a
// YES. Only names are listed: no file is opened, read or renamed. An empty directory does not show
// that an audit answered YES: a batch's audit erases batch.fresh at a NO as well, and emptying the
// directory by hand has the same effect. cp_prove rejects an isk for which this returns true
// (Prover::read_isk() in prover/prover/isk.cuh). isk/audit is opened from the descriptor in `isk`,
// so it is looked for in the directory in which the counts are raised.
static inline bool audit_is_owed(const SecretFolder &isk) {
  const int folder = openat(isk.fd, AUDIT_FOLDER, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (folder < 0) return errno != ENOENT;
  DIR *const list = fdopendir(folder);
  if (!list) return close(folder), true;
  bool something = false;
  errno = 0;
  while (const struct dirent *const e = readdir(list)) something = something || (strcmp(e->d_name, ".") && strcmp(e->d_name, ".."));
  something = something || errno; // a listing that broke off says nothing
  closedir(list);
  return something;
}
// Only with cp_prove's -keep-for-the-audit, which needs -in and -out (a call that proves one
// batch): the 32 bytes, raw, that cp_prove took from the operating system when the batch began (a
// test build's -test-coins supplies them instead). From these bytes and seed.bin every mask, pad
// and salt of that batch's proofs can be computed. cp_prove writes the file once; neither cp_prove
// nor cp_commit reads, overwrites or removes it. The batch's audit (tools/audit/audit_batch.sh)
// reads it and, unless run with --rehearsal, erases it at every verdict: YES, NO or NOT WHOLE. The
// audit leaves the file alone if, for example, it has a second hard link or was replaced meanwhile.
constexpr const char *AUDIT_BATCH_FILE = "batch.fresh";
// Writes these 32 bytes to isk/audit/batch.fresh. Every step starts from the open descriptor in
// `isk`. Creates isk/audit (mode 0700) if it is missing. The directory, new or not, must not be a
// symbolic link, must belong to this process's effective user and must have mode 0700: the audit
// accepts no other. The file is created exclusively under its final name, so the call fails if the
// name exists, and gets mode 0600 whatever the umask. It is written and synced, and then isk/audit
// and isk are synced. There is no temporary name and no rename, so a kill or a loss of power during
// the call can leave a short file of that name. Returns false if a step fails; the file, complete
// or short, may then exist.
static inline bool keep_a_batchs_bytes(const SecretFolder &isk, const uint8_t bytes[32]) {
  if (mkdirat(isk.fd, AUDIT_FOLDER, 0700) && errno != EEXIST) return false;
  const int folder = openat(isk.fd, AUDIT_FOLDER, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (folder < 0) return false;
  struct stat found;
  if (fstat(folder, &found) || found.st_uid != geteuid() || (found.st_mode & 0777) != 0700) return close(folder), false;
  const int file = openat(folder, AUDIT_BATCH_FILE, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
  bool good = file >= 0 && !fchmod(file, 0600) && write(file, bytes, 32) == 32 && !fsync(file);
  if (file >= 0) good = !close(file) && good;
  good = !fsync(folder) && good;
  close(folder);
  return !fsync(isk.fd) && good;
}
static inline std::string isk_leaves(const std::string &isk) { return isk + "/leaf"; }
static inline std::string isk_leaf_dir(const std::string &isk, uint32_t k) { return isk + "/leaf/" + std::to_string(k); }
static inline std::string isk_leaf_record(const std::string &isk, uint32_t k) { return isk_leaf_dir(isk, k) + "/wcard.bin"; }
static inline std::string isk_leaf_nodes(const std::string &isk, uint32_t k) { return isk_leaf_dir(isk, k) + "/hcnodes.bin"; }
// The 32 bytes from the operating system (zeros under -test-seed and -public-weights) that go into
// leaf table k's key, from which its masks are derived: coin_keys::copy_table_key() in
// prover/common/coin_keys.h.
constexpr const char *LEAF_FRESH_FILE = "fresh.bin";
static inline std::string isk_leaf_fresh(const std::string &isk, uint32_t k) { return isk_leaf_dir(isk, k) + "/" + LEAF_FRESH_FILE; }
constexpr const char *LEAF_ENTRY_FILE = "entry.bin"; // cp_commit's copy of ivk/leaf/<k>.bin
static inline std::string isk_leaf_entry(const std::string &isk, uint32_t k) { return isk_leaf_dir(isk, k) + "/" + LEAF_ENTRY_FILE; }
// Leaf table k's count of opened columns (see Counts), raised by count_table_columns_held() and
// count_a_batch().
constexpr const char *LEAF_COUNT_FILE = "count.bin";
static inline std::string isk_leaf_count(const std::string &isk, uint32_t k) { return isk_leaf_dir(isk, k) + "/" + LEAF_COUNT_FILE; }
static inline std::string ivk_key(const std::string &ivk) { return ivk + "/key.bin"; }
static inline std::string ivk_leaves(const std::string &ivk) { return ivk + "/leaf"; }
static inline std::string ivk_entry(const std::string &ivk, uint32_t k) { return ivk + "/leaf/" + std::to_string(k) + ".bin"; }

// ---- bytes ----
static inline void put_word(std::string &to, uint32_t word) {
  for (int i = 0; i < 4; i++) to.push_back((char)(uint8_t)(word >> (8 * i)));
}
static inline uint32_t word_at(const std::string &from, size_t at) { // at + 4 <= from.size()
  uint32_t word = 0;
  for (int i = 0; i < 4; i++) word |= (uint32_t)(uint8_t)from[at + i] << (8 * i);
  return word;
}
// The conditions on a file of secret bytes and on a count.bin: a regular file that belongs to this
// process's effective user, has one name (link count 1) and has no permissions for group and
// others. The functions here check them on the open file; batches_begun() alone checks by name.
// The owner is checked here because a process run by root can open any user's file, and that user
// could read the secret or shorten the count. Through a second name (a hard link) the file could be
// read, kept or shortened from a directory that no program checks. A second name that is created
// and removed between two checks goes unnoticed.
static inline bool own_and_closed(const struct stat &is) { return S_ISREG(is.st_mode) && is.st_uid == geteuid() && is.st_nlink == 1 && !(is.st_mode & 077); }
// Reads a file of 32 secret bytes, a leaf table's fresh.bin. It must not be a symbolic link, must
// have exactly 32 bytes and must meet own_and_closed(); otherwise the function returns false and
// fills `out` with zeros. The bytes pass through no buffer of the C library, which would be freed
// without being zeroed. The bytes themselves are not checked. folder: an open directory, against
// which a relative path is resolved.
static inline bool read_32_secret_bytes(int folder, const std::string &path, uint8_t out[32]) {
  const int fd = openat(folder, path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK); // O_NONBLOCK: a fifo does not hold us up
  struct stat is;
  uint8_t bytes[32 + 1] = {0};
  bool good = fd >= 0 && !fstat(fd, &is) && own_and_closed(is) && is.st_size == 32;
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
  good = good && have == 32;
  memset(out, 0, 32);
  if (good) memcpy(out, bytes, 32);
  explicit_bzero(bytes, sizeof bytes);
  return good;
}
static inline bool read_32_secret_bytes(const std::string &path, uint8_t out[32]) { return read_32_secret_bytes(AT_FDCWD, path, out); }
// Reads an entire file. Returns false if it cannot be read to its end. Not for secret bytes: its
// buffers are not zeroed. folder: as above.
static inline bool read_file(int folder, const std::string &path, std::string &bytes) {
  bytes.clear();
  const int fd = openat(folder, path.c_str(), O_RDONLY | O_CLOEXEC);
  FILE *f = fd >= 0 ? fdopen(fd, "rb") : nullptr;
  if (!f) {
    if (fd >= 0) close(fd);
    return false;
  }
  char room[1 << 16];
  size_t got;
  while ((got = fread(room, 1, sizeof room, f)) > 0) bytes.append(room, got);
  const bool whole = !ferror(f);
  fclose(f);
  return whole;
}
static inline bool read_file(const std::string &path, std::string &bytes) { return read_file(AT_FDCWD, path, bytes); }

// ---- the counts of opened columns ----
// An opening, below, is a proof that opens some columns of a weight table (a column is one symbol
// of every row): on the anchor, the key file's one-time test and every leaf table's LINK proof; on
// a leaf table, its own LINK proof and every batch for which that table was sampled (a batch:
// requests that cp_prove proves together). A
// row's masks allow a limited number of opened columns over the table's life: `bears` below,
// the same for every table. A batch opens `a_batch` columns. Both follow from the configuration
// files (Configured::table_columns() in blocks/frame/configure.h); the limit is written in no file.
// A table's count is the length of count.bin in its directory (isk/anchor or isk/leaf/<k>).
// cp_commit creates the file empty with the table, before anything of the table is opened. An
// opening appends one byte for each of its columns when it begins, whatever becomes of it: openings
// differ in their number of columns, so columns are counted. No number is read, changed and written
// back, so a process that dies can only leave the count too high, and all processes that use one
// directory share the count.
// The count covers this directory only. A second directory with the same table has a separate
// count, and a directory restored to an earlier state has its earlier count: no program can detect
// either (docs/operators-duties.md, "Backups and restored copies").
//
// The counting functions' result:
//   ROOM: the opening is counted and fits within the limit. After count_a_batch() the caller must
//     still wait for the sync (see CountSync) before the opening begins; look_at_the_counts() and
//     the_counts_can_be_raised() return ROOM without counting anything.
//   SPENT: the limit leaves no room for the opening.
//   FAULT: a count cannot be read or raised.
struct Counts {
  enum Say { ROOM, SPENT, FAULT } say = FAULT;
  uint32_t leaf = 0;  // SPENT: a full leaf table. FAULT: one whose count cannot be read or raised
  uint64_t begun = 0; // SPENT: its count. ROOM from `count_a_batch`: table k's, with this batch
  int error = 0;      // FAULT: the failed call's errno, or 0 (unfit file or directory, short write)
};
// Reads leaf table k's count, which is in columns despite the function's name. leaves: the open
// directory isk/leaf. Returns false unless count.bin meets the conditions that open_the_count()
// sets for the file: it is not a symbolic link and meets own_and_closed().
static inline bool batches_begun(int leaves, uint32_t k, uint64_t &begun) {
  struct stat is;
  errno = 0;
  if (fstatat(leaves, (std::to_string(k) + "/" + LEAF_COUNT_FILE).c_str(), &is, AT_SYMLINK_NOFOLLOW) || !own_and_closed(is)) return false;
  begun = (uint64_t)is.st_size;
  return true;
}
// Reads the counts of the live leaf tables, first .. M - 1 (see IskText). ROOM: each of them has
// room for a further batch. A count that cannot be read is a FAULT, never ROOM.
static inline Counts look_at_the_counts(int leaves, uint32_t first, uint32_t M, uint64_t bears, uint64_t a_batch) {
  Counts c;
  for (c.leaf = first; c.leaf < M; c.leaf++) {
    if (!batches_begun(leaves, c.leaf, c.begun)) return c.error = errno, c;
    if (a_batch > bears || c.begun > bears - a_batch) return c.say = Counts::SPENT, c;
  }
  return c.say = Counts::ROOM, c;
}
// Opens isk/leaf from the open descriptor of isk, so no path is resolved again.
static inline int open_the_leaves(const SecretFolder &isk) { return openat(isk.fd, "leaf", O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC); }
static inline Counts look_at_the_counts(const SecretFolder &isk, uint32_t first, uint32_t M, uint64_t bears, uint64_t a_batch) {
  const int leaves = open_the_leaves(isk);
  Counts c;
  c.leaf = first, c.error = errno;
  if (leaves < 0) return c;
  c = look_at_the_counts(leaves, first, M, bears, a_batch);
  close(leaves);
  return c;
}
// Opens isk/leaf and takes an exclusive lock on it, which is released when the descriptor is
// closed. Returns -1, with errno set, if either fails.
static inline int lock_the_leaves(const SecretFolder &isk) {
  const int leaves = open_the_leaves(isk);
  if (leaves < 0) return -1;
  int locked;
  do locked = flock(leaves, LOCK_EX);
  while (locked && errno == EINTR);
  if (!locked) return leaves;
  const int why = errno;
  close(leaves);
  errno = why;
  return -1;
}
// Opens leaf table k's directory, which must not be a symbolic link, and through it its count.bin
// for appending. count.bin must not be a symbolic link, must meet own_and_closed() and is never
// created here. Returns the file's descriptor and sets `folder` to the directory's. On failure both
// are -1, with errno as in Counts::error.
// The directory must belong to this process's effective user or to root, and group and others must
// have no write permission for it: whoever can write in it could rename another of this user's
// files, a shorter one, to count.bin. batches_begun() does not check the directory. Of the
// directories above it, isk is checked once, when it is opened (SecretFolder::take()); no program
// checks the owner or mode of isk/leaf.
static inline int open_the_count(int leaves, uint32_t k, int &folder) {
  errno = 0;
  folder = openat(leaves, std::to_string(k).c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  const int fd = folder < 0 ? -1 : openat(folder, LEAF_COUNT_FILE, O_WRONLY | O_APPEND | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
  struct stat is, around;
  if (fd >= 0 && !fstat(fd, &is) && own_and_closed(is) && !fstat(folder, &around) && (around.st_uid == geteuid() || !around.st_uid) && !(around.st_mode & 022)) return fd;
  const int why = errno;
  if (fd >= 0) close(fd);
  if (folder >= 0) close(folder);
  folder = -1, errno = why;
  return -1;
}
// A dry run at start-up: takes the lock and opens every live leaf table's count.bin as
// count_a_batch() does, but appends nothing. So a directory, a mount or a file system on which the
// lock or an open fails is detected at the start, not at the first batch. A write or a sync that
// would fail (a full disk, for example) is not detected. Returns ROOM or FAULT.
static inline Counts the_counts_can_be_raised(const SecretFolder &isk, uint32_t first, uint32_t M) {
  Counts c;
  c.leaf = first;
  const int leaves = lock_the_leaves(isk);
  if (leaves < 0) return c.error = errno, c;
  for (; c.leaf < M; c.leaf++) {
    int folder;
    const int fd = open_the_count(leaves, c.leaf, folder);
    if (fd < 0) break;
    close(fd), close(folder);
  }
  c.error = errno;
  if (c.leaf == M) c.say = Counts::ROOM;
  close(leaves);
  return c;
}
// Syncs one batch's append to count.bin to the disk, on a separate thread. The sync guards against
// a loss of power: a process that dies loses nothing that the kernel has accepted. A leaf
// table's masks are used up by columns that are opened, and a batch's columns are chosen in its
// opening, after its requests' own proofs. So the count is written at the batch's start, by
// count_a_batch(), and must be on the disk before the batch's opening begins, not earlier. Waiting
// for the sync in count_a_batch() would leave the GPU idle: it took 34.9 s once, on a busy machine.
// The caller must call lasting() before it starts the opening's thread, and continue only if the
// answer is yes.
//   - One object serves one batch. A yes for an earlier position in the file does not cover a later
//     one.
//   - fsync() is called once, on the descriptor that made the append. Its first result is final: on
//     Linux a second call can return 0 although the bytes were never written. Anything but that
//     call's 0 is a no: an error, an interrupted call, a thread that could not be started, an
//     object that was never started. A failed close() is a no as well.
//   - The descriptor stays open until the thread has been joined. The kernel keeps an open file's
//     length in memory, so, unless somebody shortens the file (its owner or root could: see
//     own_and_closed(); no code here does), no process on this machine sees the length decrease
//     meanwhile, and the descriptor's number cannot be reused for another file while the thread
//     uses it.
//   - The thread is joined, without a time limit, in lasting(), or earlier if the object is
//     destroyed or assigned to.
class CountSync {
 public:
  struct Said {
    bool yes = false;   // the appended bytes are on the disk
    int error = 0;      // with a no: the errno of the call that failed (the sync, the close; EAGAIN: no thread), or 0 if that call set none or nothing was ever started
    double seconds = 0; // what the sync took
  };
  CountSync() = default;
  CountSync(CountSync &&o) noexcept : fd_(std::exchange(o.fd_, -1)), end_(std::move(o.end_)), said_(std::exchange(o.said_, Said())) {}
  CountSync &operator=(CountSync &&o) noexcept {
    if (this != &o) {
      settle();
      fd_ = std::exchange(o.fd_, -1), end_ = std::move(o.end_), said_ = std::exchange(o.said_, Said());
    }
    return *this;
  }
  ~CountSync() { settle(); }
  bool under_way() const { return end_.valid(); }
  // The first call waits for the thread; the answer is kept, and later calls return it.
  const Said &lasting() {
    settle();
    return said_;
  }
  // For count_a_batch() only. fd: count.bin's descriptor, after the append. An earlier sync of this
  // object is joined first, and its answer is dropped.
  void start(int fd) {
    settle();
    said_ = Said(), fd_ = fd;
    try {
      end_ = std::async(std::launch::async, [fd] {
        struct timespec from, to;
        clock_gettime(CLOCK_MONOTONIC, &from);
        Said s;
        s.yes = !fsync(fd); // the call's own value and nothing else
        if (!s.yes) s.error = errno;
        clock_gettime(CLOCK_MONOTONIC, &to);
        s.seconds = (double)(to.tv_sec - from.tv_sec) + 1e-9 * (double)(to.tv_nsec - from.tv_nsec);
        return s;
      });
    } catch (...) { // no thread: a no
      said_.error = EAGAIN;
      close(fd_), fd_ = -1;
    }
  }

 private:
  void settle() {
    if (!end_.valid()) return;
    said_ = end_.get();
    if (close(fd_) && said_.yes) said_.yes = false, said_.error = errno;
    fd_ = -1;
  }
  int fd_ = -1;
  std::future<Said> end_;
  Said said_;
};
// For cp_prove -measure: the seconds that each step of count_a_batch() took. On a busy machine a
// batch's start has stalled for minutes, with the GPU idle; the times show in which call. Taking
// them changes no step and no order of steps.
struct CountLaps {
  double lock = 0, look = 0, open = 0, write = 0, rest = 0;
};
// Counts a batch for which leaf table k was sampled (draw_leaf()). The caller must read nothing of
// leaf table k for the batch before this call. Under one lock on isk/leaf, which every process that
// serves from this isk takes here, the function
//   1. reads the live leaf tables' counts. The batch continues only if each has room for it, so
//      whether a batch is rejected for lack of room depends on the counts only, not on k;
//   2. opens leaf table k's directory, once. cp_commit creates a new directory each time it
//      generates a leaf table, so what the caller reads through that descriptor belongs to the
//      table counted. cp_prove reads fresh.bin and entry.bin through it, hcnodes.bin later by its
//      path;
//   3. appends `a_batch` bytes to count.bin, reads the length again and starts the sync.
// Returns ROOM only if every call has succeeded, the write has written all its bytes, the length is
// now at most `bears` and the sync's thread has started. The count is then written but not yet on
// the disk: see CountSync. With any other result sync.lasting() answers no. The lock is released
// without waiting for the sync: other processes on this machine read lengths, which the append has
// changed at once.
// The length is compared after the append: each append has a different position in the file, and a
// process sees a length that is at least the end of its own append. So the batches that continue
// stay within `bears` even where a file system does not honour the lock, as long as it honours
// O_APPEND. After the write nothing is retried, and no code here shortens or rewrites count.bin.
// folder: with ROOM, leaf table k's open directory, which the caller must close; otherwise -1.
// laps: null if no times are wanted.
static inline Counts count_a_batch(const SecretFolder &isk, uint32_t first, uint32_t M, uint32_t k, uint64_t bears, int &folder, uint64_t a_batch, CountSync &sync, CountLaps *laps = nullptr) {
  auto now = [] {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
  };
  double mark = laps ? now() : 0;
  auto lap = [&](double CountLaps::*step) { // preserves errno: a failed step's is read afterwards
    if (!laps) return;
    const int kept = errno;
    const double t = now();
    laps->*step += t - mark, mark = t;
    errno = kept;
  };
  folder = -1;
  sync = CountSync(); // no answer of an earlier batch outlives this call
  Counts c;
  c.leaf = k;
  const int leaves = lock_the_leaves(isk);
  lap(&CountLaps::lock);
  if (leaves < 0) return c.error = errno, c;
  c = look_at_the_counts(leaves, first, M, bears, a_batch);
  lap(&CountLaps::look);
  if (c.say == Counts::ROOM) {
    c.say = Counts::FAULT, c.leaf = k;
    const int fd = open_the_count(leaves, k, folder);
    lap(&CountLaps::open);
    struct stat is;
    const std::string bytes((size_t)a_batch, '\0');
    const bool written = fd >= 0 && write(fd, bytes.data(), bytes.size()) == (ssize_t)bytes.size();
    lap(&CountLaps::write);
    const bool raised = written && !fstat(fd, &is);
    c.error = errno;
    if (raised) c.begun = (uint64_t)is.st_size, c.say = c.begun <= bears ? Counts::ROOM : Counts::SPENT;
    if (c.say != Counts::ROOM) {
      if (fd >= 0) close(fd); // a rejected batch opens nothing: its bytes need not reach the disk
    } else if (sync.start(fd), !sync.under_way()) { // no thread: fail now, before the batch's work
      c.say = Counts::FAULT, c.error = sync.lasting().error;
    }
    if (c.say != Counts::ROOM && folder >= 0) close(folder), folder = -1;
  }
  close(leaves); // the lock goes with it
  lap(&CountLaps::rest);
  return c;
}

// ---- counting cp_commit's own openings ----
// Counts an opening of `columns` columns that cp_commit begins on a table. (Batches are counted by
// count_a_batch(), in the same file.) The table's directory, which must not be a symbolic link, is
// locked for the call. count.bin must meet the conditions that open_the_count() sets for the file;
// the directory's owner and mode are not checked here.
// If the length and `columns` together exceed `bears`, the result is SPENT and nothing is appended.
// Otherwise `columns` bytes are appended and, unlike in count_a_batch(), synced before the return,
// and the length is read again. ROOM only if every call has succeeded, all bytes were written and
// the length is now at most `bears`. (SPENT is also the result if the length read after the append
// exceeds `bears`; the bytes then stay.) After a write nothing is retried: a fault leaves the count
// too high at worst. In the result, `begun` is the length as last read, and `leaf` is not used.
// count_table_columns_held() is for a caller that itself has an exclusive flock() on `folder`, the
// table's open directory: it locks nothing and leaves that descriptor open. cp_commit keeps a leaf
// table's directory locked while it generates the table, and a second descriptor of the same
// process would wait for that lock forever.
static inline Counts count_table_columns_held(int folder, uint64_t columns, uint64_t bears) {
  static_assert(std::string_view(ANCHOR_COUNT_FILE) == std::string_view(LEAF_COUNT_FILE), "one name for every table's count");
  Counts c;
  errno = 0;
  const int fd = openat(folder, ANCHOR_COUNT_FILE, O_WRONLY | O_APPEND | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK);
  struct stat is;
  if (fd >= 0 && !fstat(fd, &is) && own_and_closed(is)) {
    c.begun = (uint64_t)is.st_size;
    if (columns > bears || c.begun > bears - columns) {
      c.say = Counts::SPENT;
    } else {
      const std::string bytes((size_t)columns, '\0');
      const bool raised = write(fd, bytes.data(), bytes.size()) == (ssize_t)bytes.size() && !fsync(fd) && !fstat(fd, &is);
      if (raised) c.begun = (uint64_t)is.st_size, c.say = c.begun <= bears ? Counts::ROOM : Counts::SPENT;
    }
  }
  c.error = errno;
  if (fd >= 0 && close(fd)) c.error = errno, c.say = c.say == Counts::ROOM ? Counts::FAULT : c.say;
  return c;
}
// at: an open directory, against which a relative `tables_folder` is resolved, or AT_FDCWD.
static inline Counts count_table_columns(int at, const std::string &tables_folder, uint64_t columns, uint64_t bears) {
  Counts c;
  errno = 0;
  const int folder = openat(at, tables_folder.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (folder < 0) return c.error = errno, c;
  int locked;
  do locked = flock(folder, LOCK_EX);
  while (locked && errno == EINTR);
  if (locked) c.error = errno;
  else c = count_table_columns_held(folder, columns, bears);
  close(folder); // the lock goes with it
  return c;
}
static inline Counts count_table_columns(const std::string &tables_folder, uint64_t columns, uint64_t bears) { return count_table_columns(AT_FDCWD, tables_folder, columns, bears); }
static inline Counts count_first_table_columns(const std::string &isk, uint64_t columns, uint64_t bears) { return count_table_columns(isk_anchor_dir(isk), columns, bears); }
// The same, with isk/anchor given as an open directory and a path relative to it.
static inline Counts count_first_table_columns(int at, const std::string &anchor_folder, uint64_t columns, uint64_t bears) { return count_table_columns(at, anchor_folder, columns, bears); }

// ---- key.bin ----
// key.bin's header: KEY_TAG and the version word. key.bin contains neither the configuration files
// nor M, the number of leaf tables, which is in isk.txt.
struct Key {
  uint32_t version = 0;
  std::string rest; // the bytes after the header: docs/format/FORMAT.md, R13.2
};
// Splits key.bin's bytes. Returns false unless they begin with KEY_TAG and a version word. The
// caller must check that word and the length of `rest`, which follows from the configuration files and, for an opening, from the sampled positions (docs/format/FORMAT.md, R13.2).
static inline bool cut_key(const std::string &bytes, Key &key) {
  if (bytes.size() < KEY_HEAD_BYTES || word_at(bytes, 0) != KEY_TAG) return false;
  key.version = word_at(bytes, 4);
  key.rest = bytes.substr(KEY_HEAD_BYTES);
  return true;
}
static inline bool read_key(const std::string &path, Key &key) {
  std::string bytes;
  return read_file(path, bytes) && cut_key(bytes, key);
}

// ---- isk.txt ----
// Four to seven lines, in this order, each "name value" and a line feed, with numbers in decimal
// without leading zeros. h_par is blk::configuration_digest() of the configuration files.
//   version <the build's version word>
//   M <the number of leaf tables>
//   first <the first live one>      present if and only if the number is above 0
//   keep <s>                        hcnodes.bin stores level s of each table's Merkle tree
//   weights <generated, made-up-blocks, or the absolute path of the directory given to cp_commit>
//   ranges <h_par, eight hex digits per word>  written if and only if the worst-case check passes
//   seed test                       written if and only if cp_commit has -test-seed
//   seed public                     in its place, if and only if cp_commit has -public-weights
// No key protects isk.txt: anyone who can write into isk can write any of its lines.
// M, first: the leaf tables first .. M - 1 are live, and cp_prove samples among them. The call that
// creates a commitment writes no line `first`, and cp_commit rejects a later call that would raise
// M or `first` (`cp-commit-first-form-alone`).
// ranges: cp_commit's worst-case check is meant to show that, with the committed weights and the
// configuration files that have this digest, no prompt can drive a value of the forward pass out of
// its range (prover/weights_commitment/worst_case.cuh). cp_prove starts only if the line has h_par
// of its own configuration files (Prover::the_ranges_are_guarded() in prover/prover/isk.cuh); a
// test build's cp_prove also starts if the line "seed test" is present. A commitment to public
// weights ("seed public") may have no line ranges: under -public-weights cp_commit leaves the check
// out unless -worst-case forces it, and cp_prove given -public-weights then starts without the line
// (nothing is hidden, so a weight-dependent "failed" reveals nothing). The "if and only if" above
// still holds: no line ranges is ever written unless the check ran and passed.
// seed test: cp_commit read the seed from a file given with the test option -test-seed, so nothing
// is hidden. A production build has no -test-seed, and its cp_prove rejects a commitment that has
// the line [cp-isk-seed-test]. Its cp_commit has the same check in the code for a call that adds
// to an existing commitment, and never reaches it: it rejects every such call earlier
// (`cp-commit-first-form-alone`).
// seed public: a commitment to public weights. Its seed is derived from the file given with
// -public-weights, which may be published, so nothing is hidden: see
// prover/common/public_weights.h. cp_prove accepts the line and its own -public-weights only
// together [cp-isk-seed-public].
struct IskText {
  uint32_t version = 0, M = 0, first = 0;
  int keep = 0;
  std::string weights; // "generated", "made-up-blocks" or a directory
  std::string ranges;  // h_par in hex, as digest_hex() writes it; empty: the line is absent
  bool test_seed = false, public_seed = false; // whether the line "seed test" or the line "seed public" is present; never both
};
// Whether the line weights names synthetic weights (`made_up` in identifiers), which the programs
// generate without reading a file. Any other value names a directory with a model's files.
static inline bool made_up(const IskText &t) { return t.weights == "generated" || t.weights == "made-up-blocks"; }
// A digest's words, each as eight hex digits, the first word first: as the programs print h_par.
static inline std::string digest_hex(const uint32_t *words, size_t n) {
  std::string hex;
  char eight[9];
  for (size_t i = 0; i < n; i++) snprintf(eight, sizeof eight, "%08x", (unsigned)words[i]), hex += eight;
  return hex;
}
// Leaves out the line `first` if t.first is 0, the line `ranges` if t.ranges is empty and the seed
// line if neither flag is set. So a text in this form that lacks one of these lines is,
// byte for byte, what this function returns for its values, and cut_isk_text() accepts it.
static inline std::string isk_text(const IskText &t) {
  return "version " + std::to_string(t.version) + "\nM " + std::to_string(t.M) +
         (t.first ? "\nfirst " + std::to_string(t.first) : std::string()) + "\nkeep " + std::to_string(t.keep) + "\nweights " +
         t.weights + "\n" + (t.ranges.empty() ? std::string() : "ranges " + t.ranges + "\n") + (t.test_seed ? "seed test\n" : t.public_seed ? "seed public\n" : "")
      ;
}
// Returns false unless (1) the text is exactly what isk_text() gives for the values read, (2)
// `first` is 0 or below M, (3) the line `ranges`, if present, has 64 lower-case hex digits, and (4)
// the line `weights` has a value. After false, t may be partly overwritten.
static inline bool cut_isk_text(const std::string &text, IskText &t) {
  unsigned version = 0, M = 0, first = 0;
  int keep = -1, at = 0;
  if ((sscanf(text.c_str(), "version %u\nM %u\nkeep %d\nweights %n", &version, &M, &keep, &at) != 3 || !at) &&
      (sscanf(text.c_str(), "version %u\nM %u\nfirst %u\nkeep %d\nweights %n", &version, &M, &first, &keep, &at) != 4 || !at))
    return false;
  const size_t end = text.find('\n', (size_t)at);
  if (end == std::string::npos) return false;
  t.version = version, t.M = M, t.first = first, t.keep = keep, t.weights = text.substr((size_t)at, end - (size_t)at);
  t.ranges.clear();
  size_t rest = end + 1;
  if (!text.compare(rest, 7, "ranges ")) {
    const size_t stop = text.find('\n', rest);
    if (stop == std::string::npos) return false;
    t.ranges = text.substr(rest + 7, stop - rest - 7), rest = stop + 1;
    if (t.ranges.size() != 64 || t.ranges.find_first_not_of("0123456789abcdef") != std::string::npos)
      return false;
  }
  t.test_seed = !text.compare(rest, std::string::npos, "seed test\n");
  t.public_seed = !text.compare(rest, std::string::npos, "seed public\n");
  return !t.weights.empty() && (!t.first || t.first < t.M) && isk_text(t) == text;
}
static inline bool read_isk_text(const std::string &path, IskText &t) {
  std::string text;
  return read_file(path, text) && cut_isk_text(text, t);
}
static inline bool read_isk_text(const SecretFolder &isk, IskText &t) { // through isk's descriptor
  std::string text;
  return read_file(isk.fd, ISK_TEXT_FILE, text) && cut_isk_text(text, t);
}

// ---- the draw of a batch's leaf table ----
// Kbat is the 32 bytes of a batch's first key (coin_keys::batch_first_key() in
// prover/common/coin_keys.h). A batch's leaf table is number first + draw_leaf(Kbat, M - first): it
// is sampled among the live ones, and the parameter M here is their number. v = the first 8 bytes
// of f(Kbat; "CPLF", t, 0, 0), little-endian, for t = 0, 1, ..; the first v below
// M * floor(2^64 / M) is used, and the result is v mod M. M must not be 0. f is the ChaCha20 block
// function as prover/common/coin_chacha.h defines it.
// Source: the rejection rule: Lemire, "Fast Random Integer Generation in an Interval",
// arXiv:1805.10941, section 3.1, Algorithm 3 with the variation in the section's last paragraph (of
// 2^L equally likely values the last 2^L mod s are rejected, and an accepted value is reduced
// modulo s). Here L = 64 and s = M. The paper reviews the rule as a known technique.
static inline uint32_t draw_leaf(const uint8_t Kbat[32], uint32_t M) {
  const uint64_t over = (UINT64_MAX % M + 1) % M; // 2^64 mod M; v >= 2^64 - over is skipped
  for (uint32_t t = 0;; t++) {
    uint8_t block[32];
    cp_key(block, Kbat, CP_LF, t, 0, 0);
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)block[i] << (8 * i);
    explicit_bzero(block, sizeof block);
    const bool taken = !over || v < 0 - over;
    const uint32_t k = (uint32_t)(v % M);
    explicit_bzero(&v, sizeof v);
    if (taken) return k;
  }
}
// ---- a proof file's header ----
// Three words, little-endian: PROOF_TAG, the version word and the leaf table's number k (from 0).
static inline std::string proof_head(uint32_t version, uint32_t k) {
  std::string head;
  for (uint32_t word : {PROOF_TAG, version, k}) put_word(head, word);
  return head;
}
} // namespace cp
