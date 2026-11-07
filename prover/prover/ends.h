// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// How the prover cp_prove exits. After arm_ends() an exit goes through stop(): whatever the cause,
// it blocks the release of responses, runs the wipe, prints the last line (if any), calls _exit().
// stop() lists where the wipe is skipped, cut short or fails. Exit codes (`status` in code):
//   0  the run finished its work, every response due has its final file name, no wipe step failed
//   9  a self-check on secret random values or keys failed (a name that begins "own-test-")
//   1  anything else; before arm_ends() see end_of_main()
// After 9, find the cause before starting the program again: a supervisor that restarts it on
// the same machine within a minute defeats the purpose of the stop.
// A production build has restricted output (`closed_front`, prover/prover/closed_front.h): once it
// holds a secret, it prints about its exit only the last line, "p258 end <code>" and a name if one
// was set (prover/common/end_name.h), and before it, if not every step of the wipe ran, a line that
// says so (CLOSED_FRONT_WIPE_LINE). It never prints the cause's number, which can depend on hidden
// values.
// After a failure a test build's last line, "ended: ..", also has the number, but not at the wipe's
// time limit. A cause that depends on hidden values (weights, tokens, private randomness) neither
// ends the process early nor changes the exit code, except for code 9: see g_batch_cause.
#pragma once
#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <fcntl.h>
#include <new>
#include <string>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>
#include "base.h" // blocks/base.h: at_a_failed_guard
#include "prover/closed_front.h"

// The causes, by number; `Refused` in a name means rejected. A slot is a request's place in its
// batch; names in capitals are bits of its failure word (SlotFlag, prover/common/slot_flags.h).
enum class End {
  None = 0,  // success; for a slot: the request is proved
  Usage = 1, // an uncaught exception except bad_alloc; in `end_on_exit`, a status other than 1 or 2
  // An input file is missing, malformed or not supported; the system's random generator failed; or
  // an internal check that calls exit(2) failed. For a slot: a value of the model's forward pass
  // left its range (UNFIT). No overflow of a lookup's count is reported (ForestTable::planes(),
  // prover/hiding/forest_table.cuh).
  Refused = 2,
  FoldRange = 8,      // a combined vector's entry left its window (WINDOW); no overflow of a lookup's count is reported
  Gpu = 9,            // a CUDA call failed, out of memory included (the GPU engine's own calls: GuardFailed)
  // A signal asked the program to end: TERM, INT or HUP; with restricted output, every signal of
  // each_signal_that_asks() (prover/prover/closed_front.h).
  Terminated = 14,
  Coins = 15,       // what the draws gave cannot be used (NO_ELEMENT, ZERO_ELEMENT)
  WriteFailed = 17, // a file or directory cannot be created, written or synced
  HostMemory = 18,  // no host memory
  NotZeroed = 19,   // everything else went well, but something could not be zeroed at the end. Its name is also printed for a wipe given up at its time limit: see g_wipe_seconds
  Stopped = 20,     // the prover's check of its own work failed: a fault of the program or the machine
  GuardFailed = 21, // a BLK_REQUIRE check or a CUDA call of the GPU engine failed: see `stop`
  // A number that is derived from a request's values and used as an index was outside its table
  // (INDEX; prover/common/index_guard.cuh).
  IndexOutside = 22,
  // Two alarms of a batch. No request loses its files because of them. The spool directory
  // (`Spool`, prover/prover/spool.h) has the waiting requests in in/ and the responses in out/.
  // MemoryGrew is not reset: from the first rise on, every batch without another cause gets it. It
  // can come without a fault: see ForkMemory::note_what_is_taken() (prover/prover/fork_memory.h).
  MemoryGrew = 24, // the peak of the prover's own GPU memory rose above its peak at the end of the process's first pass
  CutShort = 25,   // in/ had more names than one scan reads, so those served need not be the oldest
  // For a slot only: the checked tokens' costs exceed the parameter file's bounds (ABOVE). This is
  // a result, not an error.
  Above = 26,
  // A self-check on secret random values whose result is carried as a flag has failed: on the
  // sub-proofs' one-time pads, or on the masks, the salts or the second copy of the key of the
  // batch's test on the weight table (the committed weights): Prover::self_check_failed,
  // prover/prover/prover.h. The process exits at the next end of a batch, with the self-check's
  // name in the last line; with the option -overlap that can be an earlier batch's end, while a
  // later batch is being proved. Stopped is a different cause: it also covers checks that depend on
  // weights and prompt, and it does not end the process.
  SelfCheck = 29,
  // An alarm of the service, and no batch's cause: in/ holds a file with a request's name and
  // length that the service may not read (Spool::unreadable). The file stays there, unanswered.
  Unreadable = 30
};
// The cause for a slot's failure word. ABOVE, by itself or with OWN_TEST only, gives End::Above.
// Otherwise the lowest set bit up to UNFIT decides; without one (FLAGS_UNREADABLE alone, say) the
// cause is End::Usage.
static inline End cause_of(unsigned slot_flags) {
  return !slot_flags       ? End::None
         : (slot_flags | 16u) == (64u | 16u) ? End::Above // ABOVE
         : slot_flags & 3  ? End::Coins         // NO_ELEMENT, ZERO_ELEMENT
         : slot_flags & 4  ? End::FoldRange     // WINDOW
         : slot_flags & 8  ? End::IndexOutside  // INDEX
         : slot_flags & 16 ? End::Stopped       // OWN_TEST
         : slot_flags & 32 ? End::Refused       // UNFIT
                           : End::Usage;
}

// The wipe: removes and zeroes the program's secrets; stop() calls it whatever the cause. It
// returns whether every step ran and none failed; a GPU fault fails its GPU steps (see stop()).
// True does not mean that no secret stays: the GPU engine's memory pool (prover/engine/device.cuh)
// is never wiped. More of what stays: prover/docs/ends-in-short.md, "In memory and on disk".
// Code that adds a secret installs a function that also calls the old one.
static bool (*g_end_wipe)() = nullptr;
// Called at the start of stop() and of end_on_exit(), before the wipe's first step: it blocks the
// release of responses (Spool::shut(), installed by Prover::open_the_spool(),
// prover/prover/service.cuh). The other threads run until _exit(), while the wipe zeroes keys and
// masks that they use, so nothing that they produce from then on may reach anyone. It must take no
// lock, allocate nothing and wait for nothing.
static void (*g_end_first)() = nullptr;
// The wipe's time limit, in seconds; a test sets a shorter one. The wipe waits for the GPU near its
// start and removes the staged files last. Without a limit, a GPU that no longer answers would
// block it for ever: no last line, the spool's lock held, SIGTERM without effect. At the limit
// stop() gives up the wipe and exits with the name "not-all-wiped", or with code 9 and that code's
// name. What the wipe has not reached stays: GPU memory may still hold weights and masks,
// and staged files may remain. The limit is long because a wipe cut short leaves secrets, while
// waiting costs only time. It does not help while the main thread sleeps uninterruptibly in the
// kernel (a dead driver, a hung disk): the kernel offers the alarm's signal to that thread first.
static unsigned g_wipe_seconds = 600;
// Set by any code that finds, before the exit, that something could not be zeroed.
static std::atomic<bool> g_zeroing_failed{false};
// Set when the run has finished its work: the service has ended and released every response due.
static bool g_end_at_the_last_line = false;

// Causes that depend on hidden values. A request's own cause (a bit in its slot's failure word)
// costs only that request its files: its response is a `.failed` file, not a proof. A cause that
// belongs to no single request (fail_the_batch(): the batch's test on the weight table, a bit set
// between two requests, a table of set-up constants that a run has changed) costs every request of
// the current batch its files; with -overlap it can also cost the other batches in progress theirs
// (Prover::end_the_batch(), prover/prover/service.cuh). The next batch starts clean. In both cases
// the batch is proved to its end, on substitute values where necessary, through the same stages,
// and the process does not exit because of the cause. Within a stage the GPU launches can differ,
// and whether the running times are equal has not been measured. Only a failure of the machine or
// the program, or an error of the user, ends the process early: a rejected command line or input
// file, a failed CUDA call, a failed internal check, no memory, a file that cannot be written, a
// signal.
//
// A failed self-check on secret random values or keys is an exception: it always ends the service,
// because a machine or a program that has lost one random value must make no further proof. Some
// of these checks compare at a fixed point of the program and end the process there, by exit() or
// by BLK_REQUIRE_NAMED (blocks/base.h: an internal check with a name for the last line). They are
// about random values and keys, not weights, and no way is known for a client to make one fail, so
// that point is taken to show only where the fault was noticed. The others end nothing where they
// fail: see End::SelfCheck. Whatever has no final file name when such a check fails (the batches in
// progress) gets none from this process. docs/limits-all-rows.tsv, row C3, lists three exceptions.
// One is a batch whose files are being named at that moment: it is named to the end (Spool::shut(),
// prover/prover/spool.h).
//
// g_batch_cause is the first cause in the current batch that belongs to no single request, or
// End::None.
static std::atomic<int> g_batch_cause{0};
static void fail_the_batch(End why) {
  int none = 0;
  g_batch_cause.compare_exchange_strong(none, (int)why);
}
// Call it before the batch's release: it clears g_batch_cause for the next batch.
static End take_the_batchs_cause() {
  return (End)g_batch_cause.exchange(0);
}

// The name for the last line, for the operator only, or null: for the causes that a slot's failure
// word or g_batch_cause can give (1, 2, 8, 15, 20, 22, 26) a name would depend on hidden values. A
// failed CUDA call gets its name, "cuda-call", where it is noticed (CK in prover/common/common.h,
// prover/engine/device.cuh, end_on_exit()) and not here: exit(1), which becomes Gpu, is also what
// dozens of internal checks call.
static const char *name_of(End machines) {
  return machines == End::Terminated    ? "asked-to-end"
         : machines == End::WriteFailed ? "write-failed"
         : machines == End::HostMemory  ? "no-host-memory"
         : machines == End::NotZeroed   ? "not-all-wiped"
         : machines == End::GuardFailed ? "guard"
                                        : nullptr;
}

[[noreturn]] static void stop(End code) {
  // After arm_ends() these exits come here: a return from main() (end_of_main()), exit()
  // (end_on_exit()), an uncaught exception, a signal that asks the program to end (at the next
  // stop_if_asked()), and GuardFailed. GuardFailed is a failed internal check of
  // blocks/ (BLK_REQUIRE, `guard` in identifiers: a mistake of the program itself, never a proof's
  // bytes) or a failed CUDA call of the GPU engine (prover/engine/device.cuh), out of GPU memory
  // included. Its line "file:line: text" goes to stderr, then blocks/base.h calls the function that
  // arm_ends() has installed, also from the translation units that own the weight table
  // (prover/prover/opening_in_process.cuh). A search for a proof-of-work nonce that finds none is
  // GuardFailed: a BLK_REQUIRE check in EngineGpu::search_nonce(), prover/engine/engine_gpu.cuh.
  //
  // These exits do not come here. They wipe nothing, or not everything:
  //   - SIGKILL;
  //   - the fault signals SEGV, BUS, FPE and ILL, and SYS and TRAP;
  //   - every abort() (it raises SIGABRT), among them: PSC_STOP() on the host
  //     (prover/hiding/psc_device.cuh); the four in prover/hash_commitment/hf_mask_ref.h; a failed
  //     BLK_REQUIRE check before arm_ends(), or in a wipe that such a check has begun; a library's
  //     abort().
  // At a fault signal or an abort() a test build dies of the signal, without a last line. With
  // restricted output ClosedFront::on_fault() catches the signal and prints an ordinary last line
  // (exit code 1, or 9 after a failed self-check). If the program has begun to read its secrets,
  // the line before it says that not every step of the wipe ran (CLOSED_FRONT_WIPE_LINE).
  //
  // A GPU fault (such as a kernel's illegal access or trap) leaves the CUDA context unusable: an
  // exit that comes here then cannot zero GPU memory; the wipe goes on and reports a failed step.
  // The wipe is cut short at its time limit (g_wipe_seconds) and at a re-entry: only the first
  // caller runs the steps below (others wait), and its thread re-enters if a step throws, fails a
  // BLK_REQUIRE check or calls exit() (forbidden: see end_on_exit()). It then exits, not all
  // zeroed; only under restricted output does it print CLOSED_FRONT_WIPE_LINE and a last line.
  static std::atomic_flag begun = ATOMIC_FLAG_INIT;
  static thread_local bool first_caller = false;
  static std::atomic<const char *> name{nullptr}; // the end's name: see below
  if (begun.test_and_set()) {
    if (first_caller) {
      if (closed_front.on) closed_front.leave(1, name.load());
      _exit(names_a_self_check(name.load()) ? 9 : 1);
    }
    for (;;) pause();
  }
  first_caller = true;
  if (g_end_first) g_end_first();
  // The name for the last line is taken here, before the wipe's first step: the name that a
  // self-check has set by now, or else name_of(code). A self-check that fails on
  // another thread from now on (a random value requested without a key, a second copy of a key that
  // is all zeros) is a consequence of the wipe (see g_end_first), and its name would mislead. A
  // name that is set from now on is lost.
  if (const char *machines = name_of(code)) name_the_end(machines);
  name = take_the_ends_name(); // also closes the variable: a fault's handler finds no later name
  if (code == End::None && names_a_self_check(name.load())) code = End::SelfCheck; // a self-check has failed before this exit, so it is not a success
  // A cause is still in g_batch_cause: the current batch has produced nothing valid.
  if (code == End::None)
    if (const int failed = g_batch_cause.load()) code = (End)failed;
  // The wipe's time limit (g_wipe_seconds) is an alarm. Its handler replaces whatever was set for
  // SIGALRM, also if the signal was ignored, and this thread unblocks the signal. A SIGALRM that
  // comes before the limit, from outside or left pending, does not end the wipe. At the limit exit
  // code 9 keeps its name, which tells the operator to find the cause before a restart;
  // name_of(End::NotZeroed), which tells the machine's state, replaces every other name.
  static std::atomic<long long> limit{0};                   // on the monotonic clock, in seconds
  static std::atomic_flag last_line = ATOMIC_FLAG_INIT;     // taken by whoever prints the last line: the handler, or this routine after the wipe
  struct timespec now {};
  clock_gettime(CLOCK_MONOTONIC, &now);
  limit = (long long)now.tv_sec + g_wipe_seconds;
  struct sigaction at_the_limit {};
  at_the_limit.sa_flags = SA_RESTART;
  at_the_limit.sa_handler = [](int) { // of the system's functions it calls only `clock_gettime`, `write` and `_exit`, which a signal handler may call
    struct timespec at {};
    clock_gettime(CLOCK_MONOTONIC, &at);
    if (at.tv_sec < limit.load() || last_line.test_and_set()) return;
    const char *const set = name.load();
    const char *const last = names_a_self_check(set) ? set : name_of(End::NotZeroed);
    char line[256] = "ended: the wipe was given up at its time limit, so GPU memory and out/.tmp/ may still hold secrets (prover/docs/ends-in-short.md), every step of the wipe ran: 0, by ";
    size_t n = 0;
    while (line[n]) n++;
    for (const char *c = last; *c && n + 1 < sizeof line; c++) line[n++] = *c;
    line[n++] = '\n';
    const ssize_t written = closed_front.on ? 0 : write(1, line, n); // a build without restricted output says it in a sentence
    (void)written;                                                   // the process ends in the same way if the sentence cannot be written
    closed_front.leave(1, last); // `leave` turns 1 into 9 for a self-check's name; without restricted output it prints nothing and only exits
  };
  sigset_t alarm_alone;
  sigemptyset(&alarm_alone), sigaddset(&alarm_alone, SIGALRM);
  sigaction(SIGALRM, &at_the_limit, nullptr), pthread_sigmask(SIG_UNBLOCK, &alarm_alone, nullptr);
  alarm(g_wipe_seconds);
  const bool wiped = !g_end_wipe || g_end_wipe();
  alarm(0);
  if (last_line.test_and_set())
    for (;;) pause(); // the handler is ending the process on another thread
  const bool zeroed = wiped && !g_zeroing_failed; // true or not, secrets are left: see g_end_wipe
  if (zeroed) closed_front.wipe_is_due = false;   // otherwise `leave` says, before the last line, that not every step ran
  if (code == End::None && !zeroed) code = End::NotZeroed;
  if (!name.load()) name = name_of(code); // a cause that arose in the wipe: a step failed
  const int status = code == End::None ? 0 : names_a_self_check(name.load()) ? 9 : 1;
  if (closed_front.bytes_are_kept && !closed_front.on) fputs(CLOSED_FRONT_KEPT_LINE, stdout); // the line on the batch's kept random bytes; with restricted output `leave` prints it
  SERVING_OR(, if (code != End::None) printf("ended: cause %d (prover/ends.h), every step of the wipe ran: %d%s%s%s\n", (int)code, (int)zeroed,
                                             name.load() ? ", by " : "",
                                             name.load() ? name.load() : "",
                                             code == End::GuardFailed ? ", an internal check or a CUDA call of the GPU engine failed: see the line \"file:line: ..\" on standard error" : ""));
  fflush(stdout);
  fflush(stderr);
  if (closed_front.on) {
    // An exit with End::None before g_end_at_the_last_line is set comes in the middle of the work
    // and gets exit code 1.
    closed_front.leave(!status && !g_end_at_the_last_line ? 1 : status, name.load());
  }
  _exit(status);
}

constexpr int END_HANDLERS = 64; // how often `arm_ends` registers the handler: see `end_on_exit`
// exit(1), as after a failed CUDA call under CK (prover/common/common.h), becomes Gpu; exit(2), as
// after rejected input, Refused; every other status Usage. Status 5 is used only after a failed
// CUDA call (CK in prover/weights_commitment/common.cuh; prover/weights_commitment/lent_memory.h;
// prover/hash_commitment/hf_keeper.cu), by code that sets no name, so the name is set here.
static void end_on_exit(int status, void *) {
  // Every exit() must come here, not only the first. One that is called on another thread while an
  // exit is in progress must wait for that exit in stop(), and not run on to the older handlers and
  // to _exit() during the first caller's wipe. But the C library removes a handler from its list
  // before it calls it (glibc, stdlib/exit.c), so the handler first registers itself again. The
  // library also releases the list's lock between the removal and the call, so arm_ends() registers
  // the handler END_HANDLERS times: each thread removes one and adds one, and none returns, so
  // another thread's exit() finds one as long as fewer than END_HANDLERS threads are between these
  // two steps at one instant. How many threads the program can have has not been counted.
  // What was registered later than this handler, such as the destructors of statics constructed
  // since arm_ends(), has by then run on the thread that called exit(), possibly during the wipe.
  // No step of the wipe may call exit(): in some C libraries (glibc from 2.41 on, and older ones
  // that carry the patch) exit() holds one lock for its entire duration, a thread that waits here
  // never releases it, and the wipe would hang until its time limit.
  on_exit(end_on_exit, nullptr);
  // before the flushes, which can block on a full pipe: once an exit has begun, nothing more gets a
  // final name in out/, but for the exceptions of row C3 (docs/limits-all-rows.tsv) (`stop` calls
  // it again)
  if (g_end_first) g_end_first();
  fflush(stdout); // the message of the code that called exit() is printed before anything else
  fflush(stderr);
  if (status == 5) name_the_end("cuda-call");
  // exit(0) in the middle of the work is not a success: only end_of_main() passes End::None.
  stop(status == 1 ? End::Gpu : status == 2 ? End::Refused : End::Usage);
}
// The handler of the signals that ask the program to end only sets this flag. The main thread
// checks it between steps with stop_if_asked(), because zeroing memory and deleting files are not
// safe inside a signal handler.
static std::atomic<int> g_stop_asked{0};
static_assert(std::atomic<int>::is_always_lock_free,
              "g_stop_asked is written by a signal handler, so it must be lock-free");
static void stop_if_asked() {
  if (g_stop_asked) stop(End::Terminated);
}
static bool g_ends_armed = false;
static void arm_ends() {
  // Call it after the CUDA runtime is initialized: exit handlers run in the reverse order of their
  // registration, and end_on_exit() must still be able to zero GPU memory. No key is chosen and no
  // weight is read before this call. From here on the kernel writes no image of the process's
  // memory after a trap or a fault. The limit prevents a core file; a machine that pipes core dumps
  // to a program does not apply the limit (core(5)), so the process is also made non-dumpable.
  // (With restricted output ClosedFront::enter() has done both already.) A debugger that starts the
  // program, or that root attaches, still works. Neither step covers what a library does on its own
  // under an environment variable, such as the CUDA driver's dump of GPU memory. Against that,
  // cp_prove starts again with a reduced environment. With restricted output ClosedFront::enter()
  // always does so. Otherwise second_start_unless_nothing_is_hidden() (prover/prover/cp_prove.cu)
  // does, unless the weights are synthetic and the seed is a test seed, `isk.txt` in the secret key
  // directory cannot be read, or a test's environment variable is set.
  const struct rlimit no_core = {0, 0};
  if (setrlimit(RLIMIT_CORE, &no_core) || prctl(PR_SET_DUMPABLE, 0)) {
    printf("core files cannot be disabled\n");
    exit(2); // before on_exit() below: the process holds no secret yet
  }
  g_ends_armed = true;
  for (int i = 0; i < END_HANDLERS; i++) on_exit(end_on_exit, nullptr);
  blk::at_a_failed_guard = [](const char *name) noexcept { // the check's own name, if any, is set first, and the first name stays (else "guard")
    if (name) name_the_end(name);
    stop(End::GuardFailed);
  }; // runs on the thread where the check failed. noexcept: an exception from the wipe must not reach the check's callers, who might catch it
  std::set_terminate([] { // an uncaught exception
    End code = End::Usage;
    try {
      if (auto e = std::current_exception()) std::rethrow_exception(e);
    } catch (const std::bad_alloc &) {
      code = End::HostMemory;
    } catch (...) {
    }
    // M5g: an uncaught exception that is not bad_alloc reaches stop() as Usage with no name and
    // prints a bare `p258 end 1`. Name it so the operator sees which class of end it was. First
    // name wins, so a real name a self-check set before the throw stays.
    if (code == End::Usage) name_the_end("uncaught-exception");
    stop(code);
  });
  struct sigaction ask {};
  ask.sa_handler = [](int) {
    g_stop_asked = 1;
  };
  ask.sa_flags = SA_RESTART;
  // With restricted output no signal of each_signal_that_asks() may end the process with nothing
  // zeroed and no last line. Until this call ClosedFront::on_early_signal() catches them.
  std::vector<int> asking = {SIGTERM, SIGINT, SIGHUP};
  if (closed_front.on) asking.clear(), each_signal_that_asks([&](int s) { asking.push_back(s); });
  for (int s : asking) {
    struct sigaction old {};
    sigaction(s, nullptr, &old);
    // A signal that the parent ignored, as nohup does, stays ignored. A handler that a library has
    // installed since ClosedFront::enter() stays too: the CUDA runtime is initialized by now.
    if (old.sa_handler == SIG_DFL || old.sa_handler == ClosedFront::on_early_signal) sigaction(s, &ask, nullptr);
  }
}

// For main(): turns the run's return status into an exit, 0 into a successful one. It calls stop()
// instead of returning from main(): inside exit(), the static objects of a library loaded after
// end_on_exit() was last registered are destroyed before end_on_exit() runs, and the library could
// then not zero everything that it holds. Before arm_ends() (`g_ends_armed` is false) the process
// holds no secret: a test build returns the status as it is (1: the command line; 2: a rejected
// input, as [cp-in-name]), a production build 1 for any status but 0. SERVING_OR(a, b)
// (prover/common/setting.h) is a in a production build and b in a test build.
static int end_of_main(int status) {
  if (g_ends_armed && !status) stop(End::None);
  if (g_ends_armed) end_on_exit(status, nullptr);
  return SERVING_OR(status ? 1 : 0, status);
}
