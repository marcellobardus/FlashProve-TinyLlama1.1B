# When something goes wrong

This document follows on from [Operator's duties](operators-duties.md) and explains what to do when
`cp_commit` or `cp_prove` rejects something, fails, crashes or hangs. It uses five terms that
Operator's duties [defines](operators-duties.md#how-the-service-works): *hold*, *named person*, *in
doubt*, *good exit* and *bad exit*. What each ending means inside the programs is told in [how a run
ends](../prover/docs/ends-in-short.md). Whether you may start again is decided here. A row ID such
as R18.4 names a row of [the table of all rows](limits-all-rows.tsv).

If a program's message and these two documents disagree, follow whichever is stricter. That goes for
removing things too: do not remove anything unless both allow it. Never change permissions, or move
or remove anything, to get past an error. The only exceptions are the ones these two documents name,
each with its own conditions. Here they are [`refused (cp-isk)`](#cp-isk),
[`cp_prove -clear`](#clear), [the alarm `unreadable-request`](#unreadable-request), [a closed
directory of your own](#a-closed-directory-of-your-own) and, for what a killed `cp_commit` leaves
behind, [A killed or failed cp_commit](#a-killed-or-failed-cp_commit). The runbook lists [its own
two](operators-duties.md#read-this-first). If in doubt, leave the service down.

## Always first

1. **Put the service on hold** in any of these cases:
   - `cp_prove` ends in a [bad exit](operators-duties.md#how-the-service-works). The two exceptions:
     a `TERM` that you sent yourself to a working service, with no line on the wipe before the last
     line (row `asked-to-end` below), and a start that was rejected as [In use](#cp-isk).
   - Your own `cp_verify` check of a response ends with a line that begins `NO:` or is
     `NOT-A-VERIFICATION`, or gives no answer twice. First rule out a mistake in your own call: a
     wrong policy gives `NO: policy`, a mistyped commitment ID gives `NO: model-statement`, and
     leaving out `-id` gives `NOT-A-VERIFICATION`. So does `-no-proof-of-work` where a round needs
     its proof of work to meet the target. With that option the line comes for every honest response
     at such a setting, so by itself it says nothing against the service, as long as
     `no-proof-of-work` is the only cause that the lines after `---- the call ----` name. In every
     other case, if the call was right, the hold stands.

   To put it on hold:
   - First, stop sending clients anything that this process, or a later one on the same machine,
     produced under this commitment. That includes accepted responses and "failed" answers.
   - Make sure your front end stops handing out stored responses too. With the example front end,
     run `python3 integrations/proof_middleware.py hold CONFIG POLICY` as the server's account, for
     every policy on that machine, or stop vLLM, which is more reliable. If you stop it, lay the
     hold before it starts again: a server that is started anew knows nothing of a rejection that
     the old one could not write down ([integrations/LIMITS.md](../integrations/LIMITS.md)). The
     `hold` file doesn't stop `cp_prove` ([the example front
     end](service.md#the-example-front-end)).
   - Make sure nothing restarts `cp_prove`.
   - Then stop every `cp_prove` still running on that secret key directory, using its spool
     directory's [`stop` file](operators-duties.md#stopping-and-starting). Stopped this way,
     `cp_prove` exits with code 0, so a front end that reacts only to a bad exit won't notice.
   - Only the named person lifts the hold. With the example front end that takes its `lift-hold`
     command ([the hold file](../integrations/vllm/REFERENCE.md#start-the-proving-service)).
2. **Keep what the program printed,** and its exit code. Nothing else records how it ended, and a
   restarted service just keeps serving.
3. **Do not give anyone unreleased responses.** After a bad exit, staged responses can be left under
   `out/.tmp/`. Nothing has checked them. A response that `cp_verify` rejects can contain unmasked
   hidden values, and a staged response may be one of these. A response that your own `cp_verify`
   did not accept, and the rest of its batch, are not among these: once their requests have left
   `work/`, they stay in `out/` under their final names, and no start and no `cp_prove -clear`
   removes them. Before the hold is lifted, make sure that your front end can never hand them out.
   With the vLLM integration: look first for a response that turned up late
   ([integrations/LIMITS.md](../integrations/LIMITS.md#narrow-exceptions)). At a rejection the
   example front end overwrites each proof of the batch, deletes it and lays a "failed" in its place
   (narrow exceptions, none open to a client:
   [integrations/LIMITS.md](../integrations/LIMITS.md#narrow-exceptions)); a file
   `withheld.POLICY.TIME.json` in its jobs directory whose name lacks `.done.json` means that it has
   not finished: do not remove it by hand; that no such file lies there does not show that it has
   finished. At every other outcome of its check it overwrites nothing, and the files stay in
   `out/`: keep that directory as secret as the secret key directory.
   - Do not read, copy, remove or pass on anything from there or from `work/`.
   - Keep the spool directory as secret as the secret key directory.
   - Once the hold is lifted, a next start on that spool directory whose requests have the same
     length as those left in `work/` overwrites what is staged with zeros and removes it itself. If
     no service is started there again, then before you give up the directory, run
     [`cp_prove -clear`](#clear) on it, and on no other. Neither can stop a disk or a file system,
     such as a copy-on-write one, from keeping the old contents of overwritten blocks.
4. **Find your row in the tables below.** Start again only when everything in [Before you serve
   again](#before-you-serve-again) holds.

## What you see, and what to do

Until it reads its first secret, the production build of `cp_prove` prints ordinary messages, with
one sentence for each rejection. Then it prints a line that begins `p258 closed`, and after that
only fixed lines ([the list](../prover/docs/closed_front.md)). The last one is
`p258 end <exit code>`, usually followed by a name. Which line appears can depend on hidden values,
so these lines are only for you: do not pass any of them, or the exit code, on to a client.
`cp_commit` prints its messages in full.

<a id="an-empty-policy"></a>Read the exit code and the last line together ([the
names](../prover/docs/ends-in-short.md#the-names-in-the-last-line); what the short cells below leave
out: row R35). A row may tell you to make a new commitment ([A new
commitment](operators-duties.md#a-new-commitment)). A response that `cp_verify` rejects, a failed
self-check and an unexplained exit can each be the first sign of a fault that began earlier
([limits](limits.md#a-lost-mask)).

**A line before the last.** From reading the code, the production build of `cp_prove` can print one
more line before its last line. It begins `p258 not every step of the wipe ran`, and goes on to say
what may still hold secrets. Then treat the machine's memory and the staged files as holding secrets
([Always first](#always-first)). After `KILL`, a loss of power or a crash that the program cannot
catch, neither line is printed, and nothing was wiped. Only if the last line comes after `p258
closed` without this line has the program counted every step of the wipe as run. Even then secrets
can still be in memory ([Keys, seeds and machines](operators-duties.md#keys-seeds-and-machines)).
After exit code 9, treat the machine's memory as not wiped whether or not this line was printed.

**`cp_prove`**

| Exit code | Last line or name | What it means | What to do |
|---|---|---|---|
| 0 | `p258 end 0` | It stopped on request and released every response that was due, and no step of its wipe failed. That does not show that any request was proved, because every response can be "failed". Nor does it show that a proof verifies: `cp_prove` runs no verifier. The wipe does not reach all GPU memory ([Keys, seeds and machines](operators-duties.md#keys-seeds-and-machines)). | A good exit. |
| 1 | a sentence, then `p258 end 1` with or without a name, and no line `p258 closed` | It rejected the call, a file or the machine before it read a secret. If the name begins `own-test-`, the exit code is 9, so use that row. | Read the sentence and fix what it names. Do not remove, move or restore anything to make the call succeed. The exceptions, and single names: [A rejection at the start](#a-closed-directory-of-your-own). |
| 1 | a sentence that holds `parameter-range:` and ends `answers` | The line `answers` of the secret key directory's `policy.txt` says another number than 1. `cp_commit` and `cp_prove` refuse a parameter file whose `answers` is not 1, on synthetic weights and under `-public-weights` too (`parameter-range`). | Do not edit the file, and do not add `-public-weights`: it does not lift the rejection, and under it nothing is hidden. This repository's `cp_commit` refuses such a file too, so either other programs made the commitment or someone has written to the secret key directory: the commitment is [in doubt](operators-duties.md#if-the-count-is-lost-or-in-doubt). Do not give its public key directory to anyone, and make [a new commitment](operators-duties.md#a-new-commitment) from a parameter file that says `answers 1` ([A record](operators-duties.md#a-record)). |
| 1 | `refused (cp-isk)` | The name given with `-isk`, the path, the directory, its lock or `isk.txt` is wrong. | See [`refused (cp-isk)`](#cp-isk). |
| 1 | `cp-isk-seed-public` | The option `-public-weights`, its 32-byte file and the line `seed public` in `isk.txt` do not match ([public weights](operators-duties.md#public-weights)). | If you committed with the option, pass it again with the same file. If you did not, leave it out, or check `-isk`. If `isk.txt` says `seed public`, `-isk` names the right directory and you did not commit with the option, then someone else has written to the secret key directory. Do not remove the line. The commitment is [in doubt](operators-duties.md#if-the-count-is-lost-or-in-doubt). |
| 1 | `refused (cp-isk-leaves)` | A file of a leaf table in use is missing or cannot be read. | Check that `-isk` names the right directory. Do not repair the secret key directory from a backup or a copy: not the whole directory, not a leaf table, not a `count.bin` ([Backups and restored copies](operators-duties.md#backups-and-restored-copies)). The commitment is [in doubt](operators-duties.md#if-the-count-is-lost-or-in-doubt). |
| 1 | `refused (cp-in-name)` | The directory named with `-in` holds something that is not a request file. | Do not remove anything because of this message or any other about the `-in` directory, such as `refused (cp-batch)`. First check that `-in` names the directory you mean. If you passed the secret key directory by mistake, whatever the sentence names, such as `seed.bin` or `isk.txt`, belongs there, and without it the commitment is lost for good. |
| 1 | `asked-to-end` | A signal asked it to exit. The name does not tell you whether every step of the wipe succeeded, so treat the machine's memory as still holding weights and masks. The batch in progress (with `-overlap` or `-overlap-two`: the batches) is lost, and the next start on that spool directory answers its requests with "failed". | If you sent a `TERM` yourself, and not because the program hung, and no line on the wipe came before the last line, this exit is normal and needs no hold. The batch in progress was never finished, so an exit after a signal does not rule out a failed self-check. But nothing of it was released, and a persistent fault faces the same self-checks on the next run, so that is no reason for a hold. The example front end's start command still puts the policy on hold, and a restart does not lift that ([the hold file](../integrations/vllm/REFERENCE.md#start-the-proving-service)). Start again, and next time use the `stop` file. If you did not send the signal, find out what did, and follow [Always first](#always-first). |
| 1 | `cp-copy-spent` | A leaf table has no room for another batch. | Do not change any `count.bin`. Make [a new commitment](operators-duties.md#a-new-commitment). If the leaf table filled up much earlier or later than your record led you to expect, someone else may be writing to the directory, and the commitment is [in doubt](operators-duties.md#if-the-count-is-lost-or-in-doubt). |
| 1 | `cp-copy-count` | A `count.bin` cannot be read, locked, extended or synced, or its owner, links or permissions, or its directory's, are not as they must be. | Never create, truncate, replace or copy in a `count.bin`. `cp_prove` does not show the cause, so look for it on the machine: free space, the state of the mount, the kernel log. Unless all three conditions [below](#after-cp-copy-count) hold, the commitment is [in doubt](operators-duties.md#if-the-count-is-lost-or-in-doubt). |
| 1 | `cp-prove-kept-not-written` | With `-keep-for-the-audit`, the file `audit/batch.fresh` could not be written. Nothing was proved or counted. | Run the audit on that batch with `--erase-kept-files` ([The audit with the key](operators-duties.md#the-audit-with-the-key)), which erases the file if it exists, unless the audit's last line says otherwise. Whatever it says, do not serve from that directory again, and make [a new commitment](operators-duties.md#a-new-commitment). If `audit` exists and is not a directory of yours with mode 0700, do not change its owner or mode: go straight to the new commitment. |
| 1 | `hc-key-file`, `cp-leaf-record`, `hc-nodes-root` or `hc-own-node` | The seed file, a leaf table's files or the weights are missing, damaged or changed, or others can access the seed file. `hc-own-node` can also mean a faulty machine, or a random entry of a leaf table that was created incorrectly. | Do not restore or remove anything. If others could read the seed file, assume they did. Find out who changed what, and when. If no one can say, the commitment is [in doubt](operators-duties.md#if-the-count-is-lost-or-in-doubt). |
| 1 | `hc-key-test` | `seed.bin` holds a test seed that appears in the source code. | Make [a new commitment](operators-duties.md#a-new-commitment). |
| 1 | `cp-weights-files` | The weight files cannot be read, or are not in a form the program accepts. | Check that the directory named in `isk.txt` holds the committed files, unchanged, as real files ([From nothing to the first proof](operators-duties.md#from-nothing-to-the-first-proof), step 7). At the first start on a new commitment, look at the model files' storage types too ([which types](../prover/docs/ends-in-short.md#the-names-in-the-last-line)). |
| 1 | `policy-tables-absent` or `prover-only-hiding-conditions` | `policy.txt` is missing the digests of the four public tables, or a condition that hiding requires. | Do not add or edit a line by hand. Make [a new commitment](operators-duties.md#a-new-commitment) with a complete `policy.txt`. |
| 1 | `write-failed`, `no-host-memory` or `cuda-call` | The machine failed: a write, host memory or a CUDA call. After a GPU fault the wipe can zero no GPU memory at all. | Fix the cause, then follow [Before you serve again](#before-you-serve-again). A GPU that is out of memory can end as `cuda-call`: check that nothing else uses the GPU, or try `-memory lean` ([options](service.md#options-of-the-service)). It can also end with no name: then the row for an ending with no name comes first, with its ban on a new start. After a GPU fault, treat the GPU as still holding weights and masks. See also [bytes under no name](#bytes-under-no-name). |
| 1 | `source-short` or `source-zero` | The system's random number generator returned too few bytes, or 32 zero bytes. | Do not start again on that machine. See the items on the random source in [the list of doubts](operators-duties.md#if-the-count-is-lost-or-in-doubt). |
| 1 | `not-all-wiped` | A step of the wipe failed or was skipped, or an entry that `cp_prove` did not make is in the staging directory, or the wipe was given up at its time limit of 600 seconds. | Follow [Always first](#always-first). Treat the machine's memory as holding secrets. A next start of a service on that spool directory whose requests have the same length as those left in `work/` overwrites and removes what is left under `out/.tmp/`. After a single batch with `-out`, [`cp_prove -clear`](#clear) does that: name that directory, and no other. If an entry that `cp_prove` did not make is still at the top of `out/.tmp/`, neither clears anything until you have moved that entry out ([what to do](#clear)). Give it to no one: it may be a staged response, or a copy of one. See also [bytes under no name](#bytes-under-no-name). |
| 1 | `guard`, `no-first-chain`, or no name | One of the program's internal checks failed. Two of them, on the random generator's counters, end without a name, and one of those two is the only guard against repeating one-time masks. Or the program crashed. Many of the public set-up refusals that come after `p258 closed` (a rejected `-memory` value, a table digest that does not match the machine's, a layout file that is missing or breaks a rule, a call shape that does not fit) now end with a fixed name of their own ([the names in the last line](../prover/docs/ends-in-short.md#the-names-in-the-last-line)), not a bare ending. A few configuration cases still have no name: from reading the code, one is a parameter file of your own making in which a line `lmS`, `lmh`, `lmo`, `lmbA` or `lmq` is below 11: `cp_prove` then ends so at every start ([the parameter file](../verifier/docs/parameter-file.md)). Or, from reading the code, a part of a weight tensor that is kept as blocks has more than 33,025 rows, which neither program checks at its start. Tensors stored as Q4_K, Q5_K, Q8_0, F16 or BF16 can be kept so. The usual wipe then still runs (row P4.2.3 in [the table of all rows](limits-all-rows.tsv)). Or, also from reading the code, the GPU ran out of memory at one of a few places that give the ending no name (once that is known to be the cause: the row for `cuda-call`). Whether the usual wipe runs in that case is not known. After a crash, nothing is wiped. | Do not start again on that machine until the cause is known. See [a hardware fault](#looking-for-a-hardware-fault) and [Before you serve again](#before-you-serve-again). |
| 9 | a name that begins `own-test-` | A self-check on secret random values or keys failed. From reading the code, one of these names, `own-test-coin-library`, can also mean that the operating system gave no memory page that a fork wipes, which needs Linux 4.14 or later: then no self-check has failed. Anything not yet in `out/` stays unreleased, with three exceptions, known from reading the code (rows C3 and HW-583). The last line does not show whether every step of the wipe ran. By a reading of [other sources](glossary.md#other-sources), exit code 1 can mean a like fault: the system's random source gave too few bytes or zeros, the seed file or a leaf table's key is zero or damaged, a column of a leaf table made again no longer fits its tree, or the counter of the generator that draws the sub-proofs' pads did not rise. | Follow [Always first](#always-first). Find the cause before you start again, especially on this machine. Do not let a supervisor restart it. |
| none, or 128 + the signal's number | no last line | It was stopped by `KILL`, a power cut or, rarely, a crash the program cannot catch. Nothing was wiped, and staged files remain until a next start on that spool directory whose requests have the same length as those left in `work/`. | Follow [Always first](#always-first). After a power cut, also follow [Before you serve again](#before-you-serve-again). |
| - | `p258 alarm memory-grew` or `p258 alarm cut-short` | An alarm, printed at most once. The service keeps running, and no request loses its response ([the three alarms](../prover/docs/ends-in-short.md#the-three-alarms)). | For `cut-short`, lower the front end's limit on waiting requests. Note either alarm in your record. |
| - | <a id="unreadable-request"></a>`p258 alarm unreadable-request` | An alarm, printed at most once, and the service keeps running. From reading the code, `in/` holds a file with a request's name and length that `cp_prove` is not allowed to read. What becomes of such a file: [service](service.md#a-request-file). The line does not name the file, and nothing announces a later file of this kind ([the three alarms](../prover/docs/ends-in-short.md#the-three-alarms)). | List `in/` with `ls -l` to see each file's owner and mode: a file written as root or under `sudo` belongs to root. The file's owner, or root, may give the account that runs `cp_prove` read permission on it through the file's mode: that one file, by its name, never `in/` or the spool directory as a whole. As long as the spool directory stays closed to group and others, which `cp_prove` checks at every start, this opens the file to no other account (reasoned; not tried). This is one of the exceptions to the rule on permissions. Do not change the file's owner instead. Find out what wrote it, or it will happen again. Look into `in/` again later, and note the alarm in your record. |
| - | it is still running and prints nothing | It is slow, or the GPU has hung. | See [A program that hangs](#a-program-that-hangs). |

**`cp_commit`**

| Exit code | Last line or name | What it means | What to do |
|---|---|---|---|
| 1 or 2 | `NO: <name>`, or its message on standard error, such as `refused (file-there-already)` | It rejected the call, an input file or the machine. | Read the message, and fix what it names. Keep the output private: at least one message with exit code 2 says that a gain in a named tensor is out of range, which reveals something about the weights. Do not remove, move or restore anything to make the call succeed. The names one by one are below this table. |
| 1 | `GPU error:` or `CUDA error`, on standard output | The GPU or one of its libraries failed. | Do not commit again on that machine until the cause is known. |
| 2 | `cp_commit: refused (own-test-coins)`, or the same with `own-test-generator` | The routines that make its secret random values failed their self-test. The program, its build or the machine is at fault. No seed was drawn, and nothing was written. | Do not commit again on that machine until the cause is known. |
| 3, 6, 7 or 16 | its message | The weights were rejected: a tensor cannot be converted to integers (3), the weights do not match `wsums.txt` (6), the worst-case check failed (7), an integer is out of range (16). | Keep the output private, because it reveals something about the weights. See [weight_sums](../weight_sums/README.md) and [scales](../scales/README.md) for how to prepare the files. |
| 4 | a line `cp_commit: NOT done` | Files disappeared while it worked, so not every leaf table exists. Nothing is published, and it removes what the run made. | Find out what removed the files before you commit again. |
| 4, 5 or 134 | its message | The program or the machine had a fault: a GPU fault (5), a failed check (134), an internal error, a working directory that cannot be locked or was replaced, or a file that `cp_verify` did not pass (4). Too little GPU memory can end with several of these codes, and with 2. | Do not commit again on that machine until the cause is known. Keep the output private, also the line and the length that the message after a file that `cp_verify` did not pass points to ([step 10](operators-duties.md#creating-a-commitment)). |
| 128 + the signal's number | a line that says `removed by name` | Ctrl-C, `TERM`, `HUP`, a broken pipe or `ABRT` (exit code 134) stopped it, and it removed what the call had made. | See below. |
| none, or 128 + the signal's number | no last line | It was stopped by `KILL` or another signal it does not catch, or the machine failed. Nothing was cleaned up. | See below. |

After any bad exit from `cp_commit`, see [A killed or failed
cp_commit](#a-killed-or-failed-cp_commit). If no row covers your case, keep the hold and ask the
named person.

**What `cp_commit`'s rejection names:**

- **A `.weights-are-public` file** (`cp-public-weights-marker`): read [Public
  weights](operators-duties.md#public-weights) before you follow the message.
- **An `answers` line that is not 1** (a sentence that holds `parameter-range:` and ends `answers`):
  `cp_commit` and `cp_prove` refuse a parameter file whose `answers` is not 1, on synthetic weights
  and under `-public-weights` too (`parameter-range`). Take a parameter file that says `answers 1`
  ([A record](operators-duties.md#a-record)). Do not add `-public-weights` unless the weights are
  public.
- **A path** (`cp-commit-path`): from reading the code, nothing was touched: name a place under
  directories that only you and root can write to, with no symbolic link in its path ([step
  4](operators-duties.md#from-nothing-to-the-first-proof)). Where the message offers `chmod go-w`,
  step 4's conditions for that still hold.
- **The file system** (`cp-commit-file-system`): from reading the code, no weight was loaded and no
  file was written: if that is a shared or network file system, use a local one.
- **A name that exists already** (`file-there-already`): do not remove anything: give `-isk` and
  `-ivk` two new names. The directory named with `-isk` may hold a commitment's seed, and once
  removed, a seed cannot be recovered ([more](#a-killed-or-failed-cp_commit)).
- **The system's random number generator** (`no-randomness`, `hc-key-test`): do not commit on that
  machine until the cause is known.
- **A GPU in use** (`gpu-in-use`): no seed was drawn and nothing was written. Do not stop another
  account's work on that GPU: wait, or use another GPU.
- **An entry of the tensor table** (a line that begins `gguf:` and ends with
  `entry N of the tensor table)`): from reading the code, the line gives a weight file's path and
  the entry's number, counted from 1, and never a tensor's name. If it says
  `a tensor name occurs twice`, the path and the number are those of the second entry with that
  name. The first can be in another file of the model, and the line does not name it. These pages
  name no tool that lists a file's tensor table, so use one of your own. Keep the line as secret as
  the weights ([why](../prover/docs/ends-in-short.md#the-names-in-the-last-line)).

<a id="a-closed-directory-of-your-own"></a>**A rejection at the start of `cp_prove`:**

- **A directory inside the spool directory cannot be opened** (`Permission denied`): look at that
  directory with `ls -ld`. Its line has to begin with `d` (a directory, not a link) and name you as
  the owner. If it does, and the mode in that line shuts out the owner too, and you are not root,
  then you may give yourself its permissions back, as the message says (`chmod 700` on that one
  directory). This is one of the exceptions to the rule on permissions. Otherwise do not change
  anything: name a new, empty spool directory of your own (the requests in the old one then get no
  response), and run every step without `sudo`. The message also offers to run every step with
  `sudo`: read [Under `sudo`](operators-duties.md#under-sudo) first.
- **An entry at the top of `out/.tmp/` that `cp_prove` did not make:** the one exception to the ban
  on moving ([what to do](#clear)).
- **A GPU in use:** do not stop another account's work on that GPU: wait, or ask its owner.
- **`cp-prove-environment`:** `HOME` and `TMPDIR` must both name existing directories ([the
  rule](operators-duties.md#serve)).
- **`refused (cp-spool)` with `mark.txt`:** see [The mark file](operators-duties.md#the-mark-file).
- **`cp-isk-seed-test`** (the commitment was made for a test, or someone edited `isk.txt`),
  **`ranges-not-guarded`** (`isk.txt` has no line `ranges` for these configuration files): check
  that `-isk` names the right directory, and that `cp_prove` comes from the build that made the
  commitment: another build's programs can form another commitment ID from the same files (read in
  the code), and another digest for the line `ranges`: `cp_prove` then rejects a directory that the
  other build's `cp_commit` made, until you make [a new
  commitment](operators-duties.md#a-new-commitment) with the programs that you serve with. If all of
  that checks out, the commitment is [in
  doubt](operators-duties.md#if-the-count-is-lost-or-in-doubt). Do not edit `isk.txt`.
  `ranges-not-guarded` is the one check at a start that the configuration files are the ones
  committed to, and it compares them with a line in the same directory: whoever can write the files
  can write the line. `cp_prove` does not read the key file, so damage to it is not found; it knows
  nothing of the commitment ID that you published; and it does not make the checks on the public
  files that `cp_verify` and `cp_commit` make, the one on the target of 128 bits among them. The
  check comes before `cp_prove` opens the spool directory, so a start that it rejects changes
  nothing there. What slips through shows, if at all, only as proofs that a verifier rejects.

<a id="looking-for-a-hardware-fault"></a>**Looking for a hardware fault.** Stop the service and run
a test build on that machine, on synthetic weights, with separate key directories. A test build
prints everything, including the messages from its checks, but a fault that depends on the model's
own values does not show up there. Never run a test build on the secret key directory you serve
from.

<a id="after-cp-copy-count"></a>**After `cp-copy-count`** you may fix the cause and go to [Before
you serve again](#before-you-serve-again) only if all three of these hold:

- Every `count.bin` is as `cp_prove` left it.
- The directory was never copied, restored or rolled back, and no other account can write to it or
  to any directory above it.
- The cause is a full disk, a quota, a mount you made read-only yourself, the file system's locks,
  or a shortage of threads, memory or open files. Or it is an I/O error, a failed sync, a read-only
  remount by the kernel or a file-system repair, and you also know that the disk does not lose
  synced writes. You must know that because nothing at startup notices a count that is too low.

Otherwise, or if the cause is not found or no one can say who changed what, the commitment is in
doubt.

<a id="bytes-under-no-name"></a>**Bytes under no name.** From reading the code, a staged file that
`cp_prove` could not overwrite is still removed. Its bytes can then stay on the disk under no name,
out of reach of a start and of `-clear`, and no line on the wipe need come ([what the disks
need](operators-duties.md#keys-seeds-and-machines)).

**A name does not show that the wipe ran.** `ABRT`, `SEGV`, `BUS`, `FPE`, `ILL`, `SYS` and `TRAP`
count as a crash even if someone sends them, so there is no wipe and usually no name. A crash that
comes after a name was set prints that name.

<a id="cp-isk"></a>
## `refused (cp-isk)`

**The name or the path.** The name must begin with `/` and must not contain a symlink. If every link
on the way is your own or root's, give the directory's real, absolute path. If one belongs to
another account, do not go by where it leads now: the commitment is [in
doubt](operators-duties.md#if-the-count-is-lost-or-in-doubt). `cp_prove` cannot see that case. It
checks every directory on the path that it is given, not how you came by that path. If you change
into a directory through another account's link and then name what `pwd -P` prints, the path has no
link in it, `cp_prove` accepts it, and the batch's count is written in a directory that the other
account chose. What is left: the directory must still be your own and closed to group and others. An
older copy of this same directory, with its older count, meets both conditions, and so does another
commitment's directory of yours. This was tried, not on the released programs, through a small host
program on the programs' own headers, as root and as an ordinary user: the path was accepted both
times. Do not copy or move the secret key directory to get such a path.

Every directory on the path must be owned by root or by you, and must not be writable by group or
others unless it has the sticky bit. If one of them fails this test, the commitment is [in
doubt](operators-duties.md#if-the-count-is-lost-or-in-doubt), unless you know that no other account
has been able to write there since you committed. In that case, first check that `-isk` names the
directory you mean. Then, if the failing directory is yours and you are not root, take away the
write permission as the message says (`chmod go-w` on that one directory) and start again. If you
are root, or the directory is not yours, do not change the mode, because others may share the
directory, and do not move the secret key directory. Instead, make [a new
commitment](operators-duties.md#a-new-commitment) somewhere else, and treat the old one as retired
([Backups and restored copies](operators-duties.md#backups-and-restored-copies), step 5).

**The directory itself.** Start `cp_prove` as the account that ran `cp_commit`, which under `sudo`
is root, and do not change the directory's owner. If group or others have any access to the
directory, do not just fix the mode, because the permissions show who could have reached the
directory, not who did. Nothing was exposed if you know three things: who on your side set those
permissions, when, and that since then no other account existed or could reach the directory, as a
user or through a group. A backup or an image that copied the directory with these permissions
counts as access. In that case alone you may fix the mode, on that one directory and not on its
contents, and go to [Before you serve again](#before-you-serve-again). Otherwise, or if no one can
say, the commitment is in doubt.

**In use.** The sentence begins `refused (cp-isk): the folder named with -isk is in use`. Another
`cp_prove` is running on this directory, the `cp_commit` that creates it has not finished, or
another program of yours holds a lock on the directory. `cp_prove -clear` and the audits, among
others, do not take this lock. The rejected call changed nothing, so this exit needs no hold. Wait
for the other process to end, or [stop the service](operators-duties.md#stopping-and-starting), and
start again. If that process hangs, see [A program that hangs](#a-program-that-hangs). There is no
lock file to remove. Do not copy the directory to run a second `cp_prove`: two processes on two
copies share one seed and count apart. If you wanted a
single batch while the service runs, send the same prompts to the service as ordinary requests
instead.

If the sentence says instead that the directory cannot be locked, the file system does not support
the lock. Do not copy or move the directory. Make [a new
commitment](operators-duties.md#a-new-commitment) on a local disk. We know all this from reading the
code (row R13.1.2). That `cp_prove` starts does not show that the file system is fit: some shared
and network file systems may grant the lock on each machine by itself (not tried).

**`isk.txt`.** The file cannot be read, is not what `cp_commit` writes, or comes from another build,
format or version. Check that `-isk` names the right directory and that `$B` holds the build that
made the commitment. Do not edit the file, and do not repair it or the directory from a backup or a
copy. A restored directory has an older count, so a leaf table would serve past its limit, clients
would get exact equations in the weights, and no program would notice ([Backups and restored
copies](operators-duties.md#backups-and-restored-copies)). If the directory and the build are both
right, the commitment is in doubt.

<a id="clear"></a>
## `cp_prove -clear`

`cp_prove -clear DIRECTORY` clears what is staged, just as a next start on that directory whose
requests have the same length as those left in `work/` would ([Always first](#always-first), step
3), but without starting the service (particulars: rows R18.4 and H26.1).

- Run it after a bad exit, or before you replace a spool directory with a new one.
- Run it as the account that ran `cp_prove`.
- Name the directory you passed with `-spool` or `-out`, and no other. Give its absolute path, which
  must not contain a symlink. Name the spool directory itself, not its `out/`.

**What it touches.** It clears a staging directory only if a `mark.txt` says that the directory is
`cp_prove`'s. So if by mistake you name another directory of yours, it removes nothing there.
`cp_prove` knows no commitment ID, so it clears any commitment's marked directory. It recognises
staged responses only by their names. It leaves every other entry in place and lists up to twenty of
them for each staging directory. If there are more, it says so. It does not touch `in/`, `work/` or
released responses, and does not fix a rejection that names `work/`.

**Its exit codes.** 0: nothing staged is left, also where nothing was staged. An `-out` directory
keeps `.tmp/`, and the `mark.txt` in it if there was one. 1: the call or the directory was rejected,
or something is left or could not be examined; the printed lines say which, and what to do. 2:
nothing was touched: a `cp_prove` is running there, there is no such directory, or it has no staging
directory.

**An entry listed as `LEFT`.** The printed lines list an entry as `LEFT` if `cp_prove` did not make
it, or if it has more than one hard link. If an entry at the top of a staging directory has a name
that `cp_prove` never uses there, nothing at all was cleared, and what `cp_prove` staged stays until
you have moved that entry out.

- If you named the wrong directory, do not change anything in it.
- Otherwise, do not give the entry to anyone, because it may be a copy of a staged response. Only in
  this case may you inspect an entry and move it. Once you know what it is, move it into a new
  directory only you can read, and run the command again. `cp_prove -spool` will not start on the
  directory while such an entry is there.

**A directory that it rejects.** First make sure that it is the directory you passed with `-spool`
or `-out`. If it is not, do not change anything in it, and name the right one. If it is:

- **A staging directory that is not empty and has no `mark.txt`.** Nothing was touched. If the
  message says that this is a marked spool directory with a `.tmp` at its top, `cp_prove` did not
  make that `.tmp`: move it out as you would an entry listed as `LEFT`, and run the command again.
  If the `mark.txt` is gone or damaged, no `cp_prove` call clears the directory or starts on it.
  What is staged there stays, and the requests waiting there get no response. Do not write a new
  `mark.txt`. Do not give anything from the directory to anyone, keep it as secret as the secret key
  directory, and start the next service on a new directory.
- **A `mark.txt` of another kind.** The file is not wrong: the directory serves another purpose.
  Leave it as it is.
- **Another account owns it.** Rerun the command as the account that ran `cp_prove`.
- **Group or others can access it.** What is staged there stays. Do not give anything from it to
  anyone, and find out who could read or write there. If another account could read it while it held
  staged responses, treat those responses as read, because tightening the permissions afterwards
  undoes nothing. After a fault they can contain unmasked hidden values. Without a fault, each is an
  ordinary proof with its client's prompt and reply. After that, if the directory is yours and you
  are not root, you may lock it down (`chmod 700` on the one directory the message names) and run
  the command again. If you are root, start the next service on a new directory with mode 0700
  instead, and keep the old one as secret as the secret key directory.

## A killed or failed cp_commit

Give the next call two new names for `-isk` and `-ivk`. Nothing has to be removed first, and no tool
removes what a failed call leaves behind. While any leftovers exist, keep them inaccessible to every
other account and as secret as the weights, and treat a leftover `<isk>` like a retired commitment
([Backups and restored copies](operators-duties.md#backups-and-restored-copies), step 5). Before you
make the next call, make sure that no `cp_verify` check that `cp_commit` had started is still
running (rows R13 and R13.1). A small supervisor ends each check when `cp_commit` dies. There are
two exceptions: a check whose supervisor is killed together with `cp_commit` keeps running, though
it normally ends by itself, and a check that is stuck in a system call ends only when that call
returns. The next `cp_commit` does not look for a check that was left behind. You can recognise such
a check by its supervisor's command line, which begins `/proc/self/exe --supervise-cp_verify`;
`cp_verify` is that process's child, in the same process group. Go by the command line: the
supervisor's short name reads `exe`. If you have to end one, end it by its process ID, never by
name. That the checks end with `cp_commit` is tested only on a CPU, with small substitutes for both
programs (row `prover.gate_children` of [SUITE.md](../SUITE.md)).

What is left behind depends on how the call ended.

- **`cp_commit` exited on its own, or Ctrl-C, `TERM`, `HUP`, a broken pipe or `ABRT` stopped it.**
  It then removes everything the call made. From reading the code, it overwrites each file that the
  call created with zeros before it removes it, with exceptions, and a disk can keep overwritten
  bytes. A file with a second hard link is removed
  without that. So is one that is no longer a regular file, one that it cannot examine, or one whose
  device and inode numbers are no longer those that were noted when it was made, and no line says
  so. A file that it cannot overwrite keeps its name, and a line says so (after a signal, only
  `NOT ALL`). What it removes includes a directory that already has its final name, so use neither
  `<isk>` nor `<ivk>` before `cp_commit` has ended with exit code 0. After a signal or an abort, it
  neither waits for the GPU nor wipes any memory. Its last line shows how far it got.
  - If the last line is the rejection itself, neither directory was made, unless the line says that
    something `is left as it is`.
  - If it ends `, and all that the call had made before it`, `cp_commit` has normally removed
    everything. Confirm that with `ls -ld` on `<isk>`, `<ivk>`, `<isk>.making` and `<ivk>.making`,
    but do not delete anything because of what the listing shows.
  - Treat any other ending like the next case. That includes a last line that ends
    `, but for the names above` or says `NOT ALL`, and any line that says a name could not be
    overwritten or removed.
- **`KILL`, `SEGV`, `BUS`, `QUIT` or another signal that it does not catch stopped it, or the
  machine failed.** Nothing was cleaned up, and the seed file may still be there, with the kept
  files next to it if you passed `-keep-for-the-audit`. Keep everything that is left as secret as
  the weights. Do not serve from, publish, rename or complete any of it: a call that stopped counts
  for nothing and is not continued (row H4.4.2).
  - What the call made is under `<isk>.making` and `<ivk>.making` or, if the call died in its very
    last step, partly or wholly under the final names. An empty directory from a call's first moment
    can stay (rows R3, R2.4 and R11.1).
  - A final name belongs to this call only if the call's output has the line that begins
    `cp_commit: the two folders are made as`.
  - The leftovers hold that call's seed and values that depend on the weights, so they should not be
    kept indefinitely. The two directories whose names end in `.making` can be removed, and nothing
    else can. That means each one whole, by the exact name that the call's output printed, with no
    wildcard and no guessed name. In a log shared by several calls, only this call's lines count. If
    anyone else can write to a directory above them, first check what each name points to now. That
    does not cover what is under `<isk>` or `<ivk>`. Keep that as you would a retired commitment
    ([Backups and restored copies](operators-duties.md#backups-and-restored-copies), step 5).
    Removing the two `.making` directories does not overwrite the seed file ([what the disks
    need](operators-duties.md#keys-seeds-and-machines)).

**The file named with `-worst-case-table`** remains after a successful run, and after exit code 7,
where it is the answer you asked for. A call that fails in any other way normally removes it, but a
kill, a signal, an abort or an early exit can leave it behind. Keep it as secret as the weights, and
give the next call a different name for it.

**If you repeat the same call,** it is rejected, nothing is changed, and the message says what it
found and what to do. No message tells you to remove anything, though some say what can be removed.
Follow the message, but keep in mind what it cannot tell you:

- The words "exists already" do not always mean a leftover. The name may belong to a finished
  commitment that is in use, or to a call that is running. If a call is running in those
  directories, let it finish.
- The program looks at this machine only, and it cannot tell a dead call's directories from those of
  another commitment of yours. It never names `<isk>` or `<ivk>` as something that can be removed.
- If it cannot tell whether a call is running, do not end a process or remove anything because of
  that message.
- The leftovers of a killed call with `-keep-for-the-audit` hold that call's seed and the secrets it
  kept. The message itself warns you not to take it as permission to remove them.
- If `<isk>` and `<ivk>.making` both exist, a call with these names is just finishing, or was
  interrupted between its last two steps. In this one state the program does not check for a running
  call, so look for one yourself. Wait, then repeat the call. If it is then rejected because `<ivk>`
  exists, the earlier call reached its last step. Do not remove anything, and go by that call's own
  exit code.

**Never mistake a secret key directory for a leftover.** Do not touch one that existed before that
call, that has or once had its public key directory next to it, or that has ever served.

## A program that hangs

Neither program sets a timeout on a batch or a run, and nothing in their output says a run is stuck,
so go by how long batches have taken on your own machine. The `stop` file does not help with a hung
batch, because `cp_prove` checks for it only when it picks up the next batch. A slow start on a busy
machine need not be a hang ([A slow start at full size](quickstart.md#a-slow-start-at-full-size)),
nor need a slow end ([how long an exit
takes](../prover/docs/ends-in-short.md#how-long-an-exit-takes-at-full-size)).

Only the named person decides that `cp_prove` is stuck, and they then end that one process. No test
kills or signals `cp_prove` on purpose at the default setting, which asks for proof of work
([service](service.md#how-the-service-ends)).

1. Look up its process ID just before you end it. Never end it by name.
2. Send TERM. The wipe still runs.
3. Wait well over ten minutes. `cp_prove` first finishes the step it is in, and only then does the
   time limit start. After 600 seconds `cp_prove` gives the wipe up, prints
   `p258 end 1 not-all-wiped`, or its usual line with exit code 9, and exits. From reading the
   code, the limit covers only the wipe, and at full size even a clean end takes a while ([how
   long](../prover/docs/ends-in-short.md#how-long-an-exit-takes-at-full-size)). Until then, another
   `TERM` or Ctrl-C has no effect, and the process keeps its locks on the spool directory and on the
   secret key directory. Whatever the wipe had not reached is neither zeroed nor removed. Do not
   copy the secret key directory to start another `cp_prove` meanwhile ([in use](#cp-isk)). This
   limit has been tested only on CPUs with stand-ins for the programs, and never on a GPU (the limit itself: row R7.1).
4. Look the process up again. If it is gone, do not send anything, because its process ID may by now
   belong to another process. If the same process ID still has the same command line, send KILL.
   Killing a program in the middle of a GPU kernel can make the driver take the whole machine out of
   service ([further limits](limits-further.md#r1.3)), so do this only after that wait. In at least
   two cases it never exits on its own: it is blocked on a hung GPU in the middle of a batch and
   never starts shutting down, or its main thread is stuck in the kernel (row R7.1). Also, a signal
   that was ignored when the program started, as under `nohup`, stays ignored. After KILL, nothing
   is wiped or removed.
5. Follow [Always first](#always-first). A hang is itself a fault, so the exception there for a
   `TERM` you sent yourself does not apply, and the hold stays until the cause is known.

If `cp_commit` is stuck, the named person ends it by its process ID, with TERM first. Then give it
time: from reading the code, it overwrites what the call made before it removes it, and by the
source's own comment each gigabyte takes seconds. A `KILL` in that time leaves the rest as it is.
Then see [A killed or failed cp_commit](#a-killed-or-failed-cp_commit).

## Before you serve again

No known check makes a restart safe, so start again on the same commitment only if all of the
following are true, and so is every condition that your row, or the paragraph that it points to,
sets. Otherwise, make [a new commitment](operators-duties.md#a-new-commitment). While the cause is
unknown, that machine stays out of service.

- The cause is known, gone and harmless, and it is in your record. Examples are a mistyped path, a
  full disk or a mount you made read-only yourself. A faulty GPU that was found and replaced is
  known and gone, but not harmless ([limits](limits.md#a-lost-mask)). After a power cut, a crash, an
  I/O error, a failed sync, a read-only remount by the kernel or a file-system repair, you must also
  know that the disk does not lose synced writes.
- The machine and its disks came back untouched. Nothing was restored, rolled back or moved.
- No `count.bin` is shorter than it was at your [last reading](operators-duties.md#the-count).
- The output of the ownership check ([step 11](operators-duties.md#from-nothing-to-the-first-proof))
  differs from the last output you kept only in each `count.bin` entry and the `total` line above
  it.
- If your own `cp_verify` did not accept a response: your front end cannot hand out that response or
  any other of its batch, which can still be in `out/` ([Always first](#always-first), step 3).
- The named person has lifted the hold.
