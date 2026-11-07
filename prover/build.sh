#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Builds the GPU programs from the sources in this directory: cp_commit, which commits to a model's
# weights, cp_prove, which proves requests, and their test programs.
# Requires nvcc 13.2.86 (CUDA 13.2 Update 2), gcc and g++ 13, bash 4.4 or later, sed, awk,
# and GNU binutils, coreutils, findutils and grep. nvcc is looked up on PATH first and then in
# /usr/local/cuda/bin. Every target that runs nvcc first runs check_nvcc_version.sh, which explains
# the required version, and then compiles compiler_sample.cu and, if the machine has a GPU, runs it
# there for a moment. That run has no time limit: on a GPU that does not answer, the build waits.
# Building needs no GPU; the programs need an H100 (sm_90a) and an x86-64 CPU with the SHA
# extensions. The default build took 26 to 35 minutes on 11 or more cores in the few runs that were
# timed, none of them on an idle machine (prover/docs/build.md, "Build"). Before it calls this
# script, examples/quickstart-gpu.sh requires 10 GB of available memory, for this build, and 1 GB of
# free disk space in the repository.
# Build the verifier first (verifier/build.sh): cp_commit runs cp_verify, and its targets copy
# bin/cp_verify from the repository's top next to it. `serving` and serving_cp_commit require a
# cp_verify that was built from the present sources (from_these_sources()); the other targets do
# not check it. cp_prove needs no verifier.
#
#   ./build.sh [<target> [<argument>]]
#
#   serving      (the default) The production build (`serving` in names and messages): cp_prove and
#                cp_commit into $OUT/hc, with cp_verify. They have no test option and read
#                no test variable. Once cp_prove is about to read its first secret, it prints only a
#                fixed list of lines (`closed front`: prover/prover/closed_front.h). Each gets a
#                build record, <program>.build. <argument>: the commit that you state for this
#                tree, 7 to 64 hex digits; it is recorded as given.
#   serving_cp_prove, serving_cp_commit
#                one of the two. serving_cp_commit, if it succeeds, removes the cp_verify in
#                $OUT/hc, and only `serving` installs a new one there.
#   tests        The test build: cp_prove and cp_commit with the test options, into
#                $OUT/hc/test_build, and make_model, into $OUT.
#   cp_prove, cp_commit, make_model
#                one of the three; make_model is a CPU program that writes a small synthetic model
#                as GGUF files
# Every other target builds a test program or a benchmark, belongs to neither `serving` nor `tests`,
# and is described at its branch of the first case statement. Of all builds of cp_prove and
# cp_commit, and of programs that include their source, only the production build accepts a model's
# files.
# -h or --help, anywhere among the arguments, prints the usage message (status 0). A first argument
# that is no target gets it too, on standard error (status 2). In both cases nothing is created,
# checked or compiled.
# A script that sources this one can add targets of its own; sourced, this script rejects the
# serving targets. No script in this repository sources it. See more_targets, in the last branch
# of that case statement.
#
#   --field 2  accepted before the target, without effect: the only field is p = 2^64 - 2^32 + 1
#   --dw 8     accepted for scripts that pass it; a digest always has 8 words
#   OUT        the output directory (`folder` in some messages), relative to this one; default:
#              build/. A named OUT that another account may be able to write in or replace is
#              rejected.
#   EIDB       22, 24 or 28: the build supports lookup identities of up to 2^EIDB keys (default 28)
#   ETAU       the random combinations in a sub-proof's approximate range check (`smallness` in
#              identifiers); default 128
#   CUDA_VISIBLE_DEVICES
#              set to the empty string, it keeps the build off the GPU: compiler_sample then finds
#              no GPU, and a message says that only the compiler's version was checked
# EIDB and ETAU are explained in prover/common/setting.h. A serving target rejects them, and its
# compilers see only PATH of the caller's environment.
#
# A build writes only in OUT, which it creates if it is missing, with the directories above it (but in a target that is not a serving one, a variable of yours such as DEPENDENCIES_OUTPUT can make g++ write elsewhere). Each
# call creates a new work directory there, building.XXXXXX. The compilers and this script write all
# their files into it, temporary files too. Only after every step and check has passed do renames
# give the target's files their names (the directories below OUT that receive them are created
# earlier). So a call that fails or is interrupted installs nothing and removes nothing that an
# earlier build left, unless one of these renames itself fails (see finish()). `serving` and `tests`
# call this script once for each program, and each call installs its own. So a `serving` or `tests`
# that fails or is interrupted may leave programs of two builds in OUT, and such a `serving` may
# leave no cp_verify in $OUT/hc: build again.
# Ctrl-C stops `serving`, the default, where env knows --default-signal (GNU coreutils 8.31 or
# later): its call for cp_prove runs in the background, where bash would have it ignore SIGINT, and
# is started through env with that signal set back to its default. Each call that still runs then
# installs nothing, unless it is already installing, and the script exits with status 130
# (prover/docs/build.md, "Build", has the trials). Not if this script was itself started with SIGINT
# ignored: then no call hears it. With an older env the call for cp_prove ignores SIGINT itself
# (see finish()); the compiler that it is running may still die of the signal, and then that call
# fails and installs nothing. Otherwise it goes on and, if it succeeds, installs cp_prove. The
# script waits for it, ignoring SIGHUP, SIGINT and SIGTERM
# meanwhile, and then exits with status 130.
# This script sends no signal. So a signal that reaches only the process that runs `serving` or
# `tests` (`kill` with its process ID, not Ctrl-C) does not stop a call of this script that it has
# started: that call goes on and, if it succeeds, installs its program.
# A work directory can stay behind, as after SIGKILL; remove it once no build is running.
# A build removes only its work directory with the files in it, the files that its renames replace
# and, in serving_cp_commit, the earlier cp_verify in $OUT/hc. The installed files stay, and so does
# every other directory that a call created, even a call that failed. Run as root, or under sudo,
# the script creates all of them as root's.
#
# The variable form_of, a suffix of the directory names build and ../bin, is always empty.
set -euo pipefail
unset CDPATH # with it set, the cd below could end in a folder of the same name elsewhere
# sourced: 1 if another script sources this one (see more_targets below), else empty. Outside a
# function, `return` succeeds only in a sourced script. BASH_SOURCE would not do: bash takes a value
# for that name from the caller's environment. The script that sources this one must already have
# changed to this directory.
sourced=''
if (return 0 2>/dev/null); then sourced=1; fi
if [ -z "$sourced" ]; then cd "$(dirname "$0")"; fi
if [[ $PATH != *:/usr/local/cuda/bin ]]; then export PATH=$PATH:/usr/local/cuda/bin; fi # added once: "serving" calls this script again
dw=8 field=2 form_of='' # this script's calls of itself pass --dw $dw --field $field on
while [[ ${1:-} == --dw || ${1:-} == --field ]]; do
  if [[ $1 == --field ]]; then
    if [[ ${2:-} != 2 ]]; then
      echo "build.sh: there is one field, number 2 (p = 2^64 - 2^32 + 1): --field takes 2 and nothing else" >&2
      exit 2
    fi
    field=$2
  elif [[ ${2:-} != 8 ]]; then
    echo "build.sh: --dw takes 8, the words of a digest, and nothing else" >&2
    exit 2
  else
    dw=$2
  fi
  shift 2
done
if [ $# -eq 0 ]; then set -- serving; fi # no word: THE programs
field_flags=(-DBLK_FIELD=$field -I../blocks)
for word in "$@"; do
  if [[ $word == --dw || $word == --field ]]; then
    echo "build.sh: $word stands before the target: ./build.sh $word .. <target>; behind it, it would be read by nobody" >&2
    exit 2
  fi
done
# The usage message, and the arguments that get it before anything is created. A script that
# sources this one has targets of its own: its more_targets(), below, rejects what it does not know.
usage_line="usage: ./build.sh [serving]|tests|in_pieces_test|moe_at_layer_test|tree_hash_test|tail_gam_test|psc_shell_test|token_sumcheck_psc_test|attention_sumchecks_test|forest_lookup_test|forest_lookup_bench|frame_as_the_verifier_reads|compiler_sample|make_model|cp_prove|cp_commit|serving_cp_prove|serving_cp_commit|hc_unit_test tests/<name>|hc_host_test tests/<name>"
if [ -z "$sourced" ]; then
  for word in "$@"; do
    if [[ $word == -h || $word == --help ]]; then
      echo "$usage_line"
      echo "The head of prover/build.sh and prover/docs/build.md describe the targets."
      exit 0
    fi
  done
  case $1 in
    serving | tests | in_pieces_test | moe_at_layer_test | tree_hash_test | tail_gam_test | psc_shell_test | token_sumcheck_psc_test | attention_sumchecks_test | \
      forest_lookup_test | forest_lookup_bench | frame_as_the_verifier_reads | compiler_sample | make_model | cp_prove | cp_commit | serving_cp_prove | serving_cp_commit | \
      hc_unit_test | hc_host_test) ;;
    all)
      echo "build.sh: there is no target \"all\". ./build.sh, with no word, builds the programs; ./build.sh tests builds cp_prove and cp_commit" \
        "with the test options, and make_model, for the tests." >&2
      exit 2
      ;;
    *)
      echo "$usage_line" >&2
      exit 2
      ;;
  esac
fi
# A script that sources this one builds no production program: a production program's build record
# covers this script, not that one.
# The second check: a serving target uses the defaults of prover/common/setting.h.
if [[ -n $sourced && ${1:-serving} == serving* ]]; then
  echo "build.sh: the programs without the test options are built by ./build.sh itself, not through a script that sources it" >&2
  exit 2
fi
if [[ ${1:-} == serving* && -n ${EIDB:-}${ETAU:-} ]]; then
  echo "build.sh: a serving build takes no format switch from the environment: unset EIDB and ETAU" >&2
  exit 2
fi
named=${OUT:-} # empty: the default output directory, inside the checkout
[ -n "$named" ] || unset OUT # set and empty, it would hand its export mark, and so the default's full path as a named OUT, to this script's own calls
OUT=${OUT:-build$form_of}
if [[ $OUT == -* ]]; then OUT=./$OUT; fi # so that no tool takes it for an option
# verifiers_word: prints the verifier's source digest (`word` in names and messages), a SHA-256 over
# the verifier's sources, by the same command as verifier/build.sh. That script compiles the digest
# into cp_verify, which prints it as build=; verifier_unit() compiles it into the verifier's unit.
verifiers_word() {
  (cd ../verifier/src && export LC_ALL=C && { sha256sum * ../../prover/*/plan_rounds.h ../../prover/hiding/token_plan.h; cd ../../blocks && find . -type f -name '*.h' ! -path '*/tests/*' | sort | xargs sha256sum; } | sha256sum | cut -c1-64)
}
# from_these_sources <file>: succeeds if <file> is an executable file that contains that digest. The
# digest covers the sources only: not the compiler or the flags.
from_these_sources() {
  local word
  word=$(verifiers_word) && [[ $word =~ ^[0123456789abcdef]{64}$ ]] && [ -f "$1" ] && [ -x "$1" ] && LC_ALL=C grep -a -q -F -- "$word" "$1"
}
not_from_these_sources="is not there, is no program, or was built from other sources than those that lie here (it does not hold their word, which it would print as build=): cp_commit checks its own output with it, so it must match. First run $(realpath -m ../verifier/build.sh) --field $field"
# verifier_beside <directory>: copies cp_verify, which cp_commit runs, into the work
# directory, and adds it to the files that finish() installs: it goes into <directory>, a path that
# begins with $OUT/. Installing is a rename, which does not disturb a process that is running the
# cp_verify that was there before.
# In a serving target the copy must pass from_these_sources(), or the build fails: cp_commit checks
# its own output with it, so it must come from the same sources. The other targets do not check the
# copy.
verifier_beside() {
  if ! in_work cp -T -- "$(pwd -P)/../bin$form_of/cp_verify" cp_verify; then
    echo "build.sh: $target wants a cp_verify beside it, a copy of ../bin$form_of/cp_verify: build the verifier first" >&2
    exit 1
  fi
  if [[ $target == serving* ]] && ! from_these_sources "$work/cp_verify"; then # only a serving target checks the copy: the other targets take what is there
    echo "build.sh: $target wants a cp_verify beside it, a copy of ../bin$form_of/cp_verify, which $not_from_these_sources" >&2
    exit 1
  fi
  more_files+=" ${1#"$OUT"/}/cp_verify"
}
# What a serving target checks before it creates anything. $said is the commit that the caller
# states for this tree; it goes into the build records unchecked.
said=${2:-}
if [[ ${1:-} == serving* ]]; then
  here=$(pwd -P) out=$(realpath -m -- "$OUT")
  odd=$(find ./ ../blocks ../verifier/src ! -type f ! -type d ! -path './build/*' ! -path './build-dw10/*' -print -quit)
  dotted=$(find ./ ../blocks ../verifier/src -mindepth 1 -name '.*' ! -path './build/*' ! -path './build-dw10/*' -print -quit)
  LC_ALL=C printf -v quoted '%q\n' "$PATH" "$out" "$here" # what of the caller's goes into every command's words (run())
  why=
  # The hex digits are spelled out because in some locales the range a-f matches more.
  if [[ $# -gt 2 || (-n $said && ! $said =~ ^[0123456789abcdef]{7,64}$) ]]; then
    why="after the target there may be at most one argument, the commit that you state for this tree: 7 to 64"
    why+=" characters out of 0-9 and a-f"
  elif [[ $'\n'$quoted == *$'\n'"\$'"* ]]; then # so %q writes a word that not every reader of shell words can read
    why="PATH, OUT or this folder's own path contains a control character or a byte outside ASCII, which no record can hold"
  elif [ -n "$odd" ]; then
    why="$odd is neither a file nor a folder, and the digest of the three folders would not cover what it points to. Build"
    why+=" from a clean checkout in a new directory"
  elif [ -n "$dotted" ]; then
    why="$dotted has a name that begins with a dot. It would go into the digest of the three folders, and nobody with a"
    why+=" clean checkout could then reproduce the digest. Build from a clean checkout in a new directory"
  elif [[ $out == "$here" || ($out == "$here"/* && $out != "$here"/build$form_of && $out != "$here"/build$form_of/*) ]]; then
    why="OUT ($out) lies in build.sh's folder and outside its build$form_of/, so the digest of that folder would include"
    why+=" what is built: leave OUT unset, or name a folder under build$form_of/ or outside $here"
  elif [[ $1 != serving_cp_prove ]] && ! from_these_sources ../bin$form_of/cp_verify; then # checked here, before compiling starts, and again on the copy in verifier_beside()
    why="the third program, cp_verify, is a copy of ../bin$form_of/cp_verify, which $not_from_these_sources"
  fi
  if [ -n "$why" ]; then
    echo "build.sh: a serving build is refused: $why" >&2
    exit 2
  fi
  verifier_sum=
  if [[ $1 != serving_cp_prove ]]; then verifier_sum=$(sha256sum ../bin$form_of/cp_verify | cut -c1-64); fi # goes into cp_commit's record; `serving` compares the installed copy with this sum
fi
# more_refusals <target> ..: a script that sources this one may define it, to reject a call here,
# before anything is created. It goes with more_targets, below.
if [ -n "$sourced" ] && declare -F more_refusals >/dev/null; then more_refusals "$@"; fi
# What is built in OUT is run, and later builds install over it. So this script builds only where,
# by owners and mode bits, no other account can put a file into OUT, or another directory in OUT's
# place. Three checks follow: of the directories above OUT (open_above()), of OUT itself, and of
# what OUT contains.
# An OUT that the caller names must pass all three. For the default OUT, inside the checkout, a
# place above OUT that another account may be able to change gives only a note, the two other checks
# then give only notes too, and the build goes on: whoever can change such a place can change this
# script and its sources too, so a rejection would protect nothing.
# open_above() accepts a sticky directory above OUT. The check of the run path's directory, further
# down, does not: the programs look up names there at every later start, and in a sticky directory
# another account can take a name that is free, now or later.
# open_above() has the same text in every script of this repository that has it.

# <absolute path of a directory>: walks it from the top as the system does, through each symbolic
# link. Returns 0 at the first place where another account may be able to swap what lies below: one
# whose owner is not this account, root or, under sudo, the account that ran sudo; a directory that
# its group or others can write in without the sticky bit; or one that is missing, is no directory
# or cannot be examined. It then sets place, why and cure for the message. Else it returns 1 and
# sets real, the path without links that it has looked at: the path to use. Keep it in step with
# others_can_change() in run_suite.py.
open_above() {
  local at=/ here=/ part target owner mode kind me links=0 parts=()
  place='' why='' cure='' real=''
  # A here-string needs a writable TMPDIR before bash 5.1. Up to a NUL: names can have newlines.
  # id, not $EUID: bash takes a value for that name from the caller's environment.
  if ! me=$(id -u 2>/dev/null) || [[ ${1:-} != /* ]] ||
    ! IFS=/ read -r -d '' -a parts < <(printf '%s\0' "$1"); then
    printf -v place %q "${1:-}"
    why='it could not be examined'
    return 0
  fi
  set -- ${parts[@]+"${parts[@]}"}
  while :; do
    if ! IFS=' ' read -r owner mode kind < <(
      LC_ALL=C stat -c '%u %a %F' -- "$here" 2>/dev/null
    ); then
      why='it could not be examined'
      if [ ! -e "$here" ] && [ ! -L "$here" ] && [ -x "$at" ]; then why='it does not exist'; fi
    elif [ "$owner" != 0 ] && [ "$owner" != "$me" ] &&
      { [ "$me" != 0 ] || [ "$owner" != "${SUDO_UID:-}" ]; }; then
      why='owned by another account'
      [ "$kind" != 'symbolic link' ] || why='a symbolic link that belongs to another account'
    elif [ "$kind" = directory ]; then
      if ((8#$mode & 8#22)) && ! ((8#$mode & 8#1000)); then
        why='its group or others can write in it, and it has no sticky bit'
        [ "$owner" != "$me" ] || [ "$me" = 0 ] || cure=yes # root's may be shared on purpose
      fi
    elif [ "$kind" != 'symbolic link' ]; then
      why='it is not a directory'
    else # the link's parts come before the rest of the path
      links=$((links + 1))
      if [ "$links" -gt 40 ] ||
        ! IFS= read -r -d '' target < <(readlink -z -- "$here" 2>/dev/null) ||
        ! IFS=/ read -r -d '' -a parts < <(printf '%s\0' "$target"); then
        why='it could not be examined'
      fi
    fi
    if [ -n "$why" ]; then
      printf -v place %q "$here"
      cure=${cure:+"; if $place is yours, close it with chmod go-w"}
      return 0
    fi
    if [ "$kind" = directory ]; then
      at=$here
    else
      set -- ${parts[@]+"${parts[@]}"} "$@"
      [[ $target != /* ]] || at=/
    fi
    while :; do # the next name; ".." as the system takes it, from the directory reached
      if ! (($#)); then
        real=$at
        return 1
      fi
      part=$1
      shift
      case $part in
        '' | .) ;;
        ..) at=${at%/*} at=${at:-/} ;;
        *) break ;;
      esac
    done
    here=${at%/}/$part
  done
}
# not_closed <finding> <remedy> <end of the note>: prints the finding with the remedy and exits with
# status 2. The exception is the default OUT after open_above() has found a place above it that
# another account may be able to change ($excused): then not_closed() prints the finding with
# <end of the note>, as a note, and returns.
excused=''
not_closed() {
  if [[ -n $named || -z $excused ]]; then
    echo "build.sh: nothing was built: $1: $2" >&2
    exit 2
  fi
  echo "build.sh: note: $1$3" >&2
}
[[ $OUT == /* ]] || OUT=$PWD/$OUT
while [[ $OUT == */ && $OUT != / ]]; do OUT=${OUT%/}; done
if [[ ${OUT##*/} == @(|.|..) ]]; then # (no case statement here: two tests find the targets' by the first "esac" at a line's start)
  echo "build.sh: nothing was built: OUT ($OUT) must end in a name of its own, not in \"/\", \".\" or \"..\"" >&2
  exit 2
fi
# Directories that are missing above OUT are created one at a time: each at the path without links
# that open_above() reports, below a parent that has passed its check, by a plain mkdir, which fails
# if the name has been taken and follows no link. `place` is written by printf %q: if it contains no
# backslash and no single quote, it is the path itself. A path that needs quoting is not created,
# and the message below then says that it does not exist.
while open_above "${OUT%/*}/" && [ "$why" = 'it does not exist' ] && [[ $place != *[\\\']* ]]; do
  (umask go-w && mkdir -- "$place") || break
done
if open_above "${OUT%/*}/"; then
  if [[ $why == it\ * ]]; then # it does not exist (and could not be made); it is not a directory; it could not be examined
    echo "build.sh: nothing was built: OUT ($OUT) cannot be used: $place: $why: set OUT to another directory" >&2
    exit 2
  fi
  found="OUT ($OUT) lies below $place, which another account may be able to change ($why): it could put another directory in OUT's place"
  found+=" while the build runs and choose what you later run as the programs"
  way="set OUT to a directory below directories that only you or root can change (such as one in your home directory, or in /tmp, whose"
  [ -n "$named" ] || excused=1
  not_closed "$found" "$way sticky bit keeps others from replacing it)$cure" \
    ", and change this script itself; for a build that no other account can touch, keep the checkout below directories that only you can write in$cure"
  real=$(cd -P -- "${OUT%/*}" && pwd) || exit 2 # the default, behind the note
fi
cure=''
OUT=${real%/}/${OUT##*/} # the path that the walk has looked at, without links, and OUT's own name
# Then OUT itself, which is created if it does not exist. It must be a directory, not a symbolic
# link, that neither its group nor others can write to; the sticky bit excuses nothing here (the
# same rule as in blocks/Makefile). It must belong to the caller, to root or, under sudo, to the
# account that ran sudo, the owners that open_above() and run_suite.py accept, so that
# "sudo ./build.sh" can use the directory of that account's earlier build. ACLs are not read. If the
# mkdir fails, the check after it finds no directory. The message does not suggest chmod: what
# another account put into OUT while it was writable would still be there.
if [ ! -e "$OUT" ] && [ ! -L "$OUT" ]; then (umask go-w && mkdir -- "$OUT") || true; fi
upper=$(id -u) || upper=0 # the second owner that counts, beside root: this account or, for root under sudo, the account that ran sudo
if [ "$upper" = 0 ] && [[ ${SUDO_UID:-} =~ ^[1-9][0-9]*$ ]]; then upper=$SUDO_UID; fi
way="set OUT to another directory, for example a new one" so=": another account may be able to change what you later run as the programs; to be safe from that, $way"
if [[ $(stat -c '%u %A' -- "$OUT" 2>/dev/null) != @(0|$upper)\ d????-??-? ]]; then
  found="OUT ($OUT) is not a directory (a symbolic link does not count) that belongs to you (or to root, or under sudo to the account that"
  not_closed "$found ran sudo) and that neither its group nor other users can write to, and what is built there is run" "$way" "$so"
fi
# Then what OUT contains. A link that another account has put at hc, say, would send the programs
# elsewhere, and another account can still put one into a directory below OUT that its group or
# others can write in, as a mkdir under umask 002 leaves it. find selects every owner other than
# root and $upper by "greater than" and "less than", never by "not": if a file vanishes while find
# runs (another build at the same time creates and removes files), every test on its owner is false,
# and a "not" would turn that into a match. If another call's work directory vanishes while find
# runs, find ends with a failure status (-ignore_readdir_race covers files only), so the search is
# run again, up to three times in all.
found='' searched=''
for _ in 1 2 3; do
  if found=$(find "$OUT" -ignore_readdir_race -mindepth 1 \( -uid +0 -uid "-$upper" -o -uid "+$upper" -o -type d -perm /022 \) -print -quit 2>/dev/null); then
    searched=1
    break
  fi
  if [ -n "$found" ]; then break; fi
done
if [[ -z $searched || -n $found ]]; then
  not_closed "OUT ($OUT) holds something that belongs to another account, or a directory that its group or others can write in${found:+ ($(printf %q "$found"))}, or could not be searched" \
    "$way" "$so"
fi
umask go-w # nothing that the build makes below OUT can be written by its group or by others, whatever the caller's umask
# Only one call at a time builds the production programs in a directory: of two calls at the same
# time, one's program could end up next to the other's record. The lock is on the directory
# $OUT/hc itself (no lock file is created) and ends with the process. `serving` calls this
# script again for each program; BUILD_SH_HOLDS_THE_LOCK tells those calls that the lock is held.
# Other targets take no lock and may be built at the same time into one OUT: each call builds in
# its own work directory, and cp_verify, which cp_commit installs, gets its name by a
# rename.
if [[ ${1:-} == serving* && -z ${BUILD_SH_HOLDS_THE_LOCK:-} ]]; then
  mkdir -p "$OUT/hc"
  exec 9<"$OUT/hc"
  if command -v flock >/dev/null && ! flock -n 9; then # (on a machine without util-linux's flock there is no lock)
    echo "build.sh: another call of build.sh is building the programs in $OUT/hc, or flock cannot lock that folder, as on some network file systems: wait until that call has ended, or name another folder on a local disk (OUT=<folder>)" >&2
    exit 2
  fi
  export BUILD_SH_HOLDS_THE_LOCK=1
fi
# The work directory. finish() installs what the target built in it, then tries to remove it.
if ! work=$(mktemp -d -- "$OUT/building.XXXXXX"); then
  echo "build.sh: nothing was built: no work directory can be created in $OUT: see the message above. If you have no" \
    "permission to write there, set OUT to another directory." >&2
  exit 2
fi
if [[ $work != /* ]]; then work=$PWD/$work; fi # in_work changes the directory, and one unit is compiled from another
work_id=$(stat -c %d:%i -- "$work") # device and inode
# in_work <command>: runs the command inside the work directory, after checking that this is still the directory that this
# call created. So nothing is moved out of, removed from or written into another directory that has since taken that path.
in_work() (cd -- "$work" && [ "$(stat -c %d:%i ./)" = "$work_id" ] && "$@")
# Commands for in_work(), which act in the current directory: write_new <file> creates the file from
# standard input and fails if the name is taken; add_to <file> appends standard input to the file;
# remove_files removes all files, hidden ones too, but no directory.
write_new() {
  set -C
  cat >"./$1"
}
add_to() { cat >>"./$1"; }
remove_files() {
  shopt -s nullglob dotglob
  local files=(./*)
  rm -f -- ${files[@]+"${files[@]}"}
}
# The work directory must be empty, so that whatever finish() finds in it later was created or moved
# there by this call.
if ! found=$(in_work ls -Aq) || [ -n "$found" ]; then
  echo "build.sh: nothing was built: the new work directory $work is not empty, or it is no longer the directory that this" \
    "call created. It was left as it is. Build again." >&2
  exit 2
fi
export TMPDIR=$work
# finish: runs when the script exits. After status 0 it installs what the target built, by renames
# out of the work directory: the files of $programs, then those of $more_files, so a program comes
# before its build record. Each entry is a path relative to $OUT; the file in the work directory has
# the path's last component as its name. A rename replaces a symbolic link and does not follow it.
# If a rename fails, what was renamed before it stays installed, and a message lists it. After any
# other status finish() installs nothing. Then it tries to remove the work directory and its files.
finish() {
  local status=$? file why to=$OUT moved='' stuck='' how=installed
  set +e # go on to the end even if a step fails
  trap '' HUP INT TERM # so that no signal parts a program from its record
  # After a failed step or a signal, a background command may still be running (bash starts
  # background commands with SIGINT ignored). It writes into the work directory, or it is a call of
  # this script that installs its own program.
  if [ -n "$(jobs -r)" ]; then echo "build.sh: waiting for a part of the build that still runs in the background" >&2; fi
  wait
  # The script can end with status 0 before the record is written, so for a serving program the status alone is not enough.
  if [[ $status = 0 && -n $made && " $more_files " != *" $made.build "* ]]; then
    echo "build.sh: $OUT/$made was not installed: the build ended before its record was written: build again" >&2
    status=1
  fi
  if [ $status = 0 ]; then
    if [[ $to != /* ]]; then to=$PWD/$to; fi # in_work changes the directory
    # An older cp_verify must not stay next to a new production cp_commit. It is moved into the work
    # directory here and not at the start of the target, so that a failed build leaves an earlier
    # build's program, record and cp_verify together. `serving` installs the new copy. The rename
    # counts as failed only if the name still exists afterwards: there may be no older copy.
    if [ "$made" = hc/cp_commit ] && ! why=$(in_work mv -fT -- "$to/${made%/*}/cp_verify" earlier-cp_verify 2>&1) && [[ -e $to/${made%/*}/cp_verify || -L $to/${made%/*}/cp_verify ]]; then
      echo "$why" >&2
      stuck=${made%/*}/cp_verify how="moved out of the way"
    fi
    for file in $programs $more_files; do
      if [ -n "$stuck" ]; then break; fi
      if ! in_work mv -fT -- "${file##*/}" "$to/$file"; then
        stuck=$file
        break
      fi
      moved+=" $file"
    done
    if [ -n "$stuck" ]; then
      status=1
      why="$target was built, but $OUT/$stuck could not be $how: see the other messages. Installed before that:${moved:- nothing}."
      why+=" Set OUT to another directory."
      if [ -n "$made" ]; then why+=" $OUT/$made is no serving build unless the record beside it describes it."; fi
      echo "build.sh: $why" >&2
    elif [ -n "$last_line" ]; then
      echo "$last_line"
    fi
  elif [ -n "$programs" ]; then
    echo "build.sh: $target failed with status $status; not installed: $programs. This call has replaced and removed nothing in" \
      "$OUT." >&2
  elif [ -n "$calls_itself" ]; then
    why="A call for a single program that failed says so above. One that ended well has installed its program"
    if [ "$target" = serving ]; then
      why+=", and cp_commit's call has taken an earlier cp_verify away, so $OUT may now hold programs of two builds, or no cp_verify"
    else
      why+=", so $OUT may now hold programs of two builds"
    fi
    echo "build.sh: $target failed with status $status. $why. Build again." >&2
  fi
  if ! { in_work remove_files && [ "$(stat -c %d:%i -- "$work")" = "$work_id" ] && rmdir -- "$work"; }; then
    echo "build.sh: warning: the work directory $work could not be removed: it is not empty, or it is no longer the" \
      "directory that this call created. The build does not need it any more." >&2
  fi
  exit $status
}
target=${1:-} made='' checked='' may_hold='' no_rpath='' rpath_note='' programs='' more_files='' last_line='' calls_itself='' sample_passed='' sample_word='' # set here, so that no value is inherited from the caller's environment
trap finish EXIT
# After a signal the script ends with a status that is not 0, and finish() installs nothing. bash runs these traps once the
# command that is running, such as the compiler, has ended.
# The trap on INT has no effect in a call that was started with SIGINT ignored, as bash starts a
# script's background commands: a script cannot trap such a signal. See the branch `serving`.
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
host_targets="make_model in_pieces_test moe_at_layer_test hc_host_test"
# digest: prints the SHA-256 that a production program's build record (<program>.build) gives for
# the sources. It covers the three directories from which the programs are compiled, verifier/src
# for cp_prove's record too, though cp_prove compiles nothing from it: this one without build/, build-dw10/, Python's caches, *.md and the directories named tests;
# blocks/ by the same rule; and all of verifier/src. It covers files of any name, because -I. is
# searched before the system's directories, so a stray file named like a system header would be
# compiled in. No production program is compiled from a *.md file or from a file in a directory
# named tests. One file of prover/tests is covered, because every serving target reads it: the list
# of strings that no production program may contain.
# What an operator compares the record with: docs/operators-duties.md, "Keys, seeds and machines".
digest() {
  { find ./ -type f ! -path './build/*' ! -path './build-dw10/*' ! -path '*/__pycache__/*' ! -name '*.md' \
    \( ! -path '*/tests/*' -o -path './tests/serving_build_words.txt' \) -print0 &&
    find ../blocks -type f ! -path '*/__pycache__/*' ! -name '*.md' ! -path '*/tests/*' -print0 && find ../verifier/src -type f -print0; } | LC_ALL=C sort -z |
    LC_ALL=C xargs -0 sha256sum | sha256sum | cut -c1-64
}
# begins <program>: the first step of a serving target. It records the program's path, relative to
# $OUT, for finish(), names the list of commands that run() fills in the work directory, and takes
# the digest.
begins() {
  made=$1
  mkdir -p "$OUT/${made%/*}"
  programs=$made # for finish()
  commands=commands
  if ! before=$(digest); then
    echo "build.sh: the digest of the three folders cannot be computed" >&2
    exit 1
  fi
}
# run <command>: every command that makes a part of a production program (compiler, ld, objcopy)
# goes through here, and so do the commands of keeper_unit() and verifier_unit() in every target.
# Outside the serving targets run() only runs the command.
# In a serving target the command runs under env -i ($clean): of the caller's environment it sees
# only PATH, and its TMPDIR is the work directory. nvcc adds the contents of NVCC_PREPEND_FLAGS and
# NVCC_APPEND_FLAGS to every command line, and the host compiler looks for headers on CPATH: either
# could undo -DSERVING_BUILD=1 or set another macro. The command also goes into the build record: a
# line "command <working directory>", then a line "flag <word>" for each word of $clean and of the
# command. Under LC_ALL=C, %q keeps each word on one line and in ASCII. The record gives the work
# directory's path without its random suffix, so that two builds give the same record.
# The compiler sample, below, is no part of a program: a serving target compiles it under $clean
# too, but not through run(), and no record lists it.
clean=(env -i "PATH=$PATH" "TMPDIR=$work") # a variable that some machine's compiler needs is added here, with the reason
run() {
  if [[ $target != serving_* ]]; then
    "$@"
    return
  fi
  local word listed=() words
  for word in "${clean[@]}" "$@"; do listed+=("${word//"$work"/"${work%.*}"}"); done
  LC_ALL=C printf -v words 'flag %q\n' "${listed[@]}"
  LC_ALL=C printf -v words 'command %q\n%s' "$PWD" "$words"
  if [[ $'\n'$words == *$'\n'"flag \$'"* || $words == "command \$'"* ]]; then # the look at the script's head has passed the caller's three; this one sees every word
    echo "build.sh: a serving build is refused: a word of a command contains a control character or a byte" \
      "outside ASCII: check PATH, OUT and this folder's own path" >&2
    exit 2
  fi
  printf '%s' "$words" | in_work add_to "$commands"
  "${clean[@]}" "$@"
}
# compiles <nvcc command for the program's own unit>: runs it, and once more with --dryrun. The dry
# run's output ($dry) shows tools() where the compile finds its programs.
compiles() {
  run "$@"
  dry=$("${clean[@]}" "$@" --dryrun 2>&1) || dry=
}
# file_of <name> <path>: prints two lines of the record: the file that the path resolves to, and its
# SHA-256.
file_of() {
  local path sum
  path=$(readlink -e "$2") && sum=$(sha256sum <"$path") || return 1
  LC_ALL=C printf '%s %q\n%s-sha256 %s\n' "$1" "$path" "$1" "${sum%% *}"
}
# tools <env and three compilers>: prints the record's lines on these four programs, on cicc (the
# device compiler) and on ptxas: path, SHA-256 and, for the three compilers, version. env and the
# first compiler are found on PATH. The others are found where the dry run shows that the compile
# looks: cicc in the dry run's CICC_PATH ($middle), the rest on its PATH ($looks). tools() fails
# unless the dry run prints each of the two variables once, with an absolute path, every file can be
# found and read, and every compiler prints a version. The "|| return 1" are needed because the
# caller runs tools() in $(...), where set -e has no effect.
tools() {
  local name says middle looks found=$PATH
  middle=$(sed -n 's/^#\$ CICC_PATH=//p' <<<"$dry") looks=$(sed -n 's/^#\$ PATH=//p' <<<"$dry")
  [[ $middle == /* && $middle != *$'\n'* && $looks == /* && $looks != *$'\n'* ]] || return 1
  for name in env-program compiler host-compiler host-linker; do
    if [[ $name == host-* ]]; then found=$looks; fi
    file_of $name "$(PATH=$found command -v "$1")" || return 1
    if [ $name != env-program ]; then
      says=$(env -i "PATH=$found" "$1" --version) || return 1
      if [ -z "$says" ]; then return 1; fi
      if [ $name = compiler ]; then says=$(tail -n 2 <<<"$says" | paste -sd ';' | sed 's/;/; /'); fi
      printf '%s-says %s\n' $name "${says%%$'\n'*}"
    fi
    shift
  done
  file_of device-compiler "$middle/cicc" || return 1
  file_of device-assembler "$(PATH=$looks command -v ptxas)"
}
nvcc_base=(nvcc -O3 -std=c++17 -arch=sm_90a --extended-lambda --default-stream per-thread -I. -Xcompiler -fno-ipa-modref -DCCCL_DISABLE_NVTX -DDIGEST_WORDS=$dw)
# -fno-ipa-modref, for the host compiler here and in the g++ lines below, works around a bug in g++
# 13.3.0, which can silently delete the stores that fill an object passed by reference
# (verifier/tests/compiler_pass_control.sh reproduces it). cuBLASLt is for
# prover/hiding/gram_products.cuh.
# -DCCCL_DISABLE_NVTX: where the toolkit has NVIDIA's NVTX headers (include/nvtx3), a header of its
# CUDA C++ Core Libraries that the programs' compiles take in (cccl/cuda/__nvtx/nvtx.h) includes
# them unless CCCL_DISABLE_NVTX is defined; they add to a program NVTX's start-up code, which reads
# NVTX_INJECTION64_PATH with getenv() and loads the library named there (read in that header and in
# nvtx3/nvtxDetail/nvtxInit.h). No program here uses NVTX. tests/serving_build_words.txt has that
# name, so a production program that contains it is not installed.
nvcc_base+=(-lcublasLt "${field_flags[@]}")
# EIDB and ETAU, if the environment sets them.
form=()
if [ -n "${EIDB:-}" ]; then form+=("-DEIDB=$EIDB"); fi
if [ -n "${ETAU:-}" ]; then form+=("-DETAU=$ETAU"); fi
sha=(-Xcompiler "-msha,-msse4.1,-mssse3")
# keeper_unit <directory> [<flags>]: compiles the table holder (`keeper` in identifiers), the one
# translation unit of cp_prove that reads the weight table's internals
# (prover/hash_commitment/hf_keeper_unit.h), into <directory>/hf_keeper.o, hf_keeper_2.o and
# hf_keeper_3.o. cp_commit does not link them: it makes the tables in its own unit.
# The unit and the program include the same headers, which were written for programs of one
# translation unit: kernels, functions and the allocation operators, all with external linkage,
# would be defined twice. So only the unit's entry point, hf::hf_keeper_unit(), stays visible to the
# linker. First ld -r dissolves the section groups that a linker would merge with the program's
# equal ones; then objcopy makes every other name local. Each side then has its own copy of every
# function and static object; they share only the entry point's arguments and what these refer to.
# (GNU binutils: ld, "--force-group-allocation"; objcopy, "--keep-global-symbol", "--redefine-sym".)
# Three objects, because the unit keeps state in static variables (memory pools, failure flags on
# the host and on the GPU), so one copy serves one opening of the weight table at a time. cp_prove
# uses two copies with -overlap and three with -overlap-two. hf_keeper_2.o and hf_keeper_3.o are
# hf_keeper.o with only the entry point renamed, to hf::hf_keeper_unit_2() and _3().
keeper_unit() {
  local to=$1 door n further
  shift
  # -fno-gnu-unique: else g++ marks inline variables as unique in the process, a binding that objcopy leaves as it is.
  run "${nvcc_base[@]}" "${form[@]}" "${sha[@]}" -Xcompiler -fno-gnu-unique -cudart shared "$@" -c -o "$to/hf_keeper_as_compiled.o" \
    hash_commitment/hf_keeper.cu
  # The entry point's mangled name.
  door=$(nm "$to/hf_keeper_as_compiled.o" | awk '$2 == "T" && $3 ~ /^_ZN2hf14hf_keeper_unitE/ { print $3 }')
  if [[ ! $door =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "build.sh: hf_keeper.cu's object file does not define hf::hf_keeper_unit() exactly once" >&2
    return 1
  fi
  run ld -r --force-group-allocation -o "$to/hf_keeper_one_piece.o" "$to/hf_keeper_as_compiled.o"
  run objcopy --keep-global-symbol="$door" "$to/hf_keeper_one_piece.o" "$to/hf_keeper.o"
  for n in 2 3; do
    further=${door/14hf_keeper_unit/16hf_keeper_unit_$n}
    if [[ $further == "$door" ]]; then
      echo "build.sh: the door's name, $door, does not hold 14hf_keeper_unit: the further copies' doors cannot be named after it" >&2
      return 1
    fi
    run objcopy --redefine-sym "$door=$further" "$to/hf_keeper.o" "$to/hf_keeper_$n.o"
  done
}
# verifier_unit <directory>: compiles verifier/src/cp_verify.cpp once more, without its main(), into
# <directory>/cp_configured.o, for the function cp_with_configured() (verifier/src/cp_configured.h).
# The verifier's and the prover's sources both define plain structs of the same names outside any
# namespace (ACall, for example), with different members. A linker would take the functions that the
# compiler generates for them (destructors, the containers' functions) for the same and keep one of
# each, so that one side would run the other's code; a program linked that way failed on an
# allocation inside the unit. So, by the same steps as in keeper_unit(), only the entry point stays
# visible to the linker. <directory> must be an absolute path: the compiler runs in verifier/src.
verifier_unit() {
  local to word door
  to=$1
  word=$(verifiers_word)
  (cd ../verifier/src && run g++ -O2 -fno-ipa-modref -std=c++17 -pthread -I. -I../../blocks -I../../prover "${field_flags[@]/#-I..\/blocks/-I../../blocks}" -DSERVING_BUILD=1 -DDIGEST_WORDS=$dw \
    -DVF_SOURCE_DIGEST="\"$word\"" -DCP_VERIFY_AS_A_UNIT -fno-gnu-unique -c cp_verify.cpp -o "$to/cp_configured_as_compiled.o")
  door=$(nm "$to/cp_configured_as_compiled.o" | awk '$2 == "T" && $3 ~ /^_Z18cp_with_configured/ { print $3 }')
  if [[ ! $door =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "build.sh: cp_verify.cpp's object file does not define cp_with_configured() exactly once" >&2
    return 1
  fi
  run ld -r --force-group-allocation -o "$to/cp_configured_one_piece.o" "$to/cp_configured_as_compiled.o"
  run objcopy --keep-global-symbol="$door" "$to/cp_configured_one_piece.o" "$to/cp_configured.o"
}
# The compiler version check. It runs for every target except those in host_targets, which do not run nvcc. (Of a script that sources this one, an argument
# that is not a target gets the usage message only if the compiler passes the check.) A serving target that skipped the check gets no record.
if [[ " $host_targets " != *" $target "* ]]; then
  ./check_nvcc_version.sh || exit 1
  checked=1
fi
# Every program that nvcc links gets a run path (-rpath) to the library directory of the toolkit
# that compiles it: the first of lib64 (where NVIDIA's installers put the libraries) and lib (where
# its Python packages put them), next to nvcc's own directory, that contains libcudart.so, with
# links resolved. The loader then finds the CUDA runtime and cuBLAS without LD_LIBRARY_PATH, which
# is lost when a program starts itself again with a reduced environment (reduced_env_start(),
# prover/common/reduced_env_start.h), and without an entry in its configuration, which only root can add.
# libcuda.so.1 comes with the driver: the loader must still find it through its configuration.
# Whoever can write to that directory, to a directory above it or to a library in it chooses code
# that the programs load. So the run path is set only if, by owners and mode bits at build time,
# root and the user who builds are the only ones who can; ACLs are not read. Whoever can write the
# builder's files can replace the programs themselves, so the run path gives nobody more than they
# have, as long as the programs stay the builder's files on this machine
# (prover/docs/build.md, "What you need").
# Without a run path the build goes on, and a warning after the case statement gives the reason. For
# a serving target, a note there names the first directory or library that is not root's.
# `serving` and `tests` compile nothing themselves: they make the same checks in order to print, at
# their end, that warning or the run path that cp_prove and cp_commit have. The targets in
# host_targets, which do not run nvcc, skip all this.
if [[ " all $host_targets " != *" $target "* ]]; then # "serving" and "tests" compile nothing: they look in order to say the outcome at their end
  toolkit=$(dirname -- "$(realpath -- "$(command -v nvcc)" 2>/dev/null)")/..
  cuda_libraries=
  for name in lib64 lib; do
    if [ -z "$cuda_libraries" ] && [ -e "$toolkit/$name/libcudart.so" ]; then
      cuda_libraries=$(realpath -e -- "$toolkit/$name" 2>/dev/null) || cuda_libraries=
    fi
  done
  if [ -z "$cuda_libraries" ]; then
    no_rpath="the CUDA runtime, libcudart.so, is neither in $toolkit/lib64 nor in $toolkit/lib, beside the compiler's own directory"
  # nvcc passes the path to a shell unquoted and splits it at a comma; the loader splits it at a colon and expands "$".
  elif [[ $cuda_libraries == *[!A-Za-z0-9_./+@-]* ]]; then
    no_rpath="the path $cuda_libraries contains a character other than an ASCII letter, a digit, or one of _./+@-"
  fi
  builder=$(id -u) whose="root and you" above=$cuda_libraries not_roots=
  advice="use a toolkit in a directory that only root and you can write to, with every directory above it"
  while [ -z "$no_rpath" ]; do
    seen=$(stat -c '%u %A' -- "$above") || seen=
    # The two "-": no write permission for the group or for others.
    if [[ $seen != @(0|$builder)\ d????-??-? ]]; then
      no_rpath="$above belongs to, or can be written by, a user other than $whose ($advice; then build again)"
    fi
    if [[ -z $not_roots && $seen != 0\ * ]]; then not_roots=$above; fi
    if [ "$above" = / ]; then break; fi
    above=$(dirname -- "$above")
  done
  # The same for every lib*.so* in that directory, not only CUDA's: the loader looks there, before
  # its own configuration, for each library that a program needs. A link must lead to a file in that
  # directory.
  for library in "$cuda_libraries"/lib*.so*; do
    if [ -n "$no_rpath" ]; then break; fi
    if [ ! -e "$library" ] && [ ! -L "$library" ]; then continue; fi # no such name
    file=$(realpath -e -- "$library" 2>/dev/null) || file=
    if [ -L "$library" ] && [[ $(readlink -- "$library") == */* ]]; then file=; fi
    seen=$(stat -c '%u %A' -- "$file" 2>/dev/null) || seen=
    if [[ ${file%/*} != "$cuda_libraries" ]]; then
      no_rpath="$library is a link that points to another directory, or to nothing (for a run path, put the file itself there or use another toolkit; then build again)"
    elif [[ $seen != @(0|$builder)\ -????-??-? ]]; then
      no_rpath="$file is not a regular file that only $whose can write ($advice; then build again)"
    fi
    if [[ -z $not_roots && $seen != 0\ * ]]; then not_roots=$file; fi
  done
  if [ -z "$no_rpath" ]; then
    nvcc_base+=(-Xlinker -rpath -Xlinker "$cuda_libraries")
    if [[ $target == serving* && -n $not_roots ]]; then
      name=$(id -un 2>/dev/null) || name="user $builder" # an account without a name, as in some containers
      subject=$target
      if [ "$target" = serving ]; then subject="each of the two programs"; fi
      rpath_note="$subject will load the CUDA libraries from $cuda_libraries, and $not_roots belongs to $name, not to root: whoever"
      rpath_note+=" can write there chooses code that the program loads, no matter who runs it or where it is copied; see prover/docs/build.md."
    fi
  fi
fi
# The check of the compiler by value. compiler_sample.cu, whose head says what it checks, is
# compiled with $nvcc_base into the work directory and, if the machine has a GPU, run. A wrong value
# stops the build, and so does an exit status that the program never returns (it cannot be started,
# or it is killed). Without a GPU, or with one that cannot be used, a message says that only the
# compiler's version was checked, and the build goes on. The run has no time limit: nothing in this
# script ends a GPU program from outside. A serving target's record keeps the outcome.
# The sample is started inside the work directory, as ./compiler_sample. env takes every word with
# "=" that comes before the command as an assignment: given the sample's full path with "=" in it,
# the env of $clean would start nothing and exit with status 0. Status 0 counts only together with
# the line that the sample prints when all values are equal. A message after the build repeats it.
if [[ " all tests serving $host_targets " != *" $target "* ]]; then
  sample=$work/compiler_sample under=() gpus=()
  if [[ $target == serving_* ]]; then # as run() starts a serving target's commands; the run alone also gets the caller's choice of GPUs
    under=("${clean[@]}") gpus=(${CUDA_VISIBLE_DEVICES+"CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"})
  fi
  "${under[@]}" "${nvcc_base[@]}" -o "$sample" compiler_sample.cu
  sample_status=0
  sample_says=$(in_work "${under[@]}" "${gpus[@]}" ./compiler_sample 2>&1) || sample_status=$?
  if [ $sample_status = 77 ] || [ $sample_status = 2 ]; then sample_word="not-run status $sample_status" # no GPU; the GPU could not be used
    echo "build.sh: the compiler sample was built but not run to its end, so this build has not checked the compiler by value (status $sample_status: $sample_says): only the compiler's version has been checked; see prover/docs/build.md" >&2
  elif [ $sample_status = 1 ]; then
    echo "build.sh: nothing is built: a small kernel that this compiler has just compiled gives a wrong result or fails on this machine's GPU ($sample_says). compiler_sample.cu says what it checks. Do not use programs that this compiler built; use another version of nvcc. If the brackets are empty or name cd, the sample was not started: the work directory has gone or is no longer the one that this call made: build again." >&2
    exit 1
  elif [ $sample_status != 0 ]; then
    echo "build.sh: nothing is built: the compiler sample, which this compiler has just compiled, ended with status $sample_status, which it never returns ($sample_says). compiler_sample.cu says what it checks." >&2
    exit 1
  elif [[ $sample_says != *"compiler_sample: "*" values are the same on the GPU and on the CPU"* ]]; then
    echo "build.sh: nothing is built: the compiler sample ended with status 0 but did not print that its values are the same on the GPU and on the CPU, so it may not have run ($sample_says). compiler_sample.cu says what it checks." >&2
    exit 1
  else
    sample_passed=$sample_says sample_word=passed
  fi
fi
programs=$target # the file that a target builds has the target's name and goes into $OUT, unless its branch or the second case statement says otherwise
usage() {
  echo "$usage_line" >&2
  programs=
  exit 2
}
case "${1:-}" in
  serving)
    programs= calls_itself=1 # its two calls of this script install the programs, each its own, and say so if they do not
    echo "building two programs, side by side: about half an hour in all. Little is printed meanwhile but the compiler's warnings. The last line of a good end begins THE PROGRAMS ARE BUILT"
    # In parallel: each call writes separate files (its program, its record, its work directory),
    # and together they take under 10 GB of memory. The script waits for both, even if one fails.
    # cp_commit's record gives the SHA-256 that the cp_verify to be copied had when that call started
    # (line "verifier-beside"). The copy made at the end must match it; if that cp_verify changed
    # during the build, the copy is not installed and the build fails.
    # bash starts a background command with SIGINT ignored, and a script cannot trap a signal that
    # was ignored when it started. env sets SIGINT back to its default for the background call, so
    # that Ctrl-C, which a terminal sends to every process of the build, ends it like the others.
    # Not where this call itself was started with SIGINT ignored (its own trap did not take): then
    # no call of the build hears it. Nor where env lacks the option (GNU coreutils before 8.31).
    hears=(env --default-signal=INT)
    if [[ $(trap -p INT) != *"exit 130"* ]] || ! "${hears[@]}" true 2>/dev/null; then hears=(); fi
    echo "building serving_cp_prove"
    ${hears[@]+"${hears[@]}"} ./build.sh --dw $dw --field $field serving_cp_prove ${said:+"$said"} &
    first=$!
    echo "building serving_cp_commit"
    second=0
    ./build.sh --dw $dw --field $field serving_cp_commit ${said:+"$said"} || second=$?
    wait $first || exit $?
    if [ $second != 0 ]; then exit $second; fi
    verifier_beside "$OUT/hc"
    # the record names the verifier as it lay at the call's start: the copy is that file
    if [ "$(sed -n 's/^verifier-beside //p' "$OUT/hc/cp_commit.build")" != "$(sha256sum "$work/cp_verify" | cut -c1-64)" ]; then
      echo "build.sh: ../bin$form_of/cp_verify has changed while the two programs were built: the copy is not the file that cp_commit's record names, and is not installed. Build once more" >&2
      exit 1
    fi
    last_line="THE PROGRAMS ARE BUILT: cp_commit, cp_prove and cp_verify lie in $(readlink -f "$OUT/hc")" # finish() prints it
    ;;
  # CPU test: work on a long range of bytes, divided among threads.
  in_pieces_test)
    g++ -O2 -fno-ipa-modref -std=c++17 -DDIGEST_WORDS=$dw -Wall -Wextra -pthread -I. "${field_flags[@]}" -o "$work/in_pieces_test" tests/in_pieces_test.cpp
    ;;
  # CPU test: moe_at_layer() marks a layer MoE only when the model has routed experts (NE > 0).
  moe_at_layer_test)
    g++ -O2 -std=c++17 -Wall -Wextra -I. -o "$work/moe_at_layer_test" tests/moe_at_layer_test.cpp
    ;;
  # Test: the implementations of the tree hash against the repository's test vectors.
  tree_hash_test)
    "${nvcc_base[@]}" "${form[@]}" "${sha[@]}" -o "$work/tree_hash_test" tests/tree_hash_test.cu -lcuda -ldl
    ;;
  # Test: the range check's kernel that contracts the matrix's columns, against plain sums.
  tail_gam_test)
    "${nvcc_base[@]}" -I../blocks -o "$work/tail_gam_test" tests/tail_gam_test.cu
    ;;
  # Test: the sub-proofs' transcript over a hash chain (prover/hiding/transcript_shell.cuh).
  psc_shell_test)
    "${nvcc_base[@]}" "${form[@]}" "${sha[@]}" -o "$work/psc_shell_test" tests/psc_shell_test.cu -lcuda -ldl
    ;;
  # Stand-alone tests of single sumchecks (prover/tests/door_test_host.cuh).
  token_sumcheck_psc_test | attention_sumchecks_test)
    "${nvcc_base[@]}" "${form[@]}" "${sha[@]}" -o "$work/$1" "tests/$1.cu" -lcuda -ldl
    ;;
  # Test: a group's whole lookup on the GPU (prover/hiding/forest.cuh), for
  # verifier/tests/forest_lookup_check.cpp.
  forest_lookup_test)
    "${nvcc_base[@]}" "${form[@]}" "${sha[@]}" -I../blocks -o "$work/forest_lookup_test" tests/forest_lookup_test.cu \
      -lcuda -lcublasLt -ldl
    ;;
  # Benchmark: the time of a group's whole lookup on the GPU.
  forest_lookup_bench)
    "${nvcc_base[@]}" "${form[@]}" "${sha[@]}" -I../blocks -o "$work/forest_lookup_bench" tests/forest_lookup_bench.cu -lcuda -lcublasLt -ldl
    ;;
  # The program of the compiler check above, installed so that the test table can run it on a GPU
  # (SUITE.md, gpu.compiler_sample).
  compiler_sample)
    "${nvcc_base[@]}" -o "$work/compiler_sample" compiler_sample.cu
    ;;
  make_model)
    g++ -O2 -fno-ipa-modref -std=c++17 -DDIGEST_WORDS=$dw -msha -msse4.1 -mssse3 -pthread -I. "${field_flags[@]}" -o "$work/make_model" \
      tests/weights_blocks/make_model.cpp
    ;;
  # From here to hc_host_test the targets build into $OUT/hc/test_build, except the two serving
  # targets, which build into $OUT/hc.
  cp_prove)
    mkdir -p "$OUT/hc/test_build"
    keeper_unit "$work" & # beside the long compile of the program itself
    keeper=$!
    "${nvcc_base[@]}" "${form[@]}" -DCHAIN_HOST_ENGINE=1 -cudart shared -c -o "$work/cp_prove.o" prover/cp_prove.cu
    wait $keeper # its status is this line's: a failed compile of a unit ends the script
    "${nvcc_base[@]}" -cudart shared -o "$work/cp_prove" "$work/cp_prove.o" "$work"/hf_keeper{,_2,_3}.o -lcublas -lcuda -ldl
    ;;
  cp_commit)
    # cp_commit gets its blk::Configured from the verifier's cp_with_configured(), compiled by
    # verifier_unit().
    # -DHC_RANGE_N_MOST has no effect: no source reads it. The limit on the keys of a part of the
    # range lookup is 2^HF_RANGE_N_MOST = 2^40 (prover/hash_commitment/hf_range_format.h). With
    # prover/examples/target_full_f2/policy.txt a part has up to 2^36 keys: 2^(18 + 18), from the
    # line ranges-part and the last number of the line table (worked out, not measured).
    mkdir -p "$OUT/hc/test_build"
    verifier_unit "$work" &
    "${nvcc_base[@]}" "${form[@]}" "${sha[@]}" -I../verifier/src -DHC_RANGE_N_MOST=40 -cudart shared -c -o "$work/cp_commit.o" hash_commitment/hf_commit.cu
    wait $!
    "${nvcc_base[@]}" -cudart shared -o "$work/cp_commit" "$work/cp_commit.o" "$work/cp_configured.o" -lcublasLt -lcublas -lcuda -ldl
    verifier_beside "$OUT/hc/test_build"
    ;;
  # Test: compares what the prover's code and the verifier's code derive from a commitment's public
  # files (tests/frame_as_the_verifier_reads.cu).
  frame_as_the_verifier_reads)
    mkdir -p "$OUT/hc/test_build"
    keeper_unit "$work" &
    keeper=$!
    verifier_unit "$work" &
    unit=$!
    "${nvcc_base[@]}" "${form[@]}" -DCHAIN_HOST_ENGINE=1 -I../verifier/src -cudart shared -c -o "$work/frame_as_the_verifier_reads.o" tests/frame_as_the_verifier_reads.cu
    wait $keeper # its status is this line's: a failed compile of a unit ends the script
    wait $unit
    "${nvcc_base[@]}" -cudart shared -o "$work/frame_as_the_verifier_reads" "$work/frame_as_the_verifier_reads.o" "$work"/hf_keeper{,_2,_3}.o "$work/cp_configured.o" -lcublas -lcuda -ldl
    ;;
  # The two serving targets: the commands of cp_prove and cp_commit above, with -DSERVING_BUILD=1 on
  # the program's own unit and on the table holder's (the verifier's unit has it in every build).
  # The commands run one after the other, so that the record lists them in a fixed order. See run().
  serving_cp_prove)
    begins hc/cp_prove
    parts=$work # under a name of its own: by it a test tells the serving targets' commands from the others'
    keeper_unit "$parts" -DSERVING_BUILD=1
    compiles "${nvcc_base[@]}" "${form[@]}" -DCHAIN_HOST_ENGINE=1 -DSERVING_BUILD=1 -cudart shared -c -o "$parts/cp_prove.o" prover/cp_prove.cu
    run "${nvcc_base[@]}" -cudart shared -o "$work/cp_prove" "$parts/cp_prove.o" "$parts"/hf_keeper{,_2,_3}.o -lcublas -lcuda -ldl
    with=$(tools env nvcc gcc g++ && file_of object-linker "$(command -v ld)" && file_of object-copier "$(command -v objcopy)") || with=
    ;;
  serving_cp_commit)
    # -cudart shared, as in cp_commit: the static CUDA runtime reads the environment for itself, and
    # a check below limits which such functions a production cp_commit may import.
    begins hc/cp_commit
    parts=$work # under a name of its own: by it a test tells the serving targets' commands from the others'
    verifier_unit "$parts"
    compiles "${nvcc_base[@]}" "${form[@]}" "${sha[@]}" -I../verifier/src -DHC_RANGE_N_MOST=40 -DSERVING_BUILD=1 -cudart shared -c -o "$parts/cp_commit.o" hash_commitment/hf_commit.cu
    run "${nvcc_base[@]}" -cudart shared -o "$work/cp_commit" "$parts/cp_commit.o" "$parts/cp_configured.o" -lcublasLt -lcublas -lcuda -ldl
    with=$(tools env nvcc gcc g++ && file_of object-linker "$(command -v ld)" && file_of object-copier "$(command -v objcopy)") || with=
    ;;
  hc_unit_test)
    # hc_unit_test tests/<name> compiles prover/hash_commitment/tests/<name>.cu as
    # $OUT/hc/test_build/tests/<name>; hc_unit_test today/<name> compiles prover/tests/<name>.cu as
    # $OUT/hc/test_build/<name>.
    if [[ ! ${2:-} =~ ^(tests|today)/[a-z_0-9]+$ || $# -gt 2 ]]; then
      echo "build.sh: hc_unit_test wants one word behind it: tests/<name>, and it compiles" \
        "hash_commitment/tests/<name>.cu; or today/<name>, and it compiles" \
        "tests/<name>.cu; a name has small letters, digits and _" >&2
      exit 2
    fi
    from=hash_commitment/$2.cu to=hc/test_build/$2
    if [[ $2 == today/* ]]; then from=tests/${2#today/}.cu to=hc/test_build/${2#today/}; fi
    programs=$to
    mkdir -p "$OUT/${to%/*}"
    "${nvcc_base[@]}" "${form[@]}" "${sha[@]}" -cudart shared -o "$work/${to##*/}" "$from" -lcublas -lcuda -ldl
    ;;
  # hc_host_test tests/<name>: a CPU test, prover/hash_commitment/tests/<name>.cpp, as
  # $OUT/hc/test_build/tests/<name>.
  hc_host_test)
    if [[ ! ${2:-} =~ ^tests/[a-z_0-9]+$ || $# -gt 2 ]]; then
      echo "build.sh: hc_host_test wants one word behind it, tests/<name>: it compiles" \
        "hash_commitment/tests/<name>.cpp; a name has small letters, digits and _" >&2
      exit 2
    fi
    # Where there is no <name>.cpp: the host part of <name>.cu, which some tests have, as
    # <name>_host.
    from=hash_commitment/$2.cpp to=hc/test_build/$2
    if [ ! -f "$from" ]; then from=hash_commitment/$2.cu to=hc/test_build/$2_host; fi
    programs=$to
    mkdir -p "$OUT/hc/test_build/tests"
    g++ -O2 -fno-ipa-modref -std=c++17 -DDIGEST_WORDS=$dw -Wall -Wextra -pthread -msha -msse4.1 -mssse3 -I. "${field_flags[@]}" -o "$work/${to##*/}" -x c++ "$from"
    ;;
  all)
    programs=
    echo "build.sh: there is no target \"all\". ./build.sh, with no word, builds the programs; ./build.sh tests builds cp_prove and cp_commit" \
      "with the test options, and make_model, for the tests." >&2
    exit 2
    ;;
  tests)
    programs= calls_itself=1 # each call below installs its own
    for t in cp_prove cp_commit make_model; do
      echo "building $t"
      ./build.sh --dw $dw --field $field "$t"
    done
    ;;
  *)
    # A script that sources this one may define more_targets <target> ..: it builds a target of its
    # own with the settings and functions above, and calls usage() for an argument that it does not
    # know. It is called as a plain command: in a condition, bash would go on after a command of
    # more_targets() that fails.
    if [ -z "$sourced" ] || ! declare -F more_targets >/dev/null; then usage; fi
    more_targets "$@"
    ;;
esac
# Prints which nvcc compiled, and its release, and that the compiler sample passed, if it ran. This
# comes after the case statement, so that an argument that a sourcing script does not know gets only
# the usage message.
if [ -n "$checked" ]; then # set where the compiler's version was looked at: for every target that nvcc compiles
  echo "Compiled by $(command -v nvcc), $(env -i "PATH=$PATH" "$(command -v nvcc)" --version 2>/dev/null | sed -n 's/^Cuda compilation tools, //p')"
fi
if [ -n "$sample_passed" ]; then echo "build.sh: the compiler passed the check by value on this machine's GPU ($sample_passed)"; fi
if [ -n "$no_rpath" ]; then
  echo "build.sh: warning: the programs get no rpath to the CUDA libraries: $no_rpath. They start only where the loader finds" \
    "these libraries through its own configuration; see prover/docs/build.md." >&2
fi
if [ -n "$rpath_note" ]; then echo "build.sh: note: $rpath_note" >&2; fi
if [[ -n $calls_itself && -z $no_rpath ]]; then echo "cp_prove and cp_commit carry a run path to $cuda_libraries: they load the CUDA libraries from there."; fi
# Checks on what was built. cp_commit's seed and the 32 random bytes that cp_prove takes for each
# batch must come from the C library's getrandom(): no program may define that name, and the
# programs in $draws must import it.
draws=
case "$1" in
  cp_prove) programs=hc/test_build/cp_prove draws=$programs ;;
  cp_commit) programs=hc/test_build/cp_commit draws=$programs ;;
  frame_as_the_verifier_reads) programs=hc/test_build/frame_as_the_verifier_reads ;;
  serving_cp_prove) programs=hc/cp_prove draws=$programs ;;
  serving_cp_commit) programs=hc/cp_commit draws=$programs ;;
  serving_*)
    echo "build.sh: what $1 built was not checked: name its file in build.sh's second case statement" >&2
    exit 1
    ;;
  # Not checked: make_model and the tests that are not named above. The targets of tests and
  # serving are checked in their own calls.
  *)
    # more_programs <target>, which goes with more_targets(): for a target of the sourcing script it
    # sets programs, draws and may_hold for the checks below, or exits with status 0. If the script
    # does not define it, what its target built is not checked.
    if [ -z "$sourced" ] || ! declare -F more_programs >/dev/null; then exit 0; fi
    more_programs "$1"
    ;;
esac
# The checks below pass text to grep through a pipe, not as a here-string. bash writes a long
# here-string into a temporary file; where it cannot, grep does not run and the status is 1, the
# same as for "not found", so a check would pass without having searched. No grep below has -q: it
# reads to the end of its input, so that printf is not ended by SIGPIPE, which pipefail would
# report. No target builds a *.so file, so the branch for *.so never runs.
for program in $programs; do
  why=
  if ! symbols=$(nm "$work/${program##*/}"); then
    why="nm cannot read the program"
  elif printf '%s\n' "$symbols" | grep ' [^U] getrandom$' >/dev/null; then
    why="the program defines getrandom() itself"
  elif [[ " $draws " == *" $program "* ]] && ! printf '%s\n' "$symbols" | grep ' U getrandom' >/dev/null; then
    why="the program does not import it"
  elif [[ $program == *.so ]] &&
    printf '%s\n' "$symbols" | grep -E ' U (getrandom|getentropy|arc4random[a-z_]*|s?rand|s?random|rand_r|[a-z]rand48|_ZNSt13random_device[A-Za-z0-9_]*)(@|$)' >/dev/null; then
    why="the opening library, which is handed all its keys as bytes, imports a generator of random bytes"
  fi
  if [ -n "$why" ]; then
    echo "build.sh: $OUT/$program was not installed: fresh bytes must come from the C library's getrandom(), but $why" >&2
    exit 1
  fi
  # No target of this script builds the malicious prover, whose code is not
  # part of this repository: no macro makes one from these sources. Such a program prints lines that
  # the first check below searches for, in case sources that hold its code are built here, or nvcc
  # adds code through NVCC_APPEND_FLAGS or NVCC_PREPEND_FLAGS (read outside the serving targets).
  # In may_hold a script that sources this one names what its target's program is meant to contain:
  # each of the three checks below is skipped if may_hold has the word that its condition names.
  if [[ " $may_hold " != *" door "* ]]; then
    found=2 # 0: the words are there; 1: they are not; anything else: nobody knows, which counts as there
    if lines=$(strings -a "$work/${program##*/}"); then
      found=0
      printf '%s\n' "$lines" | grep -F -e 'DOOR_CHEAT: kind' -e "DOOR_CHEAT: the keeper's unit" -e "LIE: passed the prover's own stop" >/dev/null || found=$?
    fi
    if [ $found != 1 ]; then
      echo "build.sh: $OUT/$program was not installed: it contains a string by which build.sh detects a malicious prover (a program for tests; its hooks: hiding/sumcheck_door.cuh and common/lie.cuh), or the tool strings cannot read it, and" \
        "no target of build.sh may build one: look for a switch in the environment's NVCC_APPEND_FLAGS and NVCC_PREPEND_FLAGS" >&2
      exit 1
    fi
  fi
  # No target of this script builds the malicious committer: the build that fills its test hook is
  # not part of this repository. Every line that the hook prints begins with the string searched for.
  if [[ " $may_hold " != *" commit "* ]]; then
    found=2 # as above
    if lines=$(strings -a "$work/${program##*/}"); then
      found=0
      printf '%s\n' "$lines" | grep -F -e 'COMMIT-LIE:' >/dev/null || found=$?
    fi
    if [ $found != 1 ]; then
      echo "build.sh: $OUT/$program was not installed: it contains the string by which build.sh detects a malicious committer (a program for tests; its hooks: hash_commitment/commit_lie.cuh), or the tool strings cannot read it, and" \
        "no target of build.sh may build one: look for a switch in the environment's NVCC_APPEND_FLAGS and NVCC_PREPEND_FLAGS" >&2
      exit 1
    fi
  fi
  # No target of this script sets a macro that blocks/fence.h lists, which the message below calls
  # "a switch of the blocks' tests". A program built with one contains the
  # string searched for. The environment could pass such a macro to a target too: a program with a
  # plain name and, say, leaves' salts of zeros would pass every verifier and hide nothing.
  if [[ " $may_hold " != *" fence "* ]]; then
    found=2 # as above
    if lines=$(strings -a "$work/${program##*/}"); then
      found=0
      printf '%s\n' "$lines" | grep -F -e 'BLK-NOT-FOR-SERVING' >/dev/null || found=$?
    fi
    if [ $found != 1 ]; then
      echo "build.sh: $OUT/$program was not installed: it is built with a switch of the blocks' tests (../blocks/fence.h), or the tool strings cannot read it," \
        "and no target of build.sh may set one: look for a switch in the environment's NVCC_APPEND_FLAGS and NVCC_PREPEND_FLAGS" >&2
      exit 1
    fi
  fi
done
# The rest is for a production program only. None of its printable strings may equal a line of
# prover/tests/serving_build_words.txt: names of test and tuning variables, messages printed only
# in a test's run, and NVTX's variable (see nvcc_base, above).
# Of the functions that read the environment, the production cp_prove may
# import any. The production cp_commit may import getenv() alone, not weakly, and only if SUDO_UID
# is one of its printable strings: the import is for one call, which reads that variable
# (take_who_typed_sudo()).
# A program does not show with which arguments it calls a function, so this check cannot tell
# whether it reads another variable too. Any other production program may import none.
if [[ $1 != serving_* ]]; then exit 0; fi
list=tests/serving_build_words.txt
why=
if [ ! -s $list ]; then
  why="$list, whose lines a serving program must not contain, is missing or empty"
elif ! lines=$(strings -a "$work/${program##*/}"); then
  why="strings cannot read the program"
elif found=$(printf '%s\n' "$lines" | grep -Fx -f $list); then
  why="it contains lines of $list: ${found//$'\n'/; }. Read a variable with TESTENV(), and exclude a sentence from a"
  why+=" serving build at compile time; NVTX_INJECTION64_PATH comes from NVIDIA's NVTX headers: keep -DCCCL_DISABLE_NVTX in nvcc_base"
elif [ $? -ne 1 ]; then
  why="grep cannot read $list"
elif [[ $1 != serving_cp_prove ]] && reads=$(printf '%s\n' "$symbols" | grep -E ' [Uwv] (__libc_|__)?(secure_)?getenv') &&
  { [[ $1 != serving_cp_commit ]] || printf '%s\n' "$reads" | grep -v -E ' U getenv(@|$)' >/dev/null ||
    [[ $'\n'$lines$'\n' != *$'\nSUDO_UID\n'* ]]; }; then
  why="it imports getenv() or a function like it, beyond what it may: the serving prover may import these; the serving"
  why+=" committer getenv() alone, and only if the name SUDO_UID is in it. Read a test or tuning variable with TESTENV()"
  why+=" (common/setting.h), and keep -cudart shared in the compile command, since the static CUDA runtime imports such"
  why+=" functions too"
elif ! flags=$(in_work cat "$commands") || [ -z "$flags" ]; then
  why="the list of the commands that made it cannot be read or is empty"
elif [ -z "$dry" ]; then
  why="the compile command of the program's own unit, asked with --dryrun what it would start, failed or printed nothing"
elif [ -z "$with" ]; then
  why="env or one of the seven other programs that a record names (cicc, ptxas, ld and objcopy among them)"
  why+=" leads to no file that can be read, or says nothing to --version; or the dry run does not"
  why+=" say where the compile looks"
elif ! after=$(digest); then
  why="the digest of the three folders cannot be computed"
elif [ "$after" != "$before" ]; then
  why="a file of build.sh's folder, of ../blocks or of ../verifier/src changed, came or went while the program was compiled: build again, with nothing else"
  why+=" writing there"
elif ! sum=$(sha256sum <"$work/${program##*/}"); then
  why="sha256sum cannot read the program"
elif [ -z "$checked" ]; then
  why="the compiler version check did not run for $1"
elif [ -z "$sample_word" ]; then
  why="the compiler sample was not tried for $1"
fi
# The build record is written last, and finish() installs it after its program. A program that failed a check is not installed, and a
# program without a record is not a serving build.
if [ -z "$why" ]; then
  record="record 8
program $program
sha256 ${sum%% *}
sources $after${verifier_sum:+
verifier-beside $verifier_sum}
$flags
$with
compiler-sample $sample_word${said:+
said-commit $said}"
  # The second pattern matches a path that %q wrote as $'...'. A failure of grep counts as a match.
  if printf '%s\n' "$record" | LC_ALL=C grep -e '[^ -~]' -e "^[^ ]* [$]'" >/dev/null || [ $? -ne 1 ]; then
    why="its record would contain a character outside printable ASCII: check what the compilers print about themselves,"
    why+=" and the folders in which they and env are"
  elif ! printf '%s\n' "$record" | in_work write_new "${program##*/}.build"; then
    why="its record cannot be written"
  fi
  more_files+=" $program.build"
fi
if [ -n "$why" ]; then
  echo "build.sh: $OUT/$program was not installed: $why" >&2
  exit 1
fi
