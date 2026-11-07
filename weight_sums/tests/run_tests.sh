#!/bin/bash
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

# Terms: docs/glossary.md.

# Builds bin/weight_sums and runs every test of weight_sums/. The two programs under test,
# bin/weight_sums (built from weight_sums/weight_sums.cpp) and weight_sums/weight_sums.py, each
# write a model's sums file: one line, with a SHA-256, for each committed tensor (a weight tensor
# that the committer program cp_commit commits to). The tests are described in
# weight_sums/README.md, "Tests".
# usage: weight_sums/tests/run_tests.sh [--keep] [--without-unit-test]
#   --keep               keeps the temporary folder
#   --without-unit-test  leaves out the step "unit" and counts it as skipped
# It can be called from any folder.
#
# It needs g++ 13 or later (tested with g++ 13.3.0 only), GNU coreutils, python3 3.9 or later on
# Linux 5.3 or later (weight_sums/tests/refusals.py calls os.pidfd_open) and about 0.6 GB of free
# space for its synthetic files. It takes a few minutes.
# Every input is a synthetic file that a test writes into a temporary folder, made where the
# environment variable TMPDIR says (by default in /tmp). The script refuses to run if another
# account may be able to replace what is made there, or to change a directory or symbolic link on
# the way: the comment at open_above() states the rule. The tests' temporary files go into the same
# folder, and so do the compiler's, except in weight_sums/build.sh, which keeps them in its own work
# folder. Without --keep the temporary folder is removed at the end, also after a failure, SIGINT or
# SIGTERM, but not after SIGKILL. Outside that folder the script leaves only bin/weight_sums, which
# replaces an earlier one, and a work folder bin/weight_sums.building.XXXXXX if SIGKILL ends
# weight_sums/build.sh or g++ leaves a temporary file there.
#
# The steps "model.cpp" and "model.py" need make_model, a test program of the prover
# (prover/tests/weights_blocks/make_model.cpp). If the environment variable MAKE_MODEL names a built
# make_model, that file is used. Otherwise it is built in the temporary folder. It runs on the CPU,
# which must have the SHA instructions of x86-64: on another CPU both steps fail.
#
# The steps that run weight_sums/weight_sums.py need numpy and gguf-py 0.19.0 (the llama.cpp
# project's Python package `gguf`), the only version that weight_sums.py accepts without
# --any-gguf-version. Where python3 does not have them, these steps are skipped. The other steps
# need no Python package.
#
# Each step prints one line: "passed: <step>", "failed: <step>" or "skipped: <step>: <reason>". The
# output of a failed step is printed above its line. A failed step does not end the run. The last
# line counts the failed steps, if any, and the skipped ones.
status_failed=1 # the exit status if a step failed, or this script's folders cannot be entered, or the temporary folder cannot be made
status_usage=2  # the exit status of a wrong command line, and of a TMPDIR that the script refuses
status_skipped=77 # the exit status if no step failed and at least one step was skipped
# The exit status is 0 if every step ran and passed.
set -uo pipefail
unset CDPATH
export LC_ALL=C
ulimit -c 0 # a program that a signal ends leaves no core file
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

keep=no unit_test=yes
for argument in "$@"; do
  case $argument in
    --keep) keep=yes ;;
    --without-unit-test) unit_test=no ;;
    *)
      echo "usage: run_tests.sh [--keep] [--without-unit-test]" >&2
      exit $status_usage
      ;;
  esac
done
tests=$(cd "$(dirname "$0")" && pwd) || exit $status_failed
top=$(cd "$tests/../.." && pwd) || exit $status_failed
# The tests build programs in the temporary folder and run them, and the script removes the folder
# by its path at the end. So it uses the path that open_above() examined, without links.
tmp=${TMPDIR:-/tmp}
[[ $tmp == /* ]] || tmp=$PWD/$tmp
if open_above "$tmp/"; then
  case $why in
    'it does not exist' | 'it is not a directory')
      echo "run_tests: $place: $why. Set TMPDIR to the path of an existing directory; nothing was" \
        "run" >&2
      ;;
    'it could not be examined')
      echo "run_tests: $place could not be examined, so no temporary folder is made there. Set" \
        "TMPDIR to a directory that only you or root can change, below such directories; nothing" \
        "was run" >&2
      ;;
    *)
      printf -v tmp %q "$tmp"
      echo "run_tests: the temporary folder that this script makes in $tmp would lie below" \
        "$place, which another account may be able to change ($why): that account could put" \
        "another directory in the temporary folder's place while the tests run, and choose what" \
        "they run as you and what the script removes at the end. Set TMPDIR to a directory that" \
        "only you or root can change, below such directories (such as one in your home directory," \
        "or a /tmp with the sticky bit, which keeps others from replacing what you make in" \
        "it)$cure; nothing was run" >&2
      ;;
  esac
  exit $status_usage
fi
export TMPDIR=$real
work=$(mktemp -d) || exit $status_failed
export TMPDIR=$work # for the compiler's temporary files and the tests' own
if [ $keep = yes ]; then
  echo "run_tests: the temporary folder is $work"
else
  trap 'rm -rf "$work"' EXIT
fi

failed=0 skipped=0

# step <name> <command ...>: runs the command. The step passes if the command ends with status 0.
step() {
  local name=$1
  shift
  if "$@" >"$work/step_output.txt" 2>&1 </dev/null; then
    echo "passed: $name"
  else
    failed=$((failed + 1))
    cat "$work/step_output.txt"
    echo "failed: $name"
  fi
}

# skip <name> <reason>
skip() {
  skipped=$((skipped + 1))
  echo "skipped: $1: $2"
}

# python_step <name> <command ...>: a step that runs weight_sums.py and so needs its two packages.
# The check runs in weight_sums.py's folder, where a run of that script looks for modules first:
# python3 -c looks first in the current folder, which would otherwise be the caller's.
if (cd -- "$top/weight_sums" && python3 -B -c 'import sys, importlib.metadata as m
import numpy, gguf
sys.exit(0 if m.version("gguf") == "0.19.0" else 1)') 2>/dev/null; then
  python_step() { step "$@"; }
else
  python_step() { skip "$1" "python3 does not have numpy and gguf-py 0.19.0"; }
fi

# The two programs under test.
cpp() { "$top/bin/weight_sums" --threads 2 "$@"; }

# py() runs weight_sums.py on the files of oracle.py and make_model. It stops a run that has not
# ended after this many seconds, and the run's step fails. timeout (GNU coreutils) sends SIGABRT to
# weight_sums.py's main process, and under PYTHONFAULTHANDLER=1 Python then prints the stack of each
# of that process's threads (Python's documentation, module faulthandler). The worker processes end
# when the main process does. With --foreground, timeout stays in run_tests.sh's process group, so
# that a signal to the group still reaches it and weight_sums.py. Without timeout on PATH there is
# no limit.
py_limit=240
if type -P timeout >/dev/null; then
  limited() { timeout --foreground --signal=ABRT --kill-after=10 $py_limit "$@"; }
else
  limited() { "$@"; }
fi
py() {
  local status=0
  PYTHONFAULTHANDLER=1 limited python3 -B "$top/weight_sums/weight_sums.py" --workers 2 "$@" || status=$?
  if [ $status -eq 124 ] || [ $status -eq 137 ]; then # the statuses of timeout after SIGABRT and after SIGKILL
    echo "run_tests: weight_sums.py had not ended after $py_limit s and was stopped. Its threads' stacks are above."
  fi
  return $status
}

# same_sums <cpp or py> <expected sums file> <the further arguments of the command, without --out>:
# runs the program and compares the sums file that it writes with the expected one, byte for byte.
same_sums() {
  local program=$1 expected=$2 status=0
  shift 2
  { "$program" --out "$work/sums.txt" "$@" && cmp "$expected" "$work/sums.txt"; } || status=$?
  rm -f "$work/sums.txt"
  return $status
}

# unit: tests four parts of weight_sums/weight_sums.cpp: SHA-256, the widening of binary16 to
# binary32, the rounding to an integer, and which error decides the exit status when two tensors
# have one. The compiler options are those of weight_sums/build.sh, without its warnings.
unit() {
  g++ -std=c++17 -O2 -fno-ipa-modref -ffp-contract=off -pthread -o "$work/unit_test" "$tests/unit_test.cpp" &&
    "$work/unit_test"
}

# oracle <cpp or py> <seed> <e_res> [<further arguments of the command>]: oracle.py writes two
# shards (GGUF files of one model) and, by exact rational arithmetic, the sums file that they must
# give at that e_res, the embedding's exponent (its values are scaled by 2^e_res). The program must
# write the same file. The files of a seed and an e_res are written once, by the first step that
# needs them.
oracle() {
  local program=$1 folder=$work/oracle_$2_$3
  [ -e "$folder" ] || python3 -B "$tests/oracle.py" "$2" "$3" "$folder" || return
  shift 3
  same_sums "$program" "$folder/expected.txt" "$@" \
    "$folder/oracle-00001-of-00002.gguf" "$folder/oracle-00002-of-00002.gguf"
}

# model <cpp or py>: make_model writes a synthetic model in two shards of 0.53 GB together (4
# layers, 16 experts, 38 committed tensors) and its sums file, which it computes with a decoder of
# the prover (prover/weights_blocks/fold_rule.h) at e_res 11. The program must write the same file.
# The arguments of make_model are those of prover/examples/target_small_file_f2/model.txt.
model() {
  local folder=$work/model make_model=${MAKE_MODEL:-}
  if [ ! -e "$folder" ]; then
    if [ -z "$make_model" ]; then
      # build.sh makes this new folder itself. It refuses an output folder that holds a directory
      # which its group or others can write in, and under umask 002 the earlier steps have left
      # such directories in $work.
      make_model=$work/built/make_model
      (cd "$top/prover" && OUT=$work/built ./build.sh make_model) || return
    fi
    "$make_model" "$folder" 4 40 8 2048 1024 16 2048 30 || return
  fi
  same_sums "$1" "$folder/wsums.txt" "$folder/made-00001-of-00002.gguf" "$folder/made-00002-of-00002.gguf"
}

step build "$top/weight_sums/build.sh"
if [ $unit_test = yes ]; then
  step unit unit
else
  skip unit "--without-unit-test was given"
fi

# The two gain-folding constants are a function of the width and head dimension (fold_constants);
# this pins their values at Maverick's and TinyLlama's shapes. It needs numpy and gguf.
python_step fold-constants python3 -B "$tests/fold_constants_test.py"

# Over four seeds in a row, oracle.py gives each tensor of its list `COMMITTED` each of the four
# storage types (F32, Q4_K, Q5_K, Q6_K) once. A command without --eres uses its default, 11.
# weight_sums.py's --decode-twice, which saves memory, must not change its output.
for seed in 1 2 3 4; do
  step oracle.$seed.cpp oracle cpp $seed 11
  python_step oracle.$seed.py oracle py $seed 11
done
# Over three seeds in a row from 100 on, it gives each tensor each of the types Q8_0, F16 and BF16 once.
for seed in 100 101 102; do
  step oracle.$seed.cpp oracle cpp $seed 11
  python_step oracle.$seed.py oracle py $seed 11
done
python_step oracle.2.py.decode-twice oracle py 2 11 --decode-twice
step oracle.1.eres10.cpp oracle cpp 1 10 --eres 10
python_step oracle.1.eres10.py oracle py 1 10 --eres 10

# edge.py has matrices whose largest absolute value is where their exponent changes, or next to it.
# refusals.py has inputs and command lines that must be rejected, and similar ones that must pass.
for script in edge refusals; do
  step $script.cpp python3 -B "$tests/$script.py" "$work/${script}_cpp" --kind cpp -- "$top/bin/weight_sums"
  python_step $script.py python3 -B "$tests/$script.py" "$work/${script}_py" --kind py -- \
    python3 -B "$top/weight_sums/weight_sums.py"
done

# fork_failure.py makes the start of weight_sums.py's second worker fail. weight_sums.py must then
# exit with status 26 within 60 seconds and leave no file and no process.
python_step fork-failure.py python3 -B "$tests/fork_failure.py" "$work/fork_failure" "$top/weight_sums/weight_sums.py"

step model.cpp model cpp
python_step model.py model py

if [ $failed -gt 0 ]; then
  echo "run_tests: $failed failed; $skipped skipped"
  exit $status_failed
fi
echo "run_tests: every step that ran passed; $skipped skipped"
[ $skipped -eq 0 ] || exit $status_skipped
