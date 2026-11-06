// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0

// Terms: docs/glossary.md.

// The strict reader of the parameter file (`policy.txt`) for the committer and prover programs,
// cp_commit and cp_prove. load() reads the file once; parse() checks the form and order of its
// lines; limit_missed() is meant to bound every number that sizes a buffer, counts a loop or serves
// as a shift count, a divisor or an index (its comment lists the exceptions).
//
// The file has two parts. The frame's lines, from version to table, set the proof system's
// parameters, such as the target in bits (`sec`), proof of work, the codes and the weight table's
// shape. blk::read_parameters() (blocks/frame/parameters.h) reads them, as in the verifier program
// cp_verify. parse() reads the other lines, from prof on: the model's shape, the acceptance policy
// (the rule that the checked tokens must meet), and the sizes and lookup calls of the sub-proofs
// (each the argument for one part of the forward pass). Each line's meaning:
// verifier/docs/parameter-file.md, "All lines in their order".
// Host code only: a CPU test can include it.

#pragma once
#include "common/setting.h"
#include "frame/parameters.h" // blocks/: the frame's lines
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <sys/stat.h>
#include <vector>
namespace parameter_file {
// A parameter file after parse(). The value lines ("name value") follow the frame's lines, so
// lines[i] is line frame_lines + i + 1 of the file.
struct Parameters {
  blk::Parameters frame;  // the frame's lines
  size_t frame_lines = 0; // how many they are
  struct Line {
    std::string name;
    uint32_t value;
  };
  typedef std::vector<std::string> Call; // a call line after its keyword: name, kind, segments
  std::string bytes;                     // the file as read; take its digests from these bytes
  std::vector<Line> lines;               // the value lines, in the file's order
  std::map<std::string, uint32_t> m;     // the same by name; an omitted optional line is absent
  std::vector<Call> call, callA, callH;
  std::string calls, callsA, hcalls; // the call, callA and callH lines' bytes, with line feeds
  bool has(const std::string &name) const { return m.count(name) != 0; }
  // The line's value or, if the file omits it, its default. No file may write the default of an
  // optional line (lidb to xl): limit_missed() rejects it.
  uint32_t value(const std::string &name) const {
    const auto it = m.find(name);
    if (it != m.end()) return it->second;
    return name == "lidb"    ? 22
           : name == "nminH" ? m.at("nmin")
           : name == "eg"    ? 8
           : name == "smax"  ? 4095
           : name == "xl"    ? 4
                             : 0;
  }
  // The number in the file, from 1, of `name`'s value line, or 0.
  size_t line_of(const std::string &name) const {
    for (size_t i = 0; i < lines.size(); i++)
      if (lines[i].name == name) return frame_lines + i + 1;
    return 0;
  }
};
// The names of the fixed value lines, those that a file cannot omit, in their order. NE: the number
// of routed experts in a layer with experts, or 0 in a model without experts. MF, KF: the final
// norm, before the output head, multiplies by a reciprocal scaled by 2^MF and then shifts right by
// KF bits.
// dh: the width of one attention head; there are H heads for queries and G for keys and values.
inline std::vector<std::string> fixed_names() {
  std::vector<std::string> names = {
      "prof", "lt",      "L",       "ll",      "d",       "H",   "G",         "dh",        "f",
      "fe",    "NE",      "V",    "eps1_lo", "eps1_hi", "eps2_lo", "eps2_hi", "cs",  "theta0_lo", "theta0_hi", "hq",
      "hv",    "ho",      "hgD",  "huD",     "hdD",     "hr",      "ut",      "la",  "nmin",      "D",         "tau",
      "lmaf",  "NAf",     "lmad", "NAd",     "lmar",    "lmb",     "lmq",     "lmx"};
  for (const char *kind : {"qk", "v", "o", "g", "u", "gs", "us", "gr", "ur", "dn", "dnr", "r"})
    names.push_back(std::string("wb_") + kind);
  for (int c = 0; c < 8; c++) names.push_back("iota_" + std::to_string(c));
  for (const char *n : {"seed_lo", "seed_hi"}) names.push_back(n);
  for (const char *n : {"lB", "S", "rx", "kFM", "kFD", "lmS", "NAs", "lmh", "lmo", "lmbA", "iotaA_0"})
    names.push_back(n);
  for (const char *n :
       {"MF",      "KF",      "epsF_lo", "epsF_hi", "toksrc",  "lmhf",    "NHf",     "lmhd", "lmhr",  "lmbH", "wb_emb",
        "wb_head", "iotaH_0", "iotaH_1", "iotaH_2", "iotaH_3", "iotaH_4", "iotaH_5", "C",    "theta", "cap",  "s",
        "sh",      "sT",      "F"})
    names.push_back(n);
  return names;
}
// Builds a rejection. Its rejection name is one that verifier/src/vf_rejection_names.txt lists (some by
// prefix), w-policy-Nmax, or one that begins with "prover-only-". For the same file cp_verify often
// prints another name: it checks in another order and has several names for one line's faults.
// Line 0: the fault is in a number of the frame's lines that blk::read_parameters() accepted.
inline std::string refusal(const std::string &name, size_t line, const std::string &what) {
  return name + ": line " + std::to_string(line) + ": " + what;
}
// Parses the bytes of a parameter file into p. Returns "" or the rejection, which refusal() builds:
// "<name>: line <number>: <what is wrong>". A rejection never quotes the file, so it carries no
// byte of the file to a terminal or a log.
// After the frame's lines a file holds its value lines, in one fixed order, then its call, callA
// and callH lines, and nothing else. Each of these lines has at most 1,023 bytes, the line feed
// that ends it included. A value line is a name, one space and a decimal number below 2^32 with no
// sign and no leading zero. A call line describes one lookup call (a "call" below). Its fields are
// the keyword call (layer sub-proofs), callA (attention) or callH (output head), the call's name,
// the kind of its public table, and one or more segments, the named lists of keys that it looks
// up. Fields consist of letters, digits, ".", "_" and "-"; single spaces separate them.
inline std::string parse(const std::string &bytes, Parameters &p) {
  p = Parameters();
  p.bytes = bytes;
  {
    // blk::read_parameters() reports a fault as "<name>[: <which line>] (line <number>)".
    std::string fault;
    if (!blk::read_parameters(bytes, p.frame, fault)) {
      const size_t at = fault.rfind(" (line "), colon = fault.find(": ");
      const std::string name = fault.substr(0, std::min(at, colon)), which = colon < at ? fault.substr(colon + 2, at - colon - 2) : "";
      return refusal(name, at == std::string::npos ? 0 : std::stoul(fault.substr(at + 7)),
                     "the frame's lines have one order and one spelling (blocks/frame/parameters.h)" + (which.empty() ? "" : ": " + which));
    }
    for (uint64_t wide : {p.frame.table_data, p.frame.table_random})
      if (wide >> 32) return refusal("parameter-table-shape", 0, "these programs hold the committed table's numbers in 32 bits");
    // 32 is cp_verify's cap on the bits of proof of work before a verifier's message, of every kind
    // (WORK_BITS_MOST in verifier/src/vf_common.h); blk::read_parameters() accepts up to 58. Each
    // further bit doubles the time of the GPU's search for a nonce, and above 32 bits that time
    // would be spent on a proof that cp_verify rejects.
    for (const auto &kind : p.frame.work)
      for (const auto bits : kind)
        if (bits > 32) return refusal("policy-work-most", 0, "no work line may ask for more than 32 bits: cp_verify accepts no more, and the search for a nonce takes 2^bits tries on average");
  }
  const size_t frame_bytes = bytes.size() - p.frame.rest.size();
  for (size_t i = 0; i < frame_bytes; i++) p.frame_lines += bytes[i] == '\n';
  std::vector<std::string> text(p.frame_lines); // lines without line feeds; the frame's stay empty
  for (size_t at = frame_bytes; at < bytes.size();) {
    const size_t end = bytes.find('\n', at), line = text.size() + 1;
    if (end == std::string::npos) return refusal("policy-line-form", line, "the file's last line has no line feed");
    if (end + 1 - at > 1023) return refusal("policy-line-long", line, "the line is longer than 1,023 bytes");
    text.push_back(bytes.substr(at, end - at));
    if (text.back().find('\0') != std::string::npos)
      return refusal("policy-zero-byte", line, "the line holds a zero byte");
    at = end + 1;
  }
  size_t next = p.frame_lines; // the index in text of the line that comes next
  std::string skipped; // the optional lines that could also come here, for a rejection's text
  std::string why;     // the rejection, once there is one
  // Consumes the next line and returns true if it is `name`'s value line. If not, rejects the file
  // under `missing`, unless that is empty (optional line). Always rejects a malformed number.
  auto value_line = [&](const std::string &name, const std::string &missing) {
    if (next >= text.size() || text[next].compare(0, name.size() + 1, name + " ")) {
      if (missing.empty())
        skipped += name + ", ";
      else
        why = refusal(missing, next + 1,
                      (next < text.size() ? "this must be the line " : "the file ends where there must be the line ") +
                          (skipped.empty() ? name : skipped + "or " + name));
      return false;
    }
    const std::string digits = text[next].substr(name.size() + 1);
    if (digits.empty() || digits.size() > 10 || digits.find_first_not_of("0123456789") != std::string::npos ||
        (digits[0] == '0' && digits.size() > 1) || std::stoull(digits) >> 32) {
      why = refusal("policy-line-form", next + 1,
                    "after the name " + name +
                        " and one space there must be a decimal number below 2^32, with no sign and no "
                        "leading zero, and nothing more");
      return false;
    }
    p.lines.push_back({name, (uint32_t)std::stoull(digits)});
    p.m[name] = p.lines.back().value;
    next++;
    skipped.clear();
    return true;
  };
  auto optional_line = [&](const std::string &name) { // false: the file is rejected
    return value_line(name, ""), why.empty();
  };
  if (!value_line("prof", "policy-name-prof")) return why;
  if (p.m["prof"] != 20) return refusal("policy-profile", next, "prof must be 20");
  const std::vector<std::string> fixed = fixed_names();
  for (size_t i = 1; i < fixed.size(); i++)
    if (!value_line(fixed[i], "policy-name-" + fixed[i])) return why;
  if (!optional_line("lidb") || (p.has("lidb") && !optional_line("nminH"))) return why;
  for (const char *name : {"eg", "smax", "rcl", "gcl", "xl"})
    if (!optional_line(name)) return why;
  // Each of the four public tables' digests takes DW lines: TBLW = 4 DW (prover/common/setting.h).
  const bool tables = value_line("tsilu0", ""); // public tables' digests: all TBLW lines or none
  if (!why.empty()) return why;
  for (int i = 1; tables && i < TBLW; i++)
    if (!value_line(std::string(i < DW       ? "tsilu"
                                : i < 2 * DW ? "tsig"
                                : i < 3 * DW ? "trope"
                                             : "texp") +
                        std::to_string(i % DW),
                    "policy-tables"))
      return why;
  int kinds_begun = 0; // 1: a call line has been read; 2: a callA line; 3: a callH line
  for (; next < text.size(); next++) {
    Parameters::Call words(1);
    bool good = true;
    for (const char c : text[next])
      if (c == ' ')
        words.emplace_back();
      else
        words.back().push_back(c), good = good && ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                                                   (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-');
    for (const std::string &word : words) good = good && !word.empty();
    const int kind = words[0] == "call"                         ? 1
                     : words[0] == "callA"                      ? 2
                     : words[0] == "callH"                      ? 3
                                                                : 0;
    if (!good || words.size() < 4 || !kind || kind < kinds_begun)
      return refusal(
          "policy-call-line", next + 1,
          "from here on a file holds only its call lines, then its callA lines, then its callH lines; each has at "
          "least four words of letters, digits, \".\", \"_\" and \"-\", with single spaces between them");
    kinds_begun = kind;
    (kind == 1 ? p.calls : kind == 2 ? p.callsA : p.hcalls) += text[next] + "\n";
    (kind == 1 ? p.call : kind == 2 ? p.callA : p.callH).emplace_back(words.begin() + 1, words.end());
  }
  return "";
}
constexpr int LEAFD_LIMIT = 8; // the largest leafd: 2^8 is PLACES_MOST in prover/prover/slot.h
// The table of limits, for a file that parse() has accepted. Returns "" or the rejection, in
// parse()'s form. First come checks that depend only on the parameter file and follow cp_verify's,
// under its names, except w-policy-Nmax; some, such as tau (only ETAU, a build's constant in
// prover/common/setting.h) and the last number of the line `table`, are narrower here. Then come
// further rules in which cp_commit and cp_prove are narrower than cp_verify, under names that begin
// with "prover-only-". These are not all: a program's set-up may still reject a file that passes
// here.
// Not checked: whatever needs the placement of a call's segments, the plan of a sub-proof's lookups
// (forest_plan() in prover/hiding/forest_plan.h), a computed table, another file or the weight
// matrix's number of rows.
// No upper bound (parse()'s is 2^32) is set on:
//   - theta0, seed, C, theta (at most C) and the public tables' digests, none of which sizes a
//     buffer, counts a loop or serves as a shift count, a divisor or an index;
//   - the numbers of the frame's lines, other than the last of `table` and `slots`: this table adds
//     no bound to those of blk::read_parameters() and parse() (`work`: 32), and HfParams::derive()
//     checks the weight table's before anything is sized by them;
//   - iotaA_0 (32 + log2 of the keys in one product tree of attention's lookups) and the iota line
//     of a lookup call that the file has (log2 of the number of that call's product trees): a
//     sub-proof's set-up derives the number itself, compares the line with it and uses only the
//     derived number.
inline std::string limit_missed(const Parameters &p) {
  typedef uint64_t u64;
  auto v = [&](const std::string &name) {
    return (u64)p.value(name);
  };
  // A 64-bit value takes two lines, <name>_lo and <name>_hi.
  auto wide = [&](const std::string &name) {
    return v(name + "_lo") | v(name + "_hi") << 32;
  };
  auto power_of_two = [](u64 x) {
    return x && !(x & (x - 1));
  };
  auto log2_up = [](u64 x) {
    int l = 0;
    while (((u64)1 << l) < x) l++;
    return (u64)l;
  };
  auto one_bits = [](u64 x) {
    int n = 0;
    for (; x; x &= x - 1) n++;
    return n;
  };
  // The index of the first call that repeats an earlier call's name, or the number of calls.
  auto twice = [](const std::vector<Parameters::Call> &calls) {
    std::set<std::string> names;
    for (size_t i = 0; i < calls.size(); i++)
      if (!names.insert(calls[i][0]).second) return i;
    return calls.size();
  };
  const bool experts = v("NE") > 0;
  const u64 most = 0xFFFFFFFFull, lt = v("lt"), la = v("la");
  // log2 of the columns of W, the matrix of the committed weights; log2 of the most requests in a
  // batch.
  const u64 block_log = p.frame.block_log, slots_log = blk::ceil_log2(p.frame.slots);
  // A line's inclusive range, and the rejection name for a value outside it. `frame`: for a number
  // of the frame's lines, the number; `line` then only says what it is.
  struct Range {
    std::string line;
    u64 lowest, highest;
    const char *name;
    const u64 *frame = nullptr;
  };
  // A range applies only if the file has its line: an optional line may be absent.
  // A row with `frame` always applies.
  std::vector<Range> ranges = {
      {"toksrc", 1, 1, "statement-required"},
      {"lt", 4, 16, "shape-words"},
      {"ll", 1, 6, "shape-words"},
      {"d", 1, 1 << 20, "shape-words"},
      {"f", 1, 1 << 20, "shape-words"},
      {"la", lt + 1, 16, "shape-la"},
      {"cs", 12, 12, "policy-cs"},
      {"hq", 0, 2, "policy-forms"},
      {"hv", 0, 2, "policy-forms"},
      {"ho", 0, 2, "policy-forms"},
      {"hgD", 0, 2, "policy-forms"},
      {"huD", 0, 2, "policy-forms"},
      {"hdD", 0, 2, "policy-forms"},
      {"hr", 0, experts ? 2u : 0u, experts ? "policy-forms" : "header-absent-class"},
      {"ut", 1, 1, "policy-ut"},
      {"lmq", la, 24, "shape-la"},
      {"lmaf", la, 24, "shape-request-matrix"},
      {"lmad", lt, 24, "shape-request-matrix"},
      {"lmar", lt, 24, "shape-request-matrix"},
      {"NAf", 1, 64, "shape-request-matrix"},
      {"NAd", 1, 64, "shape-request-matrix"},
      {"lmx", la, 24, "shape-request-matrix"},
      {"lmb", 5, 21, "forest-planes"}, // set-up: lmb, lmbH, lmbA <= 5 + log2 of the largest table
      {"tau", ETAU, ETAU, "hid-tau"},
      {"V", 2, most, "shape-V"},
      {"wb_emb", 21, 21, "policy-weight-width-emb"},
      {"wb_head", 13, 13, "policy-weight-kind"},
      {"the last number of the line table", ELKW_LEAST, ELKW_MOST, "w-shape-lk", &block_log},
      {"log2 of the line slots", 0, 16, "w-policy-Nmax", &slots_log},
      {"log2 of the line slots", 0, LEAFD_LIMIT, "policy-leafd", &slots_log},
      {"lidb", 23, 30, "policy-lidb"},
      {"nminH", 2, 28, "policy-nminH"},
      {"eg", 4, 7, "policy-gate-table"},
      {"rcl", 1, 1, "policy-rcl"},
      {"gcl", 1, 1, "policy-gcl"},
      {"xl", 5, 5, "policy-xl"},
      {"MF", 24, 47, "policy-norm-F"},
      {"KF", 17, 32, "policy-norm-F"},
      {"s", 0, 63, "policy-gap-words"},
      {"sh", 7, 8, "policy-gap-words"},
      {"F", 24, 24, "policy-gap-words"},
      {"lmhf", la, 24, "shape-request-matrix-H"},
      {"NHf", 1, 64, "shape-request-matrix-H"},
      {"lmhd", lt, 24, "shape-request-matrix-H"},
      {"lmhr", lt, 24, "shape-request-matrix-H"},
      {"lmbH", 5, 21, "forest-planes"},
      {"lmh", lt, 24, "shape-request-matrix-A"},
      {"lmbA", 5, 21, "forest-planes"},
      {"NAs", 1, 64, "shape-request-matrix-A"},
      {"lB", 0, lt - 2, "af-shape-tiles"},
      {"S", 1, 14, "af-policy-softmax"},
      {"rx", 1, 12, "af-policy-softmax"},
      {"lmS", 1, 24, "af-shape-lmS"},
      {"kFD", 8, 16, "af-policy-kF"},
      {"nmin", 10, 28, "af-policy-nmin"},
      {"lmo", la, 24, "af-shape-la"},
  };
  for (const Range &r : ranges) {
    const u64 value = r.frame ? *r.frame : v(r.line);
    if ((r.frame || p.has(r.line)) && (value < r.lowest || value > r.highest))
      return refusal(r.name, p.line_of(r.line),
                     r.line + " must be " +
                         (r.lowest == r.highest ? std::to_string(r.lowest)
                          : r.highest == most
                              ? "at least " + std::to_string(r.lowest)
                              : "from " + std::to_string(r.lowest) + " to " + std::to_string(r.highest)));
  }
  // All rules are evaluated, then the first that does not hold is reported. From here on every
  // shift count is inside its range and no divisor is 0. The arithmetic is unsigned: where a
  // product or a difference can wrap, an earlier rule fails.
  struct Rule {
    bool holds;
    const char *name;       // the rejection's name
    std::string line, what; // the line to report, by name or by number; the rejection's text
  };
  const u64 L = v("L"), ll = v("ll"), d = v("d"), f = v("f"), fe = v("fe"), NE = v("NE"), H = v("H"), G = v("G"),
            dh = v("dh"), V = v("V"), strip = (u64)1 << (la - lt), sh = v("sh"), sT = v("sT");
  const u64 least_cost = sh < 63 ? (u64)1 << sh : ~(u64)0; // of a token
  const std::string last_value_line = std::to_string(p.frame_lines + p.lines.size()); // the call lines follow it
  std::vector<Rule> rules = {
      {!experts || (power_of_two(fe) && f == 2 * fe && !(ll == 1 && fe == 1)), "shape-words", "fe",
       "with experts fe must be a power of two and f must be 2 fe; fe 1 does not go with ll 1"},
      {L && L % ((u64)1 << ll) == 0, "shape-L", "L", "L must be a multiple of 2^ll, and not 0"},
      {L >> ll <= 28, "plan-groups-base", "L", "L / 2^ll, the number of layer groups, must be at most 28"},
      {d % strip == 0 && f % strip == 0 && (!experts || fe % strip == 0), "shape-la", "la",
       "2^(la - lt) must divide d and f, and fe with experts"},
      {wide("eps1") >= 1 && wide("eps1") >> 59 == 0, "policy-norm", "eps1_lo",
       "eps1 (eps1_lo + 2^32 eps1_hi) must be at least 1 and below 2^59"},
      {wide("eps2") >= 1 && wide("eps2") >> 59 == 0, "policy-norm", "eps2_lo",
       "eps2 (eps2_lo + 2^32 eps2_hi) must be at least 1 and below 2^59"},
      {H && G && dh >= 2 && power_of_two(dh) && H * dh == d && H + G <= 64, "shape-heads", "H",
       "H and G must be at least 1, dh a power of two and at least 2, H dh = d, and H + G at most 64"},
      {(G * dh) % strip == 0, "shape-la", "la", "2^(la - lt) must divide G dh"}, // and so (H + G) dh = d + G dh
      {v("D") == 2048 || v("D") == 32, "shape-D", "D", "D must be 2048 (or 32, for a toy model)"},
      {!experts || (NE >= 2 && NE <= 4096 && power_of_two(NE)), "shape-NE", "NE",
       "with experts NE must be a power of two from 2 to 4096"},
      {!p.has("gcl") || experts, "policy-gcl", "gcl", "the line gcl needs a model with experts"},
      {!p.has("rcl") || experts, "policy-rcl", "rcl", "the line rcl needs a model with experts"},
      {p.call.size() >= 1 && p.call.size() <= 8, "policy-call-list", last_value_line,
       "after this line a file must have from 1 to 8 call lines"},
      {twice(p.call) == p.call.size(), "policy-call-list", std::to_string(p.frame_lines + p.lines.size() + 1 + twice(p.call)),
       "this call line's name stands in an earlier call line"},
      {twice(p.callA) == p.callA.size(), "policy-call-list-A",
       std::to_string(p.frame_lines + p.lines.size() + 1 + p.call.size() + twice(p.callA)),
       "this callA line's name stands in an earlier callA line"},
      {twice(p.callH) == p.callH.size(), "policy-call-list-H",
       std::to_string(p.frame_lines + p.lines.size() + 1 + p.call.size() + p.callA.size() + twice(p.callH)),
       "this callH line's name stands in an earlier callH line"},
      {v("xl") != 5 || (wide("eps1") >> 48 == 0 && wide("eps2") >> 48 == 0 && wide("epsF") >> 48 == 0),
       "no-wrap-ledger-eps", "xl", "with xl 5, eps1, eps2 and epsF must be below 2^48"},
      {log2_up(V) <= block_log, "w-list-token-factor", "0", "2 to the last number of the line table must be at least V"},
      {!p.has("nminH") || v("nminH") > v("nmin"), "policy-nminH", "nminH",
       "nminH must be above nmin; for nmin itself, leave the line out"},
      {!p.has("smax") || v("smax") == 2047 || v("smax") == 1023 || v("smax") == 511 || v("smax") == 255,
       "policy-gate-table", "smax", "smax must be 2047, 1023, 511 or 255"},
      {!p.has("tsilu0") || experts || NE == 0, "policy-tables", "tsilu0",
       "the tables' digest lines need a model that uses the public tables (experts, or a dense model)"},
      // A dense model (NE 0) uses three tables, not the FS/SIG table, so it carries no tsig line.
      {!p.has("tsig0") || experts, "policy-tables", "tsig0",
       "a dense model (NE 0) does not use the FS/SIG table, so it has no tsig digest lines"},
      {wide("theta0") != 0, "shape-theta0", "theta0_lo", "theta0 (theta0_lo + 2^32 theta0_hi) must not be 0"},
      {v("KF") + 14 >= v("MF"), "policy-norm-F", "KF", "KF must be at least MF - 14"},
      {wide("epsF") >= 1 && wide("epsF") >> 59 == 0, "policy-norm-F", "epsF_lo",
       "epsF (epsF_lo + 2^32 epsF_hi) must be at least 1 and below 2^59"},
      {v("theta") <= v("C") && v("cap") <= (u64)1 << lt && v("cap") < 1 << 16, "policy-gap-words", "theta",
       "theta must be at most C, and cap at most 2^lt and below 2^16"},
      {sT + 4 == sh || (sT >= sh + 4 && sT <= sh + 10), "policy-gap-words", "sT",
       "sT must be sh - 4, or from sh + 4 to sh + 10"},
      {V > strip, "policy-gap-words", "V", "V must be above 2^(la - lt)"},
      {p.callH.size() <= 6, "policy-call-list-H", last_value_line, "a file must have at most 6 callH lines"},
      {(d << 24) + 1 + ((u64)1 << (sh + 15)) <= (u64)1 << 37 &&
           (d << 24) + 1 + ((u64)1 << 37) + ((u64)1 << 32) <= (((u64)1 << 16) + 1) << (sh + 15),
       "no-wrap-H-three-digits", "d", "d must be at most 8191 with sh 8, and at most 7936 with sh 7"},
      {!p.callA.empty(), "policy-call-list-A", last_value_line,
       "a file must have at least one callA line"},
      {power_of_two(G) && H % G == 0, "af-shape-heads", "G", "G must be a power of two that divides H"},
      {la - lt <= log2_up(dh), "af-shape-la", "la", "la - lt must be at most log2 dh"},
      {lt + v("S") + 1 <= 24, "af-policy-softmax", "S", "lt + S must be at most 23"},
      {v("lmS") <= log2_up(G) + lt + v("lB"), "af-shape-lmS", "lmS",
       "lmS must be at most log2 G + lt + lB"},
      {ll + log2_up(f) + lt >= 9 && (!experts || ll - 1 + log2_up(fe) + lt >= 9), "shape-product-layer", "lt",
       "ll + log2 f + lt must be at least 9, and with experts ll - 1 + log2 fe + lt too (logarithms rounded up)"},
      {log2_up(d) + lt >= 9, "shape-product-layer-H", "lt", "log2 d (rounded up) + lt must be at least 9"},
  };
  const char *const kinds[12] = {"qk", "v", "o", "g", "u", "gs", "us", "gr", "ur", "dn", "dnr", "r"};
  for (int k = 0; k < 12; k++) {
    const bool there = experts || k < 5 || k == 9;
    rules.push_back(
        {v(std::string("wb_") + kinds[k]) == (there ? 13u : 0u), there ? "policy-weight-kind" : "header-absent-kind",
         std::string("wb_") + kinds[k],
         std::string("wb_") + kinds[k] + (there ? " must be 13" : " must be 0 in a model without experts")});
  }
  // The iota line of a call that the file does not have must be 0 (of one that it has: see the
  // comment on limit_missed()). With no callH line cp_verify uses a built-in list of six calls.
  for (int head = 0; head < 2; head++) {
    const std::string iota = head ? "iotaH_" : "iota_";
    for (u64 q = head ? (p.callH.empty() ? 6 : p.callH.size()) : p.call.size(); q < (head ? 6u : 8u); q++)
      rules.push_back({v(iota + std::to_string(q)) == 0, head ? "header-absent-call-H" : "header-absent-call",
                       iota + std::to_string(q), iota + std::to_string(q) + " must be 0: the file has no such call"});
  }
  // The "prover-only-" rules.

  // The index of the first call whose kind is not in kinds_served or whose name does not begin with
  // its kind, or the number of calls.
  auto first_misnamed = [](const std::vector<Parameters::Call> &calls,
                           std::initializer_list<const char *> kinds_served) {
    size_t i = 0;
    for (; i < calls.size(); i++) {
      bool served = false;
      for (const char *kind : kinds_served) served = served || calls[i][1] == kind;
      if (!served || calls[i][0].compare(0, calls[i][1].size(), calls[i][1])) break;
    }
    return i;
  };
  const size_t misnamed = first_misnamed(p.call, {"R16", "R8", "FC", "FS", "FL"}),
               misnamed_H = first_misnamed(p.callH, {"R16", "R8", "FC", "FE"}), first_call = p.frame_lines + p.lines.size() + 1,
               first_callA = first_call + p.call.size(), first_callH = first_callA + p.callA.size();
  std::string attention_lists;
  for (const Parameters::Call &c : p.callA) {
    for (size_t i = 1; i < c.size(); i++) attention_lists += (i > 1 ? " " : "") + c[i];
    attention_lists += "|";
  }
  for (const Rule &r : std::vector<Rule>{
           {v("D") == 2048, "prover-only-D", "D", "these programs support only D 2048"},
           {d / (d & -d) <= 32, "prover-only-d", "d",
            "d divided by the largest power of two that divides it must be at most 32"},
           // The forward pass computes its matrix products from products of signed bytes, which
           // prover_gemm() (prover/prover/forward.cuh) adds into 32-bit words without a check. A product
           // is at most 2^14, so a word holds 131,071 of them. A word gets at most 2 d of them
           // (gemm3(), gemm3at(); prover/head/forward.cuh), 2 f (a dense layer's down projection)
           // or, with experts, 4 fe = 2 f (the shared expert's and one routed expert's). d = H dh is
           // at most 63 * 128 by the rules on H, G and dh. cp_verify has no rule on f of this kind:
           // it takes files under which these programs could not prove.
           {f < (1 << 16), "prover-only-f", "f",
            "f must be below 2^16: the forward pass adds 2 f products of signed bytes into one 32-bit sum"},
           // A lookup call's segment is cut into one piece for each one-bit of its length
           // (make_layer_layout() in prover/prover/layout.h, make_head_layout() in prover/head/forward.cuh),
           // and the kernels take at most eight pieces (Pos2 in prover/prover/forward.cuh). Only f
           // without experts and V, rounded up, can have more: by the other rules d, G dh and
           // (H + G) dh have at most six one-bits, and with experts NE and f have one. The head's
           // two parts of the rounded V have no more than it has. The other lengths are fixed
           // numbers below 256.
           {one_bits(f) <= 8, "prover-only-f", "f", "f must have at most eight one-bits in binary"},
           {one_bits((V + strip - 1) / strip * strip) <= 8, "prover-only-V", "V",
            "V, rounded up to a multiple of 2^(la - lt), must have at most eight one-bits in binary"},
           {fe <= 1 << 20, "prover-only-fe", "fe", "fe must be at most 2^20"},
           {!experts || fe >= 32, "prover-only-fe", "fe", "with experts fe must be at least 32"},
           {dh == 32 || dh == 64 || dh == 128, "prover-only-attention", "dh", "dh must be 32, 64 or 128"},
           {lt <= 14 && 2 * lt + log2_up(G) <= 32, "prover-only-attention", "lt",
            "lt must be at most 14, and 2 lt + log2 G at most 32"},
           {v("lB") >= 4, "prover-only-attention", "lB", "lB must be at least 4"},
           {ll <= 4, "prover-only-attention", "ll", "ll must be at most 4"},
           // A lookup call of a layer sub-proof has at least 2^nmin keys, 2^(ll + lt) for each of
           // its columns. With this rule it has the 2^(la - lt) columns that cp_verify requires
           // (shape-call-<name>; make_layer_layout() in prover/prover/layout.h checks the same). cp_verify
           // also accepts a lower nmin, from 10 on, if each call's segments make it long enough.
           {v("nmin") >= la + ll, "prover-only-nmin", "nmin", "nmin must be at least la + ll"},
           {H <= 16 * G, "prover-only-attention", "H", "H / G must be at most 16"},
           {!experts || log2_up(fe) + log2_up(NE) >= block_log, "prover-only-lkW", "0",
            "with experts the last number of the line table must be at most log2 fe + log2 NE"},
           {block_log <= log2_up(V) + 3, "prover-only-lkW", "0", "the last number of the line table must be at most log2 V (rounded up) + 3"},
           {v("kFM") <= 30, "prover-only-kFM", "kFM", "kFM must be at most 30"},
           {v("xl") != 5 || v("KF") > 23 || v("KF") + 13 >= v("MF"), "prover-only-norm-F", "KF",
            "with xl 5 and KF at most 23, KF must be at least MF - 13"},
           // cp_verify takes only the exact match and the clipped gap as acceptance policies, both
           // with sT = sh - 4. Under another sT it passes no proof ("NO: policy"), so whatever
           // cp_prove released under such a file would fail at every client, and the work of
           // every proof would be lost. Such a file is rejected here, when cp_commit or
           // cp_prove starts.
           {sT + 4 == sh, "prover-only-policy-kind", "sT",
            "sT must be sh - 4: cp_verify takes the exact match and the clipped gap alone, and both have sT = sh - 4"},
           // Three rules against a policy under which a result would mean nothing (policyEmpty() in
           // verifier/src/vf_common.h). cp_verify answers "NO: policy" to a caller that names such a
           // policy, and a client that names the file's policy gets "NO: policy".
           // Under a file that broke the first or the second rule, every response would get
           // that answer. Under one that broke the third, a request that covers from 1
           // to cap of its own tokens would get it: cp_prove would release that
           // response, and the client's check would fail
           // (docs/limits-all-rows.tsv, row R16.3). With these rules and theta at most C (above) no
           // request gets this answer. The exact match (C = theta = 2^sh, cap 0) passes all three.
           {v("C") >= least_cost, "prover-only-policy-says-nothing", "C", "C must be at least 2^sh, the least cost of a token: under a smaller C no cost is read"},
           {v("theta") >= least_cost, "prover-only-policy-says-nothing", "theta", "theta must be at least 2^sh, the least cost of a token: no request can meet a smaller one"},
           {v("theta") != v("C") || v("cap") == 0, "prover-only-policy-says-nothing", "cap",
            "with theta = C, cap must be 0: a request that covers at most cap of its own tokens would have none of them barred, and cp_verify refuses it"},
           {misnamed == p.call.size(), "prover-only-call-names", std::to_string(first_call + misnamed),
            "a call line's kind must be R16, R8, FC, FS or FL, and its name must begin with its kind"},
           {attention_lists == "KR RM RD|KFD FM FD|R16 WL IX|WH WH|", "prover-only-callA-lines",
            std::to_string(first_callA),
            "the callA lines must be these four, kind and segments: KR RM RD; KFD FM FD; R16 WL IX; WH WH"},
           {!p.callH.empty(), "prover-only-callH-lines", std::to_string(first_callH),
            "here a file must have at least one callH line"},
           {misnamed_H == p.callH.size(), "prover-only-callH-lines", std::to_string(first_callH + misnamed_H),
            "a callH line's kind must be R16, R8, FC or FE, and its name must begin with its kind"},
       })
    rules.push_back(r);
  // Each stage of a proximity test cuts its message into 2^k rows (k: the first of the stage's four
  // numbers), so the last fold's message has 2^(l - the sum of all k) places (l: the third number
  // of the group's line). In a test of three or more stages fold_rounds_prove() in
  // blocks/iop/fold.h writes that fold's coefficients out over these places, and it ends the
  // process if there is only one. cp_verify accepts some such files. The frame's lines have one
  // order, so the `stages` line of group i, from 0, is line 16 + 2 i.
  const blk::Parameters &frame = p.frame;
  const blk::GroupLine *const groups[] = {&frame.link,        &frame.folded,     &frame.omega,    &frame.own,
                                          &frame.once_folded, &frame.once_omega, &frame.once_own, &frame.once_ranges};
  for (size_t i = 0; i < 8; i++) {
    u64 cuts = 0;
    for (const blk::StageChoice &stage : groups[i]->stages) cuts += stage.k;
    rules.push_back({cuts < groups[i]->log_ell, "prover-only-stages", std::to_string(16 + 2 * i),
                     "the 1st, 5th, 9th, .. numbers of this line must add up to less than the third number of the line "
                     "above it: these programs cannot run a proximity test whose last message has one place"});
  }
  for (const Rule &r : rules)
    if (!r.holds) return refusal(r.name, r.line[0] <= '9' ? std::stoul(r.line) : p.line_of(r.line), r.what);
  return "";
}
enum class Found { Read, Absent, Unreadable };
// Size limits, in bytes, for read_once()'s callers. No valid file comes near: a parameter file that
// load() accepts has fewer than 300 lines of at most 1,023 bytes, and the full-size example's
// layout file, prover/examples/target_full_f2/layout.txt, has 0.22 MB.
constexpr size_t PARAMETER_FILE_MOST = (size_t)1 << 20, LAYOUT_FILE_MOST = (size_t)1 << 26;
// Reads a whole file, once. Absent: no file has that name. Unreadable: it cannot be opened or read
// to its end, has more than `most` bytes, or is not a regular file (a device or a pipe may never
// end). Both leave `bytes` empty. The path is checked before the open, which would block on a named
// pipe with no writer, and the open file again, because the path may name another file by then.
inline Found read_once(const std::string &path, std::string &bytes, size_t most) {
  bytes.clear();
  struct stat is;
  if (stat(path.c_str(), &is)) return errno == ENOENT ? Found::Absent : Found::Unreadable;
  if (!S_ISREG(is.st_mode)) return Found::Unreadable;
  FILE *f = fopen(path.c_str(), "rb");
  if (!f) return Found::Unreadable;
  bool whole = !fstat(fileno(f), &is) && S_ISREG(is.st_mode);
  char piece[1 << 16];
  for (size_t n; whole && (n = fread(piece, 1, sizeof piece, f)) > 0; whole = bytes.size() <= most)
    bytes.append(piece, n);
  whole = whole && !ferror(f);
  fclose(f);
  if (!whole) bytes.clear();
  return whole ? Found::Read : Found::Unreadable;
}
// Reads the file once and applies parse() but not limit_missed(). Returns "" or the rejection.
inline std::string load_form(const std::string &path, Parameters &p) {
  std::string bytes;
  if (read_once(path, bytes, PARAMETER_FILE_MOST) != Found::Read)
    return "input-unreadable-policy: the parameter file cannot be read, is no regular file, or has more than 2^20 bytes";
  return parse(bytes, p);
}
// The same, followed by limit_missed().
inline std::string load(const std::string &path, Parameters &p) {
  const std::string why = load_form(path, p);
  return why.empty() ? limit_missed(p) : why;
}
} // namespace parameter_file
