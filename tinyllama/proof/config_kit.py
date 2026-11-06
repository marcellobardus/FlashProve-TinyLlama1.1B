"""Build the TinyLlama FlashProve configuration kit named by PROVING_PLAN M1.

The kit holds policy.txt and KL.bin.  layout.txt is written when a usable
cp_verify and prover/tools/pack_layout.py are supplied, because that file is
derived by cp_verify's planning call rather than by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

from tinyllama import contract

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_PATH = REPO_ROOT / "prover" / "examples" / "target_small_f2_roomy" / "policy.txt"
PACK_LAYOUT_PATH = REPO_ROOT / "prover" / "tools" / "pack_layout.py"

DEFAULT_MF = 43
DEFAULT_KF = 32
DEFAULT_S = 17
DEFAULT_E_RES = 11
DEFAULT_LIDB = 24
DEFAULT_NMIN = 20
DEFAULT_TABLE_ROWS_LOG = 8
DEFAULT_TABLE_RANDOM = 65535
DEFAULT_TABLE_LOG_N = 23
DEFAULT_TABLE_BLOCK_LOG = 11

LM_SMALL = {
    "lmaf": 13,
    "NAf": 3,
    "lmad": 12,
    "NAd": 1,
    "lmar": 12,
    "lmb": 13,
    "lmq": 13,
    "lmx": 13,
    "lmS": 11,
    "NAs": 1,
    "lmh": 12,
    "lmo": 13,
    "lmbA": 13,
    "lmhf": 14,
    "NHf": 1,
    "lmhd": 12,
    "lmhr": 12,
    "lmbH": 13,
}

LM_FULL = {
    "lmaf": 16,
    "NAf": 3,
    "lmad": 15,
    "NAd": 1,
    "lmar": 13,
    "lmb": 16,
    "lmq": 16,
    "lmx": 16,
    "lmS": 16,
    "NAs": 1,
    "lmh": 13,
    "lmo": 16,
    "lmbA": 14,
    "lmhf": 16,
    "NHf": 4,
    "lmhd": 14,
    "lmhr": 13,
    "lmbH": 15,
}

DENSE_CALLS = (
    ("R16a", "R16", ("G", "U")),
    ("R16b", "R16", ("N1", "N1H", "N2", "N2H", "O", "D", "V")),
    ("R16c", "R16", ("QK", "QKH")),
    ("R8", "R8", ("GH", "UH", "OH", "DH", "VH", "ROW")),
    ("FC", "FC", ("CN1", "CN2", "CQK", "CV")),
    ("FL", "FL", ("F",)),
    ("R16d", "R16", ("UT",)),
)

CALL_A = (
    ("KR", "KR", ("RM", "RD")),
    ("KFD", "KFD", ("FM", "FD")),
    ("R16", "R16", ("WL", "IX")),
    ("WH", "WH", ("WH",)),
)

CALL_H = (
    ("R16a", "R16", ("NF", "NFH", "HD0")),
    ("R16b", "R16", ("HD1",)),
    ("R8", "R8", ("ROW", "HD2")),
    ("FEa", "FE", ("HE0",)),
    ("FEb", "FE", ("HE1",)),
    ("FC", "FC", ("CNF",)),
)


class KitError(ValueError):
    """The requested kit cannot be built from the given source."""


def _ceil_log2(value):
    if value < 1:
        raise KitError("cannot take the logarithm of %r" % (value,))
    return (value - 1).bit_length()


def _default_eps(e_res):
    return (4096 * (4**e_res) + 100000) // 200000


def _contract_params():
    eps = _default_eps(DEFAULT_E_RES)
    return {
        "contract": contract.CONTRACT_ID,
        "L": contract.L,
        "d": contract.D,
        "H": contract.H,
        "G": contract.G,
        "dh": contract.DH,
        "f": contract.F,
        "V": contract.V,
        "NE": contract.NE,
        "lt": contract.LT,
        "cs": contract.CS,
        "theta0": contract.THETA0,
        "S": contract.S,
        "rx": contract.RX,
        "MF": DEFAULT_MF,
        "KF": DEFAULT_KF,
        "s": DEFAULT_S,
        "eps1": [eps] * contract.L,
        "eps2": [eps] * contract.L,
        "epsF": eps,
    }


def _read_source(artifact, kl_path):
    path = Path(artifact)
    if path.is_dir():
        params_path = path / "params.json"
        kl_path_in = path / "KL.bin"
        try:
            params = json.loads(params_path.read_text(encoding="ascii"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise KitError("cannot read %s: %s" % (params_path, exc)) from exc
        try:
            kl = kl_path_in.read_bytes()
        except OSError as exc:
            raise KitError("cannot read %s: %s" % (kl_path_in, exc)) from exc
        return params, kl, path

    if artifact in (contract.CONTRACT_ID, "TL-INT-1"):
        kl = None
        if kl_path is not None:
            try:
                kl = Path(kl_path).read_bytes()
            except OSError as exc:
                raise KitError("cannot read %s: %s" % (kl_path, exc)) from exc
        return _contract_params(), kl, None

    raise KitError(
        "artifact must be a directory or the contract id %r" % contract.CONTRACT_ID
    )


def _require_int(params, name):
    value = params.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise KitError("params.json field %s is not an integer" % name)
    return value


def _normalise(params, kl):
    if not isinstance(params, dict):
        raise KitError("params.json is not an object")
    if params.get("contract") != contract.CONTRACT_ID:
        raise KitError(
            "artifact contract is %r, expected %r"
            % (params.get("contract"), contract.CONTRACT_ID)
        )
    names = ("L", "d", "H", "G", "dh", "f", "V", "NE", "cs", "theta0", "S", "rx", "MF", "KF", "s")
    values = {name: _require_int(params, name) for name in names}
    if values["NE"] != 0:
        raise KitError("the M1 kit is dense: NE is %d, expected 0" % values["NE"])
    if values["L"] < 2 or values["L"] % 2:
        raise KitError("L is %d: it must be even and at least 2 for ll 1" % values["L"])
    if values["V"] < 2:
        raise KitError("V is %d, expected at least 2" % values["V"])
    if values["G"] < 1 or values["H"] % values["G"]:
        raise KitError("H %d is not a multiple of G %d" % (values["H"], values["G"]))
    if values["H"] * values["dh"] != values["d"]:
        raise KitError("H * dh does not equal d")

    lengths = {}
    for field in ("eps1", "eps2"):
        entry = params.get(field)
        if not isinstance(entry, list) or len(entry) != values["L"]:
            raise KitError("%s is not a list of length L" % field)
        if not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in entry):
            raise KitError("%s does not hold positive integers" % field)
        if len(set(entry)) != 1:
            raise KitError("%s is not constant across layers" % field)
        lengths[field] = entry[0]
    eps_f = params.get("epsF")
    if not isinstance(eps_f, int) or isinstance(eps_f, bool) or eps_f <= 0:
        raise KitError("epsF is not a positive integer")
    if lengths["eps1"] != lengths["eps2"] or lengths["eps1"] != eps_f:
        raise KitError("eps1, eps2 and epsF differ, but policy.txt has one value each")

    if kl is None:
        raise KitError(
            "KL.bin is needed; pass --artifact DIR or give the scales with --kl FILE"
        )
    row_bytes = 13 * 4
    if len(kl) != values["L"] * row_bytes:
        raise KitError(
            "KL.bin is %d bytes; the artifact's L %d needs %d"
            % (len(kl), values["L"], values["L"] * row_bytes)
        )
    rows = []
    for layer in range(values["L"]):
        row = list(struct.unpack_from("<13I", kl, layer * row_bytes))
        if any(shift < 1 or shift > 32 for shift in row[:7]):
            raise KitError("KL.bin layer %d has a shift outside 1 .. 32" % layer)
        if row[12] not in (0, 1):
            raise KitError("KL.bin layer %d has rp %d, expected 0 or 1" % (layer, row[12]))
        rows.append(row)

    result = dict(values)
    result["eps1"] = lengths["eps1"]
    result["eps2"] = lengths["eps2"]
    result["epsF"] = eps_f
    result["kl_rows"] = rows
    result["lt"] = params.get("lt", contract.LT)
    return result


def _frame_template():
    try:
        text = TEMPLATE_PATH.read_text(encoding="ascii")
    except OSError as exc:
        raise KitError("cannot read the frame template %s: %s" % (TEMPLATE_PATH, exc)) from exc
    lines = text.splitlines()
    try:
        table_at = next(i for i, line in enumerate(lines) if line.startswith("table "))
    except StopIteration as exc:
        raise KitError("the frame template has no table line") from exc
    if table_at != 31:
        raise KitError("the frame template has %d frame lines, expected 32" % (table_at + 1))
    return lines[:table_at], lines[table_at].split()


def _digit_form(largest_shift):
    if largest_shift <= 16:
        return 0
    if largest_shift <= 24:
        return 1
    if largest_shift <= 32:
        return 2
    raise KitError("a shift of %d needs a digit form above 2" % largest_shift)


def _la_for(values, lt):
    candidates = [
        values["d"],
        values["f"],
        values["G"] * values["dh"],
        (values["H"] + values["G"]) * values["dh"],
    ]
    delta = 0
    for step in range(1, 9):
        if all(value % (1 << step) == 0 for value in candidates):
            delta = step
    if delta == 0:
        raise KitError("no la above lt makes 2^(la - lt) divide d, f, G dh and (H + G) dh")
    return min(16, lt + delta)


def _group_rows(digits, groups, columns_log):
    left = columns_log
    rows = digits
    for extent, cls, _high in groups:
        if cls == "A":
            wanted = _ceil_log2(extent)
            taken = wanted if wanted <= left else 0
            left_count = 1 if taken else extent
        else:
            zeros = (extent & -extent).bit_length() - 1
            taken = min(zeros, left)
            left_count = extent >> taken
        left -= taken
        rows *= left_count
    return rows


def _weight_matrix_rows(values, columns_log):
    d = values["d"]
    dh = values["dh"]
    H = values["H"]
    G = values["G"]
    f = values["f"]
    V = values["V"]
    units = []
    for _layer in range(values["L"]):
        units.append(_group_rows(2, ((64, "A", False), (dh, "A", False), (d, "E", False)), columns_log))
        units.append(_group_rows(2, ((d, "E", False), (G * dh, "E", True)), columns_log))
        units.append(_group_rows(2, ((d, "E", False), (H * dh, "E", True)), columns_log))
        units.append(_group_rows(2, ((d, "E", False), (f, "E", True)), columns_log))
        units.append(_group_rows(2, ((d, "E", False), (f, "E", True)), columns_log))
        units.append(_group_rows(2, ((f, "E", False), (d, "E", True)), columns_log))
    units.append(_group_rows(4, ((V, "A", False), (d, "E", True)), columns_log))
    units.append(_group_rows(2, ((V, "A", False), (d, "E", False)), columns_log))
    total = sum(units)
    return (total + contract.D - 1) // contract.D * contract.D


def _table_line(values, template_fields):
    rows_log = int(template_fields[1])
    log_n = int(template_fields[2])
    random_entries = int(template_fields[4])
    block_log = max(int(template_fields[6]), _ceil_log2(values["V"]))
    if block_log > 20:
        raise KitError("V %d needs table block log %d, above 20" % (values["V"], block_log))
    if rows_log > block_log:
        raise KitError("table rows log %d is above its block log %d" % (rows_log, block_log))
    matrix_rows = _weight_matrix_rows(values, block_log)
    data = matrix_rows * (1 << (block_log - rows_log))
    answers = 1
    pads = data + 3 * answers
    run = 1 << (block_log - rows_log)
    mask_run = 1 << _ceil_log2(random_entries)
    step = max(2, run, mask_run)
    table_data = (pads + step - 1) // step * step
    table_log_n = max(
        log_n,
        _ceil_log2(table_data + random_entries) + 1,
        rows_log,
    )
    if table_log_n > 32:
        raise KitError("the table needs code length 2^%d, above 2^32" % table_log_n)
    if table_data >= (1 << 32):
        raise KitError("the table needs %d data entries, above cp_verify's 32-bit field" % table_data)
    line = "table %d %d %d %d 0 %d" % (
        rows_log,
        table_log_n,
        table_data,
        random_entries,
        block_log,
    )
    return line


def _segment_length(segment, values):
    dkv = values["G"] * values["dh"]
    nqk = values["H"] * values["dh"] + dkv
    if segment == "ROW":
        return 2 * (99 if values["xl"] == 5 else 74)
    if segment == "MR":
        return 1
    groups = (
        (("G", "U", "UT", "GH", "UH", "G0", "F", "CG"), values["f"]),
        (("QK", "QKH", "CQK"), nqk),
        (("N1", "N1H", "N2", "N2H", "O", "OH", "D", "DH", "XR", "CN1", "CN2"), values["d"]),
        (("V", "VH", "CV"), dkv),
        (("LR", "LT", "DD", "LH", "SEL", "L3", "CLR"), values["NE"]),
    )
    for names, length in groups:
        if segment in names:
            return length
    raise KitError("unknown lookup segment %s" % segment)


def _layer_iotas(values, calls):
    counts = []
    for _name, _kind, segments in calls:
        columns = sum(_segment_length(segment, values) for segment in segments)
        counts.append(
            max(
                values["ll"] + _ceil_log2(columns) + values["lt"],
                values["nmin"],
            )
        )
    depth = min([values["lidb"]] + counts)
    return {index: counts[index] - depth if index < len(counts) else 0 for index in range(8)}


def _attention_iota(values):
    lgk = _ceil_log2(values["G"])
    ldh = _ceil_log2(values["dh"])
    heads = values["H"] // values["G"]
    layers = 1 << values["ll"]
    nM = lgk + 2 * values["lt"] - 1
    nD = lgk + values["lt"] + values["lB"]
    wl = 1 + lgk + values["lt"] + ldh
    wh = lgk + values["lt"] + ldh
    blocks = layers * heads
    keys = (
        blocks * ((1 << nM) + (1 << nD)),
        blocks * (1 << nD),
        blocks * ((1 << wl) + (1 << (lgk + values["lt"]))),
        blocks * (1 << wh),
    )
    counts = [max(_ceil_log2(key_count), values["nmin"]) for key_count in keys]
    depth = min([values["lidb"]] + counts)
    return 32 + depth


def _head_iotas(values):
    step = 1 << (values["la"] - values["lt"])
    JV = (values["V"] + step - 1) // step * step
    JVa = 1 << (_ceil_log2(JV) - 1)
    if JVa % step:
        raise KitError("JVa %d is not divisible by S2 %d" % (JVa, step))
    lengths = {
        "NF": values["d"],
        "NFH": values["d"],
        "HD0": JV,
        "HD1": JV,
        "ROW": 99 + 21,
        "HD2": JV,
        "CNF": values["d"],
        "HE0": JVa,
        "HE1": JV - JVa,
    }
    floor = values["nmin"] - values["lt"]
    if floor < 0:
        floor = 0
    counts = []
    for _name, _kind, segments in CALL_H:
        columns = sum(lengths[segment] for segment in segments)
        columns_log = max(_ceil_log2(columns), floor)
        counts.append(columns_log + values["lt"])
    depth = min([values["lidb"]] + counts)
    return {index: counts[index] - depth for index in range(6)}


def _lm_lines(values):
    base = dict(LM_FULL if values["lt"] >= 10 else LM_SMALL)
    for name in ("lmaf", "lmq", "lmx", "lmo", "lmhf"):
        base[name] = max(base[name], values["la"])
    for name in ("lmh", "lmhd", "lmhr", "lmad", "lmar"):
        base[name] = max(base[name], values["lt"])
    return base


def build_policy(values, kl, lt):
    frame, table_fields = _frame_template()
    shifts = values["kl_rows"]
    largest = {
        "kv": max(row[2] for row in shifts),
        "ko": max(row[3] for row in shifts),
        "kg": max(row[4] for row in shifts),
        "ku": max(row[5] for row in shifts),
        "kd": max(row[6] for row in shifts),
    }
    values = dict(values)
    values["lt"] = lt
    values["ll"] = 1
    values["la"] = _la_for(values, lt)
    values["lB"] = min(6, lt - 2)
    if values["lB"] < 0:
        raise KitError("lt %d is too small for lB" % lt)
    values["nmin"] = DEFAULT_NMIN
    values["lidb"] = DEFAULT_LIDB
    values["xl"] = 5
    lm = _lm_lines(values)

    table_line = _table_line(values, table_fields)
    lines = list(frame)
    lines.append(table_line)
    lines.extend(
        [
            "prof 20",
            "lt %d" % lt,
            "L %d" % values["L"],
            "ll 1",
            "d %d" % values["d"],
            "H %d" % values["H"],
            "G %d" % values["G"],
            "dh %d" % values["dh"],
            "f %d" % values["f"],
            "fe 0",
            "NE 0",
            "V %d" % values["V"],
            "eps1_lo %d" % values["eps1"],
            "eps1_hi 0",
            "eps2_lo %d" % values["eps2"],
            "eps2_hi 0",
            "cs %d" % values["cs"],
            "theta0_lo %d" % values["theta0"],
            "theta0_hi 0",
            "hq 2",
            "hv %d" % _digit_form(largest["kv"]),
            "ho %d" % _digit_form(largest["ko"]),
            "hgD %d" % _digit_form(largest["kg"]),
            "huD %d" % _digit_form(largest["ku"]),
            "hdD %d" % _digit_form(largest["kd"]),
            "hr 0",
            "ut 1",
            "la %d" % values["la"],
            "nmin %d" % values["nmin"],
            "D %d" % contract.D,
            "tau 128",
            "lmaf %d" % lm["lmaf"],
            "NAf %d" % lm["NAf"],
            "lmad %d" % lm["lmad"],
            "NAd %d" % lm["NAd"],
            "lmar %d" % lm["lmar"],
            "lmb %d" % lm["lmb"],
            "lmq %d" % lm["lmq"],
            "lmx %d" % lm["lmx"],
            "wb_qk 13",
            "wb_v 13",
            "wb_o 13",
            "wb_g 13",
            "wb_u 13",
            "wb_gs 0",
            "wb_us 0",
            "wb_gr 0",
            "wb_ur 0",
            "wb_dn 13",
            "wb_dnr 0",
            "wb_r 0",
        ]
    )
    iotas = _layer_iotas(values, DENSE_CALLS)
    lines.extend("iota_%d %d" % (index, iotas[index]) for index in range(8))
    lines.extend(
        [
            "seed_lo 0",
            "seed_hi 0",
            "lB %d" % values["lB"],
            "S %d" % values["S"],
            "rx %d" % values["rx"],
            "kFM 15",
            "kFD 16",
            "lmS %d" % lm["lmS"],
            "NAs %d" % lm["NAs"],
            "lmh %d" % lm["lmh"],
            "lmo %d" % lm["lmo"],
            "lmbA %d" % lm["lmbA"],
            "iotaA_0 %d" % _attention_iota(values),
            "MF %d" % values["MF"],
            "KF %d" % values["KF"],
            "epsF_lo %d" % values["epsF"],
            "epsF_hi 0",
            "toksrc 1",
            "lmhf %d" % lm["lmhf"],
            "NHf %d" % lm["NHf"],
            "lmhd %d" % lm["lmhd"],
            "lmhr %d" % lm["lmhr"],
            "lmbH %d" % lm["lmbH"],
            "wb_emb 21",
            "wb_head 13",
        ]
    )
    head_iotas = _head_iotas(values)
    lines.extend("iotaH_%d %d" % (index, head_iotas[index]) for index in range(6))
    lines.extend(
        [
            "C 256",
            "theta 256",
            "cap 0",
            "s %d" % values["s"],
            "sh 8",
            "sT 4",
            "F 24",
            "lidb %d" % values["lidb"],
            "xl 5",
        ]
    )
    for name, kind, segments in DENSE_CALLS:
        lines.append("call %s %s %s" % (name, kind, " ".join(segments)))
    for name, kind, segments in CALL_A:
        lines.append("callA %s %s %s" % (name, kind, " ".join(segments)))
    for name, kind, segments in CALL_H:
        lines.append("callH %s %s %s" % (name, kind, " ".join(segments)))
    return "\n".join(lines) + "\n"


def _expected_artifact_id(directory, expected):
    manifest = Path(directory) / "manifest.json"
    try:
        raw = manifest.read_bytes()
    except OSError as exc:
        raise KitError("cannot read %s for the artifact id: %s" % (manifest, exc)) from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected.lower():
        raise KitError(
            "artifact id is %s, expected %s (manifest %s)" % (actual, expected.lower(), manifest)
        )


def _write_layout(out, policy_text, cp_verify, pack_layout):
    verifier = Path(cp_verify)
    if not verifier.is_file():
        raise KitError("cp_verify does not exist at %s" % verifier)
    packer = Path(pack_layout)
    if not packer.is_file():
        raise KitError("pack_layout.py does not exist at %s" % packer)
    row_log = None
    for line in policy_text.splitlines():
        fields = line.split()
        if len(fields) >= 5 and fields[0] == "group" and fields[1] == "link":
            row_log = fields[4]
            break
    if row_log is None:
        raise KitError("policy.txt has no group link line")
    with tempfile.TemporaryDirectory(prefix="tl-config-kit-") as temporary:
        arrays = Path(temporary) / "arrays.txt"
        planned = subprocess.run(
            [str(verifier), "-plan", "arrays", str(out), "128"],
            capture_output=True,
            text=True,
            check=False,
        )
        if planned.returncode != 4:
            raise KitError(
                "cp_verify -plan arrays ended %d: %s%s"
                % (planned.returncode, planned.stdout, planned.stderr)
            )
        arrays.write_text(planned.stdout, encoding="ascii")
        packed = subprocess.run(
            [sys.executable, str(packer), "--arrays", str(arrays), row_log, "4"],
            capture_output=True,
            text=True,
            check=False,
        )
        if packed.returncode != 0:
            raise KitError(
                "pack_layout.py ended %d: %s%s"
                % (packed.returncode, packed.stdout, packed.stderr)
            )
    (Path(out) / "layout.txt").write_text(packed.stdout, encoding="ascii")


def build_config(
    artifact,
    out,
    *,
    lt=None,
    kl_path=None,
    artifact_id=None,
    cp_verify=None,
    pack_layout=None,
    layout=True,
):
    params, kl, directory = _read_source(artifact, kl_path)
    if artifact_id is not None:
        if directory is None:
            raise KitError("--artifact-id needs an artifact directory with manifest.json")
        _expected_artifact_id(directory, artifact_id)
    values = _normalise(params, kl)
    if lt is None:
        lt = values.get("lt", contract.LT)
    if not isinstance(lt, int) or isinstance(lt, bool) or lt < 6 or lt > 10:
        raise KitError("lt must be an integer from 6 to 10, not %r" % (lt,))
    policy = build_policy(values, kl, lt)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "policy.txt").write_text(policy, encoding="ascii", newline="\n")
    (out / "KL.bin").write_bytes(kl)
    if layout:
        verifier = cp_verify
        if verifier is None:
            candidate = REPO_ROOT / "bin" / "cp_verify"
            if candidate.is_file():
                verifier = candidate
        packer = pack_layout if pack_layout is not None else PACK_LAYOUT_PATH
        if verifier is None:
            print(
                "config_kit: policy.txt and KL.bin written; layout.txt not written because "
                "cp_verify was not found (use --cp-verify)",
                file=sys.stderr,
            )
        else:
            _write_layout(out, policy, verifier, packer)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Write the TinyLlama FlashProve configuration kit (PROVING_PLAN M1)."
    )
    parser.add_argument("--artifact", required=True, help="artifact directory, or TL-INT-1")
    parser.add_argument("--out", required=True, help="configuration directory to write")
    parser.add_argument("--lt", type=int, help="log2 of the request window (6 to 10)")
    parser.add_argument("--kl", help="KL.bin for a contract id instead of an artifact directory")
    parser.add_argument("--artifact-id", help="expected artifact id (sha256 of manifest.json)")
    parser.add_argument("--cp-verify", help="cp_verify to run -plan arrays and make layout.txt")
    parser.add_argument("--pack-layout", help="prover/tools/pack_layout.py")
    parser.add_argument("--no-layout", action="store_true", help="do not try to write layout.txt")
    args = parser.parse_args(argv)
    try:
        build_config(
            args.artifact,
            args.out,
            lt=args.lt,
            kl_path=args.kl,
            artifact_id=args.artifact_id,
            cp_verify=args.cp_verify,
            pack_layout=args.pack_layout,
            layout=not args.no_layout,
        )
    except KitError as exc:
        print("config_kit: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
