"""Hash traces and optional full traces for the TL-INT-1 reference (I3).

A :class:`TraceWriter` is the ``record`` callable of
:func:`tinyllama.reference.model.forward`.  It keeps the metadata of every
record (name, layer, shape, sha256 of the int64 little-endian C-order bytes,
minimum, maximum) and, with ``full=True``, also writes the values as
``<name>.<label>.npy`` files with dtype ``<i8``.  The label is the layer number
in two decimal digits, or ``F`` for a record outside the layers (I3.2).

``finish`` writes the canonical hash trace of I3.2.  ``trace_digest(path)``
reads such a file back.  Canonical JSON is the same form as
:mod:`tinyllama.artifacts`: sorted keys, compact separators, ASCII, one final
line feed.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from ..contract import CONTRACT_ID

__all__ = ["TraceWriter", "trace_digest", "canonical_json_bytes", "int64_sha256"]


def canonical_json_bytes(obj):
    text = json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ) + "\n"
    return text.encode("ascii")


def _int64_array(values):
    array = np.asarray(values, dtype=np.int64)
    return np.ascontiguousarray(array)


def int64_sha256(values):
    """sha256 of the values as int64 little endian in C order."""
    data = _int64_array(values).astype("<i8", copy=False).tobytes(order="C")
    return hashlib.sha256(data).hexdigest()


def token_ids_sha256(token_ids):
    """sha256 of token ids as unsigned 32-bit little endian in C order."""
    data = np.asarray(token_ids, dtype="<u4").tobytes(order="C")
    return hashlib.sha256(data).hexdigest()


def _layer_label(layer):
    return "F" if layer is None else "%02d" % int(layer)


def _failure_object(failure):
    if failure is None:
        return None
    return {
        "condition": int(failure.condition),
        "name": str(failure.name),
        "layer": None if failure.layer is None else int(failure.layer),
        "position": int(failure.position),
        "index": int(failure.index),
        "value": int(failure.value),
    }


def _top16(lam_last):
    values = _int64_array(lam_last)
    order = sorted(
        range(values.shape[0]),
        key=lambda index: (-int(values[index]), index),
    )
    return [
        [int(index), int(values[index])]
        for index in order[: min(16, values.shape[0])]
    ]


class TraceWriter:
    """Collect records and write the I3.2 hash trace.

    ``path`` is the hash trace file.  With ``full=True`` the .npy files go into
    ``path`` with a trailing ``.json`` removed (so ``P0.prefix.json`` writes
    below ``P0.prefix/``), matching the I3.2 full-trace layout.  A path with no
    ``.json`` suffix gets a ``.full`` directory instead.
    """

    def __init__(self, path, full=False):
        self.path = Path(path)
        self.full = bool(full)
        self.records = []
        if self.full:
            if self.path.suffix == ".json":
                self.directory = self.path.with_suffix("")
            else:
                self.directory = self.path.with_name(self.path.name + ".full")
            self.directory.mkdir(parents=True, exist_ok=True)
        else:
            self.directory = None

    def __call__(self, name, layer, values):
        array = _int64_array(values)
        entry = {
            "name": str(name),
            "layer": None if layer is None else int(layer),
            "shape": [int(size) for size in array.shape],
            "sha256": int64_sha256(array),
            "min": int(array.min()) if array.size else 0,
            "max": int(array.max()) if array.size else 0,
        }
        self.records.append(entry)
        if self.full:
            filename = "%s.%s.npy" % (name, _layer_label(layer))
            np.save(
                self.directory.joinpath(filename),
                array.astype("<i8", copy=False),
                allow_pickle=False,
            )
        return None

    def finish(self, artifact=None, prompt_id=None, mode=None, token_ids=None, result=None):
        """Write and return the canonical hash trace of I3.2.

        ``artifact`` may be an :class:`~tinyllama.artifacts.Artifact` or the
        artifact id string.  On a range failure ``lam_last_sha256`` is null,
        ``top16`` is empty and ``next_token`` is null.
        """
        if artifact is None:
            artifact_id = getattr(self, "artifact_id", None)
        else:
            artifact_id = getattr(artifact, "id", artifact)
        if token_ids is None:
            token_ids = getattr(self, "token_ids", [])
        token_ids = [int(token) for token in token_ids]
        if result is None:
            result = getattr(self, "result", None)
        if result is None:
            raise ValueError("finish needs a forward Result")

        lam_last = result.lam_last
        trace = {
            "contract": CONTRACT_ID,
            "artifact_id": artifact_id,
            "prompt_id": prompt_id,
            "mode": mode,
            "n": int(result.n),
            "token_ids_sha256_u32le": token_ids_sha256(token_ids),
            "records": self.records,
            "range_failure": _failure_object(result.range_failure),
            "next_token": (
                None if result.next_token is None else int(result.next_token)
            ),
            "lam_last_sha256": (
                None if lam_last is None else int64_sha256(lam_last)
            ),
            "top16": [] if lam_last is None else _top16(lam_last),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(canonical_json_bytes(trace))
        return trace


def trace_digest(path):
    """Return the hash trace object stored at ``path``."""
    raw = Path(path).read_bytes()
    return json.loads(raw.decode("utf-8"))
