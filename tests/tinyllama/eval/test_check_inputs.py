import hashlib
import json
import random
import struct
from collections import UserDict

import numpy as np

import check_inputs


def small_prompt_document():
    text = "<|user|>\nHello</s>\n<|assistant|>\n"
    ids = [1, 529, 29989, 15043, 2, 29871, 13, 29966, 29989, 465, 22137, 29989, 29958, 13]
    return {
        "prompt_set": "TL-PROMPTS-1",
        "contract": "TL-INT-1",
        "prompts": [
            {
                "id": "PX",
                "messages": [{"role": "user", "content": "Hello"}],
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "n": len(ids),
                "token_ids": ids,
                "token_ids_sha256_u32le": hashlib.sha256(
                    struct.pack("<%dI" % len(ids), *ids)
                ).hexdigest(),
            }
        ],
    }


class FakeEncoding:
    def __init__(self, ids):
        self.ids = list(ids)


class FakeLibTokenizer:
    def __init__(self, ids_by_text):
        self.ids_by_text = dict(ids_by_text)

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is True
        return FakeEncoding(self.ids_by_text[text])


class FakeBatchEncoding(UserDict):
    pass


class FakeTokenizer:
    def __init__(self, text, ids, tokenize_returns_dict=False):
        self.text = text
        self.ids = list(ids)
        self.tokenize_returns_dict = tokenize_returns_dict

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True
    ):
        assert add_generation_prompt is True
        if tokenize:
            if self.tokenize_returns_dict:
                return FakeBatchEncoding({"input_ids": list(self.ids)})
            return list(self.ids)
        return self.text

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is True
        return FakeBatchEncoding({"input_ids": list(self.ids)})


class FakeTransformersTokenizer:
    def __init__(self, text_by_messages, ids_by_text):
        self.text_by_messages = dict(text_by_messages)
        self.ids_by_text = dict(ids_by_text)

    @staticmethod
    def _key(messages):
        return json.dumps(messages, sort_keys=True)

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True
    ):
        assert add_generation_prompt is True
        text = self.text_by_messages[self._key(messages)]
        if tokenize:
            return FakeBatchEncoding({"input_ids": list(self.ids_by_text[text])})
        return text

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is True
        return FakeBatchEncoding({"input_ids": list(self.ids_by_text[text])})


def independent_windows(
    ids,
    seed=0,
    draws=128,
    draws_used=48,
    draw_length=2048,
    remove_ids=(1, 2),
    window_length=1024,
    num_windows=40,
    prefix_id=1,
):
    generator = random.Random(seed)
    starts = [
        generator.randint(0, len(ids) - draw_length - 1) for _ in range(draws)
    ]
    drawn = [list(ids[start : start + draw_length]) for start in starts]
    kept = [
        value
        for row in drawn[:draws_used]
        for value in row
        if value not in remove_ids
    ]
    stride = window_length - 1
    windows = [
        [prefix_id] + kept[stride * index : stride * index + stride]
        for index in range(num_windows)
    ]
    return drawn, kept, windows


def test_ids_sha256_matches_independent_struct_pack():
    ids = [1, 529, 29989, 1792, 32000]
    expected = hashlib.sha256(struct.pack("<%dI" % len(ids), *ids)).hexdigest()
    assert check_inputs.ids_sha256_u32le(ids) == expected


def test_npy_bytes_length_for_fit_shape():
    array = np.zeros((32, 1024), dtype="<i4")
    assert len(check_inputs.npy_bytes(array)) == 131_200


def test_compare_prompt_accepts_committed_text_and_ids():
    record = check_inputs.load_prompt_set()["prompts"][0]
    result = check_inputs.compare_prompt(record, record["text"], record["token_ids"])
    assert result["ok"] is True
    assert result["text_match"] is True
    assert result["text_sha256_match"] is True
    assert result["ids_match"] is True
    assert result["ids_sha256_match"] is True
    assert result["n_match"] is True
    assert result["first_diff"] is None


def test_compare_prompt_reports_first_altered_id():
    record = check_inputs.load_prompt_set()["prompts"][0]
    altered = list(record["token_ids"])
    index = 7
    altered[index] = (altered[index] + 1) % 32000
    result = check_inputs.compare_prompt(record, record["text"], altered)
    assert result["ok"] is False
    assert result["ids_match"] is False
    assert result["first_diff"] == index
    assert result["first_differing_index"] == index
    assert result["actual_ids"] == altered
    assert result["recorded_ids"] == record["token_ids"]


def test_build_windows_matches_independent_rederivation():
    ids = list(range(5000))
    ids[100] = 1
    ids[500] = 2
    ids[1234] = 1
    ids[4321] = 2
    options = {
        "seed": 0,
        "draws": 12,
        "draws_used": 4,
        "draw_length": 64,
        "remove_ids": (1, 2),
        "window_length": 16,
        "num_windows": 5,
        "prefix_id": 1,
    }
    drawn, kept, windows = check_inputs.build_windows(ids, **options)
    expected_drawn, expected_kept, expected_windows = independent_windows(
        ids, **options
    )
    assert drawn == expected_drawn
    assert kept == expected_kept
    assert windows == expected_windows
    assert all(len(window) == options["window_length"] for window in windows)
    assert all(window[0] == options["prefix_id"] for window in windows)


def test_build_windows_defaults_match_i4_3():
    ids = list(range(5000))
    ids[17] = 1
    ids[31] = 2
    drawn, kept, windows = check_inputs.build_windows(ids)
    assert len(drawn) == 128
    assert all(len(row) == 2048 for row in drawn)
    assert len(windows) == 40
    assert all(len(window) == 1024 for window in windows)
    expected_drawn, expected_kept, expected_windows = independent_windows(ids)
    assert drawn == expected_drawn
    assert kept == expected_kept
    assert windows == expected_windows


def test_prompts_v1_json_is_internally_consistent():
    document = check_inputs.load_prompt_set()
    result = check_inputs.check_prompts_json(document)
    assert result["ok"] is True
    assert len(result["prompts"]) == 4
    assert all(record["ok"] for record in result["prompts"])


def _monkeypatch_small_inputs(monkeypatch, document, tokenizer, lib_tokenizer=None):
    if lib_tokenizer is None:
        lib_tokenizer = FakeLibTokenizer({tokenizer.text: tokenizer.ids})
    monkeypatch.setattr(check_inputs, "load_manifest", lambda *a, **k: {"revision": "rev"})
    monkeypatch.setattr(check_inputs, "load_prompt_set", lambda *a, **k: document)
    monkeypatch.setattr(check_inputs, "load_tokenizer", lambda *a, **k: tokenizer)
    monkeypatch.setattr(
        check_inputs, "resolve_tokenizer_json", lambda *a, **k: "tokenizer.json"
    )
    monkeypatch.setattr(
        check_inputs, "load_calibration_tokenizer", lambda *a, **k: lib_tokenizer
    )


def test_main_writes_report_and_returns_zero(monkeypatch, tmp_path):
    document = small_prompt_document()
    record = document["prompts"][0]
    tokenizer = FakeTokenizer(record["text"], record["token_ids"])
    _monkeypatch_small_inputs(monkeypatch, document, tokenizer)
    out = tmp_path / "report.json"
    code = check_inputs.main(
        ["--checkpoint-dir", "fake", "--out", str(out), "--skip-calibration"]
    )
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["prompts"]["ok"] is True
    assert report["prompts"]["results"][0]["ids2"]["informational_only"] is True
    assert report["calibration"] == {"skipped": True}


def test_main_returns_one_on_prompt_mismatch(monkeypatch, tmp_path):
    document = small_prompt_document()
    record = document["prompts"][0]
    altered = list(record["token_ids"])
    altered[3] = (altered[3] + 1) % 32000
    tokenizer = FakeTokenizer(record["text"], altered)
    lib_tokenizer = FakeLibTokenizer({record["text"]: record["token_ids"]})
    _monkeypatch_small_inputs(monkeypatch, document, tokenizer, lib_tokenizer)
    out = tmp_path / "report.json"
    code = check_inputs.main(
        ["--checkpoint-dir", "fake", "--out", str(out), "--skip-calibration"]
    )
    assert code == 1
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert report["prompts"]["ok"] is False
    assert report["prompts"]["results"][0]["first_diff"] == 3


def test_check_prompts_ids2_from_dict_batch_encoding():
    document = small_prompt_document()
    record = document["prompts"][0]
    tokenizer = FakeTokenizer(
        record["text"], record["token_ids"], tokenize_returns_dict=True
    )
    lib_tokenizer = FakeLibTokenizer({record["text"]: record["token_ids"]})
    report = check_inputs.check_prompts(tokenizer, document, lib_tokenizer)
    result = report["results"][0]
    assert "error" not in result["ids2"]
    assert result["ids2"]["ids"] == record["token_ids"]
    assert result["ids2"]["matches_committed_ids"] is True
    assert result["ids2"]["informational_only"] is True
    assert result["ok"] is True


def test_real_prompts_report_transformers_drops_29871(monkeypatch, tmp_path):
    document = check_inputs.load_prompt_set()
    text_by_messages = {
        json.dumps(record["messages"], sort_keys=True): record["text"]
        for record in document["prompts"]
    }
    committed = {record["text"]: record["token_ids"] for record in document["prompts"]}
    transformers_ids = {text: list(ids) for text, ids in committed.items()}

    p0 = next(record for record in document["prompts"] if record["id"] == "P0")
    dropped = list(p0["token_ids"])
    assert dropped[15] == 29871
    del dropped[15]
    transformers_ids[p0["text"]] = dropped

    tokenizer = FakeTransformersTokenizer(text_by_messages, transformers_ids)
    lib_tokenizer = FakeLibTokenizer(committed)

    monkeypatch.setattr(
        check_inputs, "load_manifest", lambda *a, **k: {"revision": "rev"}
    )
    monkeypatch.setattr(check_inputs, "load_tokenizer", lambda *a, **k: tokenizer)
    monkeypatch.setattr(
        check_inputs, "resolve_tokenizer_json", lambda *a, **k: "tokenizer.json"
    )
    monkeypatch.setattr(
        check_inputs, "load_calibration_tokenizer", lambda *a, **k: lib_tokenizer
    )
    monkeypatch.setattr(check_inputs, "load_prompt_set", lambda *a, **k: document)

    report = check_inputs.check_prompts(tokenizer, document, lib_tokenizer)
    result = next(item for item in report["results"] if item["id"] == "P0")
    assert result["ids_transformers"]["first_difference"] == 15
    assert result["ids_transformers"]["recorded_id_at_first_difference"] == 29871
    assert result["ids_transformers"]["got_id_at_first_difference"] == 13
    assert result["ids_tokenizers_lib"]["ok"] is True
    assert result["ok"] is False
    assert report["ok"] is False

    out = tmp_path / "report.json"
    code = check_inputs.main(
        ["--checkpoint-dir", "fake", "--out", str(out), "--skip-calibration"]
    )
    assert code == 1
    main_report = json.loads(out.read_text(encoding="utf-8"))
    main_result = next(
        item for item in main_report["prompts"]["results"] if item["id"] == "P0"
    )
    assert main_result["ids_transformers"]["first_difference"] == 15
    assert main_result["ids_transformers"]["ok"] is False
    assert main_report["ok"] is False
