import tl_ref_independent as ref


def _load():
    return ref.load_vectors("layer.json")


def _run(data):
    inputs = data["inputs"]
    return ref.layer(
        inputs["x_rows"],
        inputs["weights"],
        inputs["scales"],
        inputs["tables"],
    )


def test_layer_matches_every_committed_intermediate():
    data = _load()
    result = _run(data)
    expected = data["expected"]
    assert set(result) == set(expected)
    for name in expected:
        assert result[name] == expected[name], name


def test_first_norm_is_the_contract_section_8_worked_example_by_hand():
    data = _load()
    result = _run(data)
    assert data["inputs"]["x_rows"][0] == [3, -4, 0, 0]
    assert result["a"][0] == [77, -102, 0, 0]
    assert result["a"][1] == [0, 77, -102, 0]


def test_rotary_step_is_hand_derivable():
    result = _run(_load())
    assert result["q"][0][0] == [32, -16]
    assert result["k"][0][0] == [32, -16]
    assert result["q"][1][0] == [19, 29]
    assert result["k"][1][0] == [-44, 16]
    assert result["q"][1][1] == [-38, 26]


def test_attention_and_residual_mid_are_hand_derivable():
    result = _run(_load())
    assert result["scores"][0][0] == [1280]
    assert result["o"][0][0] == [19, -25]
    assert result["x_mid"] == [[13, -16, 10, -12], [10, -9, -16, 10]]
    assert result["e"][1][0][1] == 0


def test_gate_key_and_ts_offset_match_the_small_table():
    result = _run(_load())
    assert result["gw"] == [[3, -71, -5], [56, 3, -16]]
    assert result["key"] == result["gw"]
    assert result["s"] == [[13, -61, 5], [66, 13, -6]]
