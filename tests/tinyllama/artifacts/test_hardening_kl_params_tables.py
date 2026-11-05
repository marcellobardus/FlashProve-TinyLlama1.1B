"""A12, A14, A15 and A16 sub-rules for KL words, params and tables."""

import numpy as np
import pytest

import synth_artifact as synth
from test_hardening_support import (
    DIMS,
    artifacts,
    artifact_copy,
    assert_only,
    params,
    read_kl,
    rewrite_file,
    small_artifact,
    write_kl,
    write_params,
)


@pytest.mark.parametrize(
    "m_value, k_value",
    [(24, 16), (37, 22), (30, 32)],
    ids=["k-outside", "k-below-m-minus-14", "k-above-m"],
)
def test_a12_k_rule_on_the_first_norm(artifact_copy, m_value, k_value):
    words = read_kl(artifact_copy)
    words[7] = m_value
    words[8] = k_value
    directory = write_kl(artifact_copy, words)
    assert_only(directory, "A12")


@pytest.mark.parametrize(
    "m_value, k_value",
    [(48, 32), (37, 16), (37, 22), (30, 32)],
    ids=["m-outside", "k-outside", "k-below-m-minus-14", "k-above-m"],
)
def test_a12_mf_kf_in_params(artifact_copy, m_value, k_value):
    params_object = params(artifact_copy)
    params_object["MF"] = m_value
    params_object["KF"] = k_value
    directory = write_params(artifact_copy, params_object)
    assert_only(directory, "A12")
    found = artifacts.validate_artifact(directory, dims=DIMS)
    assert any(
        code == "A12" and failure_path == "params.json"
        for code, failure_path, _ in found
    ), found


@pytest.mark.parametrize("sh_value", [3, 13])
def test_a14_sh_outside_four_to_twelve(artifact_copy, sh_value):
    words = read_kl(artifact_copy)
    for start in range(0, len(words), 13):
        words[start + 11] = sh_value
    directory = write_kl(artifact_copy, words)
    assert_only(directory, "A14")


def test_a15_alias_constant_name(artifact_copy):
    params_object = params(artifact_copy)
    params_object["D"] = params_object["d"]
    directory = write_params(artifact_copy, params_object)
    assert_only(directory, "A15")
    found = artifacts.validate_artifact(directory, dims=DIMS)
    assert any(
        code == "A15" and detail == "non canonical constant name"
        for code, _, detail in found
    ), found


def test_a15_wrong_constant(artifact_copy):
    params_object = params(artifact_copy)
    params_object["smax"] += 1
    directory = write_params(artifact_copy, params_object)
    assert_only(directory, "A15")
    found = artifacts.validate_artifact(directory, dims=DIMS)
    assert any(
        code == "A15" and "smax" in detail for code, _, detail in found
    ), found


def test_a15_e_res_out_of_range(artifact_copy):
    params_object = params(artifact_copy)
    e_res = 15
    eps = (4096 * (4**e_res) + 100000) // 200000
    params_object["e_res"] = e_res
    params_object["eps1"] = [eps] * DIMS.L
    params_object["eps2"] = [eps] * DIMS.L
    params_object["epsF"] = eps
    directory = write_params(artifact_copy, params_object)
    assert_only(directory, "A15")
    found = artifacts.validate_artifact(directory, dims=DIMS)
    assert any(
        code == "A15" and "e_res" in detail for code, _, detail in found
    ), found


@pytest.mark.parametrize(
    "field", ["eps1", "eps2", "epsF"], ids=["eps1", "eps2", "epsF"]
)
def test_a15_zero_eps(artifact_copy, field):
    params_object = params(artifact_copy)
    if field == "epsF":
        params_object[field] = 0
    else:
        params_object[field][0] = 0
    directory = write_params(artifact_copy, params_object)
    assert_only(directory, "A15")


def test_a15_s_above_63(artifact_copy):
    params_object = params(artifact_copy)
    params_object["s"] = 64
    directory = write_params(artifact_copy, params_object)
    assert_only(directory, "A15")


def test_a16_table_digest_in_params_differs(artifact_copy):
    path = "tables/rope_c.i32"
    array = np.fromfile(artifact_copy / path, dtype="<i4")
    array[0] = int(array[0]) + 1
    directory = rewrite_file(
        artifact_copy, path, data=array.astype("<i4").tobytes()
    )
    params_object = params(directory)
    params_object["tables_sha256"][path] = synth.file_sha256(directory / path)
    directory = write_params(directory, params_object)
    assert_only(directory, "A16")
    found = artifacts.validate_artifact(directory, dims=DIMS)
    assert any(
        code == "A16" and detail == "digest trope differs"
        for code, _, detail in found
    ), found


def test_a16_tables_sha256_differs_alone(artifact_copy):
    path = "tables/rope_c.i32"
    params_object = params(artifact_copy)
    params_object["tables_sha256"][path] = "0" * 64
    directory = write_params(artifact_copy, params_object)
    assert_only(directory, "A16")
    found = artifacts.validate_artifact(directory, dims=DIMS)
    assert any(
        code == "A16" and "tables_sha256 for rope_c differs" in detail
        for code, _, detail in found
    ), found
