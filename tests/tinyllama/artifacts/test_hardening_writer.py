"""The streaming writer must keep every output byte identical."""

import hashlib

import pytest

artifacts = pytest.importorskip("tinyllama.artifacts")

import test_hardening_artifact_factory as factory


EXPECTED_ID = "f771d5ae582478f6f345a82565e0a1c8f3fcc8ee44732843c05df2c853c1ce2e"

EXPECTED_SHA256 = {
    "KL.bin": "adb78e6467e55b2c641ceceee6e33a9de2f9e448b0cdfb74da187dbccd79b9bf",
    "emb.i32": "45a8f89e84f493279f572e2ebf37fd0280e45fd08869999ddc096ca1177e37cd",
    "exponents.json": "2697472a7ef2804698ef4c40413884f63ebfaca046597e285a08b5ad334b63ae",
    "layers/00.wd.i16": "8ea22453f435426fd376a6fcabf2f9642e992e1f2b54ec3bcbc6a09a335ac7f1",
    "layers/00.wg.i16": "cd9c9ff8cdf95de0a64a95dc104a4f9397071e54cc4fb584dd5b94335a84f1c1",
    "layers/00.wk.i16": "8660c3b74055fa02a252856a85f633a91a7eda5587800008f35170c08cea341b",
    "layers/00.wo.i16": "cad4e954cb1e697651369b791a1c4b3d1aa6dba1335ab1d4f95a2f0f6cac0a3a",
    "layers/00.wq.i16": "73e1d4c1b0eb0b5566834532719a35f7dbfb577a2b74bedfd7b3d8c9f146803e",
    "layers/00.wu.i16": "ec2bdd6fd256f44fbe1d795c07135ae149706b2845b4b7ab973fd7a7e93953d2",
    "layers/00.wv.i16": "4e3d58325d392b78b96f8b104c0dc60ae3a99b19a80043f72c7a66474ba462f6",
    "layers/01.wd.i16": "18950eae7b289fca00a51a9b69e55fc1779fe767888b2c0e733266e0cbd7cee8",
    "layers/01.wg.i16": "9f73ae886717591859d7d4210624d7a25e9359156d955e910b924d467c613d9a",
    "layers/01.wk.i16": "e667b6e5ff2c6b73e38ca2e027bddce22b81bf97a5c9fcd7605568be7799a4a7",
    "layers/01.wo.i16": "1555dd8f09f5814bdf21fac77fdbad34183962b98140453d88d1666ca151fcf5",
    "layers/01.wq.i16": "25bf4af9f63f6e6584251e2f4f8c91d1cf4fac8a17ad1d0ecd57176db403faa1",
    "layers/01.wu.i16": "1b54a24f93f238a638edeb532dbc77766f731e13090b6e497335c9b2ab4b0626",
    "layers/01.wv.i16": "5c2c9ecf798502d3e11374b51d11e9709db3deaa4a0e94a269453324fe989883",
    "manifest.json": "f771d5ae582478f6f345a82565e0a1c8f3fcc8ee44732843c05df2c853c1ce2e",
    "params.json": "147de917fa9cb6091c8ee620da924bb11abbd24237ceff629e0cab5135e56f67",
    "tables/rope_c.i32": "ea26d25813d6d78b2d468b5a6571b0b188f5caf263dc0bb36aa6d306ab300978",
    "tables/rope_s.i32": "b1cbcd6e576500b03f55ed7b3b3dc3a73e93095bf4a73133906bbecf03f0f52b",
    "tables/silu_ts.i32": "1ab2e392d82299c94204dadcee37380aff6d8b5a64e5b8e34295e661bea14a63",
    "tables/softmax_ex.i32": "e27fdb2a6e9b273138ee66a6729de3c2c186ac1e81d24e5514ad29c155221b63",
    "wh.i16": "12b699f64101c65970b1e4c1b44fac78602a2ef336d0b2b941f963ed4f6d33c0",
}


@pytest.fixture(scope="module")
def small_artifact(tmp_path_factory):
    return factory.build_small_artifact(tmp_path_factory.mktemp("writer-artifact"))


def test_artifact_id_is_unchanged(small_artifact):
    assert small_artifact.name == EXPECTED_ID


def test_every_file_sha256_is_unchanged(small_artifact):
    actual = {}
    for path in small_artifact.rglob("*"):
        if path.is_file():
            relative = path.relative_to(small_artifact).as_posix()
            actual[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert actual == EXPECTED_SHA256


def test_written_artifact_still_validates(small_artifact):
    assert (
        artifacts.validate_artifact(small_artifact, dims=factory.DIMS) == []
    )
