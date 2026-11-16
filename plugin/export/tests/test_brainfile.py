import numpy as np
import pytest

from flybrain.brainfile import FORMAT_VERSION, read_brain, verify_brain, write_brain


def test_round_trip_and_hash(brain_small, tmp_path):
    b = brain_small[0]
    p = tmp_path / "copy.flybrain"
    digest = write_brain(p, b.meta, b.arrays)
    c = read_brain(p)
    assert c.sha256 == digest
    for k, v in b.arrays.items():
        assert np.array_equal(c[k], v) and c[k].dtype == v.dtype, k
    assert c.meta["format_version"] == FORMAT_VERSION
    assert c.meta["n_neurons"] == b.n


def test_corruption_is_refused(brain_small, tmp_path):
    b = brain_small[0]
    data = bytearray(open(b.path, "rb").read())
    data[len(data) // 2] ^= 0xFF
    bad = tmp_path / "bad.flybrain"
    bad.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="hash"):
        read_brain(bad)
    short = tmp_path / "short.flybrain"
    short.write_bytes(bytes(data[:100]))
    with pytest.raises(ValueError):
        read_brain(short)
    other = tmp_path / "other.flybrain"
    other.write_bytes(b"RIFF" + bytes(data[4:]))
    with pytest.raises(ValueError, match="not a"):
        read_brain(other)


def test_wrong_version_is_refused(brain_small, tmp_path):
    import hashlib
    data = bytearray(open(brain_small[0].path, "rb").read())
    data[8:12] = np.uint32(FORMAT_VERSION + 1).tobytes()
    body = bytes(data[:-32])
    p = tmp_path / "v2.flybrain"
    p.write_bytes(body + hashlib.sha256(body).digest())
    with pytest.raises(ValueError, match="version"):
        read_brain(p)


def test_verify_catches_shape_and_range_errors(brain_small):
    b = brain_small[0]
    a = dict(b.arrays)
    a["w_indices"] = a["w_indices"].copy()
    a["w_indices"][0] = b.n + 5
    from flybrain.brainfile import Brain
    with pytest.raises(ValueError, match="w_indices"):
        verify_brain(Brain(b.meta, a, b.sha256))
    a = dict(b.arrays)
    a["alpha"] = a["alpha"].copy()
    a["alpha"][3] = 0.0
    with pytest.raises(ValueError, match="alpha"):
        verify_brain(Brain(b.meta, a, b.sha256))
    m = dict(b.meta, capability="drumming_god")
    with pytest.raises(ValueError, match="capability"):
        verify_brain(Brain(m, b.arrays, b.sha256))


def test_missing_array_is_refused(brain_small, tmp_path):
    b = brain_small[0]
    a = dict(b.arrays)
    del a["mn_idx"]
    with pytest.raises(ValueError, match="mn_idx"):
        write_brain(tmp_path / "x.flybrain", b.meta, a)


def test_weights_are_sign_times_unsigned(brain_small):
    b = brain_small[0]
    pre = b["w_indices"]
    assert np.array_equal(b["w_values"], b["sign"][pre].astype(np.float32) * b["w_unsigned"])
    assert b.meta["silenced_count"] == int((b["alive"] == 0).sum())
    assert np.all(b["w_values"][b["sign"][pre] == 0] == 0)
