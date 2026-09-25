"""The truncated E-GMD fetch, pinned where a mistake would be silent.

``fetch_egmd_subset.py`` rewrites the corpus on the way in: it picks rows,
resamples audio and writes its own info.csv. Each of these can go wrong without
anything failing. A held-out kit could leak into train, a resampler could
differ from the loader's, or the style ids could shift. Any of those would
change what a model learns while the numbers still look like numbers.
"""
import csv
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from fetch_egmd_subset import (  # noqa: E402
    GMD_STYLES, convert_wav, entry_bytes, select_rows, write_bytes, write_flac,
    write_info,
)

KITS = [f"kit{i:02d}" for i in range(10)]
HOLDOUT = ["kit03", "kit07"]


def fake_rows(per_split=6):
    rows = []
    for split in ("train", "validation", "test"):
        for p in range(per_split):
            pid = f"drummer{p % 3 + 1}/{split}/{p}"
            for k in KITS:
                name = f"{pid}_{k}"
                rows.append({
                    "drummer": f"drummer{p % 3 + 1}", "session": f"{split}", "id": pid,
                    "style": "funk/groove1", "bpm": "100", "beat_type": "beat",
                    "time_signature": "4-4", "duration": "5.0", "split": split,
                    "midi_filename": name + ".midi", "audio_filename": name + ".wav",
                    "kit_name": k,
                })
    return rows


def test_train_never_hears_a_held_out_kit():
    picked = select_rows(fake_rows(), None, HOLDOUT, 0, ("train", "validation", "test"))
    train_kits = {r["kit_name"] for r in picked if r["split"] == "train"}
    assert train_kits == set(KITS) - set(HOLDOUT)
    for split in ("validation", "test"):
        assert {r["kit_name"] for r in picked if r["split"] == split} <= train_kits
        assert {r["kit_name"] for r in picked
                if r["split"] == f"{split}_unseen_kit"} <= set(HOLDOUT)


def test_more_kits_is_a_superset_so_the_pool_grows_in_place():
    rows = fake_rows()
    key = lambda picked: {(r["id"], r["kit_name"]) for r in picked if r["split"] == "train"}
    small = key(select_rows(rows, 2, HOLDOUT, 0, ("train",)))
    large = key(select_rows(rows, 5, HOLDOUT, 0, ("train",)))
    assert len(small) == 6 * 2 and len(large) == 6 * 5
    assert small < large


def test_every_performance_is_kept_and_kits_vary_across_them():
    picked = select_rows(fake_rows(), 1, HOLDOUT, 0, ("train", "validation", "test"))
    train = [r for r in picked if r["split"] == "train"]
    assert len({r["id"] for r in train}) == 6
    assert len({r["kit_name"] for r in train}) > 1
    for split in ("validation", "test"):
        assert sum(r["split"] == split for r in picked) == 6
        assert sum(r["split"] == f"{split}_unseen_kit" for r in picked) == 6


def test_selection_is_seeded():
    rows = fake_rows()
    a = select_rows(rows, 3, HOLDOUT, 0, ("train",))
    b = select_rows(rows, 3, HOLDOUT, 0, ("train",))
    c = select_rows(rows, 3, HOLDOUT, 1, ("train",))
    assert a == b
    assert [r["kit_name"] for r in a] != [r["kit_name"] for r in c]


def test_unknown_holdout_kit_is_refused():
    with pytest.raises(ValueError):
        select_rows(fake_rows(), 1, ["not a kit"], 0, ("train",))


def _zip_with(tmp_path, payload: bytes):
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("e-gmd-v1.0.0/first.txt", b"x" * 100, compress_type=zipfile.ZIP_STORED)
        z.writestr("e-gmd-v1.0.0/deflated.wav", payload, compress_type=zipfile.ZIP_DEFLATED)
        z.writestr("e-gmd-v1.0.0/stored.mid", payload[:999], compress_type=zipfile.ZIP_STORED)
    return path


def test_entry_bytes_reads_members_one_at_a_time(tmp_path):
    payload = np.random.default_rng(0).integers(0, 8, 50_000, dtype=np.uint8).tobytes()
    path = _zip_with(tmp_path, payload)
    blob = path.read_bytes()
    read = lambda start, n: blob[start:start + min(n, len(blob) - start)]
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            assert entry_bytes(read, info) == z.read(info)


def test_a_corrupted_transfer_is_caught_not_stored(tmp_path):
    payload = bytes(range(256)) * 200
    path = _zip_with(tmp_path, payload)
    blob = bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as z:
        info = z.getinfo("e-gmd-v1.0.0/stored.mid")
    blob[info.header_offset + 30 + len(info.filename) + 10] ^= 0xFF
    read = lambda start, n: bytes(blob[start:start + min(n, len(blob) - start)])
    with pytest.raises(ValueError, match="CRC"):
        entry_bytes(read, info)


def _mini_corpus(root: Path, wav: bytes, midi: bytes, audio_name: str, row: dict):
    root.mkdir()
    (root / audio_name).write_bytes(wav)
    (root / "a.mid").write_bytes(midi)
    with (root / "info.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)


def test_the_loader_sees_the_same_audio_from_the_converted_file(tmp_path):
    import pretty_midi
    import soundfile as sf
    from dataset import GrooveDataset

    rng = np.random.default_rng(0)
    sr = 44_100
    audio = (0.05 * rng.standard_normal(5 * sr)).astype(np.float32)
    for t in (0.3, 1.1, 2.2, 3.4, 4.1):
        audio[int(t * sr):int(t * sr) + 400] += 0.8 * np.hanning(400)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    wav = buf.getvalue()

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for t, pitch, vel in ((0.3, 36, 100), (1.1, 38, 60), (2.2, 42, 90),
                          (3.4, 36, 30), (4.1, 38, 127)):
        inst.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=t, end=t + 0.05))
    pm.instruments.append(inst)
    mbuf = io.BytesIO()
    pm.write(mbuf)
    midi = mbuf.getvalue()

    row = {"drummer": "d1", "session": "s", "id": "d1/s/1", "style": "funk/groove1",
           "bpm": "100", "beat_type": "beat", "time_signature": "4-4",
           "midi_filename": "a.mid", "audio_filename": "a.wav", "duration": "5.0",
           "split": "validation", "kit_name": "kit00"}
    _mini_corpus(tmp_path / "orig", wav, midi, "a.wav", row)

    conv = tmp_path / "conv"
    write_flac(conv / "a.flac", convert_wav(wav), 22_050)
    write_bytes(conv / "a.mid", midi)
    write_info(conv, [{**row, "audio_filename": "a.flac"}], seed=0)

    classes = ["kick", "snare", "hat_closed"]
    kw = dict(classes=classes, split="validation", seconds=2.0, random_windows=False)
    a = GrooveDataset(tmp_path / "orig", **kw)[0]
    b = GrooveDataset(conv, **kw)[0]
    assert a[0].shape == b[0].shape
    np.testing.assert_allclose(a[0].numpy(), b[0].numpy(), atol=1e-4)
    np.testing.assert_array_equal(a[1].numpy(), b[1].numpy())
    np.testing.assert_array_equal(a[2].numpy(), b[2].numpy())


def test_style_ids_match_gmd_even_without_highlife(tmp_path):
    from dataset import GrooveDataset

    row = {"drummer": "d1", "session": "s", "id": "d1/s/1", "style": "soul/groove3",
           "bpm": "100", "beat_type": "beat", "time_signature": "4-4",
           "midi_filename": "a.mid", "audio_filename": "a.flac", "duration": "1.0",
           "split": "train", "kit_name": "kit00"}
    write_flac(tmp_path / "a.flac", np.zeros(22_050, np.float32), 22_050)
    write_bytes(tmp_path / "a.mid", b"")
    write_info(tmp_path, [row], seed=0)
    # Selected but not yet fetched, as mid-download: its style must not count
    # as present, or it would get no placeholder and drop out of the vocabulary.
    missing = {**row, "style": "jazz/groove1", "audio_filename": "b.flac",
               "midi_filename": "b.mid"}
    write_info(tmp_path, [row, missing], seed=0)
    ds = GrooveDataset(tmp_path, ["kick"], split="train", seconds=0.5)
    assert ds.styles == GMD_STYLES
    assert len(ds) == 1              # the placeholders never become training rows


def test_gmd_styles_constant_matches_the_corpus():
    info = ROOT / "data" / "egmd" / "groove" / "info.csv"
    if not info.exists():
        pytest.skip("GMD corpus not on disk")
    rows = list(csv.DictReader(info.open(newline="")))
    assert sorted({r["style"].split("/")[0] for r in rows}) == GMD_STYLES
