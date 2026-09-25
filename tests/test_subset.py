"""Which clips a capped run trains on.

``max_files`` used to mean "the first N usable rows of info.csv", and GMD's
info.csv is ordered by drummer. So ``max_files: 256`` -- every velocity_*_cpu
config -- was 256 clips of drummer1, 227 of them fills, out of a training split
that covers nine drummers. Nothing in a config or a log said so.

``data.subset: stratified`` draws across drummers instead. ``first`` stays the
default, because every Phase A' arm and checkpoint so far was trained on the
first-N rows, and a comparison between arms is only a comparison if they all
saw the same clips.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build import load_config  # noqa: E402
from dataset import GrooveDataset, build_dataset, subset_rows  # noqa: E402
from decoder import DrumKit  # noqa: E402

#: GMD's shape in miniature: grouped by drummer, one drummer far larger than
#: the rest and mostly fills, one drummer with a single clip.
DRUMMERS = {"drummer1": (4, 16), "drummer2": (6, 3), "drummer3": (3, 0), "drummer4": (1, 0)}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A fake info.csv in drummer order, plus one row whose audio is missing.

    Construction only checks that the audio exists, so empty files do.
    """
    pytest.importorskip("soundfile")
    root = tmp_path_factory.mktemp("groove")
    rows = []
    for drummer, (beats, fills) in DRUMMERS.items():
        for k, beat_type in enumerate(["beat"] * beats + ["fill"] * fills):
            name = f"{drummer}_{k}_{beat_type}"
            rows.append({"drummer": drummer, "id": name, "style": "funk/groove1",
                         "bpm": "120", "beat_type": beat_type,
                         "midi_filename": f"{name}.mid", "audio_filename": f"{name}.wav",
                         "split": "train"})
            (root / f"{name}.wav").touch()
    # indexed but not shipped, as 51 of GMD's train rows are: usable-row
    # filtering must happen before the cap, under both subsets
    rows.insert(1, {**rows[0], "id": "missing", "audio_filename": "missing.wav"})
    with (root / "info.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return root


def _usable(corpus):
    rows = list(csv.DictReader((corpus / "info.csv").open()))
    return [r for r in rows if r["id"] != "missing"]


def _ids(corpus, **kw):
    ds = GrooveDataset(corpus, DrumKit.from_tier("3piece").classes, split="train", **kw)
    return [r["id"] for r in ds.rows]


# --- first: exactly what it always was -------------------------------------

@pytest.mark.parametrize("n", [None, 1, 5, 12, 33, 1000])
def test_first_is_the_leading_usable_rows_in_csv_order(corpus, n):
    want = [r["id"] for r in _usable(corpus)][:n]
    assert _ids(corpus, max_files=n) == want                    # the default
    assert _ids(corpus, max_files=n, subset="first") == want
    assert _ids(corpus, max_files=n, subset="first", subset_seed=7) == want


def test_a_config_without_the_key_is_unchanged(corpus):
    """build_dataset is how every run gets its data; no key must mean `first`."""
    cfg = {"root": str(corpus), "synthetic": False, "max_files": 12}
    ds = build_dataset(cfg, DrumKit.from_tier("3piece").classes, "train")
    assert [r["id"] for r in ds.rows] == [r["id"] for r in _usable(corpus)][:12]
    assert {r["drummer"] for r in ds.rows} == {"drummer1"}   # the problem, pinned


def test_no_shipped_config_opts_in():
    """Existing configs and checkpoints stay reproducible: none of them may
    resolve to anything but `first` until a config deliberately asks."""
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        data = load_config(path).get("data", {})
        assert data.get("subset", "first") == "first", f"{path.name} changed its corpus"


# --- stratified ------------------------------------------------------------

def test_stratified_is_deterministic_for_a_seed(corpus):
    a = _ids(corpus, max_files=12, subset="stratified", subset_seed=3)
    b = _ids(corpus, max_files=12, subset="stratified", subset_seed=3)
    assert a == b
    others = {tuple(_ids(corpus, max_files=12, subset="stratified", subset_seed=s))
              for s in range(5)}
    assert len(others) > 1, "the seed does not reach the draw"


@pytest.mark.parametrize("seed", range(5))
def test_stratified_covers_every_drummer_when_n_allows(corpus, seed):
    # exactly one slot per drummer: each gets one, including the one-clip drummer
    got = Counter(r["drummer"] for r in subset_rows(_usable(corpus), 4, "stratified", seed))
    assert got == {d: 1 for d in DRUMMERS}

    # a larger draw still reaches every drummer, and both beat and fill for
    # every drummer that has both
    rows = subset_rows(_usable(corpus), 12, "stratified", seed)
    assert len(rows) == 12
    assert {r["drummer"] for r in rows} == set(DRUMMERS)
    for d, (beats, fills) in DRUMMERS.items():
        types = {r["beat_type"] for r in rows if r["drummer"] == d}
        assert types == {t for t, k in (("beat", beats), ("fill", fills)) if k}, d


def test_stratified_follows_the_corpus_proportions(corpus):
    """A scaled-down copy of the split, not an equal share per drummer."""
    got = Counter(r["drummer"] for r in subset_rows(_usable(corpus), 17, "stratified", 0))
    # 17 of 20:9:3:1. One each first, then the other 13 split 19:8:2:0 by
    # largest remainder: 13*19/29 = 8.52, 13*8/29 = 3.59, 13*2/29 = 0.90 ->
    # 8, 3, 0 plus the two largest remainders, drummer3's and drummer2's.
    assert got == {"drummer1": 1 + 8, "drummer2": 1 + 3 + 1, "drummer3": 1 + 0 + 1,
                   "drummer4": 1}


def test_stratified_keeps_csv_order_and_no_duplicates(corpus):
    order = {r["id"]: i for i, r in enumerate(_usable(corpus))}
    ids = _ids(corpus, max_files=20, subset="stratified", subset_seed=1)
    assert len(set(ids)) == 20
    assert ids == sorted(ids, key=order.__getitem__)
    assert "missing" not in ids


def test_stratified_at_or_above_the_row_count_is_first(corpus):
    """So validation, whose 120 rows sit under a 256 cap, is untouched."""
    everything = [r["id"] for r in _usable(corpus)]
    for n in (None, len(everything), 1000):
        assert _ids(corpus, max_files=n, subset="stratified", subset_seed=9) == everything


def test_an_unknown_subset_is_refused(corpus):
    with pytest.raises(ValueError, match="subset"):
        _ids(corpus, max_files=4, subset="random")
