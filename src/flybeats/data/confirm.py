"""v0.15 confirmation sets A and B (locked.yaml data.confirmation; plan section 9).

Two sealed sets of about 150 songs each, carved from the training split. Chosen before any list was generated:
  salt   flybeats-v0.15-confirm
  unit   a drum_hash cluster (every song sharing a drum part moves together)
  rank   clusters by SHA-256 of salt + drum_hash, ascending; fill A, then B
  size   add clusters until a set holds at least 150 songs
  clips  4 windows of 16 s per song, numpy seed 20261115, drawn once per set in sorted song order

Nothing here trains or scores anything. The manifest split becomes 'confirm_a' or 'confirm_b' for the sealed songs,
so the sampler (which reads split == 'train') never draws them, and build_manifest re-applies the seal on rebuild.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SALT = "flybeats-v0.15-confirm"
SEED = 20261115
MIN_SONGS = 150
CLIPS_PER_SONG = 4
SETS = ("confirm_a", "confirm_b")


def read_list(path):
    """Song ids from a list file (one per line, '#' starts a comment). Missing file -> empty."""
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        s = ln.split("#")[0].strip()
        if s:
            out.append(s)
    return out


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rank_key(drum_hash, salt=SALT):
    return hashlib.sha256((salt + drum_hash).encode("utf-8")).hexdigest()


def select(manifest, excluded_songs, salt=SALT, min_songs=MIN_SONGS):
    """Return (A, B): two lists of song ids, from the train split, cluster-closed and disjoint.

    excluded_songs: ids that may not be sealed (BabySlakh songs, songs in the 12 Nov check draw).
    A cluster is eligible only if every one of its train songs is eligible. Clusters are ranked by
    SHA-256(salt + drum_hash); A takes clusters until it holds >= min_songs, then B does the same.
    """
    excluded = set(excluded_songs)
    train = manifest[manifest.split == "train"]
    clusters = {}
    for song, h in zip(train.song, train.drum_hash):
        clusters.setdefault(h, []).append(song)
    eligible = {h: sorted(members) for h, members in clusters.items()
                if all(s not in excluded for s in members)}
    order = sorted(eligible, key=lambda h: rank_key(h, salt))
    a, b, i = [], [], 0
    for target in (a, b):          # one pass down the ranking: A takes clusters first, then B continues
        while i < len(order) and len(target) < min_songs:
            target += eligible[order[i]]
            i += 1
    return sorted(a), sorted(b)


def draw_clips(songs, durations, clip_s, seed=SEED, per_song=CLIPS_PER_SONG):
    """Fixed windows for one set: per_song windows of clip_s seconds per song, sorted song order, one generator."""
    rng = np.random.default_rng(seed)
    out = []
    for s in sorted(songs):
        d = float(durations[s])
        for _ in range(per_song):
            out.append({"song": s, "start": round(float(rng.uniform(0, d - clip_s)), 3)})
    return out


def apply_confirmation(manifest, a_songs, b_songs):
    """Relabel the sealed songs in the manifest. Idempotent. Refuses any sealed song not currently in train."""
    manifest = manifest.copy()
    for name, songs in (("confirm_a", a_songs), ("confirm_b", b_songs)):
        if not songs:
            continue
        ids = manifest.song.isin(songs)
        found = set(manifest.song[ids])
        missing = sorted(set(songs) - found)
        if missing:
            raise ValueError(f"{name}: songs not in the manifest: {missing[:5]}")
        bad = manifest[ids & (manifest.split != "train") & (manifest.split != name)]
        if len(bad):
            raise ValueError(f"{name}: sealed songs already in another split: {list(bad.song)[:5]}")
        manifest.loc[ids, "split"] = name
    return manifest


def confirmation_gate_checks(manifest):
    """[(ok, name, detail)] for the data gate: the seal holds in the manifest as it is now."""
    checks = []
    splits = set(manifest.split)
    a = set(manifest.song[manifest.split == "confirm_a"])
    b = set(manifest.song[manifest.split == "confirm_b"])
    checks.append((not (a & b), "confirm_a and confirm_b are disjoint", f"overlap {len(a & b)}"))
    train_hashes = set(manifest.drum_hash[manifest.split == "train"])
    sealed = manifest[manifest.split.isin(SETS)]
    leaks = sealed[sealed.drum_hash.isin(train_hashes)]
    checks.append((leaks.empty, "no drum part shared between training and a confirmation set",
                   "leaking: " + ", ".join(leaks.song.head(5)) if len(leaks) else "none"))
    other = manifest[manifest.split.isin(["validation", "test"]) & manifest.drum_hash.isin(set(sealed.drum_hash))]
    checks.append((other.empty, "no drum part shared between a confirmation set and validation/test",
                   f"{len(other)} shared" if len(other) else "none"))
    checks.append((len(a) == 0 or len(a) >= MIN_SONGS, "confirm_a holds at least 150 songs" if a else
                   "confirm_a not sealed yet", f"{len(a)} songs"))
    checks.append((len(b) == 0 or len(b) >= MIN_SONGS, "confirm_b holds at least 150 songs" if b else
                   "confirm_b not sealed yet", f"{len(b)} songs"))
    checks.append((splits <= {"train", "validation", "test", *SETS}, "only known splits in the manifest",
                   ", ".join(sorted(splits))))
    return checks


LEDGER_NAME = "confirmatory_ledger.jsonl"


def read_ledger(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def open_confirmation(set_name, run_name, code_hash, tag, ledger_path, known_tags=None, pytest_result=None):
    """Guard for any scoring run on a confirmation set. Appends a ledger line BEFORE the caller scores.

    Refuses a second opening of the same set. Refuses confirm_b unless `tag` is a tag that no earlier ledger line
    used (a new tag since A was opened) and, when known_tags is given, is in that list.
    Returns the ledger entry written.
    """
    if set_name not in SETS:
        raise ValueError(f"unknown confirmation set {set_name!r}")
    entries = read_ledger(ledger_path)
    if any(e["set"] == set_name for e in entries):
        raise PermissionError(f"{set_name} was already opened: confirmation sets are scored once")
    if set_name == "confirm_b":
        if not any(e["set"] == "confirm_a" for e in entries):
            raise PermissionError("confirm_b is the reserve: open confirm_a first")
        used = {e["tag"] for e in entries}
        if tag in used:
            raise PermissionError(f"confirm_b needs a new tag since confirm_a was opened; {tag} was used before")
        if known_tags is not None and tag not in known_tags:
            raise PermissionError(f"tag {tag} does not exist")
    entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "set": set_name,
             "run": run_name, "code_hash": code_hash, "tag": tag}
    if pytest_result is not None:
        entry["pytest"] = pytest_result
    p = Path(ledger_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry
