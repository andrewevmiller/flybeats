"""Phase 2, step 3: process each song into a manifest row, its drum hits and its beat grid.

build_manifest() returns three tables:
  manifest: one row per song (see process_song for the columns)
  events:   song, time (s), piece, velocity (1-127)
  beats:    song, time (s), downbeat (bool)
"""
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pretty_midi
import soundfile as sf
import yaml

from .slakh import list_songs, stem_audio


def note_to_piece(kit_cfg):
    return {note: piece for piece, notes in kit_cfg["gm_notes"].items() for note in notes}


def drum_notes(song_dir, drum_stems):
    """All notes of the song's drum parts, sorted by (time, pitch)."""
    notes = []
    for stem in drum_stems:
        path = Path(song_dir) / "MIDI" / f"{stem}.mid"
        if path.exists():
            for inst in pretty_midi.PrettyMIDI(str(path)).instruments:
                notes += [(n.start, n.pitch, n.velocity) for n in inst.notes]
    return sorted(notes)


def process_song(song_dir, split, pieces_of, pieces):
    """(manifest row, events frame, beats frame, Counter of dropped GM notes)."""
    song_dir = Path(song_dir)
    song = song_dir.name
    # lstrip: train/Track00017's file starts with a stray space (" UUID: ..."), which YAML rejects
    meta = yaml.safe_load((song_dir / "metadata.yaml").read_text(encoding="utf-8").lstrip())
    stems = meta["stems"]
    drum_stems = sorted(s for s, m in stems.items() if m.get("is_drum"))
    # audio_rendered is unreliable (false even when audio exists): a stem is rendered if its file exists
    audio = {s: stem_audio(song_dir, s) for s in stems}
    input_stems = sorted(s for s in stems if s not in drum_stems and audio[s] is not None)
    midi_no_audio = sorted(s for s in stems if audio[s] is None and (song_dir / "MIDI" / f"{s}.mid").exists())

    frames = {s: sf.info(str(p)).frames for s, p in audio.items() if p is not None}
    rates = {sf.info(str(p)).samplerate for p in audio.values() if p is not None}
    if len(rates) != 1:
        raise ValueError(f"{song}: stems have sample rates {rates}")
    sr = rates.pop()
    duration = max(frames.values()) / sr if frames else 0.0

    notes = [(t, p, v) for t, p, v in drum_notes(song_dir, drum_stems) if t < duration]
    kept = [(t, pieces_of[p], v) for t, p, v in notes if p in pieces_of]
    dropped = Counter(p for _, p, _ in notes if p not in pieces_of)
    drum_hash = hashlib.sha1(repr([(round(t, 4), p, v) for t, p, v in notes]).encode()).hexdigest()

    pm = pretty_midi.PrettyMIDI(str(song_dir / "all_src.mid"))
    beats = pm.get_beats()
    downbeats = set(np.round(pm.get_downbeats(), 6))
    beats = beats[beats < duration]
    tempo_times, tempi = pm.get_tempo_changes()

    counts = Counter(p for _, p, _ in kept)
    row = {"song": song, "split": split, "sample_rate": sr, "duration": duration,
           "input_stems": input_stems, "drum_stems": drum_stems, "midi_no_audio": midi_no_audio,
           "stem_frames_min": min(frames.values()) if frames else 0,
           "stem_frames_max": max(frames.values()) if frames else 0,
           "tempo_change_times": [float(x) for x in tempo_times], "tempi": [float(x) for x in tempi],
           "n_beats": int(len(beats)), "dropped_hits": int(sum(dropped.values())), "drum_hash": drum_hash,
           **{f"hits_{p}": int(counts.get(p, 0)) for p in pieces}}
    events = pd.DataFrame({"song": song, "time": [t for t, _, _ in kept],
                           "piece": [p for _, p, _ in kept], "velocity": [v for _, _, v in kept]})
    beat_df = pd.DataFrame({"song": song, "time": beats, "downbeat": [round(b, 6) in downbeats for b in beats]})
    return row, events, beat_df, dropped


def read_excluded(path):
    """Song ids moved out of every split after a leak check (one per line, # comments)."""
    if not Path(path).exists():
        return set()
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return {ln.split("#")[0].strip() for ln in lines if ln.split("#")[0].strip()}


def build_manifest(root, which, cfg, excluded=(), progress=None):
    pieces = cfg["kit"]["pieces"]
    pieces_of = note_to_piece(cfg["kit"])
    skip_splits = set(cfg["data"]["slakh"]["exclude"])
    rows, events, beats, dropped = [], [], [], Counter()
    for split, songs in list_songs(root, which).items():
        if split in skip_splits:
            continue
        for song_dir in songs:
            if song_dir.name in excluded:
                continue
            row, ev, bt, dr = process_song(song_dir, split, pieces_of, pieces)
            row["path"] = song_dir.relative_to(root).as_posix()
            rows.append(row); events.append(ev); beats.append(bt); dropped.update(dr)
            if progress:
                progress(song_dir.name)
    manifest = pd.DataFrame(rows).sort_values(["split", "song"]).reset_index(drop=True)
    events = pd.concat(events, ignore_index=True).sort_values(["song", "time", "piece"]).reset_index(drop=True)
    events["velocity"] = events.velocity.astype(np.uint8)
    beats = pd.concat(beats, ignore_index=True)
    return manifest, events, beats, dropped


def set_dir(work_dir, which):
    d = Path(work_dir) / "data" / which
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_manifest(work_dir, which):
    d = Path(work_dir) / "data" / which
    if not (d / "manifest.parquet").exists():
        raise FileNotFoundError(f"no manifest in {d}: run scripts/build_manifest.py --set {which} first")
    return (pd.read_parquet(d / "manifest.parquet"), pd.read_parquet(d / "events.parquet"),
            pd.read_parquet(d / "beats.parquet"))
