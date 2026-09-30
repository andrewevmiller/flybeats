"""Phase 2, steps 5-6: measure the data (reports/data_report.json) and run the gate checks."""
import numpy as np
import pandas as pd
import pretty_midi

from .slakh import list_songs


def sixteenth_grid(beat_times):
    b = np.asarray(beat_times)
    if len(b) < 2:
        return b
    steps = np.diff(b)[:, None] * (np.arange(4) / 4)[None, :]
    return np.concatenate([(b[:-1, None] + steps).ravel(), b[-1:]])


def grid_offsets_ms(events, beats):
    """Distance of each hit to the nearest 16th-note grid point, for hits inside the beat grid."""
    out = []
    for song, ev in events.groupby("song"):
        grid = sixteenth_grid(beats.time[beats.song == song].to_numpy())
        if len(grid) < 2:
            continue
        t = ev.time.to_numpy()
        t = t[(t >= grid[0]) & (t <= grid[-1])]
        j = np.clip(np.searchsorted(grid, t), 1, len(grid) - 1)
        out.append(np.minimum(np.abs(t - grid[j - 1]), np.abs(grid[j] - t)) * 1000)
    return np.concatenate(out) if out else np.zeros(0)


def build_report(manifest, events, beats, dropped, pieces, full=False):
    off = grid_offsets_ms(events, beats)
    total_kept = len(events)
    total_dropped = sum(dropped.values())
    total = total_kept + total_dropped
    changes = manifest.tempi.map(lambda t: len(set(np.round(t, 2))) > 1)
    r = {
        "songs": int(len(manifest)),
        "hits_on_16th_grid_within_1ms": round(float((off <= 1.0).mean()), 3) if len(off) else None,
        "grid_offset_ms": {"median": round(float(np.median(off)), 1), "p90": round(float(np.percentile(off, 90)), 1)}
        if len(off) else None,
        "songs_with_tempo_changes": f"{int(changes.sum())} of {len(manifest)}",
        "distinct_velocities": int(events.velocity.nunique()),
        "median_velocity": float(events.velocity.median()),
        "notes_total": int(total),
        "dropped_hits_share": round(total_dropped / total, 3) if total else None,
        "dropped_top": {pretty_midi.note_number_to_drum_name(int(n)) or str(n): int(c)
                        for n, c in dropped.most_common(6)},
        "pieces": {p: {"hits": int((events.piece == p).sum()),
                       "share_of_all_notes": round(float((events.piece == p).sum() / total), 3)} for p in pieces},
    }
    if full:
        per = manifest.groupby("split")
        r["songs_per_split"] = per.size().to_dict()
        r["hours_per_split"] = (per.duration.sum() / 3600).round(1).to_dict()
        n_in = manifest.input_stems.map(len)
        r["input_stems_per_song"] = {"min": int(n_in.min()), "median": float(n_in.median()), "max": int(n_in.max())}
        r["parts_with_midi_but_no_audio"] = int(manifest.midi_no_audio.map(len).sum())
    return r


def gate_checks(manifest, root, which, excluded_splits, stem_tolerance_s=0.01):
    """[(ok, name, detail)] for the checks that need only the manifest."""
    checks = []
    on_disk = {s: len(v) for s, v in list_songs(root, which).items() if s not in excluded_splits}
    in_manifest = manifest.groupby("split").size().to_dict()
    checks.append((on_disk == in_manifest, "song counts per split match the folders",
                   f"folders {on_disk}, manifest {in_manifest}"))
    checks.append((not set(manifest.split) & set(excluded_splits), "omitted songs excluded",
                   f"splits used: {sorted(set(manifest.split))}"))

    dup = manifest.song[manifest.song.duplicated()]
    checks.append((dup.empty, "no song id in two splits", f"{len(dup)} duplicated"))

    train_hashes = set(manifest.drum_hash[manifest.split == "train"])
    leaks = manifest[(manifest.split != "train") & manifest.drum_hash.isin(train_hashes)]
    checks.append((leaks.empty, "no drum part shared between training and validation/test",
                   "leaking: " + ", ".join(f"{r.song} ({r.split})" for r in leaks.itertuples()) if len(leaks) else
                   "none"))

    hit_cols = [c for c in manifest.columns if c.startswith("hits_")]
    bad = manifest[(manifest.drum_stems.map(len) < 1) | (manifest.input_stems.map(len) < 1)
                   | (manifest[hit_cols].sum(axis=1) < 1)]
    checks.append((bad.empty, "every song has drums, input audio and a kit hit",
                   f"failing: {list(bad.song)}" if len(bad) else "all songs"))

    spread = (manifest.stem_frames_max - manifest.stem_frames_min) / manifest.sample_rate
    worst = manifest.song[spread.idxmax()] if len(spread) else None
    checks.append((bool((spread <= stem_tolerance_s).all()), "stem lengths within a song agree",
                   f"largest spread {spread.max():.4f} s ({worst})"))
    return checks
