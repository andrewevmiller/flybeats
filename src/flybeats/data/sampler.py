"""Phase 2, step 4: the clip sampler. Hands out clips mixed on the fly from the stems, with drum targets.

A clip is a dict of numpy arrays:
  audio  (samples,)       float32, the sum of the kept input stems
  hits   (frames, pieces) uint8, 1 on the frame a hit starts
  vel    (frames, pieces) float32, velocity / 127 on hit frames, else 0
  mask   (frames,)        bool, False on the warm-up frames ignored by the loss
  beats  (n,)             beat times in seconds from the clip start (downbeats in `downbeats`)
"""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from .slakh import stem_audio


class ClipSampler:
    def __init__(self, root, manifest, events, beats, cfg, split, seed=None):
        td = cfg["training_data"]
        self.root = Path(root)
        self.split = split
        self.pieces = cfg["kit"]["pieces"]
        self.clip_s = td["clip_seconds"]
        self.frame_s = td["frame_ms"] / 1000
        self.n_frames = int(round(self.clip_s / self.frame_s))
        self.warmup = int(round(td["warmup_ignored_seconds"] / self.frame_s))
        self.drop_p = td["stem_dropout"]["p"]
        self.keep_min = td["stem_dropout"]["keep_at_least"]
        self.rng = np.random.default_rng(td["sampling_seed"] if seed is None else seed)
        self.songs = manifest[manifest.split == split].set_index("song")
        self.songs = self.songs[self.songs.duration > self.clip_s]
        self.events = {s: g for s, g in events[events.song.isin(self.songs.index)].groupby("song")}
        self.beats = {s: g for s, g in beats[beats.song.isin(self.songs.index)].groupby("song")}
        w = self.songs.duration.to_numpy()
        self.weights = w / w.sum()

    def batch(self, size=8):
        """size different songs, weighted by duration, each with a random window and stem dropout."""
        if size > len(self.songs):
            raise ValueError(f"only {len(self.songs)} songs in {self.split}, asked for {size}")
        picks = self.rng.choice(len(self.songs), size, replace=False, p=self.weights)
        clips = []
        for i in picks:
            song = self.songs.index[i]
            start = self.rng.uniform(0, self.songs.duration.iloc[i] - self.clip_s)
            clips.append(self.clip(song, start, dropout=True))
        return clips

    def clip(self, song, start, dropout=False):
        if song not in self.songs.index:
            raise ValueError(f"{song} is not in the {self.split} split")
        row = self.songs.loc[song]
        sr = int(row.sample_rate)
        stems = list(row.input_stems)
        if dropout:
            keep = self.rng.random(len(stems)) >= self.drop_p
            if keep.sum() < self.keep_min:
                keep[self.rng.choice(len(stems), self.keep_min, replace=False)] = True
            stems = [s for s, k in zip(stems, keep) if k]
        n = int(round(self.clip_s * sr))
        first = int(round(start * sr))
        audio = np.zeros(n, np.float32)
        for s in stems:
            x, _ = sf.read(str(stem_audio(self.root / row.path, s)), start=first, frames=n,
                           dtype="float32", always_2d=True)
            audio[:len(x)] += x.mean(axis=1)

        hits = np.zeros((self.n_frames, len(self.pieces)), np.uint8)
        vel = np.zeros((self.n_frames, len(self.pieces)), np.float32)
        ev = self.events.get(song)
        if ev is not None:
            ev = ev[(ev.time >= start) & (ev.time < start + self.clip_s)]
            f = ((ev.time.to_numpy() - start) / self.frame_s).astype(int).clip(0, self.n_frames - 1)
            p = ev.piece.map({q: j for j, q in enumerate(self.pieces)}).to_numpy()
            hits[f, p] = 1
            np.maximum.at(vel, (f, p), ev.velocity.to_numpy() / 127.0)
        mask = np.ones(self.n_frames, bool)
        mask[:self.warmup] = False

        bt = self.beats.get(song)
        bt = bt[(bt.time >= start) & (bt.time < start + self.clip_s)] if bt is not None else None
        return {"song": song, "start": float(start), "stems": stems, "audio": audio, "hits": hits, "vel": vel,
                "mask": mask,
                "beats": (bt.time.to_numpy() - start) if bt is not None else np.zeros(0),
                "downbeats": (bt.time[bt.downbeat].to_numpy() - start) if bt is not None else np.zeros(0)}


def make_validation_clips(sampler, path, per_song=4, seed=1234):
    """The fixed validation list: per_song windows per song, written once and never regenerated."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; validation clips are written once")
    rng = np.random.default_rng(seed)
    clips = [{"song": s, "start": round(float(rng.uniform(0, d - sampler.clip_s)), 3)}
             for s, d in sampler.songs.duration.items() for _ in range(per_song)]
    path.write_text(json.dumps(clips, indent=1), encoding="utf-8")
    return clips


def load_validation_clips(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
