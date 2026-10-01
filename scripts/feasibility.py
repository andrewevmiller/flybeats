"""Can the main test show anything? A ceiling for the task, not a connectome: a small unrestricted model with the
model's own ear, trained on the full set's training songs and scored on its validation songs, which no network
trains on. If even this model barely beats a no-listening baseline on unseen songs, the real network and the
controls would all score near the floor and the main test could not tell them apart.

RULES, fixed before the run (30 Sep 2026; change them here before running, never after):
  Data       Slakh2100 redux 16k (the full set): training songs to train, the fixed validation clips
             (validation_clips.json, 4 per song, 270 songs) to score. Test songs are not touched.
  Ear        exactly the model's: 16 bands, 40-4,000 Hz, causal, normalised with the full set's band stats.
  Model      ear -> 2-layer GRU (128 units; causal, unlimited memory) -> a hit logit and a velocity per drum.
             About 160,000 parameters, about 4x the network's 41,000 dials.
  Training   the locked loss (answer key, drum weights, velocity term), optimiser (Adam 1e-3), schedule (cosine to
             zero), batch (8 clips of 16 s from different songs, with stem dropout) and gradient clipping. STEPS
             steps. Its own sampling seed, so it does not share batches with any network run.
  Score      the locked hit F1 (local peak >= 0.5, within 10% of the local beat), as train.py validates; the
             scorecard (Phase 4) does not exist yet. The weights at the last step are scored; nothing is chosen
             by validation score.
  Floor      the same final model on the same validation clips, but each clip hears another song's band (the
             clip 4 places on, so the next song's): what it scores from drum-like output not tied to this song.
             The same idea as the scorecard's floor (another song's drum part).
  Measure    gap = matched F1 minus floor F1, each averaged over kick, snare and closed hi-hat; a 95% interval
             from resampling validation songs (2,000 resamples, seed 0).
  Decision   gap >= 0.10 and its interval above 0:  the task can separate networks; go back to the setup
             gap <  0.05 or its interval reaches 0:  near the floor; the task needs rethinking first
             otherwise:                              unclear
  The script prints the outcome. It changes nothing.

Runs on the CPU (4 threads) so the GPU stays free. Writes reports/feasibility/ (log.jsonl, final.pt) and
reports/feasibility.json.
    .venv\\Scripts\\python.exe scripts\\feasibility.py [--trial]
--trial runs a few steps on a few validation clips to check the script, and writes nothing.
"""
import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import load_locked, load_paths, reports_dir  # noqa: E402
from flybeats.data.manifest import load_manifest, set_dir  # noqa: E402
from flybeats.data.sampler import ClipSampler, load_validation_clips  # noqa: E402
from flybeats.data.slakh import unpacked_root  # noqa: E402
from flybeats.model import ear as model_ear  # noqa: E402
from flybeats.model.loss import drum_weights, f1_from_counts, hit_counts, make_loss  # noqa: E402
from flybeats.model.training import make_schedule, to_tensors  # noqa: E402

STEPS = 3000
HIDDEN, LAYERS = 128, 2
SAMPLING_SEED, INIT_SEED = 314, 0
EVAL_EVERY = 500                          # for the log only; the decision uses the last step
DRUMS = ["kick", "snare", "hihat_closed"]
GAP_GO, GAP_NEAR = 0.10, 0.05
SHIFT = 4                                 # floor: clip i hears clip i + 4's band (validation clips are 4 per song)


class Ceiling(torch.nn.Module):
    def __init__(self, n_drums, mean, std, hit_bias):
        super().__init__()
        self.ear = model_ear.Bands(mean, std)
        self.gru = torch.nn.GRU(model_ear.N_BANDS, HIDDEN, LAYERS, batch_first=True)
        self.hit = torch.nn.Linear(HIDDEN, n_drums)
        self.vel = torch.nn.Linear(HIDDEN, n_drums)
        torch.nn.init.constant_(self.hit.bias, hit_bias)

    def from_bands(self, bands):
        h, _ = self.gru(bands)
        return {"hit_logits": self.hit(h), "vel": torch.sigmoid(self.vel(h))}

    def forward(self, audio):
        return self.from_bands(self.ear(audio))


class Prefetch(threading.Thread):
    """The sampler's next batches, made ahead in order on one thread (as in train.py)."""

    def __init__(self, sampler, size, ahead=2):
        super().__init__(daemon=True)
        self.sampler, self.size, self.ready = sampler, size, queue.Queue(maxsize=ahead)

    def run(self):
        while True:
            try:
                self.ready.put(self.sampler.batch(self.size))
            except BaseException as e:
                self.ready.put(e)
                return

    def next(self):
        item = self.ready.get()
        if isinstance(item, BaseException):
            raise item
        return item


@torch.no_grad()
def per_clip_counts(model, bands, clips, cfg, batch=32):
    """(clips, drums, 3) tp/fp/fn per clip, the locked hit F1's counts. bands: normalised ear output per clip."""
    frame_s = cfg["training_data"]["frame_ms"] / 1000
    fraction = cfg["scorecard"]["onset_tolerance"]["fraction_of_local_beat"]
    out = []
    for i in range(0, len(clips), batch):
        b = to_tensors(clips[i:i + batch], "cpu")
        logits = model.from_bands(torch.stack(bands[i:i + batch]))["hit_logits"]
        for j in range(logits.shape[0]):
            out.append(hit_counts(logits[j:j + 1], b["hits"][j:j + 1], b["mask"][j:j + 1], b["beats"][j:j + 1],
                                  frame_s, fraction, cfg["training"]["hit_threshold"]))
    return np.stack(out)


def three_drum_f1(counts, idx):
    return float(np.mean(f1_from_counts(counts.sum(0))[idx]))


ap = argparse.ArgumentParser()
ap.add_argument("--trial", action="store_true", help="a few steps on a few validation clips; writes nothing")
args = ap.parse_args()
torch.set_num_threads(4)
t0 = time.time()
cfg, paths = load_locked(), load_paths()
tc, frame_s = cfg["training"], cfg["training_data"]["frame_ms"] / 1000
steps, eval_every = (20, 10) if args.trial else (STEPS, EVAL_EVERY)
manifest, events, beats = load_manifest(paths["work_dir"], "full")
root = unpacked_root(paths["work_dir"], "full")
train = ClipSampler(root, manifest, events, beats, cfg, "train", seed=SAMPLING_SEED)
val = ClipSampler(root, manifest, events, beats, cfg, "validation")
pieces = cfg["kit"]["pieces"]
idx = [pieces.index(d) for d in DRUMS]
mean, std = model_ear.load_band_stats(set_dir(paths["work_dir"], "full") / "band_stats.json")
weights = drum_weights(manifest, pieces, frame_s, cfg["loss"]["max_drum_weight"])
loss_of = make_loss(cfg, weights)
torch.manual_seed(INIT_SEED)
model = Ceiling(len(pieces), mean, std, cfg["model"]["hit_bias_init"])
print(f"ceiling model: {sum(p.numel() for p in model.parameters()):,} parameters; {steps} steps on the CPU")

listed = load_validation_clips(set_dir(paths["work_dir"], "full") / "validation_clips.json")
if args.trial:
    listed = listed[:16]
val_clips, val_bands = [], []
for c in tqdm(listed, desc="reading validation clips", unit="clip"):
    clip = val.clip(c["song"], c["start"])
    with torch.no_grad():
        val_bands.append(model.ear(torch.from_numpy(clip["audio"])[None])[0])
    clip["audio"] = np.zeros(1, np.float32)              # the ear's output is kept instead
    val_clips.append(clip)
songs = [c["song"] for c in val_clips]
floor_bands = [val_bands[(i + SHIFT) % len(val_bands)] for i in range(len(val_bands))]
assert all(songs[i] != songs[(i + SHIFT) % len(songs)] for i in range(len(songs))), "floor clip from the same song"
print(f"{len(val_clips)} validation clips from {len(set(songs))} songs ({time.time() - t0:.0f} s)")

opt = torch.optim.Adam(model.parameters(), lr=tc["learning_rate"])
sched = make_schedule(opt, steps, cfg)
out_dir = reports_dir() / "feasibility"
if not args.trial:
    out_dir.mkdir(parents=True, exist_ok=True)
log = None if args.trial else open(out_dir / "log.jsonl", "w", encoding="utf-8")
prefetch = Prefetch(train, tc["batch_clips"])
prefetch.start()
bar = tqdm(range(1, steps + 1), unit="step", desc="training")
for step in bar:
    td = time.time()
    b = to_tensors(prefetch.next(), "cpu")
    data_s = time.time() - td
    out = model(b["audio"])
    loss, parts = loss_of(out, b["hits"], b["vel"], b["mask"])
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip_norm"]))
    opt.step()
    sched.step()
    rec = {"step": step, "loss": float(loss), **parts, "grad_norm": grad_norm, "data_s": round(data_s, 2),
           "elapsed_s": round(time.time() - t0, 1)}
    bar.set_postfix(loss=f"{float(loss):.4f}")
    if step % eval_every == 0 or step == steps:
        model.eval()
        c = per_clip_counts(model, val_bands, val_clips, cfg)
        rec["val_f1"] = dict(zip(pieces, np.round(f1_from_counts(c.sum(0)), 3).tolist()))
        model.train()
        bar.write(f"step {step}: loss {float(loss):.4f}  validation F1 {rec['val_f1']}")
    if log:
        log.write(json.dumps(rec) + "\n")
        log.flush()
if log:
    log.close()
    torch.save({"model": model.state_dict(), "step": steps}, out_dir / "final.pt")

model.eval()
matched = per_clip_counts(model, val_bands, val_clips, cfg)
floor = per_clip_counts(model, floor_bands, val_clips, cfg)
gap = three_drum_f1(matched, idx) - three_drum_f1(floor, idx)
song_names = sorted(set(songs))
by_song = {s: np.flatnonzero(np.array(songs) == s) for s in song_names}
rng = np.random.default_rng(0)
boot = []
for _ in range(2000):
    pick = np.concatenate([by_song[s] for s in rng.choice(song_names, len(song_names))])
    boot.append(three_drum_f1(matched[pick], idx) - three_drum_f1(floor[pick], idx))
lo, hi = np.percentile(boot, [2.5, 97.5])
outcome = ("the task can separate networks; go back to the setup" if gap >= GAP_GO and lo > 0 else
           "near the floor; the task needs rethinking first" if gap < GAP_NEAR or lo <= 0 else "unclear")
f1m, f1f = f1_from_counts(matched.sum(0)), f1_from_counts(floor.sum(0))
print(f"\n{'drum':<13} {'hears its song':>14} {'hears another':>14}")
for d, p in enumerate(pieces):
    print(f"{p:<13} {f1m[d]:>14.3f} {f1f[d]:>14.3f}")
print(f"\nkick, snare, closed hi-hat: {three_drum_f1(matched, idx):.3f} against floor {three_drum_f1(floor, idx):.3f}; "
      f"gap {gap:+.3f} (95% interval {lo:+.3f} to {hi:+.3f})")
print(f"decision rule (go if gap >= {GAP_GO} with interval above 0; near the floor if under {GAP_NEAR} or interval "
      f"reaches 0): {outcome}")
report = {"steps": steps, "parameters": sum(p.numel() for p in model.parameters()), "validation_clips": len(val_clips),
          "f1_matched": dict(zip(pieces, np.round(f1m, 3).tolist())),
          "f1_floor": dict(zip(pieces, np.round(f1f, 3).tolist())),
          "three_drum": {"matched": three_drum_f1(matched, idx), "floor": three_drum_f1(floor, idx), "gap": gap,
                         "interval_95": [float(lo), float(hi)]},
          "rule": {"drums": DRUMS, "go": GAP_GO, "near": GAP_NEAR}, "outcome": outcome, "version": cfg["version"]}
if args.trial:
    sys.exit(f"trial run: nothing written ({time.time() - t0:.0f} s)")
(reports_dir() / "feasibility.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"wrote {reports_dir() / 'feasibility.json'} ({time.time() - t0:.0f} s)")
