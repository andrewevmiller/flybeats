"""Phase 2 gate: every check the manifest, the sampler and the leak test can answer, plus 3 spot-check
clips (drum targets as clicks over the input) for you to listen to. Exit code = number of failures.

    .venv\\Scripts\\python.exe scripts\\check_data_gate.py --set baby
Run on BabySlakh, then again on the full set. Delete the archive parts only after both pass.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import ROOT, load_locked, load_paths, reports_dir  # noqa: E402
from flybeats.data.manifest import load_manifest, read_excluded, set_dir  # noqa: E402
from flybeats.data.report import gate_checks  # noqa: E402
from flybeats.data.sampler import ClipSampler  # noqa: E402
from flybeats.data.slakh import unpacked_root  # noqa: E402
from tqdm import tqdm  # noqa: E402

CLICK_HZ = {"kick": 120, "hihat_pedal": 3000, "snare": 900, "hihat_closed": 6000,
            "hihat_open": 5000, "tom": 300, "crash": 7000, "ride": 4000}

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["baby", "full"], required=True)
ap.add_argument("--batches", type=int, default=200)
args = ap.parse_args()
cfg, paths = load_locked(), load_paths()
root = unpacked_root(paths["work_dir"], args.set)
manifest, events, beats = load_manifest(paths["work_dir"], args.set)
excluded = read_excluded(ROOT / "config" / "excluded_songs.txt")
checks = gate_checks(manifest, root, args.set, cfg["data"]["slakh"]["exclude"], excluded)

# Sampler: 8 different training songs per batch; never a song from another split
train = ClipSampler(root, manifest, events, beats, cfg, "train", seed=99)
seen_ok = True
for _ in tqdm(range(args.batches), unit="batch", desc="sampler batches"):
    songs = [c["song"] for c in train.batch()]
    seen_ok &= len(set(songs)) == 8 and all(manifest.split[manifest.song == s].iloc[0] == "train" for s in songs)
checks.append((seen_ok, "each batch holds 8 different training songs", f"{args.batches} batches"))
test_song = manifest.song[manifest.split == "test"].iloc[0]
try:
    train.clip(test_song, 0.0)
    refused = False
except ValueError:
    refused = True
checks.append((refused, "a test song cannot be sampled for training", test_song))

rpath = reports_dir() / f"data_report_{args.set}.json"
checks.append((rpath.exists(), "data report written", str(rpath)))

# Listening spot-check
out = set_dir(paths["work_dir"], args.set) / "spotcheck"
out.mkdir(exist_ok=True)
pieces = cfg["kit"]["pieces"]
frame_s = cfg["training_data"]["frame_ms"] / 1000
for k, c in enumerate(train.batch()[:3], 1):
    sr = int(manifest.sample_rate[manifest.song == c["song"]].iloc[0])
    mix = c["audio"] / (np.abs(c["audio"]).max() + 1e-9) * 0.5
    t = np.arange(int(0.03 * sr)) / sr
    for f, p in zip(*np.nonzero(c["hits"])):
        blip = 0.4 * c["vel"][f, p] * np.sin(2 * np.pi * CLICK_HZ[pieces[p]] * t) * np.exp(-t / 0.008)
        i = int(f * frame_s * sr)
        mix[i:i + len(blip)] += blip[:len(mix) - i]
    sf.write(out / f"clip{k}_{c['song']}_{c['start']:.0f}s.wav", mix, sr)

fails = 0
for ok, name, detail in checks:
    fails += not ok
    print(f"{'OK  ' if ok else 'FAIL'}  {name:<58} {detail}")
print(f"\nListen to the 3 clips in {out}: the clicks should land on the music's beats.")
print("Then read the data report. Tick both boxes by hand; this script cannot.")
if any(not ok and name.startswith("no drum part") for ok, name, _ in checks):
    print("Leak found: add those songs to config/excluded_songs.txt, log it in config/CHANGELOG.md, rebuild.")
sys.exit(fails)
