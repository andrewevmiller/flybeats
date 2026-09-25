"""Does the fly's timing survive a drum kit it has never heard?

Every GMD recording is one Roland TD-11 kit. E-GMD re-recorded the same
performances on 43 kits of a TD-17, and its MIDI is identical to GMD's: same
notes, same times, zero offset, same audio length. Checked on 24 Sep 2026 on
the validation and test files. So a window at the same clock time holds the
same bars in both corpora, and the kit's sound is the only difference.

This scores one checkpoint's timing on the same windows three ways:

* ``gmd``    -- GMD's own recording (the kit every model so far trained on);
* ``seen``   -- the E-GMD split whose kits a model trained on the E-GMD subset
  could have heard (``<split>`` in ``fetch_egmd_subset.py``'s info.csv);
* ``unseen`` -- the E-GMD kits held out of that subset's training rows
  (``<split>_unseen_kit``).

For a model trained only on GMD, both E-GMD columns are unfamiliar kits. The
seen/unseen split starts to matter once a model has trained on E-GMD.

The timing score is the project's own: ``metrics.onset_f_sweep`` per window.
Windows are averaged within each performance, then performances are averaged
with equal weight, so a long groove cut into eight windows does not outvote
seven short fills. (``train.evaluate`` has one window per clip, so there the two
weightings coincide.) It uses the loader's own audio and targets
(``GrooveDataset``), windows of ``data.seconds``, and a fresh state per window,
like ``train.evaluate``. It is reported two ways:

* at the checkpoint's ``best_threshold``, which is what an exported bundle
  plays at;
* at each corpus's own best threshold, which separates "the output got louder
  or quieter on this kit" (recalibration fixes that) from "it cannot find the
  hits".

Differences are paired: each performance's mean F on an E-GMD kit minus its
mean F on GMD, with a 95% range from resampling performances.

    python scripts/kit_check.py --checkpoint runs/velocity_probe_cpu/best.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CORPORA = ("gmd", "seen", "unseen")


def plan_windows(durations: dict[str, float], seconds: float, per_clip: int) -> list[float]:
    """Window starts for one performance: back to back from 0, at most ``per_clip``.

    ``durations`` holds its length in each corpus. The shortest one bounds the
    windows, so every corpus scores exactly the same passages. A performance
    shorter than one window still gets one, which the loader pads.
    """
    usable = min(durations.values())
    n = max(1, min(per_clip, int(usable // seconds)))
    return [k * seconds for k in range(n)]


def paired_ci(diffs: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of per-performance differences, and its 95% range over performances."""
    if len(diffs) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def verdict(mean: float, lo: float, hi: float) -> str:
    if np.isnan(mean):
        return "no data"
    if hi < 0:
        return "WORSE (confident)"
    if lo > 0:
        return "better (confident)"
    return "no clear difference"


def build_items(cfg: dict, classes: list[str], egmd_root: Path, split: str,
                per_clip: int):
    """One GrooveDataset per corpus, with rows expanded to one per window."""
    from dataset import build_dataset

    seconds = float(cfg["data"].get("seconds", 4.0))
    base = {k: v for k, v in cfg["data"].items() if k not in ("max_files",)}
    sets = {
        "gmd": build_dataset({**base, "max_files": None}, classes, split=split),
        "seen": build_dataset({**base, "root": str(egmd_root), "max_files": None},
                              classes, split=split),
        "unseen": build_dataset({**base, "root": str(egmd_root), "max_files": None},
                                classes, split=f"{split}_unseen_kit"),
    }
    if not (sets["gmd"].styles == sets["seen"].styles == sets["unseen"].styles):
        raise RuntimeError("style vocabularies differ between corpora; style ids would not "
                           "mean the same thing (re-run fetch_egmd_subset.py, which pads them)")
    by_id = {name: {r["id"]: r for r in ds.rows} for name, ds in sets.items()}
    common = sorted(set.intersection(*(set(v) for v in by_id.values())))
    plan = {pid: plan_windows({n: float(by_id[n][pid]["duration"]) for n in CORPORA},
                              seconds, per_clip) for pid in common}
    for name, ds in sets.items():
        rows, starts = [], []
        for pid in common:
            for s in plan[pid]:
                rows.append(by_id[name][pid])
                starts.append(s)
        ds.rows = rows
        sr = ds.sample_rate
        # The loader asks where a window starts; answer with the planned second,
        # identical across corpora, instead of its per-row-index seed.
        ds._window_start = (lambda st, sr_: lambda i, span:
                            min(int(round(st[i] * sr_)), max(span - 1, 0)))(starts, sr)
        ds.window_ids = [r["id"] for r in rows]
    return sets, common, plan


def score(model, ds, cfg, thresholds, batch: int = 16) -> dict:
    """Per-window F at every threshold, plus which performance and kit it was."""
    import torch
    from torch.utils.data import DataLoader
    from metrics import onset_f_sweep

    step_ms = cfg["audio"]["step_ms"]
    tol = cfg["eval"].get("tolerance_s", 0.05)
    f = {t: [] for t in thresholds}
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    with torch.no_grad():
        for wav, y, _vel, style, _tempo in loader:
            sid = style if model.genre is not None else None
            logits, _ = model(wav, style_id=sid)
            prob = torch.sigmoid(logits.float()).numpy()
            ref = y.numpy()
            steps = min(prob.shape[1], ref.shape[1])
            for i in range(prob.shape[0]):
                for t, v in onset_f_sweep(prob[i, :steps], ref[i, :steps], step_ms,
                                          thresholds, tol).items():
                    f[t].append(v)
    return {"f": {t: np.array(v) for t, v in f.items()},
            "ids": list(ds.window_ids), "kits": [r.get("kit_name", "TD-11") for r in ds.rows]}


def per_performance(ids, values) -> dict[str, float]:
    acc = defaultdict(list)
    for pid, v in zip(ids, values):
        acc[pid].append(v)
    return {pid: float(np.mean(v)) for pid, v in acc.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--egmd", type=Path, default=ROOT / "data" / "egmd" / "e-gmd-subset")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--windows-per-clip", type=int, default=8)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: runs/kit_check/<run name>/")
    a = ap.parse_args(argv)

    import torch
    torch.set_num_threads(a.threads)
    from build import build_model, get_subgraph
    from decoder import DrumKit

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model, _ = build_model(cfg, get_subgraph(cfg), n_styles=max(int(ck.get("n_styles", 1) or 1), 1))
    model.load_state_dict(ck["model"])
    model.eval()
    classes = list(DrumKit(ck["kit"]).classes)
    ship = float(ck.get("best_threshold") or cfg["eval"].get("threshold", 0.3))
    sweep = sorted(set(cfg["eval"].get("threshold_sweep",
                                       [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
                   | {ship})

    sets, common, plan = build_items(cfg, classes, a.egmd, a.split, a.windows_per_clip)
    n_win = sum(len(v) for v in plan.values())
    print(f"{a.checkpoint}: {len(common)} {a.split} performances in all three corpora, "
          f"{n_win} windows of {cfg['data'].get('seconds')} s each, threshold {ship}, "
          f"{a.threads} thread(s)", flush=True)

    t0 = time.time()
    res = {}
    for name in CORPORA:
        res[name] = score(model, sets[name], cfg, sweep)
        print(f"  {name:<7} scored ({(time.time() - t0) / 60:.1f} min)", flush=True)

    out = {"checkpoint": str(a.checkpoint), "split": a.split, "threshold": ship,
           "performances": len(common), "windows": n_win, "corpora": {}, "paired": {}}
    for name in CORPORA:
        # Performance-weighted, like the paired differences below, so the
        # columns and the difference between them always agree in sign. Window
        # weighting would let a two-minute groove's eight windows outvote
        # seven one-window fills.
        means = {f"{t:g}": float(np.mean(list(per_performance(res[name]["ids"],
                                                                res[name]["f"][t]).values())))
                 for t in sweep}
        best = max(means, key=means.get)
        out["corpora"][name] = {"f_at_threshold": means[f"{ship:g}"], "best_threshold": best,
                                "f_at_best": means[best], "curve": means}
    gmd_pp = per_performance(res["gmd"]["ids"], res["gmd"]["f"][ship])
    for name in ("seen", "unseen"):
        pp = per_performance(res[name]["ids"], res[name]["f"][ship])
        diffs = np.array([pp[pid] - gmd_pp[pid] for pid in common])
        m, lo, hi = paired_ci(diffs)
        out["paired"][name] = {"mean_diff": m, "lo": lo, "hi": hi, "verdict": verdict(m, lo, hi)}
    kit_perf = defaultdict(lambda: defaultdict(list))
    for kit, pid, v in zip(res["unseen"]["kits"], res["unseen"]["ids"], res["unseen"]["f"][ship]):
        kit_perf[kit][pid].append(v)
    out["unseen_by_kit"] = {
        k: {"f": float(np.mean([np.mean(v) for v in perfs.values()])),
            "vs_gmd": float(np.mean([np.mean(v) - gmd_pp[pid] for pid, v in perfs.items()])),
            "performances": len(perfs)}
        for k, perfs in sorted(kit_perf.items())}
    out["minutes"] = round((time.time() - t0) / 60, 1)

    c, p = out["corpora"], out["paired"]
    lines = [
        f"Does the fly keep time on drum kits it has never heard? ({a.checkpoint.parent.name})",
        f"The same {len(common)} {a.split} performances, cut into the same {n_win} two-second "
        f"passages, heard three ways. The notes are identical; only the drum sound changes.",
        "",
        f"TIMING SCORE (0-1, higher is better), at the threshold the model plays at ({ship}):",
        f"  GMD's own kit, the one it trained on:   {c['gmd']['f_at_threshold']:.3f}",
        f"  other kits (E-GMD '{a.split}'):          {c['seen']['f_at_threshold']:.3f}  "
        f"-> {p['seen']['verdict']} [{p['seen']['mean_diff']:+.3f}, 95% range "
        f"{p['seen']['lo']:+.3f} to {p['seen']['hi']:+.3f}]",
        f"  held-out kits:                          {c['unseen']['f_at_threshold']:.3f}  "
        f"-> {p['unseen']['verdict']} [{p['unseen']['mean_diff']:+.3f}, 95% range "
        f"{p['unseen']['lo']:+.3f} to {p['unseen']['hi']:+.3f}]",
        "",
        "IF EACH KIT GOT ITS OWN BEST THRESHOLD (is the loss just a volume shift?):",
    ] + [f"  {n:<7} {c[n]['f_at_best']:.3f} at threshold {c[n]['best_threshold']}"
         for n in CORPORA] + [
        "",
        "HELD-OUT KITS ONE BY ONE (timing score; change against the same performances on "
        "GMD's kit; how many performances):",
    ] + [f"  {k:<28} {v['f']:.3f}  [{v['vs_gmd']:+.3f}]  ({v['performances']})"
         for k, v in out["unseen_by_kit"].items()] + [
        "",
        f"({out['minutes']} min to measure)",
    ]
    text = "\n".join(lines) + "\n"
    dest = a.out or ROOT / "runs" / "kit_check" / a.checkpoint.parent.name
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "kit_check.json").write_text(json.dumps(out, indent=2))
    (dest / "report.txt").write_text(text, encoding="utf-8")
    print("\n" + text + f"wrote {dest / 'report.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
