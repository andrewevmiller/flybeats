"""Measured feel. PLAN.md is explicit that there is no hand-authored swing
parameter -- the model's timing is a finding, not a setting.

Everything here is descriptive: per class, against the metric grid, report mean
onset deviation, swing ratio, and the hat-vs-kick offset. "The model rushes
closed hats by 8 ms" is a result. A swing slider would be a guess.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metrics import match_onsets, peak_pick, swing_ratio


@dataclass
class FeelReport:
    per_class: dict[str, dict]
    pairwise_offset_ms: dict[str, float]
    overall: dict

    def to_text(self) -> str:
        lines = ["Measured feel (positive = late / behind the beat)", ""]
        head = f"{'class':<14}{'n':>6}{'grid dev ms':>13}{'spread ms':>11}{'swing':>8}{'vs ref ms':>11}"
        lines += [head, "-" * len(head)]
        for name, r in self.per_class.items():
            lines.append(
                f"{name:<14}{r['n']:>6}{r['grid_dev_ms']:>13.1f}{r['grid_spread_ms']:>11.1f}"
                f"{r['swing_ratio']:>8.2f}{r['ref_dev_ms']:>11.1f}")
        if self.pairwise_offset_ms:
            lines += ["", "Pairwise timing offsets (ms, first relative to second)"]
            for k, v in self.pairwise_offset_ms.items():
                lines.append(f"  {k:<26}{v:>+8.1f}")
        lines += ["", f"overall grid deviation {self.overall['grid_dev_ms']:+.1f} ms "
                      f"(spread {self.overall['grid_spread_ms']:.1f} ms)"]
        return "\n".join(lines)


def _grid_deviation(times: np.ndarray, tempo_bpm: float, subdiv: int = 4) -> np.ndarray:
    """Signed distance from each onset to the nearest grid line, in seconds."""
    if tempo_bpm <= 0 or len(times) == 0:
        return np.array([])
    grid = 60.0 / tempo_bpm / subdiv
    return times - np.round(times / grid) * grid


def analyse(pred_act: np.ndarray, ref_act: np.ndarray, classes: list[str],
            step_ms: float, tempo_bpm: float, threshold: float = 0.3) -> FeelReport:
    """Measure one clip. ``*_act`` are ``(steps, n_classes)`` activations."""
    per_class, onsets = {}, {}
    all_dev = []

    for c, name in enumerate(classes):
        p = peak_pick(pred_act[:, c], step_ms, threshold)
        r = peak_pick(ref_act[:, c], step_ms, threshold=0.5)
        onsets[name] = p

        dev = _grid_deviation(p, tempo_bpm)
        _, _, _, ref_dev = match_onsets(p, r, tolerance_s=0.05)
        all_dev.append(dev)
        per_class[name] = {
            "n": int(len(p)),
            "grid_dev_ms": float(np.mean(dev) * 1000) if len(dev) else float("nan"),
            "grid_spread_ms": float(np.std(dev) * 1000) if len(dev) else float("nan"),
            "swing_ratio": float(swing_ratio(p, tempo_bpm)),
            "ref_dev_ms": float(np.mean(ref_dev) * 1000) if len(ref_dev) else float("nan"),
        }

    # The hat-vs-kick offset PLAN.md singles out, plus any other pair present.
    pairwise = {}
    for a, b in (("hat_closed", "kick"), ("snare", "kick"), ("hat_open", "kick")):
        if a in onsets and b in onsets and len(onsets[a]) and len(onsets[b]):
            da = _grid_deviation(onsets[a], tempo_bpm)
            db = _grid_deviation(onsets[b], tempo_bpm)
            pairwise[f"{a} - {b}"] = float((np.mean(da) - np.mean(db)) * 1000)

    flat = np.concatenate([d for d in all_dev if len(d)]) if any(len(d) for d in all_dev) \
        else np.array([])
    return FeelReport(
        per_class=per_class,
        pairwise_offset_ms=pairwise,
        overall={
            "grid_dev_ms": float(np.mean(flat) * 1000) if len(flat) else float("nan"),
            "grid_spread_ms": float(np.std(flat) * 1000) if len(flat) else float("nan"),
            "n_onsets": int(len(flat)),
        },
    )


def aggregate(reports: list[FeelReport]) -> FeelReport:
    """Pool per-clip reports, weighting each class by its onset count."""
    if not reports:
        raise ValueError("no reports to aggregate")
    names = list(reports[0].per_class)
    per_class = {}
    for name in names:
        rows = [r.per_class[name] for r in reports if r.per_class[name]["n"] > 0]
        w = np.array([r["n"] for r in rows], dtype=float)
        if not len(rows):
            per_class[name] = {"n": 0, "grid_dev_ms": float("nan"),
                               "grid_spread_ms": float("nan"),
                               "swing_ratio": float("nan"), "ref_dev_ms": float("nan")}
            continue

        def wmean(key):
            v = np.array([r[key] for r in rows], dtype=float)
            m = np.isfinite(v)
            return float(np.average(v[m], weights=w[m])) if m.any() else float("nan")

        per_class[name] = {"n": int(w.sum()), "grid_dev_ms": wmean("grid_dev_ms"),
                           "grid_spread_ms": wmean("grid_spread_ms"),
                           "swing_ratio": wmean("swing_ratio"),
                           "ref_dev_ms": wmean("ref_dev_ms")}

    pair_keys = {k for r in reports for k in r.pairwise_offset_ms}
    pairwise = {}
    for k in sorted(pair_keys):
        v = np.array([r.pairwise_offset_ms[k] for r in reports if k in r.pairwise_offset_ms])
        v = v[np.isfinite(v)]
        if len(v):
            pairwise[k] = float(v.mean())

    dev = np.array([r.overall["grid_dev_ms"] for r in reports], dtype=float)
    spread = np.array([r.overall["grid_spread_ms"] for r in reports], dtype=float)
    return FeelReport(per_class, pairwise, {
        "grid_dev_ms": float(np.nanmean(dev)) if np.isfinite(dev).any() else float("nan"),
        "grid_spread_ms": float(np.nanmean(spread)) if np.isfinite(spread).any() else float("nan"),
        "n_onsets": int(sum(r.overall["n_onsets"] for r in reports)),
    })
