"""Read the Phase A' probe outputs and apply the decision rule to all of them.

Five configs, eight classes, two scorings each: eighty intervals to hold in
your head. Deciding that by eye is how a favourite wins, so the rule is written
down once and applied by a program.

The rule, in the order it decides:

  1. **The gate.** At least one class significantly positive, and no class
     significantly negative -- "significant" meaning the bootstrap 95% interval
     is entirely on one side of zero. A head that is backwards on any class is
     not a head that reports how hard the drummer hit; it plays some hits
     inverted, which is worse than playing them all at the mean.
  2. **Spread.** Among the runs that clear the gate, the largest head output sd.
     A head predicting the mean scores a defensible loss and carries no
     dynamics, and the target's own sd is what it would have to reach.

Scored on the peak rows (y > 0.95), because that is the step the streaming path
reads velocity at. The support rows are reported for context only.

  python scripts/compare_velocity_runs.py               # every probe in results/
  python scripts/compare_velocity_runs.py a b c         # named runs, in order
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# tom_low   570   0.221   0.081   0.290   0.306  [+0.13, +0.45]
ROW = re.compile(
    r"^(?P<cls>\w+)\s+(?P<n>\d+)\s+(?P<target_sd>[-\d.]+)\s+(?P<drive>[-+\d.]+)\s+"
    r"(?P<motor>[-+\d.]+)\s+(?P<head>[-+\d.]+)\s+\[\s*(?P<lo>[-+\d.]+),\s*(?P<hi>[-+\d.]+)\]")
HEAD_SD = re.compile(r"head output at hits:\s+mean\s+([-\d.]+)\s+sd\s+([\d.]+)")
TARGET_SD = re.compile(r"target at hits:\s+mean\s+([-\d.]+)\s+sd\s+([\d.]+)")


def parse(text: str, section: str = "peak") -> dict:
    """Pull one scoring section out of a probe file."""
    marker = "== peak (y > 0.95) ==" if section == "peak" else "== support (y > 0.5) =="
    if marker not in text:
        raise ValueError(f"no {section!r} section in this probe output")
    body = text.split(marker, 1)[1]

    classes = []
    for line in body.splitlines():
        if line.startswith("=="):
            break
        m = ROW.match(line.strip())
        if m:
            lo, hi = float(m["lo"]), float(m["hi"])
            classes.append({"cls": m["cls"], "n": int(m["n"]), "r": float(m["head"]),
                            "lo": lo, "hi": hi,
                            "sign": "+" if lo > 0 else "-" if hi < 0 else "0"})

    head = HEAD_SD.search(body)
    target = TARGET_SD.search(body)
    return {"classes": classes,
            "head_sd": float(head.group(2)) if head else float("nan"),
            "target_sd": float(target.group(2)) if target else float("nan")}


def onset_f(run: str, logs: Path | None = None) -> str:
    """Best val onset F, from the run log if it is still here, else from the
    commit the queue made when the run landed.

    The log lives wherever the queue's OUT pointed, which is usually a
    scratchpad that does not survive a container restart. The commit does, so
    it is the fallback and the reason commit_result puts the number in the
    message."""
    for log in ([logs / f"{run}.log"] if logs else []) + list(ROOT.glob(f"runs/**/{run}.log")):
        if not log.exists():
            continue
        hits = re.findall(r"best val onset F: ([\d.]+)", log.read_text())
        if hits:
            return hits[-1]
    try:
        out = subprocess.run(["git", "log", "--format=%B", "-1",
                              f"--grep=Queue result: {run}$"],
                             cwd=ROOT, capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"best val onset F: ([\d.]+)", out)
        if m:
            return m.group(1)
    except (OSError, subprocess.SubprocessError):
        pass
    return "?"


def verdict(p: dict) -> tuple[bool, str]:
    pos = [c for c in p["classes"] if c["sign"] == "+"]
    neg = [c for c in p["classes"] if c["sign"] == "-"]
    if neg:
        return False, "backwards on " + ", ".join(c["cls"] for c in neg)
    if not pos:
        return False, "no class significantly positive"
    return True, "positive on " + ", ".join(c["cls"] for c in pos)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="default: every probe in results/")
    ap.add_argument("--section", choices=["peak", "support"], default="peak")
    ap.add_argument("--logs", type=Path, default=None,
                    help="the queue's OUT directory, for onset F. Without it only "
                         "runs/ and the queue's own commit messages are consulted")
    a = ap.parse_args(argv)

    runs = a.runs or sorted(p.stem[len("probe_"):] for p in RESULTS.glob("probe_*.txt"))
    if not runs:
        print(f"no probe output in {RESULTS}")
        return 1

    rows = []
    for run in runs:
        f = RESULTS / f"probe_{run}.txt"
        if not f.exists():
            print(f"{run:<24} no probe yet")
            continue
        p = parse(f.read_text(), a.section)
        ok, why = verdict(p)
        rows.append((run, p, ok, why))

    if not rows:
        return 1

    print(f"Phase A' comparison -- {a.section} rows, "
          f"target sd {rows[0][1]['target_sd']:.3f}\n")
    print(f"{'run':<24}{'gate':>6}{'+':>4}{'0':>4}{'-':>4}{'head sd':>10}"
          f"{'of target':>11}{'onset F':>10}")
    print("-" * 73)
    for run, p, ok, _ in rows:
        n = {s: sum(1 for c in p["classes"] if c["sign"] == s) for s in "+0-"}
        frac = p["head_sd"] / p["target_sd"] if p["target_sd"] else float("nan")
        print(f"{run:<24}{('PASS' if ok else 'fail'):>6}{n['+']:>4}{n['0']:>4}{n['-']:>4}"
              f"{p['head_sd']:>10.4f}{frac:>10.0%}{onset_f(run, a.logs):>10}")

    print()
    for run, _, ok, why in rows:
        print(f"  {run:<24}{why}")

    passed = [(run, p) for run, p, ok, _ in rows if ok]
    print()
    if not passed:
        print("Nothing clears the gate. No run is shippable as a velocity head yet.")
        return 0
    win, wp = max(passed, key=lambda r: r[1]["head_sd"])
    print(f"Clears the gate: {', '.join(r for r, _ in passed)}")
    print(f"Widest spread among those: {win} (head sd {wp['head_sd']:.4f}, "
          f"{wp['head_sd'] / wp['target_sd']:.0%} of the target's)")
    if len(passed) > 1:
        print("\nThese are point estimates of a spread, not intervals on it. Two runs "
              "within\na few thousandths of each other are not separated by this table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
