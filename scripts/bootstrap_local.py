"""Set this machine up to run flybeats, and report what it found.

Run it with the *system* Python, from a clone of the repo. It creates the venv,
installs the right torch for whatever card is present, runs the suite, runs the
CUDA smoke test, and writes a report you can hand back:

    python scripts/bootstrap_local.py                  # do it
    python scripts/bootstrap_local.py --check          # look, change nothing
    python scripts/bootstrap_local.py --pull           # update the clone first

Windows, from the clone:

    py -3.12 scripts\\bootstrap_local.py

Stdlib only, and it runs on the Python you already have -- it cannot import
torch or anything else from the venv it is about to create. Every subprocess
gets its arguments as a list rather than a shell string, because a path like
``C:\\Users\\you\\Documents\\AI Databases\\flybeats`` has a space in it and
shell quoting is where that goes wrong.

The report lands in ``results/local/`` as JSON: OS, CPU, RAM, GPU, driver,
torch build, the thread-count timing, and the outcome of each step. Commit it.
The numbers in this repository all came off 4 shared cloud cores, and there is
currently no record of what real hardware does with any of it.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_PY = (3, 11)

report: dict = {"steps": []}


def say(msg: str = "") -> None:
    print(msg, flush=True)


def step(name: str, ok: bool | None, detail: str = "") -> bool:
    mark = {True: "ok", False: "FAILED", None: "skipped"}[ok]
    report["steps"].append({"name": name, "status": mark, "detail": detail})
    say(f"  [{mark}] {name}" + (f" -- {detail}" if detail else ""))
    return bool(ok)


def run(args: list[str], **kw) -> subprocess.CompletedProcess:
    """Never shell=True: these paths contain spaces."""
    return subprocess.run([str(a) for a in args], capture_output=True, text=True, **kw)


# ------------------------------------------------------------------ inspect --

def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def describe_machine() -> dict:
    info = {
        "when": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "python": sys.version.split()[0],
        "python_exe": sys.executable,
    }
    try:                                    # RAM, without a third-party dependency
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names:
            info["ram_gb"] = round(
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
        elif os.name == "nt":
            digits: list[int] = []
            for probe in (["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                          ["powershell", "-NoProfile", "-Command",
                           "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"]):
                if not shutil.which(probe[0]):
                    continue
                digits = [int(s) for s in run(probe).stdout.split() if s.isdigit()]
                if digits:
                    break
            if digits:
                info["ram_gb"] = round(digits[0] / 1e9, 1)
    except Exception:                                           # noqa: BLE001
        pass
    return info


def describe_gpu() -> dict:
    """What nvidia-smi says, before torch has an opinion about it."""
    if not shutil.which("nvidia-smi"):
        return {"present": False, "why": "nvidia-smi not on PATH"}
    out = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
               "--format=csv,noheader"])
    if out.returncode != 0:
        return {"present": False, "why": (out.stderr or out.stdout).strip()[:200]}
    first = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    parts = [p.strip() for p in first.split(",")]
    gpu = {"present": True, "raw": first}
    if len(parts) >= 3:
        gpu.update(name=parts[0], driver=parts[1], memory=parts[2])
    return gpu


# -------------------------------------------------------------------- steps --

def check_git(pull: bool) -> None:
    if not (ROOT / ".git").exists():
        step("git clone", None, "not a git checkout")
        return
    branch = run(["git", "-C", ROOT, "rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
    head = run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"]).stdout.strip()
    dirty = bool(run(["git", "-C", ROOT, "status", "--porcelain"]).stdout.strip())

    fetched = run(["git", "-C", ROOT, "fetch", "origin", "--quiet"]).returncode == 0
    behind = ahead = "?"
    if fetched:
        counts = run(["git", "-C", ROOT, "rev-list", "--left-right", "--count",
                      "origin/main...HEAD"]).stdout.split()
        if len(counts) == 2:
            behind, ahead = counts

    report["git"] = {"branch": branch, "head": head, "dirty": dirty,
                     "behind_main": behind, "ahead_main": ahead}
    detail = f"{branch} at {head}, {behind} behind / {ahead} ahead of origin/main"
    if dirty:
        detail += ", uncommitted changes present"
    step("git state", True, detail)

    if behind not in ("0", "?") and not pull:
        say(f"\n  This clone is {behind} commits behind origin/main. The tests,"
            f"\n  the fixtures and the CUDA smoke test are all newer than it."
            f"\n  Re-run with --pull, or: git pull origin main\n")
    elif behind not in ("0", "?") and pull:
        if dirty:
            step("git pull", False, "uncommitted changes -- commit or stash first")
        else:
            out = run(["git", "-C", ROOT, "pull", "origin", "main"])
            step("git pull", out.returncode == 0,
                 (out.stdout or out.stderr).strip().splitlines()[-1][:120] if
                 (out.stdout or out.stderr).strip() else "")


def abi_mismatch(venv: Path, py: Path) -> str:
    """Does the interpreter in this venv still match the packages inside it?

    A venv is a launcher plus a `pyvenv.cfg` pointing at a base interpreter,
    and `python -m venv` over an existing directory rewrites both while
    leaving `site-packages` untouched. So a second Python on the machine --
    a new release, an IDE creating an environment, a stray `py -m venv` --
    can repoint the venv at itself and leave every compiled wheel in it built
    for the old ABI.

    That happened here on 16 September: `.venv` was rebuilt with 3.14 over a
    3.12 site-packages, and every run afterwards died at `import torch` with
    "PyTorch has loaded the torch/_C folder of the PyTorch repository", which
    names neither Python nor the version. The DLLs were all present and
    hash-correct; the interpreter simply could not load a `cp312` extension.

    Returns a description of the mismatch, or "" when the venv is coherent.
    """
    out = run([py, "-c", "import sys, sysconfig; "
                         "print(f'{sys.version_info[0]}.{sys.version_info[1]}'); "
                         "print(sysconfig.get_path('purelib'))"])
    if out.returncode != 0:
        return f"the interpreter in {venv} does not run: {(out.stderr or out.stdout).strip()[:200]}"
    lines = out.stdout.strip().splitlines()
    running, site = lines[0], Path(lines[-1])
    if not site.is_dir():
        return ""

    return describe_abi_mismatch(venv, running, abi_tags(site))


def abi_tags(site: Path) -> set[str]:
    """Python versions the compiled extensions in ``site`` were built for.

    Extensions carry their ABI in the filename -- ``_C.cp312-win_amd64.pyd``,
    ``_speedups.cpython-312-x86_64-linux-gnu.so``. One level down covers both
    top-level modules and each package's own extensions, which is where torch
    keeps its. Pure-Python packages contribute nothing and are not evidence
    either way.
    """
    tags: set[str] = set()
    for pat in ("*.pyd", "*.so", "*/*.pyd", "*/*.so"):
        for f in site.glob(pat):
            for part in f.name.split("."):
                if part.startswith(("cp3", "cpython-3")):
                    digits = part.split("-")[0].replace("cpython-", "").replace("cp", "")
                    if digits.isdigit() and len(digits) >= 2:
                        tags.add(f"{digits[0]}.{digits[1:]}")
    return tags


def describe_abi_mismatch(venv: Path, running: str, tags: set[str]) -> str:
    """The message, given what the venv runs and what is installed in it."""
    if not tags or running in tags:
        return ""
    want = sorted(tags)[0]
    return (f"the venv runs Python {running} but its packages were built for "
            f"{', '.join(sorted(tags))}. `python -m venv` over an existing "
            f"directory repoints the launcher and leaves site-packages alone. "
            f"Rebuild it with the matching interpreter: "
            f"`<python{want}> -m venv {venv}` (site-packages survives), "
            f"or delete {venv} and start over.")


def ensure_venv(venv: Path, check_only: bool) -> Path | None:
    py = venv_python(venv)
    if py.exists():
        bad = abi_mismatch(venv, py)
        if bad:
            step("venv", False, bad)
            return None
        step("venv", True, f"reusing {venv}")
        return py
    if check_only:
        step("venv", None, f"would create {venv}")
        return None
    out = run([sys.executable, "-m", "venv", venv])
    if out.returncode != 0:
        step("venv", False, (out.stderr or out.stdout).strip()[:200])
        return None
    step("venv", True, f"created {venv}")
    return venv_python(venv)


def install(py: Path, gpu: dict, index_url: str | None, check_only: bool) -> bool:
    """torch first, from the right index; requirements-dev then leaves it alone."""
    if check_only:
        step("install", None, "would install torch + requirements-dev.txt")
        return True

    run([py, "-m", "pip", "install", "--upgrade", "--quiet", "pip"])

    args = [py, "-m", "pip", "install", "--quiet", "torch"]
    if index_url:
        # --force-reinstall, not just --index-url: the CUDA wheel is version
        # 2.x.y+cuNNN and the CPU one is plain 2.x.y, so pip reads the bare
        # `torch` requirement as already satisfied and changes nothing. Asking
        # for an index and being handed back the CPU wheel you were trying to
        # replace is the whole failure this script exists to catch.
        args += ["--index-url", index_url, "--force-reinstall", "--no-cache-dir"]
        note = f"from {index_url} (forced)"
    elif gpu.get("present"):
        # No --index-url given. On Linux the default wheel is already CUDA; on
        # Windows it is CPU-only, and a CPU wheel on a machine with a card is
        # exactly the silent failure this script exists to prevent.
        note = "default index"
        if os.name == "nt":
            say("\n  NOTE: a card is present and no --index-url was given. On Windows\n"
                "  the default torch wheel is CPU-only. If the CUDA check below fails,\n"
                "  reinstall with --index-url for a build your driver supports\n"
                "  (pytorch.org's selector names it).\n")
    else:
        note = "default index (no GPU detected)"

    t0 = time.perf_counter()
    out = run(args)
    if out.returncode != 0:
        return step("install torch", False, (out.stderr or out.stdout).strip()[-300:])
    step("install torch", True, f"{note}, {time.perf_counter() - t0:.0f}s")

    out = run([py, "-m", "pip", "install", "--quiet", "-r", ROOT / "requirements-dev.txt"])
    return step("install requirements-dev.txt", out.returncode == 0,
                (out.stderr or out.stdout).strip()[-300:] if out.returncode else "")


def torch_facts(py: Path) -> dict:
    """Ask the venv's torch about itself -- this Python cannot import it."""
    probe = (
        "import json, torch;"
        "d = {'version': torch.__version__, 'cuda_build': torch.version.cuda,"
        " 'cuda_available': torch.cuda.is_available(),"
        " 'threads': torch.get_num_threads()};"
        "d.update({'device': torch.cuda.get_device_name(0),"
        " 'capability': '.'.join(map(str, torch.cuda.get_device_capability(0))),"
        " 'vram_gb': round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),"
        " 'bf16': torch.cuda.is_bf16_supported()}) if torch.cuda.is_available() else None;"
        "print(json.dumps(d))"
    )
    out = run([py, "-c", probe])
    if out.returncode != 0:
        step("torch", False, (out.stderr or out.stdout).strip()[-300:])
        return {}
    facts = json.loads(out.stdout.strip().splitlines()[-1])
    report["torch"] = facts
    if facts["cuda_available"]:
        step("torch sees the GPU", True,
             f"{facts['version']} (CUDA {facts['cuda_build']}) -> {facts['device']}, "
             f"{facts['vram_gb']} GB, bf16 {'yes' if facts['bf16'] else 'no'}")
    else:
        step("torch sees the GPU", False,
             f"{facts['version']} reports cuda_available=False -- "
             "a CPU-only wheel, or a driver problem")
    return facts


def run_suite(py: Path, gpu_too: bool) -> bool:
    t0 = time.perf_counter()
    out = run([py, "-m", "pytest", "-q"] + ([] if gpu_too else ["-m", "not gpu"]),
              cwd=ROOT)
    tail = [ln for ln in out.stdout.strip().splitlines() if "passed" in ln or "failed" in ln]
    summary = tail[-1] if tail else (out.stdout or out.stderr).strip()[-200:]
    report["pytest"] = {"summary": summary, "seconds": round(time.perf_counter() - t0, 1),
                        "returncode": out.returncode}
    return step("pytest", out.returncode == 0, f"{summary} in {time.perf_counter() - t0:.0f}s")


def run_smoke(py: Path, cuda: bool) -> bool:
    args = [py, ROOT / "scripts" / "cuda_smoke.py"]
    if not cuda:
        args += ["--device", "cpu"]
    out = run(args, cwd=ROOT)
    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip().startswith("[")]
    report["cuda_smoke"] = {"returncode": out.returncode, "cuda": cuda, "checks": lines}
    for ln in lines:
        say(f"    {ln}")
    if out.returncode != 0:
        return step("cuda_smoke.py", False, "see the checks above")
    # Passing on CPU says nothing about the CUDA path; the bf16 check does not
    # even run. Say so in the step, or the report reads as a GPU clean bill.
    return step("cuda_smoke.py", True,
                "" if cuda else "ran on CPU -- the CUDA path is still unverified")


def thread_timing(py: Path, default_threads: int | None = None) -> None:
    """The 13x finding, re-measured here -- it is machine-specific."""
    probe = (
        "import sys, json, torch, numpy as np;"
        "sys.path.insert(0, 'src');"
        "torch.set_num_threads(int(sys.argv[1]));"
        "from build import build_model, get_subgraph, load_config;"
        "from decoder import DrumKit; from realtime import benchmark;"
        "cfg = load_config('configs/sanity_3piece.yaml');"
        "cfg['subgraph'].update(max_nodes=10000, min_weight=5,"
        " cache='tests/fixtures/subgraph_10k.npz');"
        "sg = get_subgraph(cfg); kit = DrumKit.from_tier(cfg['kit']['tier']);"
        "m, kit = build_model(cfg, sg, n_styles=1);"
        "s = benchmark(m, kit, cfg, block_ms=20.0, n_blocks=20);"
        "print(json.dumps({'ms': s['inference_ms_mean'], 'x': s['realtime_factor']}))"
    )
    # Measure torch's own default too. It picks physical cores, so on an SMT
    # machine it is neither 1 nor cpu_count -- and it is what you get if you
    # set nothing, which makes it the one number a reader actually needs.
    counts = sorted({1, int(default_threads or 0), os.cpu_count() or 4} - {0})
    timings = {}
    for n in counts:
        out = run([py, "-c", probe, str(n)], cwd=ROOT)
        if out.returncode == 0 and out.stdout.strip():
            try:
                timings[str(n)] = json.loads(out.stdout.strip().splitlines()[-1])
            except json.JSONDecodeError:
                pass
    if not timings:
        step("thread timing", None, "could not measure")
        return
    report["threads"] = timings
    detail = ", ".join(f"{n} thread{'s' if n != '1' else ''}: {v['ms']:.1f} ms "
                       f"({v['x']:.2f}x realtime)" for n, v in sorted(timings.items(),
                                                                     key=lambda kv: int(kv[0])))
    step("thread timing", True, detail)

    best = min(timings, key=lambda k: timings[k]["ms"])
    report["threads_best"] = {"count": int(best), "ms": timings[best]["ms"]}
    one = timings.get("1", {}).get("ms")
    if one and timings[best]["ms"] > one * 2:
        say(f"\n  Threads cost you {timings[best]['ms'] / one:.1f}x here. Set "
            f"OMP_NUM_THREADS=1\n  for the live path (PowerShell: "
            f"$env:OMP_NUM_THREADS=1).\n")
    elif default_threads and int(best) != int(default_threads):
        say(f"\n  Fastest at {best} threads; torch defaults to {default_threads} here.\n"
            f"  For the live path: OMP_NUM_THREADS={best} (PowerShell: "
            f"$env:OMP_NUM_THREADS={best}).\n")


# ------------------------------------------------------------------- driver --

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report, install nothing")
    ap.add_argument("--pull", action="store_true", help="git pull origin main first")
    ap.add_argument("--venv", type=Path, default=ROOT / ".venv")
    ap.add_argument("--index-url", default=None,
                    help="torch wheel index, e.g. https://download.pytorch.org/whl/cu124")
    ap.add_argument("--no-timing", action="store_true", help="skip the thread benchmark")
    ap.add_argument("--out", type=Path, default=None, help="where to write the report")
    a = ap.parse_args(argv)

    if sys.version_info < MIN_PY:
        say(f"Python {'.'.join(map(str, MIN_PY))}+ required; this is "
            f"{platform.python_version()}. SETUP.md recommends 3.12.")
        return 2

    say("flybeats -- local bootstrap\n" + "=" * 27)
    report["machine"] = describe_machine()
    for k, v in report["machine"].items():
        say(f"  {k}: {v}")

    say("\nGPU")
    gpu = describe_gpu()
    report["gpu"] = gpu
    say(f"  {gpu.get('raw') or gpu.get('why')}")

    say("\nRepository")
    check_git(a.pull)

    say("\nEnvironment")
    py = ensure_venv(a.venv, a.check)
    if py is None:
        return finish(a, 0 if a.check else 1)
    if not install(py, gpu, a.index_url, a.check):
        return finish(a, 1)

    facts = torch_facts(py)
    cuda = bool(facts.get("cuda_available"))

    say("\nTests")
    suite_ok = run_suite(py, gpu_too=cuda)

    say("\nSmoke test")
    smoke_ok = run_smoke(py, cuda)

    if not a.no_timing:
        say("\nThreads")
        thread_timing(py, facts.get("threads"))

    say("\nWhat this machine can now do")
    say(f"  the suite, including the {'GPU tests' if cuda else 'CPU tests (no GPU tests)'}")
    if cuda:
        say("  Phase B' and Phase D -- see RUNBOOK.md, and time your first epoch")
    else:
        say("  everything except GPU training. Fix the torch install first:")
        say("  the CUDA wheel index is in RUNBOOK.md.")

    # A card that torch cannot see is a failure, and has to reach the exit
    # code: a CPU wheel on a GPU machine is the silent failure this script
    # exists to catch, and returning 0 for it reproduces that failure here.
    # No card at all is not a failure -- CPU-only is a supported way to run.
    gpu_unusable = bool(gpu.get("present")) and not cuda
    return finish(a, 0 if (suite_ok and smoke_ok and not gpu_unusable) else 1)


def finish(a, code: int) -> int:
    out = a.out or (ROOT / "results" / "local" /
                    f"bootstrap-{platform.node()}-"
                    f"{datetime.now(timezone.utc):%Y%m%d-%H%M}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    report["exit_code"] = code
    out.write_text(json.dumps(report, indent=2))
    say(f"\nReport: {out}")
    say("Commit it -- there is no record in this repo of what real hardware does.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
