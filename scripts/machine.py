"""CPU and GPU readings for the live chart, for the whole machine and for the run's own process ("this run"), from what
Windows and the NVIDIA driver expose without extra software. Nothing here needs admin rights or changes anything.

What each reading is:
  CPU clock     base clock x the "% Processor Performance" counter (the effective average clock); over 100 % means the
                CPU is boosting above its base clock
  CPU usage     machine: "% Processor Utility", the figure Task Manager shows, not capped, so it passes 100 % on boost.
                this run: the run's processor time over the interval as a share of all logical processors, scaled by
                the same performance figure so the two are on one scale
  CPU temp      the ACPI thermal zone near the CPU (a board sensor), not per-core temperatures
  CPU power     the whole package, live: Windows' "Energy Meter" counter, which passes on the CPU's own power meter
                (RAPL); missing on machines that do not expose one
  GPU           nvidia-smi, all live: graphics and memory clocks, usage, temperature, power, and the reasons the driver
                gives for holding the clock back (a bit mask, decoded on the page). No rated clocks: the driver no
                longer reports them. this run's GPU usage: its SM (compute) share from nvidia-smi pmon; Windows'
                per-app GPU counters (what Task Manager shows) read 0 for CUDA work on this card
  Memory        this run only: RAM (working set) and dedicated GPU memory of the run's processes
Voltage is not read: neither part reports a live value here.

The run's processes are the python.exe whose command line matches the run (the venv launcher and the interpreter it
starts). Cost to the run: the first version started PowerShell every 5 s and slowed memorisation from 8.4 to 11.7 s
per step. The Sampler keeps ONE PowerShell loop running at idle priority (it and nvidia-smi only get CPU time nobody
else wants), reads every 10 s, looks up the run's processes every 30 s, and exits when the process that started it does.
"""
import json
import os
import re
import subprocess
import threading
import time

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_IDLE = getattr(subprocess, "IDLE_PRIORITY_CLASS", 0)
_GPU_FIELDS = ("name,clocks.gr,clocks.mem,clocks.max.gr,utilization.gpu,temperature.gpu,power.draw,power.limit,"
               "memory.used,memory.total,clocks_event_reasons.active")
_GPU_KEYS = ["name", "clock_mhz", "mem_clock_mhz", "max_clock_mhz", "usage_pct", "temp_c", "power_w", "power_limit_w",
             "mem_used_mb", "mem_total_mb", "held_by"]
_COUNTERS = ("'\\Processor Information(_Total)\\% Processor Utility','\\Processor Information(_Total)\\% Processor Performance',"
             "'\\Thermal Zone Information(*)\\Temperature','\\Energy Meter(*)\\Power'")
_READ = f"""
$c = (Get-Counter {_COUNTERS}).CounterSamples
$o = [ordered]@{{
  util = @($c | Where-Object {{ $_.Path -like '*processor utility*' }} | ForEach-Object {{ $_.CookedValue }})
  perf = @($c | Where-Object {{ $_.Path -like '*processor performance*' }} | ForEach-Object {{ $_.CookedValue }})
  tz = @($c | Where-Object {{ $_.Path -like '*thermal zone*' }} | ForEach-Object {{ $_.CookedValue }})
  pkg_mw = @($c | Where-Object {{ $_.Path -like '*energy meter(*_pkg)*' }} | ForEach-Object {{ $_.CookedValue }})
  gpu = (& nvidia-smi --query-gpu={_GPU_FIELDS} --format=csv,noheader,nounits | Select-Object -First 1)
}}
"""
# The run's processes, then their readings (CPU seconds, working set, GPU memory, GPU share). Found either by command
# line every __PROC_EVERY__ readings (a chart watching a run from outside), or given (a chart inside the run: its own
# process, plus the venv launcher that started it)
_FIND = """
if ($n % __PROC_EVERY__ -eq 0) {
  $pids = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match '__PATTERN__' } | ForEach-Object { $_.ProcessId })
}
"""
_OWN = """
if ($n -eq 0) {
  $pids = @(__PIDS__)
  $me = Get-CimInstance Win32_Process -Filter "ProcessId=$($pids[0])"
  $up = Get-CimInstance Win32_Process -Filter "ProcessId=$($me.ParentProcessId)"
  if ($up.Name -eq 'python.exe') { $pids += $up.ProcessId }
}
$pids = @($pids | Where-Object { Get-Process -Id $_ })
"""
_RUN = """
$o.run = [bool]$pids.Count
$o.pids = $pids
if ($pids.Count) {
  $pr = @(Get-Process -Id $pids)
  $o.run_cpu_s = ($pr | Measure-Object -Property CPU -Sum).Sum
  $o.run_ws = ($pr | Measure-Object -Property WorkingSet64 -Sum).Sum
  $o.run_gmem = ((Get-Counter @($pids | ForEach-Object { "\\GPU Process Memory(pid_$($_)_*)\\Dedicated Usage" })).CounterSamples | Measure-Object -Property CookedValue -Sum).Sum
  $o.run_sm = @(& nvidia-smi pmon -c 1 -s u | Where-Object { $_ -notmatch '^\\s*#' } | ForEach-Object { $f = -split $_; if ($f[1] -match '^\\d+$' -and $pids -contains [int]$f[1] -and $f[3] -match '^[\\d.]+$') { [double]$f[3] } })
}
"""
_LOOP = """
$ErrorActionPreference = 'SilentlyContinue'
$n = 0
$pids = @()
while ($true) {
  if (-not (Get-Process -Id __PARENT__)) { exit }
  __READ__
  __RUN__
  $o | ConvertTo-Json -Compress
  [Console]::Out.Flush()
  $n++
  Start-Sleep -Seconds __SLEEP__
}
"""
_STATIC = "Get-CimInstance Win32_Processor | Select-Object -First 1 Name, NumberOfCores, NumberOfLogicalProcessors, " \
          "MaxClockSpeed | ConvertTo-Json -Compress"


def _ps_once(script, timeout=30):
    try:
        return subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], capture_output=True,
                              text=True, timeout=timeout, creationflags=_NO_WINDOW | _IDLE).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _list(x):
    return x if isinstance(x, list) else [] if x is None else [x]


def _run_part(pattern, proc_every, pids=None):
    find = _OWN.replace("__PIDS__", ",".join(str(p) for p in pids)) if pids else \
        _FIND.replace("__PATTERN__", pattern.replace("'", "''")).replace("__PROC_EVERY__", str(proc_every))
    return find + _RUN


def cpu_static():
    try:
        s = json.loads(_ps_once(_STATIC))
    except json.JSONDecodeError:
        return {}
    return {"name": (s.get("Name") or "").strip(), "cores": s.get("NumberOfCores"),
            "threads": s.get("NumberOfLogicalProcessors"), "base_mhz": s.get("MaxClockSpeed")}


def _gpu(line):
    if not line:
        return None
    g = dict(zip(_GPU_KEYS, [x.strip() for x in str(line).split(",")]))
    held = g.pop("held_by", "")
    g = {k: (v if k == "name" else _num(v)) for k, v in g.items()}
    g["held_by"] = int(held, 16) if held.startswith("0x") else None      # nvidia-smi's clock event reasons, a bit mask
    return g


def _sample(raw, static, prev=None):
    """One reading from the loop's output. prev: the previous reading's run CPU seconds, time and processes, for the
    run's CPU usage over the interval."""
    util, perf = _num((_list(raw.get("util")) or [None])[0]), _num((_list(raw.get("perf")) or [None])[0])
    tz = [_num(t) for t in _list(raw.get("tz")) if _num(t)]
    pkg_mw = _num((_list(raw.get("pkg_mw")) or [None])[0])
    base, threads, now = static.get("base_mhz"), static.get("threads"), time.time()
    pids, cpu_s, ws, gmem = sorted(_list(raw.get("pids"))), _num(raw.get("run_cpu_s")), _num(raw.get("run_ws")), \
        _num(raw.get("run_gmem"))
    run_cpu = None
    if prev and pids and prev["pids"] == pids and cpu_s is not None and prev["cpu_s"] is not None and threads:
        run_cpu = max(0.0, cpu_s - prev["cpu_s"]) / ((now - prev["t"]) * threads) * 100 * ((perf or 100) / 100)
    return {"taken_ms": now * 1000,
            "cpu": {"name": static.get("name"), "cores": static.get("cores"), "threads": threads,
                    "clock_mhz": base * perf / 100 if base and perf is not None else None, "base_mhz": base,
                    "usage_pct": util, "temp_c": max(tz) - 273.15 if tz else None, "temp_source": "ACPI thermal zone",
                    "power_w": pkg_mw / 1000 if pkg_mw is not None else None},
            "gpu": _gpu(raw.get("gpu")),
            "run": {"cpu_pct": run_cpu, "gpu_pct": sum(_num(v) or 0.0 for v in _list(raw.get("run_sm"))) if pids else None,
                    "ram_mb": ws / 2**20 if ws is not None and pids else None,
                    "gpu_mem_mb": gmem / 2**20 if gmem is not None and pids else None},
            "process_running": raw.get("run") if "run" in raw else None,
            "_prev": {"pids": pids, "cpu_s": cpu_s, "t": now}}


def sample(static, process_pattern):
    """One reading, from a one-off PowerShell (for occasional use, not a loop; the run's CPU usage needs two)."""
    script = "$ErrorActionPreference = 'SilentlyContinue'\n$n = 0\n$pids = @()\n" + _READ + \
             _run_part(process_pattern, 1) + "$o | ConvertTo-Json -Compress"
    try:
        raw = json.loads(_ps_once(script) or "{}")
    except json.JSONDecodeError:
        raw = {}
    s = _sample(raw, static)
    s.pop("_prev")
    return s


# One saved reading: [time ms, CPU MHz, CPU %, CPU °C, GPU MHz, GPU %, GPU °C, GPU W, GPU memory MB (whole card),
# this run's CPU %, this run's GPU %, this run's RAM MB, this run's GPU memory MB, GPU memory clock MHz,
# GPU clock event reasons (bit mask), CPU package W]. Rows saved before 29 Sep 2026, 22:00 stop at GPU memory; the
# CPU power column came at 22:25.
ROW = ["t", "cpu_clock", "cpu_usage", "cpu_temp", "gpu_clock", "gpu_usage", "gpu_temp", "gpu_power", "gpu_mem",
       "run_cpu_usage", "run_gpu_usage", "run_ram", "run_gpu_mem", "gpu_mem_clock", "gpu_held_by", "cpu_power"]


def row(s):
    c, g, r = s["cpu"], s["gpu"] or {}, s["run"]
    out = [s["taken_ms"], c["clock_mhz"], c["usage_pct"], c["temp_c"], g.get("clock_mhz"), g.get("usage_pct"),
           g.get("temp_c"), g.get("power_w"), g.get("mem_used_mb"), r["cpu_pct"], r["gpu_pct"], r["ram_mb"],
           r["gpu_mem_mb"], g.get("mem_clock_mhz"), g.get("held_by"), c["power_w"]]
    return [round(x, 1) if isinstance(x, float) else x for x in out]


def load_history(path, keep=100_000):
    rows = []
    if path and path.exists():
        for line in path.read_text(encoding="utf-8").splitlines()[-keep:]:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


class Sampler(threading.Thread):
    """Reads the machine every `every` seconds through one idle-priority PowerShell loop. .latest is the newest
    reading; .history holds every reading, also appended to history_path so a restarted chart keeps what came before."""

    def __init__(self, process_pattern=None, history_path=None, every=10, proc_every=3, own_process=False):
        """own_process: the run is this process (the chart runs inside it), so no search by command line."""
        super().__init__(daemon=True)
        self.pattern, self.every, self.path = process_pattern, every, history_path
        self.latest, self.prev = None, None
        self.history = sorted(load_history(history_path), key=lambda r: r[0])
        self.lock = threading.Lock()
        self.static = cpu_static()
        run = _run_part(process_pattern, proc_every, [os.getpid()] if own_process else None)
        self.loop = _LOOP.replace("__READ__", _READ).replace("__RUN__", run) \
                         .replace("__PARENT__", str(os.getpid())).replace("__SLEEP__", str(max(1, every - 1)))

    def since(self, t_ms):
        with self.lock:
            return [r for r in self.history if r[0] > t_ms]

    def run(self):
        while True:                                   # restart the loop if it ever dies
            p = subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", self.loop],
                                 stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                                 creationflags=_NO_WINDOW | _IDLE)
            for line in p.stdout:
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                s = _sample(raw, self.static, self.prev)
                self.prev = s.pop("_prev")
                r = row(s)
                with self.lock:
                    self.latest = s
                    self.history.append(r)
                if self.path:
                    with open(self.path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(r) + "\n")
            time.sleep(5)


if __name__ == "__main__":
    print(json.dumps(sample(cpu_static(), r"scripts[\\/]memorise\.py"), indent=1))
