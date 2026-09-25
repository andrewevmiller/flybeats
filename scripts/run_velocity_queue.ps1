<#  Phase A' queue, for Windows. The POSIX twin is run_velocity_queue.sh.

    Strictly sequential, and that is the whole design. Two training jobs on
    one machine is the documented 76 s -> 1,415 s per-epoch collapse: torch
    takes a thread per core in each process and the oversubscribed threads
    spin rather than progress. So each run waits for the last, the queue
    waits for anything already training before it starts, and a lock file
    stops a second copy of the queue from interleaving with the first.

    Why a port rather than Git Bash: the .sh uses `.venv/bin/python`, which is
    the POSIX venv layout -- this machine has `.venv\Scripts\python.exe` -- and
    `pgrep`, which Git Bash does not provide.

        powershell -ExecutionPolicy Bypass -File scripts\run_velocity_queue.ps1
        powershell ... -File scripts\run_velocity_queue.ps1 -WhatIf          # plan only
        powershell ... -File scripts\run_velocity_queue.ps1 -StartAt 23:45   # preflight now, train later

    The arms trained here are A'2 (velocity_lin_cpu) and A'3
    (velocity_peak_cpu), then A'1 (velocity_w5_cpu) and the baseline
    (velocity_probe_cpu) if this machine has no finished run of them. A' is
    read as a set of per-class intervals against the baseline's, so the
    baseline is probed like every other arm; the new arms go first so that a
    night cut short still produces the new information.

    All four configs resolve to `device: cpu` (pinned in v1_8piece_cpu.yaml,
    asserted by tests/test_config.py). That is not a throughput choice: A'1's
    published per-class intervals were measured on CPU, and an arm measured
    somewhere else is not a comparison.

    Cost, measured on the Ryzen 9 4900HS on 22 September: one epoch of 24 + 24
    clips took 53.8 s at 4 threads and 45.3 s at 8, with identical metrics to
    four places. Scaled to 256 train + 120 val clips that is ~7 min an epoch,
    ~1.5 h a run at 12 epochs -- not the ~4 h this header used to budget.
#>
param(
    [string]$Out = "runs/queue",
    [int]$Threads = 8,
    [string[]]$Arms = @("velocity_lin_cpu", "velocity_peak_cpu", "velocity_w5_cpu"),
    [string]$Baseline = "velocity_probe_cpu",
    # Wall-clock time (HH:mm) to begin training. Preflight runs immediately, so
    # a broken venv is found at hand-off rather than at 3 a.m.
    [string]$StartAt = "",
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
# Under `powershell -File`, "-Arms a,b" arrives as the one string "a,b".
$Arms = @($Arms | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Py = Join-Path $Repo ".venv\Scripts\python.exe"
$Q  = Join-Path $Repo $Out
# Before the first Say, which logs into it: on a clone that had never run the
# queue the directory did not exist, and the first log line threw.
if (-not $WhatIf) { New-Item -ItemType Directory -Force -Path $Q | Out-Null }

# Everything this queue trains-if-missing and probes, in order. The baseline
# goes last among the trainees but first in the summary.
$Train = @($Arms) + @($Baseline)
$Summarise = @($Baseline) + @($Arms)

function Say([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line
    if (-not $WhatIf) { Add-Content -Path (Join-Path $Q "queue.log") -Value $line -Encoding utf8 }
}

# Windows PowerShell 5.1 turns every stderr line of a redirected native
# command into an ErrorRecord, and under "Stop" the first one ends the
# script. torch prints a sparse-CSR UserWarning seconds into every run, so the
# queue as first written would have died at the start of its first arm, with
# the night still ahead. Stderr is folded into the log as plain text instead.
function Invoke-Py([string]$log, [string[]]$argv) {
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Py @argv 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Encoding utf8
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
}

function Get-TrainingProcess {
    Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
        Where-Object { $_.CommandLine -match 'src[\\/](train|ablations)\.py' -and
                       $_.CommandLine -notmatch 'spawn_main' } |
        Sort-Object CreationDate | Select-Object -First 1
}

function Wait-ForIdle {
    $p = Get-TrainingProcess
    while ($p) {
        Say ("waiting on pid {0}, running since {1}" -f $p.ProcessId, $p.CreationDate.ToString("HH:mm"))
        Start-Sleep -Seconds 60
        $p = Get-TrainingProcess
    }
}

# best.pt is written after the first epoch, so its presence says a run
# started, not that it finished. A run is done when its history holds every
# epoch the config asked for.
function Test-RunComplete([string]$name) {
    $hist = Join-Path $Repo "runs\$name\history.json"
    if (-not (Test-Path (Join-Path $Repo "runs\$name\best.pt")) -or -not (Test-Path $hist)) { return $false }
    $done = @((Get-Content $hist -Raw | ConvertFrom-Json)).Count
    return $done -ge $Epochs[$name]
}

# --- preflight: the failures that waste a whole overnight ------------------
if (-not (Test-Path $Py)) { throw "no venv interpreter at $Py -- see SETUP.md" }

# The venv can be repointed at another Python while its packages stay behind,
# which fails at `import torch` with a message about torch rather than about
# Python. bootstrap_local checks for exactly that; ask it before committing
# the night to a queue that cannot import anything.
& $Py -c "import sys; sys.path.insert(0, 'scripts'); from pathlib import Path; import bootstrap_local as b; m = b.abi_mismatch(Path('.venv'), Path(sys.executable)); sys.exit(m or 0)"
if ($LASTEXITCODE -ne 0) { throw "venv preflight failed -- run: py -3.12 scripts\bootstrap_local.py --check" }

# device is a config property, and a run on the wrong one is not comparable to
# A'1. Assert it here too: this script is what a tired operator actually runs.
$Epochs = @{}
foreach ($cfg in $Summarise) {
    $res = & $Py -c "import sys; sys.path.insert(0, 'src'); from build import load_config; t = load_config('configs/$cfg.yaml')['train']; print(t.get('device'), t.get('epochs', 10))"
    $dev, $ep = "$res".Trim() -split '\s+'
    if ($dev -ne "cpu") { throw "configs/$cfg.yaml resolves to device=$dev, not cpu -- the A' arms must share a device" }
    $Epochs[$cfg] = [int]$ep
}
Say ("preflight ok: venv coherent, all {0} arms resolve to device=cpu" -f $Summarise.Count)

if ($WhatIf) {
    foreach ($cfg in $Train) {
        $state = if (Test-RunComplete $cfg) { "complete, probe only" } else { "train $($Epochs[$cfg]) epochs, then probe" }
        Say ("would run {0}: {1}" -f $cfg, $state)
    }
    Say ("would use OMP_NUM_THREADS={0}, start {1}" -f $Threads, $(if ($StartAt) { "at $StartAt" } else { "now" }))
    Say ("would write: {0}" -f (Join-Path $Out "SUMMARY.txt"))
    return
}

# One queue at a time. A second copy would wait out the first one's training
# process, then start the next arm while the first copy starts it too.
$Lock = Join-Path $Q "queue.lock"
if (Test-Path $Lock) {
    $owner = [int](Get-Content $Lock -Raw)
    if (Get-Process -Id $owner -ErrorAction SilentlyContinue) {
        Say "refused: another queue (pid $owner) holds the lock"
        throw "another queue (pid $owner) holds $Lock -- not starting a second one"
    }
    Say "removing stale lock from pid $owner"
}
Set-Content -Path $Lock -Value $PID -Encoding ascii

try {
    # What this queue will do, for scripts\runstat.ps1's progress bar: which
    # arms train (and for how many epochs) and which only need probing.
    $plan = [ordered]@{
        out      = $Out
        start_at = $StartAt
        created  = (Get-Date -Format s)
        arms     = @($Train | ForEach-Object {
            [ordered]@{ name = $_; epochs = $Epochs[$_]; train = -not (Test-RunComplete $_) } })
    }
    $plan | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $Q "plan.json") -Encoding utf8

    if ($StartAt) {
        $at = [datetime]::ParseExact($StartAt, "HH:mm", $null)
        if ($at -lt (Get-Date)) { $at = $at.AddDays(1) }
        Say ("holding until {0:yyyy-MM-dd HH:mm}" -f $at)
        while ((Get-Date) -lt $at) { Start-Sleep -Seconds 30 }
    }

    $env:OMP_NUM_THREADS = "$Threads"
    Say "starting, OMP_NUM_THREADS=$Threads"

    function Invoke-Probe([string]$name) {
        $ck = Join-Path $Repo "runs\$name\best.pt"
        if (-not (Test-Path $ck)) { Say "no checkpoint for $name, skipping probe"; return }
        Say "probing $name"
        $rc = Invoke-Py (Join-Path $Q "probe_$name.txt") @("scripts\probe_velocity.py", "--checkpoint", $ck, "--seed", "0")
        Say "probe $name exit $rc -> $Out/probe_$name.txt"
        # train.py writes best_refit.pt beside best.pt: the same network with
        # its velocity head solved in closed form (src/refit.py). Both are
        # probed -- best.pt says what training reached, best_refit.pt what the
        # network can do -- and runs from before the refit step have only one.
        $rck = Join-Path $Repo "runs\$name\best_refit.pt"
        if (Test-Path $rck) {
            Say "probing $name (refit)"
            $rc = Invoke-Py (Join-Path $Q "probe_${name}_refit.txt") @("scripts\probe_velocity.py", "--checkpoint", $rck, "--seed", "0")
            Say "probe $name (refit) exit $rc -> $Out/probe_${name}_refit.txt"
        }
    }

    Wait-ForIdle

    foreach ($cfg in $Train) {
        if (Test-RunComplete $cfg) {
            Say "$cfg already has a complete run, not retraining"
        } else {
            Wait-ForIdle
            Say "training $cfg"
            $log = Join-Path $Q "$cfg.log"
            # --resume: an arm cut short -- by a STOP file, a crash or a restart
            # -- carries on from runs\<arm>\last.pt instead of starting over,
            # and matches the run that was never interrupted (test_resume.py).
            $rc = Invoke-Py $log @("-u", "src\train.py", "--config", "configs\$cfg.yaml", "--resume")
            # No such line when the run stopped early or crashed; indexing the
            # missing match threw and took the whole queue down with it.
            $hit = Select-String -Path $log -Pattern 'best val onset F: ([0-9.]+)' | Select-Object -Last 1
            $best = if ($hit) { $hit.Matches[0].Groups[1].Value } else { "none yet" }
            Say ("{0} exit {1} (best val onset F: {2})" -f $cfg, $rc, $best)
            # An arm that did not finish is not probed or marked done: its
            # best.pt is a partial run, and probing it would put a half-trained
            # model into SUMMARY under the finished arm's name.
            if (-not (Test-RunComplete $cfg)) {
                $done = @((Get-Content (Join-Path $Repo "runs\$cfg\history.json") -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json)).Count
                Say ("{0} stopped after {1} of {2} epochs; runs\{0}\last.pt holds its place" -f $cfg, $done, $Epochs[$cfg])
                Say "queue paused -- run this queue again to resume"
                return
            }
        }
        Invoke-Probe $cfg
        Set-Content -Path (Join-Path $Q "DONE_$cfg") -Value (Get-Date -Format s) -Encoding ascii
    }

    Say "queue done"
    $summary = Join-Path $Q "SUMMARY.txt"
    "=== Phase A' summary ===" | Set-Content -Path $summary -Encoding utf8
    $sections = foreach ($r in $Summarise) {
        @{ label = $r; file = "probe_$r.txt" }
        if (Test-Path (Join-Path $Q "probe_${r}_refit.txt")) { @{ label = "$r (refit)"; file = "probe_${r}_refit.txt" } }
    }
    foreach ($sec in $sections) {
        Add-Content -Path $summary -Value "", "--- $($sec.label) ---" -Encoding utf8
        $probe = Join-Path $Q $sec.file
        if (Test-Path $probe) {
            # From the per-class table down: the pooled numbers above it are the
            # ones that already misled once, and a summary is read, not studied.
            $lines = Get-Content $probe
            $start = ($lines | Select-String -Pattern 'peak \(y > 0\.95\)' | Select-Object -First 1).LineNumber
            if ($start) { Add-Content -Path $summary -Value $lines[($start - 1)..($lines.Count - 1)] -Encoding utf8 }
            else { Add-Content -Path $summary -Value $lines -Encoding utf8 }
        } else {
            Add-Content -Path $summary -Value "(no probe)" -Encoding utf8
        }
    }
    Say "summary -> $Out/SUMMARY.txt"
    Say "read it as per-class bootstrap intervals against the baseline's, never as point estimates"
} finally {
    Remove-Item -Path $Lock -ErrorAction SilentlyContinue
}
