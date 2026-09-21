<#  Phase A' queue, for Windows. The POSIX twin is run_velocity_queue.sh.

    Strictly sequential, and that is the whole design. Two training jobs on
    one machine is the documented 76 s -> 1,415 s per-epoch collapse: torch
    takes a thread per core in each process and the oversubscribed threads
    spin rather than progress. So each run waits for the last, and the queue
    waits for anything already training before it starts.

    Why a port rather than Git Bash: the .sh uses `.venv/bin/python`, which is
    the POSIX venv layout -- this machine has `.venv\Scripts\python.exe` -- and
    `pgrep`, which Git Bash does not provide.

        powershell -ExecutionPolicy Bypass -File scripts\run_velocity_queue.ps1
        powershell ... -File scripts\run_velocity_queue.ps1 -WhatIf   # plan only

    The arms trained here are A'2 (velocity_lin_cpu) and A'3
    (velocity_peak_cpu). A'1 (velocity_w5_cpu) is probed if its checkpoint is
    present and trained if it is not, so the queue is the same command whether
    or not that run has already happened.

    All four configs resolve to `device: cpu` (pinned in v1_8piece_cpu.yaml,
    asserted by tests/test_config.py). That is not a throughput choice: A'1's
    published per-class intervals were measured on CPU, and an arm measured
    somewhere else is not a comparison. Budget ~4 h per run on this machine --
    the ~100 min in ROADMAP.md came off cloud cores.
#>
param(
    [string]$Out = "runs/queue",
    [int]$Threads = 4,
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Py = Join-Path $Repo ".venv\Scripts\python.exe"
$Q  = Join-Path $Repo $Out

# A'1 first: trained if it has no checkpoint, probed either way.
$Arms = @("velocity_w5_cpu", "velocity_lin_cpu", "velocity_peak_cpu")
# What the summary reports, baseline included -- A' is read as a set of
# intervals against the baseline's, never as one arm's point estimate.
$Summarise = @("velocity_probe_cpu") + $Arms

function Say([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line
    if (-not $WhatIf) { Add-Content -Path (Join-Path $Q "queue.log") -Value $line -Encoding utf8 }
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
foreach ($cfg in $Summarise) {
    $dev = & $Py -c "import sys; sys.path.insert(0, 'src'); from build import load_config; print(load_config('configs/$cfg.yaml')['train'].get('device'))"
    if ($dev -ne "cpu") { throw "configs/$cfg.yaml resolves to device=$dev, not cpu -- the A' arms must share a device" }
}
Say "preflight ok: venv coherent, all four arms resolve to device=cpu"

if ($WhatIf) {
    Say ("would train: {0}" -f ($Arms -join ", "))
    Say ("would write: {0}" -f (Join-Path $Out "SUMMARY.txt"))
    return
}

New-Item -ItemType Directory -Force -Path $Q | Out-Null
$env:OMP_NUM_THREADS = "$Threads"

function Invoke-Probe([string]$name) {
    $ck = Join-Path $Repo "runs\$name\best.pt"
    if (-not (Test-Path $ck)) { Say "no checkpoint for $name, skipping probe"; return }
    Say "probing $name"
    & $Py scripts\probe_velocity.py --checkpoint $ck --seed 0 *> (Join-Path $Q "probe_$name.txt")
    Say "probe $name -> $Out/probe_$name.txt"
}

Wait-ForIdle

foreach ($cfg in $Arms) {
    if (Test-Path (Join-Path $Repo "runs\$cfg\best.pt")) {
        Say "$cfg already has a checkpoint, not retraining"
    } else {
        Say "training $cfg"
        $log = Join-Path $Q "$cfg.log"
        & $Py -u src\train.py --config "configs\$cfg.yaml" *> $log
        $best = (Select-String -Path $log -Pattern 'best val onset F: ([0-9.]+)' |
                 Select-Object -Last 1).Matches.Groups[1].Value
        Say ("{0} exit {1} (best val onset F: {2})" -f $cfg, $LASTEXITCODE, $best)
    }
    Invoke-Probe $cfg
}

Say "queue done"
$summary = Join-Path $Q "SUMMARY.txt"
"=== Phase A' summary ===" | Set-Content -Path $summary -Encoding utf8
foreach ($r in $Summarise) {
    Add-Content -Path $summary -Value "", "--- $r ---" -Encoding utf8
    $probe = Join-Path $Q "probe_$r.txt"
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
