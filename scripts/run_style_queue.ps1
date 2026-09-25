<#  Overnight style-dial queue (24-25 Sep 2026), for Windows. Armed by hand,
    started by a signal.

    An unattended session cannot start a script: `Start-Process powershell
    ... -File x.ps1` always asks for approval, even with an exact allow rule,
    and nobody is there at night to give it (dry run, 24 Sep). So this runner
    is started while Andrew is at the machine, and then it waits. The 23:00
    session builds the style-path fix and the two configs, tests them, and
    writes runs\style\GO with its file tools, which need no approval.
    Training starts within a poll of that.

        powershell -ExecutionPolicy Bypass -File scripts\run_style_queue.ps1          # arm
        powershell ... -File scripts\run_style_queue.ps1 -WhatIf                        # plan only

    Control files in runs\style\, read on every poll, so none needs a relaunch:
      GO          start. If the configs are missing or not device=cpu, the GO
                  is logged as ignored and the wait goes on; write GO again
                  after fixing them. If Windows Update has a restart
                  pending, the runner refuses and exits without training.
      DISARM      stop waiting and exit without training.
      GIVE_UP_AT  "yyyy-MM-dd HH:mm". With no accepted GO by then, exit.
                  Rewrite it to move the deadline, e.g. when the 23:00 task
                  is rescheduled.

    After GO, per arm: train (resuming from last.pt if cut short), then
    measure_variety.py and kit_check.py on its best.pt; then the side-by-side
    comparison in COMPARE.txt. Strictly sequential, like
    run_velocity_queue.ps1, whose lock, log lines and plan.json this copies,
    so that
        powershell -File scripts\runstat.ps1 -Queue runs/style
    draws its progress bar. A relaunch after a crash or restart finds GO
    already there, skips every arm with a DONE_ marker and resumes the one in
    progress.
#>
param(
    [string]$Out = "runs/style",
    [int]$Threads = 8,
    [int]$MeasureThreads = 4,
    [string[]]$Arms = @("style_pc1_cpu", "style_pc1_gaps_cpu"),
    [string]$Baseline = "velocity_probe_cpu",
    [string]$GiveUpAt = "2026-09-25 03:00",
    [int]$PollSeconds = 30,
    [switch]$WhatIf
)

$ErrorActionPreference = "Stop"
# Under `powershell -File`, "-Arms a,b" arrives as the one string "a,b".
$Arms = @($Arms | ForEach-Object { $_ -split ',' } | Where-Object { $_ })
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$Py = Join-Path $Repo ".venv\Scripts\python.exe"
$Q  = Join-Path $Repo $Out
if (-not $WhatIf) { New-Item -ItemType Directory -Force -Path $Q | Out-Null }
$GoFile       = Join-Path $Q "GO"
$DisarmFile   = Join-Path $Q "DISARM"
$DeadlineFile = Join-Path $Q "GIVE_UP_AT"
$Stamp        = "yyyy-MM-dd HH:mm"

function Say([string]$msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Write-Host $line
    if (-not $WhatIf) { Add-Content -Path (Join-Path $Q "queue.log") -Value $line -Encoding utf8 }
}

# Stderr folded into the log as text: under "Stop", 5.1 turns each stderr line
# of a native command into a terminating error (see run_velocity_queue.ps1).
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

# Done means the history holds every epoch the config asked for; best.pt
# alone only says a run started.
function Test-RunComplete([string]$name) {
    $hist = Join-Path $Repo "runs\$name\history.json"
    if (-not (Test-Path (Join-Path $Repo "runs\$name\best.pt")) -or -not (Test-Path $hist)) { return $false }
    $done = @((Get-Content $hist -Raw | ConvertFrom-Json)).Count
    return $done -ge $script:Epochs[$name]
}

# Why the arms can't start yet, or "" when they can. Fills $script:Epochs.
# The configs are written by the 23:00 session, so they are checked at GO,
# not at arming.
function Test-Configs {
    $script:Epochs = @{}
    foreach ($cfg in $Arms) {
        if (-not (Test-Path (Join-Path $Repo "configs\$cfg.yaml"))) { return "configs\$cfg.yaml does not exist" }
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        $res = & $Py -c "import sys; sys.path.insert(0, 'src'); from build import load_config; t = load_config('configs/$cfg.yaml')['train']; print(t.get('device'), t.get('epochs', 10))" 2>&1
        $rc = $LASTEXITCODE
        $ErrorActionPreference = $prev
        # The answer line, not the last line: a warning on stderr can follow it.
        $line = @($res | ForEach-Object { "$_".Trim() } | Where-Object { $_ -match '^\S+\s+\d+$' }) | Select-Object -Last 1
        if ($rc -ne 0 -or -not $line) { return "configs\$cfg.yaml does not load: $(@($res)[-1])" }
        $dev, $ep = $line -split '\s+'
        if ($dev -ne "cpu") { return "configs\$cfg.yaml resolves to device=$dev, not cpu" }
        $script:Epochs[$cfg] = [int]$ep
    }
    return ""
}

# Windows Update's pending-restart flags. A restart killed an overnight queue
# at 01:59 on 23 Sep. Read here, by the runner, because a session's
# `Test-Path HKLM:\...` prompts even under a `Test-Path *` allow rule (dry run,
# 24 Sep), and the session that writes GO is unattended.
function Test-RebootPending {
    (Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired') -or
    (Test-Path 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending')
}

function Read-Deadline([datetime]$current) {
    if (-not (Test-Path $DeadlineFile)) { return $current }
    $t = (Get-Content $DeadlineFile -Raw).Trim()
    try { return [datetime]::ParseExact($t, $Stamp, $null) }
    catch {
        # Said once per bad value, not on every poll.
        if ($t -ne $script:BadDeadline) { Say "GIVE_UP_AT holds '$t', not $Stamp; keeping $($current.ToString($Stamp))" }
        $script:BadDeadline = $t
        return $current
    }
}

# --- preflight: the failures that waste a whole overnight ------------------
if (-not (Test-Path $Py)) { throw "no venv interpreter at $Py -- see SETUP.md" }
& $Py -c "import sys; sys.path.insert(0, 'scripts'); from pathlib import Path; import bootstrap_local as b; m = b.abi_mismatch(Path('.venv'), Path(sys.executable)); sys.exit(m or 0)"
if ($LASTEXITCODE -ne 0) { throw "venv preflight failed -- run: py -3.12 scripts\bootstrap_local.py --check" }
$deadline = [datetime]::ParseExact($GiveUpAt, $Stamp, $null)

if ($WhatIf) {
    Say "preflight ok: venv coherent"
    Say ("Windows Update restart pending: {0}" -f (Test-RebootPending))
    Say ("would wait for {0}\GO until {1} (or the time in {0}\GIVE_UP_AT)" -f $Out, (Read-Deadline $deadline).ToString($Stamp))
    $why = Test-Configs
    if ($why) { Say "configs not ready yet: $why" }
    else { foreach ($cfg in $Arms) { Say ("would train {0}: {1} epochs, then measure variety and kit check" -f $cfg, $Epochs[$cfg]) } }
    Say ("would compare against runs\variety\{0}\variety.json -> {1}\COMPARE.txt" -f $Baseline, $Out)
    return
}

# One queue at a time, as in run_velocity_queue.ps1.
$Lock = Join-Path $Q "queue.lock"
if (Test-Path $Lock) {
    $owner = [int](Get-Content $Lock -Raw)
    if (Get-Process -Id $owner -ErrorAction SilentlyContinue) {
        Say "refused: another copy (pid $owner) holds the lock"
        throw "another copy (pid $owner) holds $Lock -- not starting a second one"
    }
    Say "removing stale lock from pid $owner"
}
Set-Content -Path $Lock -Value $PID -Encoding ascii

try {
    # A relaunch keeps a deadline someone moved; a first arming writes it.
    if (-not (Test-Path $DeadlineFile)) { Set-Content -Path $DeadlineFile -Value $GiveUpAt -Encoding ascii }
    $deadline = Read-Deadline $deadline
    Say ("armed (pid {0}): waiting for {1}\GO, giving up at {2}" -f $PID, $Out, $deadline.ToString($Stamp))

    $rejected = $null      # the LastWriteTime of a GO that failed Test-Configs
    while ($true) {
        if (Test-Path $GoFile) {
            $at = (Get-Item $GoFile).LastWriteTime
            if ($at -ne $rejected) {
                $why = Test-Configs
                if (-not $why -and (Test-RebootPending)) {
                    # Training into a pending restart loses the night; refuse
                    # rather than wait, since only Andrew may deal with updates.
                    Say "refused: Windows Update has a restart pending (reboot flags set), nothing trained. Restart or pause updates, then re-arm"
                    return
                }
                if (-not $why) { Say ("GO accepted (written {0:HH:mm})" -f $at); break }
                Say "GO ignored: $why. Fix it and write GO again"
                $rejected = $at
            }
        }
        if (Test-Path $DisarmFile) { Say "disarmed: found $Out\DISARM, exiting without training"; return }
        $moved = Read-Deadline $deadline
        if ($moved -ne $deadline) { $deadline = $moved; Say ("deadline moved to {0}" -f $deadline.ToString($Stamp)) }
        if ((Get-Date) -gt $deadline) { Say ("gave up: no accepted GO by {0}, nothing trained" -f $deadline.ToString($Stamp)); return }
        Start-Sleep -Seconds $PollSeconds
    }

    # For scripts\runstat.ps1 -Queue runs/style. Written only now: the epochs
    # come from configs that did not exist at arming.
    $plan = [ordered]@{
        out      = $Out
        start_at = ""
        created  = (Get-Date -Format s)
        arms     = @($Arms | ForEach-Object {
            [ordered]@{ name = $_; epochs = $Epochs[$_]; train = -not (Test-RunComplete $_) } })
    }
    $plan | ConvertTo-Json -Depth 4 | Set-Content -Path (Join-Path $Q "plan.json") -Encoding utf8

    $env:OMP_NUM_THREADS = "$Threads"
    Say "starting, OMP_NUM_THREADS=$Threads"

    function Invoke-Measure([string]$name) {
        $ck = Join-Path $Repo "runs\$name\best.pt"
        if (-not (Test-Path $ck)) { Say "no checkpoint for $name, skipping measurement"; return }
        Say "probing $name (variety)"
        $rc = Invoke-Py (Join-Path $Q "variety_$name.txt") @("scripts\measure_variety.py", "--checkpoint", $ck, "--threads", "$MeasureThreads")
        Say "variety $name exit $rc -> runs/variety/$name/report.txt"
        Say "probing $name (kit check)"
        $rc = Invoke-Py (Join-Path $Q "kit_check_$name.txt") @("scripts\kit_check.py", "--checkpoint", $ck, "--threads", "$MeasureThreads")
        Say "kit check $name exit $rc -> $Out/kit_check_$name.txt"
    }

    foreach ($cfg in $Arms) {
        if (Test-Path (Join-Path $Q "DONE_$cfg")) { Say "$cfg already done, skipping"; continue }
        if (Test-RunComplete $cfg) {
            Say "$cfg already has a complete run, not retraining"
        } else {
            Wait-ForIdle
            Say "training $cfg"
            $log = Join-Path $Q "$cfg.log"
            $rc = Invoke-Py $log @("-u", "src\train.py", "--config", "configs\$cfg.yaml", "--resume")
            $hit = Select-String -Path $log -Pattern 'best val onset F: ([0-9.]+)' | Select-Object -Last 1
            $best = if ($hit) { $hit.Matches[0].Groups[1].Value } else { "none yet" }
            Say ("{0} exit {1} (best val onset F: {2})" -f $cfg, $rc, $best)
            # A partial run is neither measured nor marked done: its best.pt
            # would be reported under the finished arm's name.
            if (-not (Test-RunComplete $cfg)) {
                $done = @((Get-Content (Join-Path $Repo "runs\$cfg\history.json") -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json)).Count
                Say ("{0} stopped after {1} of {2} epochs; runs\{0}\last.pt holds its place" -f $cfg, $done, $Epochs[$cfg])
                Say "queue paused -- run this queue again to resume"
                return
            }
        }
        Invoke-Measure $cfg
        Set-Content -Path (Join-Path $Q "DONE_$cfg") -Value (Get-Date -Format s) -Encoding ascii
    }

    if (-not (Test-Path (Join-Path $Q "DONE_compare"))) {
        $base = "runs\variety\$Baseline\variety.json"
        if (-not (Test-Path $base)) {
            Say "no baseline variety for $Baseline; measuring it"
            $rc = Invoke-Py (Join-Path $Q "variety_$Baseline.txt") @("scripts\measure_variety.py", "--checkpoint", "runs\$Baseline\best.pt", "--threads", "$MeasureThreads")
            Say "variety $Baseline exit $rc"
        }
        $files = @(@($base) + @($Arms | ForEach-Object { "runs\variety\$_\variety.json" }) | Where-Object { Test-Path $_ })
        $rc = Invoke-Py (Join-Path $Q "COMPARE.txt") (@("scripts\measure_variety.py", "--compare") + $files)
        Say "compare exit $rc -> $Out/COMPARE.txt"
        Set-Content -Path (Join-Path $Q "DONE_compare") -Value (Get-Date -Format s) -Encoding ascii
    }
    Say "queue done"
} finally {
    Remove-Item -Path $Lock -ErrorAction SilentlyContinue
}
