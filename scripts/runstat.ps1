<#  Is the flybeats training run active, and how far along?
    Usage:  powershell -File runstat.ps1        one-shot
            powershell -File runstat.ps1 -Watch refresh every 30s
    Also draws a progress bar for the Phase A' queue when runs\queue has one
    (-Queue points it at another --Out directory).
#>
param([switch]$Watch, [string]$Queue = "runs/queue")

# [int] on a TimeSpan component ROUNDS (0.92h -> 1), which inflated every
# duration here by up to an hour. Floor the hours and take whole minutes.
function Fmt-Span([TimeSpan]$t) {
    "{0}h{1:d2}m" -f [math]::Floor($t.TotalHours), $t.Minutes
}

# The repo root, so --out paths (which are relative to it) resolve. Derived
# from the script's own location rather than hardcoded, so a clone anywhere
# reports on itself.
$Repo = Split-Path -Parent $PSScriptRoot

function Show-Snapshot {
    Write-Host ""
    Write-Host ("flybeats run  " + (Get-Date -Format "HH:mm:ss")) -ForegroundColor White

    $p = Get-CimInstance Win32_Process -Filter "Name like '%python%'" |
         Where-Object { $_.CommandLine -match 'ablations\.py|train\.py' -and
                        $_.CommandLine -notmatch 'spawn_main' } |
         Sort-Object CreationDate | Select-Object -First 1

    if (-not $p) {
        Write-Host "  process   IDLE" -ForegroundColor Red -NoNewline
        Write-Host "  no ablations.py / train.py running"
        return
    }

    $el = (Get-Date) - $p.CreationDate
    Write-Host "  process   ACTIVE" -ForegroundColor Green -NoNewline
    Write-Host ("  pid {0}  since {1}  elapsed {2}" -f `
        $p.ProcessId, $p.CreationDate.ToString("HH:mm"), (Fmt-Span $el))

    # how many child dataloader workers it currently has
    $kids = @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $($p.ProcessId)").Count
    Write-Host ("  workers   {0} dataloader child process(es)" -f $kids)

    $g = (nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu `
          --format=csv,noheader,nounits 2>$null)
    if ($g) {
        $f = $g -split ',' | ForEach-Object { $_.Trim() }
        Write-Host ("  gpu       {0}% util   {1}/{2} MiB   {3} C" -f $f[0],$f[1],$f[2],$f[3])
    }

    # progress, read from the run's own --out directory
    $outdir = $null
    if ($p.CommandLine -match '--out\s+(\S+)') { $outdir = $Matches[1].Trim('"') }
    if (-not $outdir) { return }
    $full = Join-Path $Repo $outdir
    Write-Host ("  outdir    {0}" -f $outdir)
    if (-not (Test-Path $full)) { return }

    $pts = @(Get-ChildItem "$full\*.pt" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending)
    if ($pts.Count -gt 0) {
        $last = $pts[0]
        $age  = (Get-Date) - $last.LastWriteTime
        Write-Host ("  seeds     {0} done   last {1} at {2} ({3} ago)" -f `
            $pts.Count, $last.Name, $last.LastWriteTime.ToString("HH:mm"), (Fmt-Span $age))
        if ($pts.Count -ge 2) {
            $pace = $last.LastWriteTime - $pts[1].LastWriteTime
            if ($pace.TotalSeconds -gt 0) {
                Write-Host ("  pace      {0}/seed   next eta ~{1}" -f `
                    (Fmt-Span $pace), $last.LastWriteTime.Add($pace).ToString("HH:mm"))
            }
        }
    } else {
        Write-Host "  seeds     0 done (first seed still training)"
    }

    if (Test-Path "$full\ablation_results.json") {
        Write-Host "  results   WRITTEN" -ForegroundColor Green -NoNewline
        Write-Host ("  {0}\ablation_results.json" -f $outdir)
    } else {
        Write-Host "  results   pending (written only after the final seed)"
    }
}

function Bar([double]$frac, [int]$width = 30) {
    $frac = [math]::Max(0.0, [math]::Min(1.0, $frac))
    $n = [int][math]::Floor($frac * $width)
    "[" + ("#" * $n) + ("." * ($width - $n)) + "]"
}

# The Phase A' queue (scripts\run_velocity_queue.ps1), read from the plan it
# writes at start, its own log, and each arm's training log. Progress is
# counted in epochs, the unit every arm is made of; the ETA is the mean epoch
# measured so far tonight, or the 22 Sep estimate of ~7 min before there is one.
function Show-Queue {
    $qdir = Join-Path $Repo $Queue
    $planf = Join-Path $qdir "plan.json"
    if (-not (Test-Path $planf)) { return }
    $plan = Get-Content $planf -Raw | ConvertFrom-Json
    $log = Join-Path $qdir "queue.log"
    $qlines = if (Test-Path $log) { @(Get-Content $log) } else { @() }
    $text = $qlines -join "`n"

    $total = 0; $done = 0; $secs = @(); $rows = @(); $pending = 0
    foreach ($a in @($plan.arms)) {
        $name = $a.name
        $eps = @()
        $alog = Join-Path $qdir "$name.log"
        if ($a.train -and (Test-Path $alog)) {
            $eps = @(Select-String -Path $alog -Pattern '^ep\s+\d+ .*\| ([0-9.]+)s\s*$' |
                     ForEach-Object { [double]$_.Matches[0].Groups[1].Value })
            $secs += $eps
        }
        $n = if ($a.train) { [int]$a.epochs } else { 0 }
        $total += $n
        if (Test-Path (Join-Path $qdir "DONE_$name")) {
            $done += $n
            $f = if ($text -match [regex]::Escape($name) + ' exit \d+ \(best val onset F: ([0-9.]+)\)') { "best F $($Matches[1])" } else { "probed" }
            $rows += ("    {0,-20} done      {1}" -f $name, $f)
        } elseif ($text -match "probing $([regex]::Escape($name))") {
            $done += $n
            $rows += ("    {0,-20} probing" -f $name)
        } elseif ($text -match "training $([regex]::Escape($name))") {
            $k = [math]::Min($eps.Count, $n)
            $done += $k
            $pending += $n - $k
            $rows += ("    {0,-20} training  epoch {1}/{2} {3}" -f $name, $k, $n, (Bar ($k / [math]::Max($n, 1)) 12))
        } else {
            $pending += $n
            $rows += ("    {0,-20} waiting   {1}" -f $name, $(if ($a.train) { "$n epochs" } else { "probe only" }))
        }
    }

    $last = if ($qlines.Count) { $qlines[-1] } else { "" }
    $lock = Join-Path $qdir "queue.lock"
    $alive = $false
    if (Test-Path $lock) { $alive = [bool](Get-Process -Id ([int](Get-Content $lock -Raw)) -ErrorAction SilentlyContinue) }
    $state = if ($text -match 'queue done') { "FINISHED" }
             elseif (-not $alive -and $last -match 'queue paused') { "PAUSED -- run the queue again to resume from last.pt" }
             elseif (-not $alive) { "STOPPED -- no queue process holds the lock; see $($plan.out)\queue.log" }
             elseif ($last -match 'holding until (.+)$') { "holding until $($Matches[1])" }
             else { "running" }

    $per = if ($secs.Count) { ($secs | Measure-Object -Average).Average } else { 420.0 }
    $frac = if ($total) { $done / $total } else { 1.0 }

    Write-Host ""
    Write-Host ("A' queue  {0} {1,3:N0}%  {2}/{3} epochs" -f (Bar $frac), (100 * $frac), $done, $total) -ForegroundColor White
    Write-Host ("  state     {0}" -f $state)
    if ($state -eq "running" -or $state -like "holding*") {
        # each remaining arm also gets probed, ~2 min apiece at full size
        $remain = [TimeSpan]::FromSeconds($pending * $per + 120 * @($rows | Where-Object { $_ -notmatch ' done ' }).Count)
        $from = if ($state -like "holding*") { [datetime]::ParseExact($plan.start_at, "HH:mm", $null) } else { Get-Date }
        if ($from -lt (Get-Date).AddHours(-12)) { $from = $from.AddDays(1) }
        Write-Host ("  pace      {0:N0} s/epoch {1}   eta ~{2:HH:mm}" -f $per,
            $(if ($secs.Count) { "(measured)" } else { "(estimate)" }), $from.Add($remain))
    }
    $rows | ForEach-Object { Write-Host $_ }
}

if ($Watch) { while ($true) { Clear-Host; Show-Snapshot; Show-Queue; Start-Sleep -Seconds 30 } }
else { Show-Snapshot; Show-Queue }
