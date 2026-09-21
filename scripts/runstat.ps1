<#  Is the flybeats training run active, and how far along?
    Usage:  powershell -File runstat.ps1        one-shot
            powershell -File runstat.ps1 -Watch refresh every 30s
#>
param([switch]$Watch)

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

if ($Watch) { while ($true) { Clear-Host; Show-Snapshot; Start-Sleep -Seconds 30 } }
else { Show-Snapshot }
