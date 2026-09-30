# Phase 0, step 7 and the gate: check everything, then commit the settings and tag prereg-v1.
# Refuses (and changes nothing) unless:
#   - config\locked.yaml parses, says version: prereg-v1, and has no "# DECIDE" lines left
#   - SOURCES.md exists and has no TODO left
#   - reports\phase0_verify.txt shows all three MaleCNS files OK and no FAIL
#   - reports\phase0_gpu.txt starts with True
#   - nothing exists under src\ yet, and the tag prereg-v1 does not
# It commits and tags locally; it does not push.
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\phase0\4_lock_and_tag.ps1

param([string]$Repo = 'C:\Users\ricos\Documents\AI Databases\flybeats\.01')
$ErrorActionPreference = 'Stop'
Set-Location $Repo
$py = Join-Path $Repo '.venv\Scripts\python.exe'
$problems = @()

# Settings file
$locked = 'config\locked.yaml'
if (-not (Test-Path $locked)) { $problems += "missing $locked" }
else {
    $decide = Select-String -Path $locked -Pattern '#\s*DECIDE'
    foreach ($d in $decide) { $problems += "undecided setting, line $($d.LineNumber): $($d.Line.Trim())" }
    $ver = & $py -c "import yaml; print(yaml.safe_load(open(r'$locked', encoding='utf-8'))['version'])"
    if ($LASTEXITCODE -ne 0) { $problems += "$locked does not parse as YAML" }
    elseif ($ver -ne 'prereg-v1') { $problems += "$locked says version '$ver', expected prereg-v1" }
}

# Provenance
if (-not (Test-Path 'SOURCES.md')) { $problems += 'missing SOURCES.md (run 3_verify_data.ps1)' }
elseif (Select-String -Path 'SOURCES.md' -Pattern 'TODO' -Quiet) { $problems += 'SOURCES.md still has a TODO' }

# Gate: data
$verify = 'reports\phase0_verify.txt'
if (-not (Test-Path $verify)) { $problems += "missing $verify (run 3_verify_data.ps1)" }
else {
    $v = Get-Content $verify
    if (($v -match '^OK\s+MaleCNS').Count -ne 3) { $problems += 'not all three MaleCNS files are OK' }
    if (($v -match '^FAIL').Count -gt 0) { $problems += "$verify has a FAIL line" }
}

# Gate: GPU
$gpu = 'reports\phase0_gpu.txt'
if (-not (Test-Path $gpu) -or -not ((Get-Content $gpu -TotalCount 1) -match '^True ')) {
    $problems += "GPU check has not passed (run 2_build_env.ps1 without -SkipGpuCheck)"
}

# Gate: tag before code
if ((Test-Path 'src') -and (Get-ChildItem -Recurse -File 'src' | Select-Object -First 1)) {
    $problems += 'files already exist under src\; prereg-v1 must come first'
}
git rev-parse -q --verify 'refs/tags/prereg-v1' *> $null
if ($LASTEXITCODE -eq 0) { $problems += 'tag prereg-v1 already exists; later changes get a new tag and a CHANGELOG line' }

if ($problems) {
    'Not tagging. Fix these first:'
    $problems | ForEach-Object { "  - $_" }
    exit 1
}

# Changelog: the first entry
$log = Join-Path $Repo 'config\CHANGELOG.md'   # full path: .NET calls ignore PowerShell's current folder
if (-not (Test-Path $log)) {
    $today = Get-Date -Format 'yyyy-MM-dd'
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [IO.File]::WriteAllText($log, "# Settings changelog`n`nEach change to config/locked.yaml gets a new tag and a dated line here, written before the run that uses it.`n`n- $today - prereg-v1: first locked settings.`n", $utf8)
}

git add .gitignore SOURCES.md requirements.lock.txt config/locked.yaml config/CHANGELOG.md scripts reports
git commit -m "Phase 0: pinned environment, verified data, locked settings (prereg-v1)"
if ($LASTEXITCODE -ne 0) { throw 'git commit failed' }
git tag -a prereg-v1 -m 'Pre-registered settings, before any training code'
if ($LASTEXITCODE -ne 0) { throw 'git tag failed' }

""
"Tagged prereg-v1. Phase 0 gate passed."
"To publish: git push origin HEAD --tags"
