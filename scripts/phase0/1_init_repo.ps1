# Phase 0, steps 1, 2 and 4: start a clean repository in flybeats\.01, lay out the
# folders, write .gitignore and config/paths.local.yaml, and copy these scripts in.
#
#   New GitHub repository (create it empty on GitHub first):
#     powershell -ExecutionPolicy Bypass -File .\1_init_repo.ps1 -RemoteUrl https://github.com/<you>/<repo>.git
#   Local only (add a remote later with: git remote add origin <url>):
#     powershell -ExecutionPolicy Bypass -File .\1_init_repo.ps1
#
# Safe to re-run: it only creates what is missing. Nothing is copied from _to_delete.

param(
    [string]$RemoteUrl = '',
    [string]$Base = 'C:\Users\ricos\Documents\AI Databases\flybeats'
)
$ErrorActionPreference = 'Stop'
$Repo = Join-Path $Base '.01'
$Data = Join-Path $Base 'data'
$utf8 = New-Object System.Text.UTF8Encoding($false)   # no BOM

function Write-IfMissing($path, $text) {
    if (Test-Path $path) { "  kept      $path"; return }
    [IO.File]::WriteAllText($path, $text, $utf8)
    "  wrote     $path"
}

# 1. Clean repository
if (Test-Path (Join-Path $Repo '.git')) {
    "Repository already exists at $Repo; filling in anything missing."
} elseif ((Test-Path $Repo) -and (Get-ChildItem -Force $Repo | Select-Object -First 1)) {
    $nested = Get-ChildItem -Directory $Repo | Where-Object { Test-Path (Join-Path $_.FullName '.git') } | Select-Object -First 1
    if ($nested) {
        throw "$Repo is not a git repository, but $($nested.FullName) is: the repo was cloned one folder too deep. Move its .git up into $Repo, delete the empty folder, and run this again."
    }
    throw "$Repo exists, is not empty and is not a git repository. Move it aside first."
} elseif ($RemoteUrl) {
    git clone $RemoteUrl $Repo
    if ($LASTEXITCODE -ne 0) { throw "git clone failed" }
} else {
    New-Item -ItemType Directory -Force $Repo | Out-Null
    git -C $Repo init -b main
    if ($LASTEXITCODE -ne 0) { throw "git init failed" }
}

# 2. Folders. src/ is deliberately NOT created: the gate says prereg-v1 must exist
# before any file under src/ does.
foreach ($d in 'config', 'scripts\phase0', 'tests', 'reports') {
    New-Item -ItemType Directory -Force (Join-Path $Repo $d) | Out-Null
}
New-Item -ItemType Directory -Force (Join-Path $Data 'work') | Out-Null

# Where the large data lives (untracked; nothing is copied into the repo)
$dataFwd = $Data -replace '\\', '/'
Write-IfMissing (Join-Path $Repo 'config\paths.local.yaml') @"
malecns_dir: $dataFwd/malecns
slakh_dir:   $dataFwd/slakh
work_dir:    $dataFwd/work
"@

# 4. .gitignore. The guide's list, with two entries anchored to the repo root:
# a bare "data/" would also ignore src/flybeats/data/, the Phase 1 package.
Write-IfMissing (Join-Path $Repo '.gitignore') @"
.venv/
__pycache__/
/data/
/runs/
config/paths.local.yaml
*.npz
*.pt
"@

# Copy these scripts and the settings draft into the repo
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Copy-Item (Join-Path $here '*.ps1') (Join-Path $Repo 'scripts\phase0') -Force
Copy-Item (Join-Path $here 'README.md') (Join-Path $Repo 'scripts\phase0') -Force
Write-IfMissing (Join-Path $Repo 'config\locked.yaml') ([IO.File]::ReadAllText((Join-Path $here 'locked.yaml')))

""
"Repository ready at $Repo"
"Next: scripts\phase0\2_build_env.ps1"
