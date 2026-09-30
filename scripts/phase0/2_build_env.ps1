# Phase 0, step 3: build the pinned environment in .01\.venv and write requirements.lock.txt.
# The last step runs one small CUDA check (is the GPU visible to PyTorch?). Pass -SkipGpuCheck
# to build without touching the GPU; the lock script then refuses to tag until the check passes.
#
#   powershell -ExecutionPolicy Bypass -File .\scripts\phase0\2_build_env.ps1

param(
    [switch]$SkipGpuCheck,
    [string]$Repo = 'C:\Users\ricos\Documents\AI Databases\flybeats\.01'
)
$ErrorActionPreference = 'Stop'
Set-Location $Repo
$py = Join-Path $Repo '.venv\Scripts\python.exe'

if (-not (Test-Path $py)) {
    py -3.12 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
}

# The venv's own python is called directly, so activation (and its execution policy) is not needed.
& $py -m pip install --upgrade pip
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw "torch install failed" }
& $py -m pip install numpy pandas pyarrow scipy pretty_midi soundfile pyyaml tqdm pytest
if ($LASTEXITCODE -ne 0) { throw "package install failed" }

# Plain ASCII: PowerShell 5.1's ">" would write UTF-16, which pip reads badly.
& $py -m pip freeze | Out-File -Encoding ascii requirements.lock.txt
"wrote requirements.lock.txt ($((Get-Content requirements.lock.txt).Count) packages)"

if ($SkipGpuCheck) {
    "GPU check skipped. Re-run without -SkipGpuCheck before 4_lock_and_tag.ps1."
    exit 0
}

# Expect: True NVIDIA GeForce RTX 2060 with Max-Q Design
$gpu = & $py -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
$gpu
$gpu | Out-File -Encoding ascii reports\phase0_gpu.txt
if ($gpu -notmatch '^True ') { "GPU check FAILED: PyTorch cannot see the GPU."; exit 1 }
"GPU check OK. Next: scripts\phase0\3_verify_data.ps1"
