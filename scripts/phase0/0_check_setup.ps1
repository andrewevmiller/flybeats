# Phase 0, setup check: confirms the one-time installs from the guide are in place.
# Read-only: it changes nothing. Run it first; fix anything marked FAIL, then move on.
#
#   powershell -ExecutionPolicy Bypass -File .\0_check_setup.ps1

$fails = 0
function Report($ok, $what, $detail) {
    if ($ok) { $tag = 'OK  ' } else { $tag = 'FAIL'; $script:fails++ }
    "{0}  {1,-18} {2}" -f $tag, $what, $detail
}

# Python 3.12
$py = ''
try { $py = (& py -3.12 --version) 2>$null } catch {}
Report ($py -match '^Python 3\.12') 'Python 3.12' $(if ($py) { $py } else { 'not found: winget install -e --id Python.Python.3.12' })

# Git
$git = ''
try { $git = (& git --version) 2>$null } catch {}
Report ($git -match '^git version') 'Git' $(if ($git) { $git } else { 'not found: winget install -e --id Git.Git' })

# NVIDIA driver: needs to list the RTX 2060 and support CUDA 12.6 or higher.
# nvidia-smi only asks the driver; it does not run anything on the GPU.
$smi = ''
try { $smi = (& nvidia-smi) -join "`n" } catch {}
$cuda = 0.0
if ($smi -match 'CUDA Version:\s*([\d.]+)') { $cuda = [double]$Matches[1] }
Report ($smi -match 'RTX 2060') 'GPU' $(if ($smi -match 'RTX 2060') { 'RTX 2060 listed' } else { 'RTX 2060 not listed by nvidia-smi' })
Report ($cuda -ge 12.6) 'CUDA (driver)' "$cuda (need 12.6 or higher; update through the NVIDIA app)"

# Execution policy, so .venv\Scripts\Activate.ps1 can run
$pol = Get-ExecutionPolicy -Scope CurrentUser
Report ($pol -in 'RemoteSigned', 'Unrestricted', 'Bypass') 'Execution policy' "$pol (want RemoteSigned: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)"

# Disk. The guide wants 60 GB free after the Slakh archive is unpacked and deleted.
# Unpacking needs the archive and the unpacked files side by side, so check again before Phase 2.
$freeGB = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
Report ($freeGB -ge 60) 'Disk C: free' "$freeGB GB"

""
"Not checked (set these by hand before long runs; undo afterwards):"
"  powercfg /change standby-timeout-ac 0      (default 30)"
"  powercfg /change hibernate-timeout-ac 0    (default 180)"
"  Power Options -> 'When I close the lid' -> Do nothing, if you close the lid during runs"
""
if ($fails -eq 0) { "All setup checks passed." } else { "$fails check(s) failed." }
exit $fails
