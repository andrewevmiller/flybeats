# Phase 0 scripts

Run in order, in PowerShell. Each one prints OK/FAIL lines and stops on failure.

| Script | Guide step | Touches |
| --- | --- | --- |
| `0_check_setup.ps1` | Setup: install once | nothing (read-only) |
| `1_init_repo.ps1 [-RemoteUrl <url>]` | 1, 2, 4 | creates `flybeats\.01`, `.gitignore`, `config\paths.local.yaml`, `config\locked.yaml` draft |
| `2_build_env.ps1 [-SkipGpuCheck]` | 3 | `.venv`, `requirements.lock.txt`; one small CUDA check |
| `3_verify_data.ps1 [-RehashSlakh]` | 5, 6 | `reports\phase0_verify.txt`, `SOURCES.md` (MaleCNS check needs internet) |
| `4_lock_and_tag.ps1` | 7 + gate | commits and tags `prereg-v1` locally, only if every gate check passes |

Scripts 0 and 1 run from this folder. From script 2 on, run the copies in `.01\scripts\phase0\`.

Before step 4: read `SOURCES.md` and `config\locked.yaml` once more. The lock script refuses to tag while any `# DECIDE` line or TODO remains.
