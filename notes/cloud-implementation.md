# flybeats: cloud implementation steps

30 Sep 2026 · based on `main` at `1aa6865` (prereg-v3) · target: RunPod Secure Cloud, RTX A5000, one network volume

Companion to `cloud-costs.md`. The steps are in order. Nothing here has been run yet. Items marked **(check)** are assumptions to confirm on the first pod.

## 0. Decisions before any cloud run

- [ ] **Full-set step count.** `locked.yaml` has `steps.full: null`, and `train.py` refuses `--set full` until it's set. Set it from the cloud benchmark (step 5), with a new tag and a changelog line, before the first full run.
- [ ] **Seed spread.** `locked.yaml` has one `init_seed: 0`, so the spread comes from the five controls. If you want extra real-network seeds, lock that now: it adds runs.
- [ ] **Recorded step in `train.py`?** It's about a third faster. It's a training-code change, so it needs a fresh `check_cuda_graph.py` pass. Decide before the benchmark so the benchmark measures what the runs will use.
- [ ] **PyTorch build.** Stay on `torch 2.14.0+cu126`, so no Blackwell GPUs (see the costs notes).
- [ ] **Live page.** Only the real network's run (`full_real`) publishes. The controls don't.

## 1. Code changes

### 1a. Recorded step in `train.py` (optional, recommended)

- `train.py` calls the ordinary `train_step`; `memorise.py` already switches to `GraphedStep` when `graph_checked()` passes. Use the same pattern in `train.py`.
- The learning-rate schedule and validation stay the same. The recording runs once at the start (about 24 s on the laptop).
- Re-run `check_cuda_graph.py`. It passes on the code hash only, so run it on the cloud GPU too (step 5).

### 1b. Linux version of `machine.py` (for the live page's machine panel)

**What happens without it** (read from the code, not run): `cpu_static()` returns `{}`, and the sampler thread crashes once when it can't find `powershell`. The page then shows the loss curve and validation checks with no CPU/GPU readings. Training and uploads carry on.

**Structure**
- Keep the Windows code as it is. Add `scripts/machine_linux.py`, and have `machine.py` pick one based on `sys.platform`.
- Provide the same names: `Sampler` (`.latest`, `.since()`, `.every`, `.history`, `.start()`), `ROW`, `row()`, `cpu_static()`, `sample()`.
- Each reading has the same structure: `cpu`, `gpu`, `run`, `process_running`. Then `machine_log.jsonl` rows and the page are unchanged.

**Where each reading comes from**

| Reading | Linux source | On a RunPod pod |
|---|---|---|
| GPU clocks, usage, temperature, power, memory, what's holding the clock back | Same `nvidia-smi --query-gpu=...` fields; reuse `_gpu()` as is | ✅ |
| CPU name, cores | `/proc/cpuinfo`; the pod's allowance from `/sys/fs/cgroup/cpu.max` | ✅, but describes the host; the pod gets a share (9 vCPUs on an A5000) |
| CPU usage, whole pod | `/sys/fs/cgroup/cpu.stat` (`usage_usec`) over the interval ÷ the quota. Not `/proc/stat`, which shows the whole host. | ✅ |
| CPU clock, base clock | `/proc/cpuinfo` "cpu MHz"; `/sys/devices/system/cpu/cpu0/cpufreq/` | ⚠️ Often frozen or missing, so "–" |
| CPU temperature | `/sys/class/thermal`, `hwmon` | ❌ Usually hidden → `None` |
| CPU package power | RAPL (`/sys/class/powercap/intel-rapl`) | ❌ Root-only → `None` |
| This run's CPU % | `/proc/<pid>/stat` utime + stime, in clock ticks (`os.sysconf("SC_CLK_TCK")`) | ✅ |
| This run's RAM | `/proc/<pid>/status` VmRSS | ✅ |
| This run's GPU % and memory | `nvidia-smi pmon` / `--query-compute-apps`, matched by process ID | ⚠️ Containers often show host process IDs or none. The pod runs only this job, so report the whole card's figures and label them as such. |
| Is the run alive | Match the run's pattern against `/proc/*/cmdline` (NUL-separated; join with spaces); check `/proc/<pid>` exists | ✅ The existing patterns accept `/` |

**Design rules**
- **Don't slow training.** The Windows version once slowed memorisation from 8.4 to 11.7 s per step.
  - Read `/proc` and `/sys` directly, which takes microseconds.
  - Call `nvidia-smi` once every 10 s under `nice -n 19` as a separate process.
  - Don't do heavy Python work in the sampler thread.
- **Don't leave processes behind.** Use a daemon thread plus one `nvidia-smi` call per reading, not a long-lived `nvidia-smi -l` child, which would survive a hard kill.
- **Unknown values are `None`.** The page shows them as "–". `_num()` already turns `[N/A]` into `None`.
- Set `temp_source` to match whatever sensor is actually used (it currently says "ACPI thermal zone").

**Checking it**
- `python scripts/machine.py` prints one reading. Compare with `nvidia-smi` and `top` on the benchmark pod.
- Time steps with and without the chart running, to confirm it costs the run almost nothing.
- If WSL2 is set up on the laptop, `nvidia-smi` works there, so most of this can be checked before renting anything.

About 150 lines. Half a day to a day, mostly checking.

### 1c. Live-page fix for a missing base clock

`live_chart.html` lines 381 and 386 assume `cpu.base_mhz` is known. With a clock but no base, the page shows "base 0.00 GHz" and wrongly marks the CPU as boosting. Add a check for a missing base there and in the two graph helpers that read `base_mhz` (lines 500 and 576). A few lines.

### 1d. Not needed on Secure Cloud

- **Resume from `last.pt`.** Only matters on Community Cloud or spot instances, where a host can disappear mid-run.
- **Sparse matrix multiply.** A separate decision; see the costs notes.

## 2. RunPod account and volume

- [ ] Create an account and add credit. Set a low spending alert if one is offered.
- [ ] **Pick one datacenter that has six Secure Cloud A5000s free right now.** Network volumes are tied to one datacenter.
- [ ] Create a **100 GB network volume** (standard, $0.07 per GB per month) in that datacenter.
- [ ] Choose a base image with CUDA 12.x and the **same Python minor version as the laptop's `.venv`**. Check with `.venv\Scripts\python.exe --version`.
- [ ] Use the same image and GPU type for every pod. Record the image name.

## 3. Data onto the volume (one pod)

- [ ] Start one A5000 pod with the volume mounted (e.g. `/workspace`).
- [ ] Clone the repo onto the volume at the exact tag or commit the runs will use. Record `git rev-parse HEAD`.
- [ ] Download from Zenodo straight to the volume. Don't upload from the laptop.
  - Slakh2100 redux 16 kHz: record 7708270, 48,689,473,348 bytes.
  - MaleCNS v1.0 feathers, if you're rebuilding rather than copying.
- [ ] **Check the MD5s against `SOURCES.md`.** Slakh: `66a2301ed7b4d5f4f6d3383474e546c6`.
- [ ] Unpack: `python scripts/unpack_slakh.py --set full`. Delete the tarball afterwards to save space.
- [ ] **Built files: copy, don't rebuild.** Copy these from the laptop to the volume, so the runs use exactly the files the gates passed on:
  - the slice and the five control edge tables;
  - the full-set manifest, `validation_clips.json` and band stats;
  - `config/excluded_songs.txt` (already in git).

  If you rebuild instead, compare hashes with the laptop's.
- [ ] Write `config/paths.local.yaml` on the volume with Linux paths. It isn't committed.

## 4. Environment on the volume (build once, shared by all pods)

```bash
cd /workspace/flybeats
python -m venv .venv
.venv/bin/pip install -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
.venv/bin/python -m pytest
```

- [ ] Confirm `torch==2.14.0+cu126`, and that the arch list includes `sm_86` (the A5000).
- [ ] Record `nvidia-smi` (driver and CUDA version) in `reports/`.
- The README commands translate directly: `.venv\Scripts\python.exe scripts\x.py` becomes `.venv/bin/python scripts/x.py`. The `scripts/phase0/*.ps1` scripts don't run on Linux; steps 3–4 replace them.

## 5. Benchmark (same pod)

- [ ] `.venv/bin/python scripts/check_cuda_graph.py`. It must pass on this GPU.
  - It overwrites `reports/cuda_graph_check.json` in this copy of the repo. Keep the laptop's version in git, and save the cloud one alongside it (e.g. `cuda_graph_check_a5000.json`).
- [ ] `.venv/bin/python scripts/measure_cost.py --set full`. It defaults to the small set, so pass `--set full` to time loading the real data. Note seconds per step, and `data_seconds_per_batch`.
  - **Check:** is reading from the network volume slow compared with the laptop's 0.36 s? If so, copy the data to each pod's local disk at start-up.
- [ ] `python scripts/machine.py`, if 1b is done.
- [ ] Update the cost estimate with the measured speed.
- [ ] **Set `steps.full` → new tag + `CHANGELOG.md` line, before any full run.**
- [ ] **Test the auto-stop** (step 8) on this pod.

## 6. Live page (real-network pod only)

- [ ] Install rclone on the pod.
- [ ] Copy `rclone.conf` (the `flybeats` FTP remote) and `config/live.local.yaml` from the laptop. Neither is in git.
- [ ] Test: `.venv/bin/python scripts/live_chart.py --run test --publish` for one update, then check https://andrewevmiller.com/flybeats-live/.
- The FTP account is already limited to `public_html/flybeats-live`, which limits the damage if the pod is compromised.
- **Control pods:** don't copy the upload config, and never pass `--publish`. When run on its own, `live_chart.py` only uploads with `--publish`. To watch a control, run the chart without it and open port 8765 through an SSH tunnel.

## 7. Shakedown (one pod)

```bash
tmux new -s run
.venv/bin/python scripts/train.py --set baby --run baby_real_cloud ; runpodctl stop pod $RUNPOD_POD_ID
```

- [ ] Finishes, validates at step 500 and 1,000, and writes `best.pt` and `last.pt`.
- [ ] If there's a laptop baby run, compare the loss curves. Expect them to be close, not identical.
- [ ] The live page updates, and shows the final update when the run ends.

## 8. The six full-set runs

Start six pods: same datacenter, image and GPU type, all with the volume mounted. In each, in tmux:

```bash
# real network (the only one that publishes)
.venv/bin/python scripts/live_chart.py --run full_real --publish &    # or a second tmux window
.venv/bin/python scripts/train.py --set full --run full_real ; runpodctl stop pod $RUNPOD_POD_ID

# controls 1-5, one per pod
.venv/bin/python scripts/train.py --set full --control 1 --run full_control1 ; runpodctl stop pod $RUNPOD_POD_ID
```

- `;` rather than `&&`, so the pod stops even if training fails or hits the stop rule.
- **(check)** RunPod may only allow *terminating* a pod that has a network volume, not stopping it. In that case use `runpodctl remove pod $RUNPOD_POD_ID`, and make sure the repo and `runs/` live on the volume, because the pod's own disk is deleted.
- Every run writes `runs/<name>/` on the shared volume. The names differ, so there are no collisions.
- Record, per run: GPU, driver, image, commit, start and end times.

## 9. After the runs

- [ ] Copy `runs/` back to the laptop (`best.pt`, `last.pt`, `log.jsonl`, `args.json`, machine logs), e.g. `runpodctl send runs/` → `runpodctl receive <code>`.
- [ ] Commit reports and notes. Checkpoints go wherever you keep them outside git.
- [ ] **Terminate all pods. Delete the network volume** once the data and results are safe.
- [ ] Phase 4: score the six checkpoints on the test split, **once**. Forward passes only; the laptop or one pod is fine.

## Pre-flight checklist (every pod)

- [ ] `torch.__version__ == "2.14.0+cu126"`; the arch list includes the GPU.
- [ ] Same GPU type, driver and image on all six.
- [ ] `check_cuda_graph.py` passed on this GPU type with this code.
- [ ] `config/locked.yaml` is at the tag that sets `steps.full`.
- [ ] The auto-stop is on the command line.
- [ ] Only `full_real` has `--publish` and the upload config.
