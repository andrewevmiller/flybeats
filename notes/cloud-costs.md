# flybeats: cloud training costs

30 Sep 2026 · based on `main` at `1aa6865` (prereg-v3)

> **Status: estimates.** The GPU speeds are my estimates from memory bandwidth. Nothing in them has been measured on a cloud GPU. A 15-minute benchmark (`measure_cost.py`, `check_cuda_graph.py`) on the chosen GPU replaces them. Costs scale in a straight line with seconds per step and with the full-set step count, which `locked.yaml` leaves open (`steps.full: null`).

## Summary

- **Recommended:** RunPod Secure Cloud, one RTX A5000 per run ($0.27/hr), with one shared network volume for the data.
- **Runs:** 8 if nothing breaks (a benchmark, a small-set shakedown, the 6 full-set runs, test scoring), or 14 with one full redo of the six.
- **Total:** about **$15–35**, depending on the full-set step count. See [Costs on RunPod Secure A5000](#costs-on-runpod-secure-a5000).
- **For comparison:** 10,000 steps × 6 runs on the laptop is about six days back to back.
- **The biggest cost risk is not compute.** It's pods and storage left running after the work is done.

## The workload, from the repo

| Fact | Value | Source |
|---|---|---|
| Neurons in the slice | 3,302 | `reports/slice_report.json` |
| Connections | 108,593 | `reports/slice_report.json` |
| Weight matrix as built | dense 3,302 × 3,302 = 10.9 M entries, 44 MB (float32), about 1% filled | `model/network.py`, `weight_matrix()` |
| Timesteps per clip | 3,200 (16 s at 5 ms), run one after another | `locked.yaml`, `network.run()` |
| Batch | 8 clips, whole-clip backprop | `locked.yaml` |
| Peak GPU memory | 1.3 GB | `reports/cost.json` |
| Laptop speed (RTX 2060 Max-Q) | 7.9 s/step ordinary; 5.4 s/step recorded (CUDA graph, 1.45× faster) | `reports/cost.json`, `reports/cuda_graph_check.json` |
| Training set | 1,162 songs, 79.9 h | `reports/data_report_full.json`, `config/CHANGELOG.md` |
| Validation | 270 songs, 18.3 h; about 1,080 clips (4 per song) every 500 steps | same, `data/sampler.py` |
| Test | 151 songs, 11.0 h, scored once | same |

What this means for the choice of GPU:

- **Memory size doesn't matter.** Any 16–24 GB card is plenty.
- **Speed follows memory bandwidth.** Each timestep reads the whole 44 MB matrix, and the backward pass writes a gradient of the same size.
- **There's a floor.** The 3,200 timesteps run one after another, so past about A100/H100 class a faster card barely helps.
- **`train.py` doesn't use the recorded step yet.** Only `memorise.py` uses `GraphedStep`. The ordinary step is limited by one Python thread launching GPU work (about 2.4 s of the laptop's 7.8 s). So with `train.py` as it is, any cloud GPU is probably stuck at about **2.5–3 s per step**.

## Estimated speed by GPU (recorded step)

| GPU | Memory bandwidth | Est. s/step | Note |
|---|---|---|---|
| RTX 2060 Max-Q (laptop) | ~264 GB/s | 5.4 (measured) | |
| L4 | 300 GB/s | ~4.7 | no faster than the laptop |
| A10 / A10G | 600 GB/s | ~2.5 | |
| RTX A5000 | 768 GB/s | ~2.0 | |
| RTX 3090 | 936 GB/s | ~1.6 | |
| RTX 4090 | 1,008 GB/s | ~1.3 | 72 MB on-chip cache could help more |
| A100 40 GB | 1,555 GB/s | ~1.0 | |
| A100 80 GB | ~2,000 GB/s | ~0.8 | |
| H100 SXM | 3,350 GB/s | ~0.6 | near the step-by-step floor |
| RTX 5090 (Blackwell) | 1,792 GB/s | ~0.7–0.9 | needs a new PyTorch build; see below |
| B200 (Blackwell) | ~8,000 GB/s | ~0.4 | limited by the floor; needs a new build |

## Providers compared

Assumptions for this table:
- 30,000 steps in total (6 runs × 5,000 steps).
- The recorded step, with validation time left out (it adds about 10%).
- All six runs in parallel, so "Each run" is also the time you wait.
- Prices as listed on or near 30 Sep 2026.

| Provider | GPU, $/hr | Whole experiment | Each run | Notes |
|---|---|---|---|---|
| Laptop | RTX 2060 Max-Q | $0 | ~45 h in total, one at a time | The known setup. Retrospective: Windows Update restarts; never run two at once. |
| RunPod Community | 3090, $0.22 | ~$3 | ~2.2 h | Cheapest. No network volumes. |
| RunPod Community | 4090, $0.34 | ~$3.70 | ~1.8 h | |
| Vast.ai | 4090, ~$0.35 typical (from $0.14) | ~$3.80 | ~1.8 h | Marketplace; hosts vary |
| **RunPod Secure** | **A5000, $0.27** | **~$4.50** | **~2.8 h** | **Recommended** |
| Jarvislabs | A100 40 GB, $0.89 | ~$7.40 | ~1.4 h | |
| RunPod Secure | 5090, $0.99 | ~$7 | ~1.1 h | Needs a new PyTorch build |
| RunPod Secure | 4090, $0.74 | ~$8 | ~1.8 h | The faster alternative to the A5000 |
| RunPod Community | A100 80 GB PCIe, $1.19 | ~$8 | ~1.1 h | |
| RunPod Secure | A100, $1.59 | ~$11 | ~1.1 h | |
| Lambda | A100 40 GB, $1.99 / H100 PCIe, $3.29 | ~$17 / ~$22 | ~1.4 h / ~1.1 h | Reliable; no egress fees |
| Modal | A100 80 GB, $2.50 + CPU/RAM | ~$18, covered by the $30/month free credit | ~1.1 h | Can be interrupted unless you pay 3× |
| RunPod Secure | H100 SXM, $3.49 | ~$18 | <1 h | |
| AWS | g5.xlarge A10G, ~$1.01 | ~$21 | ~3.5 h | Windows machines available |
| GCP | A100 40 GB, $3.67 (spot ~$1.10–2.22) | ~$30 (spot $9–18) | ~1.4 h | Windows machines available; needs a GPU quota request |

**Why RunPod Secure over the cheaper options:**
- Only Secure Cloud pods can attach a **network volume**, so Slakh (48.7 GB) is downloaded once and all six pods read the same copy. On Community Cloud or Vast, each pod downloads its own 49 GB from Zenodo while you pay for the GPU.
- **`train.py` can't resume.** It saves `best.pt` at validation and `last.pt` only at the end. A host dropping out costs a whole run, so reliability matters more than a few dollars.
- The difference between Secure and Community is about $1–4 for the whole experiment.

**Cheap-looking options that don't fit:**
- **Modal:** free within the credit, but runs can be interrupted without the 3× surcharge, and it's function-based rather than a terminal you start and watch.
- **Salad:** consumer PCs, containers only, and runs can be interrupted.
- **Thunder Compute:** as far as I know its GPUs are attached over the network, which would hurt a job that launches GPU work for 3,200 timesteps in a row. Check before using it.
- **L4 anywhere:** no faster than the laptop.
- **Hyperscalers (AWS, GCP, Azure):** two to five times the price. The one real advantage is Windows machines, which would run the PowerShell scripts and `machine.py` unchanged.

## Blackwell (RTX 5090, RTX Pro 6000, B200)

- **The pinned environment can't use it.** Blackwell support arrived in CUDA 12.8, and `torch 2.14.0+cu126` is built with CUDA 12.6. It would fail with "no kernel image is available". Confirm with `torch.cuda.get_arch_list()`.
- **Switching needs no code changes.**
  - Change the index URL in `scripts/phase0/2_build_env.ps1` line 22 (`cu126` → `cu128` or `cu130`), rebuild, and re-freeze `requirements.lock.txt`.
  - The newer build needs NVIDIA driver 570 or newer (for 12.8) or 580 or newer (for 13.0) on any machine that uses it.
  - It's an environment change under the pre-registration: a changelog line, a new tag, and re-running pytest, the stability gate, `check_cuda_graph.py` and `measure_cost.py`.
  - **Trap:** the CUDA-graph check's fingerprint hashes only the code, so it won't notice a PyTorch change. Re-run it on purpose.
- **Savings are small.** A 5090 ($0.99) is about as fast as an A100 at about 60% of the price. That's roughly $4 saved on the experiment, or about 40 minutes per run compared with a 4090. The RTX Pro 6000 only adds memory you don't need. The B200 costs about three times as much for a modest gain.
- **Verdict:** stay on cu126. Revisit only if the full-set step count turns out to be much larger. If you do switch, do it before tagging the full-set step count, so all six runs share one environment.

## How many runs

| Run | Count | What it's for |
|---|---|---|
| Benchmark (`measure_cost.py`, `check_cuda_graph.py`) | 1 short | Measured speed on the cloud GPU; re-checks the recorded step there |
| Small-set shakedown (`train.py --set baby --run baby_real`, 1,000 steps) | 1 | Proves the cloud setup end to end |
| **Full-set runs** | **6** | Real network (`full_real`) plus controls 1–5 (`--control N`) |
| Test-split scoring | 1, cheap | Forward passes only, done once |
| Contingency: redo all six | +6 | Only for a run broken by a bug. A run that stops under the stop rule is a result, not a reason to rerun. |

**Open question:** `locked.yaml` has a single `init_seed: 0`, so as written the "seed spread" in the win rule comes from the five controls. If it should also include training seeds of the real network, add one full run per seed.

## Costs on RunPod Secure A5000

Assumptions:
- **$0.27/hr.**
- **About 3 s/step** with `train.py` as it is (ordinary step), or about 2 s with the recorded step.
- **Validation** adds about 10%.
- **Fixed overhead, about $5:**
  - data download and unpack: about $1;
  - environment setup and pod start-up: about $0.40;
  - the benchmark and the shakedown: about $0.50;
  - a 100 GB network volume for about two weeks: about $3.50.
- **No data-transfer fees** on RunPod.

| Steps per full run | Hours per run | Six runs | **Total** | With one redo of all six |
|---|---|---|---|---|
| 5,000 | 4.6 | $7 | **~$13** | ~$20 |
| 10,000 | 9.2 | $15 | **~$20** | ~$35 |
| 20,000 | 18.3 | $30 | **~$35** | ~$65 |

With the recorded step in `train.py`, the totals are roughly **$10 / $15 / $25**. For reference, 2,247 steps is about one pass over the training audio (79.9 h ÷ 128 s of audio per step).

## Cost risks

1. **Pods left running.** A finished pod bills until it's stopped or removed. Six A5000s forgotten overnight is about $16, as much as the whole experiment. End every run command with an automatic stop (see the implementation notes).
2. **The network volume** bills until deleted: about $7 a month for 100 GB standard, or $0.14 per GB per month for the high-performance tier.
3. **Long runs.** At 20,000 steps each run is about 18 hours, so runs go overnight without you watching. That cuts against the v0.1 rule of watching every run, and makes the auto-stop more important.

## Ways to cut the cost further

- **Use the recorded step in `train.py`:** about a third off the time. It's a change to the training code, so it needs a fresh `check_cuda_graph.py` pass.
- **Use a sparse matrix multiply:** the weight matrix is only about 1% filled. Multiplying only the 108,593 real connections would cut the per-step memory traffic sharply, maybe several times faster even on the laptop. The model and the science stay the same; only floating-point rounding changes.
  - It needs a changelog line ("no settings change") and a fresh CUDA-graph check.
  - **Catch:** `index_add_` on a GPU sums in a slightly different order each run, so two identical runs would no longer match exactly (today they show 0.0 difference). PyTorch's CSR sparse multiply may avoid that. Test both inside the recorded step.
  - It also shrinks the advantage of faster GPUs, since the step-by-step floor would then dominate.

## Sources

- [RunPod pricing (updated 27 Sep 2026)](https://www.runpod.io/pricing)
- [RunPod RTX 4090: Community $0.34, Secure $0.74](https://www.runpod.io/gpu-models/rtx-4090)
- [RunPod Community vs Secure Cloud guide](https://gpuhosted.com/en/runpod-community-cloud-guide/)
- [RunPod Secure Cloud price change, 20 Sep 2026](https://www.usagepricing.com/blueprint/activity/runpod-2026-09-20-secure-cloud-price-hike)
- [Vast.ai RTX 4090 pricing](https://vast.ai/pricing/gpu/RTX-4090)
- [GetDeploying: RTX 4090 prices](https://getdeploying.com/gpus/nvidia-rtx-4090)
- [GetDeploying: A100 prices](https://getdeploying.com/gpus/nvidia-a100)
- [Modal pricing](https://modal.com/pricing)
- [Lambda pricing](https://lambda.ai/pricing)
- [Jarvislabs H100 price guide (Lambda 1× H100 SXM)](https://jarvislabs.ai/blog/h100-price)
- [Synpix cloud GPU pricing (Lambda 1× A100 40 GB, H100 PCIe)](https://www.synpixcloud.com/blog/cloud-gpu-pricing-comparison-2026)
- [Thunder Compute: Google Cloud GPU instances](https://www.thundercompute.com/blog/google-cloud-gpu-instances)
- [Spheron: GCP A3 H100 pricing](https://www.spheron.network/blog/google-cloud-a3-h100-pricing/)
- [Economize: AWS g5.xlarge pricing](https://www.economize.cloud/resources/aws/pricing/ec2/g5.xlarge/)
- [PyTorch 2.7 release (Blackwell support via CUDA 12.8)](https://pytorch.org/blog/pytorch-2-7/)
- [Qualiteg: PyTorch versions and supported GPU compute capabilities](https://journal.qualiteg.com/pytorch_and_supported_gpu_version/)
