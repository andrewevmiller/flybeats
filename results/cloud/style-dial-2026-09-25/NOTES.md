# Style dial experiment: cloud run notes

Session start: Fri Sep 25 02:15:47 UTC 2026 (first session). Branch from origin/cuda-path-verified at 054b3a7.

## Machine
- nproc: 4
- RAM: 15 GB total
- disk: 30G free at start
- Python 3.11.15
- torch: 2.14.0+cu130 from PyPI (CUDA libs unused; torch.cuda.is_available() = False)
- disk after venv + data: 18 GB free

## Setup timeline (UTC, 25 Sep)
- 02:18 venv created; torch + requirements-dev installed by ~02:19:30
- 02:20 MaleCNS tables fetched (fetch_data.py)
- 02:21:40 GMD groove-v1.0.0 fetched and extracted, zip deleted
- Pre-change test run (clean checkout of 054b3a7, -m "not gpu"): 247 passed, 2 skipped, 4 deselected, 126 s. No pre-existing failures.

## Pre-training check of the pC1 target (untrained style_pc1_cpu, 02:30 UTC)
style_pathway.py --config configs/style_pc1_cpu.yaml (one GMD test clip, 8 s):
- push into pC1 (155 neurons): drum shift 0.0035 / 0.1218 / 0.2800 at 0.5 / 5 / 50; octopaminergic (25): 0.0006 / 0.0082 / 0.0340.
- the untrained model's own 18 styles: largest drum shift 0.0002 (embedding starts near zero).
A second check (scratch script) with every pC1 neuron held at the full cap:
- +5: rates finite, drum shift 0.341, drum probabilities 0.021..0.808, no drum pinned on or off.
- -5: rates finite, drum shift 0.010, nothing pinned.
- The network's max rate is 20.0 (the state clip) with or without tonic, so something already sits at the clip at rest; the pC1 push does not add saturation of its own. Judged safe to train.
- Full suite after the changes (-m "not gpu", 02:35 UTC): 264 passed, 4 deselected (the 2 earlier skips ran now that data is present; 10 new tests in tests/test_style_dial.py plus 7 new config rows/tests).
