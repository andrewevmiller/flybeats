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
