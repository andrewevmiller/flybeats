"""Live-path latency per 20 ms block, CPU against GPU, on one machine in one sitting.

`realtime.py --benchmark` measures on CPU only; this puts the same model on
each device in turn so the two numbers share a day and a thermal state. The
model is untrained: latency is a property of the graph and the schedule, not
of the weights.

    python scripts/stream_latency.py --config configs/v1_8piece.yaml \
        --out results/local/streaming_latency_30k.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build import build_model, get_subgraph, load_config  # noqa: E402
from realtime import benchmark  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "v1_8piece.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--blocks", type=int, default=50)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    sg = get_subgraph(cfg)
    print(sg.summary().splitlines()[0])
    model, kit = build_model(cfg, sg)
    model.eval()

    out = {"config": str(a.config.name), "tier": cfg["subgraph"]["max_nodes"],
           "block_ms": 20.0, "n_blocks": a.blocks, "rows": []}
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        m = model.to(dev)
        for threads in ((1, 8) if dev == "cpu" else (1,)):
            torch.set_num_threads(threads)
            for speed in (1.0, 2.0, 4.0):
                np.random.seed(0)
                benchmark(m, kit, cfg, block_ms=20.0, n_blocks=5, speed=speed)   # warm-up
                np.random.seed(0)
                r = benchmark(m, kit, cfg, block_ms=20.0, n_blocks=a.blocks, speed=speed)
                row = {"device": dev, "threads": threads, "speed": speed,
                       "mean_ms": round(r["inference_ms_mean"], 2),
                       "p95_ms": round(r["inference_ms_p95"], 2),
                       "realtime_x": round(r["realtime_factor"], 2),
                       "meets_budget": r["meets_budget"]}
                out["rows"].append(row)
                print(row, flush=True)

    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name(0)
    out["torch"] = torch.__version__
    out["when"] = time.strftime("%Y-%m-%d %H:%M")
    a.out.write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
