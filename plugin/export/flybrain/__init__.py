"""flybrain: export a trained flybeats network to a versioned brain file, and run it without PyTorch.

Modules
  brainfile   the .flybrain container (write, read, verify); numpy only
  reference   the numpy reference engine: ear, core, readout, decode, perturbations; numpy only
  exporter    FlyNet + slice + checkpoint -> brain file (needs torch and the flybeats package)
  synthetic   small slice-shaped networks for tests and benchmarks (no connectome needed)
  parity      the parity harness: reference engine and C++ engine against PyTorch
"""
from .brainfile import FORMAT_VERSION, read_brain, verify_brain, write_brain  # noqa: F401
