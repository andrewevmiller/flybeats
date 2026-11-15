"""memorise.py's run names and overwrite guard (scripts/run_naming.py). No data, no GPU."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_naming import pair_names, refusal, run_name  # noqa: E402


def test_default_pair_keeps_the_locked_names():
    assert pair_names(1) == ("real", "control-1")
    assert pair_names(3) == ("real", "control-3")


def test_sqrt_pair_never_uses_a_locked_folder():
    real, control = pair_names(1, "sqrt")
    assert (real, control) == ("real-loss-sqrt", "control-1-loss-sqrt")
    assert real != "real" and control != "control-1"


def test_pair_names_match_single_run_names():
    assert pair_names(2, "sqrt") == (run_name(None, "sqrt"), run_name(2, "sqrt"))


def test_explicit_name_wins():
    assert run_name(None, "sqrt", name="mine") == "mine"


def test_guard_refuses_existing_weights_unless_forced(tmp_path):
    assert refusal(tmp_path) is None                       # empty folder: fine
    (tmp_path / "final.pt").write_bytes(b"x")
    assert "final.pt" in refusal(tmp_path)
    assert refusal(tmp_path, force=True) is None
