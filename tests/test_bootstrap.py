"""The bootstrap's preflight guards.

`bootstrap_local.py` exists to catch environment failures that are silent --
a CPU-only torch wheel on a machine with a card being the first one. This adds
the second, found the hard way on 21 September: `.venv` had been rebuilt with
Python 3.14 over a 3.12 `site-packages`, so every run died at `import torch`
with "PyTorch has loaded the torch/_C folder of the PyTorch repository", a
message that names neither Python nor a version. Every DLL was present and
hash-correct against the wheel's RECORD; the interpreter simply could not load
a `cp312` extension.

Nothing in the repo noticed. `ensure_venv` saw a `python.exe` and reported
"reusing".
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from bootstrap_local import abi_tags, describe_abi_mismatch  # noqa: E402


def _touch(d: Path, *names):
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(b"")


def test_reads_the_abi_off_windows_and_posix_extensions(tmp_path):
    _touch(tmp_path, "_cffi_backend.cp312-win_amd64.pyd")
    _touch(tmp_path / "torch", "_C.cp312-win_amd64.pyd")
    _touch(tmp_path / "yaml", "_yaml.cpython-312-x86_64-linux-gnu.so")
    assert abi_tags(tmp_path) == {"3.12"}


def test_pure_python_packages_are_not_evidence(tmp_path):
    """An environment of nothing but pure-Python wheels is coherent on any
    interpreter, so it must not be reported as a mismatch."""
    _touch(tmp_path, "six.py", "typing_extensions.py")
    _touch(tmp_path / "pip", "__init__.py")
    assert abi_tags(tmp_path) == set()
    assert describe_abi_mismatch(tmp_path, "3.14", set()) == ""


def test_a_matching_interpreter_is_silent(tmp_path):
    assert describe_abi_mismatch(tmp_path, "3.12", {"3.12"}) == ""


def test_the_mismatch_names_both_versions_and_the_way_out(tmp_path):
    msg = describe_abi_mismatch(tmp_path, "3.14", {"3.12"})
    assert "3.14" in msg and "3.12" in msg
    assert "site-packages" in msg, "the message has to say why the packages stayed"
    assert "venv" in msg, "and how to rebuild it"


def test_a_two_digit_minor_version_is_read_whole(tmp_path):
    """3.10 and 3.1 are different Pythons; a lazy split would confuse them."""
    _touch(tmp_path / "torch", "_C.cp310-win_amd64.pyd")
    assert abi_tags(tmp_path) == {"3.10"}
    assert describe_abi_mismatch(tmp_path, "3.10", {"3.10"}) == ""
    assert describe_abi_mismatch(tmp_path, "3.1", {"3.10"}) != ""
