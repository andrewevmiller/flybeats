"""The overnight queue's 'is training already running?' guard.

The guard exists so a second batch can be chained behind a first without the
two racing: torch takes a thread per core in each process, and two training
jobs on four cores put one epoch from 76 s to 1,415 s.

It was written as ``pgrep -f "src/train.py"``, which asks a different question
-- does any command line anywhere mention that path -- and the shell that
launches the queue mentions it, in the check that reports whether training came
up. The guard waited on its own launcher, and a five-run batch slept all night
having trained nothing. It failed silently, and only when unattended.

So the guard is tested against the two processes that actually distinguish it:
a shell that merely names the path, and a python interpreter that is running it.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_velocity_queue.sh"

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux") or shutil.which("bash") is None
    or not Path("/proc/self/exe").exists(),
    reason="the guard reads /proc and is bash-only; it runs where the queue runs",
)


def guard_says_training_is_running():
    """Source the queue script and ask its guard, without running a batch."""
    r = subprocess.run(
        ["bash", "-c", f'source "{SCRIPT}"; train_running && echo YES || echo NO'],
        capture_output=True, text=True, timeout=60, cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
    out = r.stdout.strip().splitlines()[-1]
    assert out in ("YES", "NO"), r.stdout
    return out == "YES"


def a_real_training_job_is_in_flight():
    """Answered without the guard, on purpose.

    The first version of this file skipped when the guard said training was
    running -- which is the bug's own answer, so a broken guard skipped every
    test that would have caught it and the file reported green. A test's
    baseline cannot come from the code under test.
    """
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
            exe = os.path.basename(os.readlink(entry / "exe"))
        except OSError:
            continue                      # exited, or not ours to look at
        if b"src/train.py" in cmdline and "python" in exe:
            return True
    return False


@pytest.fixture
def idle_box():
    """Requested by name, never autouse.

    Autouse would apply it to every test in the file, and a batch running on
    this box would then skip the resume tests too -- which have nothing to do
    with live processes. A whole file reporting green because it ran nothing is
    the failure this file already exists to document.
    """
    if a_real_training_job_is_in_flight():
        pytest.skip("a training job is in flight; the decoys cannot be told apart")


class Decoy:
    """A process that lives for the length of the `with` block."""

    def __init__(self, argv):
        self.argv = argv

    def __enter__(self):
        self.p = subprocess.Popen(self.argv, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        deadline = time.time() + 10
        while time.time() < deadline:
            if subprocess.run(["pgrep", "-f", "src/train.py"],
                              capture_output=True).returncode == 0:
                return self
            time.sleep(0.05)
        raise AssertionError("decoy never became visible to pgrep")

    def __exit__(self, *exc):
        self.p.kill()
        self.p.wait()


def test_an_idle_box_reads_as_idle(idle_box):
    """The baseline the other two tests are read against."""
    assert not guard_says_training_is_running()


def test_a_shell_that_merely_names_the_path_is_not_a_training_job(idle_box):
    """The regression. This decoy is exactly the launcher that broke it: a
    bash process carrying 'src/train.py' in its command line and training
    nothing."""
    # A compound command, because bash exec-replaces itself with a trailing
    # simple command -- `bash -c '... ; sleep 30'` becomes a bare `sleep 30`
    # and loses the very command line under test.
    decoy = 'probe="pgrep -f src/train.py"; while :; do sleep 1; done'
    with Decoy(["bash", "-c", decoy]):
        assert not guard_says_training_is_running(), (
            "the guard counted a shell that only mentions src/train.py; an "
            "unattended batch will wait on its own launcher forever"
        )


def test_a_python_interpreter_running_it_is_a_training_job(idle_box):
    """And the guard still guards -- otherwise the fix is just its removal."""
    with Decoy([sys.executable, "-c", "import time; time.sleep(30)", "src/train.py"]):
        assert guard_says_training_is_running(), (
            "the guard missed a live python running src/train.py; two batches "
            "will overlap and each epoch costs ~18x"
        )


def test_sourcing_the_queue_does_not_start_a_batch():
    """The guard is only testable because the script has a __main__ guard.
    If that regresses, importing it here would launch five trainings."""
    before = set(os.listdir(ROOT / "runs")) if (ROOT / "runs").exists() else set()
    r = subprocess.run(["bash", "-c", f'source "{SCRIPT}"; echo SOURCED'],
                       capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert "SOURCED" in r.stdout
    assert "starting batch" not in r.stdout + r.stderr
    after = set(os.listdir(ROOT / "runs")) if (ROOT / "runs").exists() else set()
    assert after == before, "sourcing the queue created run directories"


# --- resuming after a restart ----------------------------------------------
#
# scripts/resume_velocity_queue.sh decides what still needs running. Getting
# that wrong is expensive in the direction nobody notices: re-running a config
# that already finished costs ~100 minutes and overwrites the checkpoint its
# result was measured on.

RESUME = ROOT / "scripts" / "resume_velocity_queue.sh"


def resume_plan(tmp_path, landed=(), half_trained=(), argv=()):
    """Run the resume script against a throwaway tree and return its plan."""
    (tmp_path / "scripts").mkdir()
    shutil.copy(RESUME, tmp_path / "scripts" / RESUME.name)
    (tmp_path / "results").mkdir()
    for run in landed:
        (tmp_path / "results" / f"probe_{run}.txt").write_text("probe output\n")
    for run in half_trained:
        d = tmp_path / "runs" / run
        d.mkdir(parents=True)
        (d / "best.pt").write_text("a checkpoint from whichever epoch it reached")

    r = subprocess.run(["bash", str(tmp_path / "scripts" / RESUME.name), *argv],
                       capture_output=True, text=True, timeout=60,
                       env={**os.environ, "DRY_RUN": "1"})
    return r


def queued(result):
    for line in result.stdout.splitlines():
        if line.startswith("queue "):
            return line.split(None, 1)[1].split()
    return []


def test_a_landed_run_is_not_run_again(tmp_path):
    r = resume_plan(tmp_path, landed=["a", "b"], argv=["a", "b", "c"])
    assert r.returncode == 0, r.stderr
    assert queued(r) == ["c"]


def test_a_half_trained_run_is_run_again(tmp_path):
    """The regression that would cost a result rather than an hour.

    A run killed mid-training leaves best.pt from whichever epoch it reached.
    Treating that as 'landed' would skip it and quietly report a partial run's
    checkpoint as a finished comparison arm.
    """
    r = resume_plan(tmp_path, landed=["a"], half_trained=["b"], argv=["a", "b"])
    assert r.returncode == 0, r.stderr
    assert queued(r) == ["b"], "a half-trained run was mistaken for a finished one"


def test_a_finished_batch_starts_nothing(tmp_path):
    r = resume_plan(tmp_path, landed=["a", "b"], argv=["a", "b"])
    assert r.returncode == 0
    assert queued(r) == [] and "nothing left to run" in r.stdout


def test_no_arguments_is_refused_rather_than_guessed(tmp_path):
    """Defaulting to some built-in list here would re-run whatever that list
    happened to say, which is how a finished arm gets overwritten."""
    r = resume_plan(tmp_path, argv=[])
    assert r.returncode == 2 and "usage" in r.stderr
