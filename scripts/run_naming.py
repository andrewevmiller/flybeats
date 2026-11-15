"""Pure helpers for scripts/memorise.py (no torch, no data): how a run is named, and the overwrite guard."""
from pathlib import Path


def run_name(control=None, loss_weight="ratio", wide_target=False, history_readout=False, name=None):
    """The run's name: --name if given, else "real" / "control-N" plus a suffix for each trial option."""
    if name:
        return name
    return (("real" if control is None else f"control-{control}")
            + "-wide-target" * bool(wide_target) + "-history-readout" * bool(history_readout)
            + "-loss-sqrt" * (loss_weight == "sqrt"))


def pair_names(control, loss_weight="ratio"):
    """The two run names --pair uses: the same ones a single run of each would get."""
    return run_name(None, loss_weight), run_name(control, loss_weight)


def refusal(out_dir, force=False):
    """Text explaining why a run must not start (its folder already holds saved weights), or None if it may."""
    final = Path(out_dir) / "final.pt"
    if final.exists() and not force:
        return (f"{final} already exists; starting would overwrite those saved weights. "
                f"Pick another --name, or pass --force to overwrite on purpose.")
    return None
