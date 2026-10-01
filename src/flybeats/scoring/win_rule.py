"""The main win rule (config/locked.yaml win_rule, defined in prereg-v5.1): the real network wins if its score is
higher than the highest of the 5 random controls' by more than the seed spread, the sample standard deviation
(divisor n - 1) of the controls' scores. Same metric, same test songs, unrounded; a tie is not a win."""
import numpy as np

SPREADS = {"sample_sd_of_the_5_control_scores": lambda c: float(np.std(c, ddof=1))}


def main_win(real, controls, cfg):
    """real: the real network's score; controls: one score per control seed, in locked.yaml's order.
    Returns {wins, real, best_control, seed_spread, margin}, margin = real - best_control - seed_spread."""
    wr = cfg["win_rule"]
    seeds = cfg["connectome"]["random_controls"]["seeds"]
    if wr.get("seed_spread") not in SPREADS:
        raise ValueError(f"win_rule.seed_spread {wr.get('seed_spread')!r} is not implemented")
    controls = np.asarray(controls, dtype=np.float64)
    if controls.shape != (len(seeds),):
        raise ValueError(f"need one score per control seed {seeds}, got {controls.shape}")
    if not np.isfinite(controls).all() or not np.isfinite(real):
        raise ValueError("scores must be finite")
    spread = SPREADS[wr["seed_spread"]](controls)
    best = float(controls.max())
    margin = float(real) - best - spread
    return {"wins": margin > 0, "real": float(real), "best_control": best, "seed_spread": spread, "margin": margin}
