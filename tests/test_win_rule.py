import pytest

from flybeats.scoring.win_rule import main_win


def test_seed_spread_is_locked_as_the_sample_sd(cfg):
    assert cfg["win_rule"]["seed_spread"] == "sample_sd_of_the_5_control_scores"


def test_win(cfg):
    r = main_win(0.34, [0.20, 0.22, 0.23, 0.25, 0.30], cfg)
    assert r["best_control"] == 0.30
    assert r["seed_spread"] == pytest.approx(0.0381, abs=1e-4)      # sample sd, divisor 4
    assert r["wins"] and r["margin"] == pytest.approx(0.34 - 0.30 - 0.0381, abs=1e-4)


def test_beating_every_control_by_less_than_the_spread_is_not_a_win(cfg):
    assert not main_win(0.33, [0.20, 0.22, 0.23, 0.25, 0.30], cfg)["wins"]


def test_below_the_best_control_is_not_a_win(cfg):
    assert not main_win(0.29, [0.20, 0.22, 0.23, 0.25, 0.30], cfg)["wins"]


def test_a_tie_is_not_a_win(cfg):
    r = main_win(0.25, [0.25] * 5, cfg)                               # spread 0: real equals best + spread
    assert r["seed_spread"] == 0 and not r["wins"]


def test_needs_one_score_per_control_seed(cfg):
    with pytest.raises(ValueError, match="one score per control seed"):
        main_win(0.3, [0.2, 0.2, 0.2, 0.2], cfg)


def test_unimplemented_spread_is_refused(cfg):
    import copy
    bad = copy.deepcopy(cfg)
    bad["win_rule"]["seed_spread"] = "range_of_the_5_control_scores"
    with pytest.raises(ValueError, match="not implemented"):
        main_win(0.3, [0.2] * 5, bad)
