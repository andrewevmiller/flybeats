"""The encoder is shared by training and live streaming, so the windowed path
must agree with the whole-clip path exactly. If it does not, a model trains on
one signal and plays back on a different one, and nothing downstream reveals it.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from encoder import AudioToJO, erb_space, gammatone_filterbank, zone_frequency_prior  # noqa: E402


def _enc(n_channels=12, **kw):
    zones = ["JO_A", "JO_B", "JO_E", "JO_other"] * (n_channels // 4)
    return AudioToJO(n_channels=len(zones), zone_of_channel=zones,
                     sample_rate=22050, step_ms=5.0, **kw)


def test_window_matches_full_clip():
    torch.manual_seed(0)
    enc = _enc()
    wav = torch.randn(2, 22050)                      # 1 s
    full = enc.forward(wav)
    n_steps = full.shape[1]

    for start, n in [(0, 8), (5, 20), (40, 33), (n_steps - 10, 10)]:
        got = enc.forward_window(wav, start, n)
        want = full[:, start: start + n]
        assert got.shape == want.shape, (got.shape, want.shape)
        assert torch.allclose(got, want, atol=1e-5), \
            f"window [{start},{start + n}) differs by {(got - want).abs().max():.2e}"


def test_streaming_block_sequence_matches_full_clip():
    """Block-by-block, the way the live path actually calls it."""
    torch.manual_seed(1)
    enc = _enc()
    wav = torch.randn(1, 22050 // 2)
    full = enc.forward(wav)

    out, step, block = [], 0, 4
    while step + block <= full.shape[1]:
        out.append(enc.forward_window(wav, step, block))
        step += block
    got = torch.cat(out, dim=1)
    assert torch.allclose(got, full[:, : got.shape[1]], atol=1e-5), \
        (got - full[:, : got.shape[1]]).abs().max()


def test_only_one_trainable_layer():
    """If the front end learns the task, the connectome is decoration."""
    enc = _enc()
    trainable = {n for n, p in enc.named_parameters() if p.requires_grad}
    assert trainable == {"to_jo.weight", "to_jo.bias"}, trainable
    assert not enc.filters.requires_grad


def test_filterbank_is_ordered_and_normalised():
    k, cf = gammatone_filterbank(16, 22050)
    assert np.all(np.diff(cf) > 0), "centre frequencies must ascend"
    assert np.allclose(np.linalg.norm(k, axis=1), 1.0, atol=1e-5)
    assert cf[0] >= 35 and cf[-1] <= 6100


def test_erb_spacing_is_monotonic():
    e = erb_space(40.0, 6000.0, 32)
    assert np.all(np.diff(e) < 0) or np.all(np.diff(e) > 0)


def test_song_band_prior_favours_jo_b():
    """JO-B should start weighted toward the 100-500 Hz courtship-song band."""
    _, cf = gammatone_filterbank(64, 22050)
    prior = zone_frequency_prior(cf, ["JO_A", "JO_B", "JO_E"])
    song = (cf >= 100) & (cf <= 500)
    mass = prior[song].sum(axis=0) / prior.sum(axis=0)
    assert mass[1] > mass[0], "JO-B should carry more song-band mass than JO-A"
    assert mass[1] > mass[2], "JO-B should carry more song-band mass than JO-E"
    assert mass[1] > 0.5


def test_decimation_averages_rather_than_samples():
    """A box mean over each hop window, not every hop-th sample: sampling
    aliases, and the impulse would land or not depending on its phase."""
    enc = _enc()
    hop = enc.hop
    for offset in (0, hop // 3, hop - 1):
        wav = torch.zeros(1, hop * 20)
        wav[0, hop * 5 + offset] = 1.0
        f = enc.features(wav)
        assert f[0, 5:8].abs().sum() > 0, f"impulse at offset {offset} vanished"


def test_output_shape_tracks_step_rate():
    enc = _enc()
    for seconds in (0.25, 1.0, 2.5):
        n = int(22050 * seconds)
        assert enc.features(torch.zeros(1, n)).shape[1] == n // enc.hop
