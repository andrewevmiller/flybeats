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


def test_calibration_standardises_the_features():
    """The DC the encoder used to fight with negative weights is removed here."""
    torch.manual_seed(2)
    enc = _enc()
    wav = torch.randn(4, 22050) * 0.3
    raw = enc.raw_features(wav)
    assert raw.mean().abs() > 0.5 * raw.std(), "test needs features with real DC"

    stats = enc.calibrate([wav])
    assert stats["calibrated"] and stats["frames"] > 0
    f = enc.features(wav).reshape(-1, enc.n_features)
    assert f.mean(0).abs().max() < 1e-4
    assert (f.std(0) - 1.0).abs().max() < 1e-3


def test_calibrated_encoder_still_streams_exactly():
    """Standardisation must not break the training/streaming agreement, which
    is why it is a fixed affine and not a per-clip statistic."""
    torch.manual_seed(3)
    enc = _enc()
    enc.calibrate([torch.randn(2, 22050)])
    wav = torch.randn(1, 22050)
    full = enc.forward(wav)
    for start, n in [(0, 8), (17, 30), (full.shape[1] - 6, 6)]:
        assert torch.allclose(enc.forward_window(wav, start, n),
                              full[:, start: start + n], atol=1e-5)


def test_standardisation_is_fixed_not_per_clip():
    """A per-clip normaliser would make a quiet clip and a loud one encode
    identically -- and would be non-causal in the live path."""
    torch.manual_seed(4)
    enc = _enc()
    wav = torch.randn(2, 22050)
    enc.calibrate([wav])
    quiet = enc.features(wav * 0.01)
    assert quiet.mean().abs() > 0.1, "loudness must survive a fixed affine"


def test_projection_keeps_to_jo_non_negative():
    """The failure mode being closed off: to_jo going negative and cancelling
    the encoder's own input."""
    enc = _enc(nonneg=True)
    enc.to_jo.weight.data.sub_(1.0)
    assert float(enc.to_jo.weight.detach().min()) < 0
    enc.project_()
    assert float(enc.to_jo.weight.detach().min()) >= 0.0

    free = _enc(nonneg=False)
    free.to_jo.weight.data.sub_(1.0)
    free.project_()
    assert float(free.to_jo.weight.detach().min()) < 0, "nonneg=False must not constrain"


def test_zone_prior_is_non_negative():
    """The constraint has to be satisfiable at initialisation, or the first
    projection would silently move the model off the prior."""
    enc = _enc()
    assert float(enc.to_jo.weight.detach().min()) >= 0.0
