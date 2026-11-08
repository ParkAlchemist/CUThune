import torch

from legacy.cyclegan.losses.cyclegan_losses import GANLoss, cqt_to_chroma, ChromaConsistencyLoss, inject_gaussian_noise, \
    build_cyclegan_loss_bundle, CycleGANLossConfig, compute_discriminator_loss


def test_lsgan_loss_real_and_fake_are_finite():
    loss_fn = GANLoss("lsgan")

    pred = torch.randn(2, 1, 10, 19)

    loss_real = loss_fn(pred, True)
    loss_fake = loss_fn(pred, False)

    assert torch.isfinite(loss_real)
    assert torch.isfinite(loss_fake)
    assert loss_real.ndim == 0
    assert loss_fake.ndim == 0


def test_cqt_to_chroma_shape():
    cqt = torch.randn(2, 1, 96, 172).clamp(-1, 1)

    chroma = cqt_to_chroma(cqt, bins_per_octave=12)

    assert chroma.shape == (2, 12, 172)


def test_chroma_loss_is_finite():
    a = torch.randn(2, 1, 96, 172).clamp(-1, 1)
    b = torch.randn(2, 1, 96, 172).clamp(-1, 1)

    loss_fn = ChromaConsistencyLoss(bins_per_octave=12)
    loss = loss_fn(a, b)

    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_noise_injection_preserves_shape_and_range():
    x = torch.zeros(2, 1, 96, 172)

    y = inject_gaussian_noise(x, std=0.03, enabled=True)

    assert y.shape == x.shape
    assert y.min() >= -1.0
    assert y.max() <= 1.0
    assert not torch.allclose(x, y)


def test_noise_injection_can_be_disabled():
    x = torch.randn(2, 1, 96, 172).clamp(-1, 1)

    y = inject_gaussian_noise(x, std=0.03, enabled=False)

    assert torch.allclose(x, y)


def test_discriminator_loss_is_finite():
    bundle = build_cyclegan_loss_bundle(CycleGANLossConfig())

    pred_real = torch.randn(2, 1, 10, 19)
    pred_fake = torch.randn(2, 1, 10, 19)

    loss = compute_discriminator_loss(
        pred_real=pred_real,
        pred_fake_detached=pred_fake,
        bundle=bundle,
    )

    assert torch.isfinite(loss.total)
    assert torch.isfinite(loss.real)
    assert torch.isfinite(loss.fake)

