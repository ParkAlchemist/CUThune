import torch

from src.models.cyclegan import (
    GeneratorConfig,
    DiscriminatorConfig,
    AudioResnetGenerator,
    PatchGANDiscriminator,
    MultiScalePatchGANDiscriminator,
)


def test_generator_preserves_cqt_shape_cpu():
    x = torch.randn(2, 1, 96, 172)

    config = GeneratorConfig(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        max_channels=256,
        num_downsamples=2,
        num_res_blocks=6,
        residual_dropout=0.0,
        use_attention=True,
        attention_position="middle",
        norm="instance",
        padding_mode="reflect",
    )

    model = AudioResnetGenerator(config)
    model.eval()

    with torch.no_grad():
        y = model(x)

    assert y.shape == x.shape
    assert y.min() >= -1.05
    assert y.max() <= 1.05


def test_patchgan_discriminator_output_shape_cpu():
    x = torch.randn(2, 1, 96, 172)

    config = DiscriminatorConfig(
        in_channels=1,
        base_channels=32,
        max_channels=512,
        num_layers=3,
        norm="instance",
        spectral_norm=False,
        use_sigmoid=False,
    )

    model = PatchGANDiscriminator(config)
    model.eval()

    with torch.no_grad():
        y = model(x)

    assert y.ndim == 4
    assert y.shape[0] == 2
    assert y.shape[1] == 1
    assert y.shape[-2] > 1
    assert y.shape[-1] > 1


def test_multiscale_discriminator_outputs_list_cpu():
    x = torch.randn(2, 1, 96, 172)

    config = DiscriminatorConfig(
        in_channels=1,
        base_channels=32,
        max_channels=512,
        num_layers=3,
        norm="instance",
        spectral_norm=False,
        use_sigmoid=False,
    )

    model = MultiScalePatchGANDiscriminator(
        config=config,
        num_scales=2,
    )
    model.eval()

    with torch.no_grad():
        outputs = model(x)

    assert isinstance(outputs, list)
    assert len(outputs) == 2

    for y in outputs:
        assert y.ndim == 4
        assert y.shape[0] == 2
        assert y.shape[1] == 1


def test_generator_backward_pass_cpu():
    x = torch.randn(2, 1, 96, 172)

    config = GeneratorConfig(
        base_channels=16,
        max_channels=128,
        num_downsamples=2,
        num_res_blocks=2,
        use_attention=False,
    )

    model = AudioResnetGenerator(config)

    y = model(x)
    loss = y.abs().mean()

    loss.backward()

    grads = [
        p.grad
        for p in model.parameters()
        if p.requires_grad
    ]

    assert any(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_generator_forward_with_balanced_attention():
    cfg = GeneratorConfig(
        base_channels=16,
        max_channels=128,
        num_downsamples=2,
        num_res_blocks=6,
        use_attention=True,
        num_att_blocks=2,
        attention_position="balanced",
        padding_mode="reflect",
    )

    model = AudioResnetGenerator(cfg)

    x = torch.randn(2, 1, 96, 172)
    y = model(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_generator_forward_with_cqt_padding():
    cfg = GeneratorConfig(
        base_channels=16,
        max_channels=128,
        num_downsamples=2,
        num_res_blocks=2,
        use_attention=False,
        padding_mode="cqt",
    )

    model = AudioResnetGenerator(cfg)

    x = torch.randn(2, 1, 96, 172)
    y = model(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_generator_forward_with_squeeze_excite():
    cfg = GeneratorConfig(
        base_channels=16,
        max_channels=128,
        num_downsamples=2,
        num_res_blocks=2,
        use_attention=False,
        padding_mode="reflect",
        use_se=True,
        se_reduction=16,
    )

    model = AudioResnetGenerator(cfg)

    x = torch.randn(2, 1, 96, 172)
    y = model(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
