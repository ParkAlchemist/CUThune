import numpy as np
import torch

from src.eval.vocoder_diagnostics import (
    tensor_to_audio_np,
    tensor_to_cqt_np,
    normalize_path_string,
)


def test_tensor_to_audio_np_shapes():
    x = torch.randn(1, 1, 128)
    y = tensor_to_audio_np(x)

    assert y.shape == (128,)
    assert y.dtype == np.float32


def test_tensor_to_cqt_np_shapes():
    x = torch.randn(1, 1, 96, 32)
    y = tensor_to_cqt_np(x)

    assert y.shape == (96, 32)
    assert y.dtype == np.float32


def test_normalize_path_string_case_and_slashes():
    a = normalize_path_string(r"E:\Foo\Bar.wav")
    b = normalize_path_string("e:/foo/bar.wav")

    assert a == b
