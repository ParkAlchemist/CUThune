import pytest
import torch
import torch.nn.functional as F

from cut.src.nes_cut.losses.components.patch_nce_loss import PatchNCELoss


def test_matching_pairs_have_lower_loss_than_shuffled() -> None:
    loss_fn = PatchNCELoss(
        temperature=0.07,
    )

    key = F.normalize(
        torch.randn(2, 32, 64),
        dim=-1,
    )
    matching_query = key.clone()

    permutation = torch.randperm(32)
    shuffled_query = matching_query[
        :,
        permutation,
        :,
    ]

    matching_loss = loss_fn(
        matching_query,
        key,
    )
    shuffled_loss = loss_fn(
        shuffled_query,
        key,
    )

    assert matching_loss < shuffled_loss


def test_identical_orthogonal_pairs_have_low_loss() -> None:
    loss_fn = PatchNCELoss(
        temperature=0.07,
    )

    # P=16, D=16 produces distinct orthogonal positives.
    key = torch.eye(16).unsqueeze(0)
    query = key.clone()

    loss = loss_fn(query, key)

    assert loss.item() < 1e-3


def test_patchnce_supports_batch_size_one() -> None:
    loss_fn = PatchNCELoss()

    query = torch.randn(1, 16, 32)
    key = torch.randn(1, 16, 32)

    loss = loss_fn(query, key)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_patchnce_detaches_key_branch() -> None:
    loss_fn = PatchNCELoss(
        detach_key=True,
    )

    query = torch.randn(
        2,
        16,
        32,
        requires_grad=True,
    )
    key = torch.randn(
        2,
        16,
        32,
        requires_grad=True,
    )

    loss = loss_fn(query, key)
    loss.backward()

    assert query.grad is not None
    assert torch.isfinite(query.grad).all()
    assert key.grad is None


def test_patchnce_query_gradients_are_finite() -> None:
    loss_fn = PatchNCELoss()

    query = torch.randn(
        2,
        16,
        32,
        requires_grad=True,
    )
    key = torch.randn(2, 16, 32)

    loss = loss_fn(query, key)
    loss.backward()

    assert query.grad is not None
    assert torch.isfinite(query.grad).all()


def test_patchnce_rejects_single_patch() -> None:
    loss_fn = PatchNCELoss()

    with pytest.raises(
        RuntimeError,
        match="at least two patches",
    ):
        loss_fn(
            torch.randn(2, 1, 32),
            torch.randn(2, 1, 32),
        )


