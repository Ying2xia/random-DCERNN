"""Loss and regularization functions used by CERNN and FedCERNN."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .model import CERNN


def _tau_tensor(
    predictions: Tensor,
    taus: Sequence[float] | Tensor,
) -> Tensor:
    values = torch.as_tensor(
        taus,
        dtype=predictions.dtype,
        device=predictions.device,
    ).reshape(-1)
    if values.numel() != predictions.shape[1]:
        raise ValueError(
            "The number of expectile levels must equal the prediction width."
        )
    if torch.any((values <= 0.0) | (values >= 1.0)):
        raise ValueError("Every expectile level must lie strictly between 0 and 1.")
    return values


def composite_expectile_loss(
    predictions: Tensor,
    targets: Tensor,
    taus: Sequence[float] | Tensor,
    reduction: str = "mean",
    tau_reduction: str = "mean",
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Compute the composite asymmetric squared expectile loss.

    ``tau_reduction`` first combines the expectile levels for each sample.
    ``reduction`` then combines samples.  Setting either option to ``"none"``
    retains the corresponding dimension.
    """

    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(1)
    if predictions.ndim != 2:
        raise ValueError("predictions must have shape (n, K).")

    targets = targets.to(
        device=predictions.device,
        dtype=predictions.dtype,
    )
    if targets.ndim == 2 and targets.shape[1] == 1:
        targets = targets[:, 0]
    if targets.ndim != 1 or targets.shape[0] != predictions.shape[0]:
        raise ValueError("targets must have shape (n,) or (n, 1).")

    tau_values = _tau_tensor(predictions, taus)
    residuals = targets.unsqueeze(1) - predictions
    asymmetric_weights = torch.where(
        residuals >= 0.0,
        tau_values.unsqueeze(0),
        1.0 - tau_values.unsqueeze(0),
    )
    pointwise = asymmetric_weights * residuals.square()

    if tau_reduction == "mean":
        per_sample = pointwise.mean(dim=1)
    elif tau_reduction == "sum":
        per_sample = pointwise.sum(dim=1)
    elif tau_reduction == "none":
        per_sample = pointwise
    else:
        raise ValueError("tau_reduction must be 'mean', 'sum', or 'none'.")

    weights: Tensor | None = None
    if sample_weight is not None:
        weights = sample_weight.to(
            device=predictions.device,
            dtype=predictions.dtype,
        ).reshape(-1)
        if weights.shape[0] != predictions.shape[0]:
            raise ValueError("sample_weight must have one entry per sample.")
        if torch.any(weights < 0.0):
            raise ValueError("sample_weight cannot contain negative values.")
        if not bool(torch.sum(weights) > 0.0):
            raise ValueError("sample_weight must have positive total weight.")

    if reduction == "none":
        if weights is None:
            return per_sample
        weight_shape = (weights.shape[0],) + (1,) * (per_sample.ndim - 1)
        return per_sample * weights.reshape(weight_shape)
    if reduction == "sum":
        if weights is None:
            return per_sample.sum()
        weight_shape = (weights.shape[0],) + (1,) * (per_sample.ndim - 1)
        return (per_sample * weights.reshape(weight_shape)).sum()
    if reduction != "mean":
        raise ValueError("reduction must be 'mean', 'sum', or 'none'.")

    if weights is None:
        return per_sample.mean(dim=0)
    weight_shape = (weights.shape[0],) + (1,) * (per_sample.ndim - 1)
    weighted = per_sample * weights.reshape(weight_shape)
    return weighted.sum(dim=0) / weights.sum()


def shared_hidden_l2_penalty(model: nn.Module) -> Tensor:
    """Return one-half of the squared norm of shared hidden-layer weights.

    Output-head parameters and all biases are deliberately excluded.
    """

    if isinstance(model, CERNN):
        weights = tuple(model.shared_hidden_weights())
    else:
        backbone = getattr(model, "backbone", None)
        if backbone is None:
            raise TypeError("model must expose a shared 'backbone'.")
        weights = tuple(
            module.weight
            for module in backbone.modules()
            if isinstance(module, nn.Linear)
        )
    if not weights:
        reference = next(model.parameters())
        return torch.zeros((), dtype=reference.dtype, device=reference.device)
    return 0.5 * sum(weight.square().sum() for weight in weights)
