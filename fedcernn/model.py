"""Neural-network model and parameter-vector utilities for FedCERNN."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def _validate_taus(taus: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(tau) for tau in taus)
    if not values:
        raise ValueError("At least one expectile level is required.")
    if any(tau <= 0.0 or tau >= 1.0 for tau in values):
        raise ValueError("Every expectile level must lie strictly between 0 and 1.")
    return values


class CERNN(nn.Module):
    """Composite expectile regression neural network.

    The hidden layers form a shared tanh backbone.  Each expectile level has
    its own scalar linear output head.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int] = (32, 16),
        taus: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if int(input_dim) <= 0:
            raise ValueError("input_dim must be positive.")

        hidden = tuple(int(width) for width in hidden_dims)
        if not hidden or any(width <= 0 for width in hidden):
            raise ValueError("hidden_dims must contain positive layer widths.")

        tau_values = _validate_taus(taus)
        self.input_dim = int(input_dim)
        self.hidden_dims = hidden
        self.tau_values = tau_values

        layers: list[nn.Module] = []
        previous_width = self.input_dim
        for width in hidden:
            layers.append(nn.Linear(previous_width, width, dtype=dtype))
            layers.append(nn.Tanh())
            previous_width = width
        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleList(
            [nn.Linear(previous_width, 1, dtype=dtype) for _ in tau_values]
        )
        self.register_buffer(
            "taus",
            torch.tensor(tau_values, dtype=dtype),
            persistent=True,
        )

    @property
    def dtype(self) -> torch.dtype:
        """Parameter dtype used by the model."""

        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        """Device on which the model parameters are stored."""

        return next(self.parameters()).device

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x must have shape (n, {self.input_dim}); got {tuple(x.shape)}."
            )
        x = x.to(device=self.device, dtype=self.dtype)
        representation = self.backbone(x)
        return torch.cat([head(representation) for head in self.heads], dim=1)

    def shared_hidden_weights(self) -> Iterable[Tensor]:
        """Yield only weight matrices in the shared hidden backbone."""

        for layer in self.backbone:
            if isinstance(layer, nn.Linear):
                yield layer.weight


def trainable_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    """Return trainable parameters in the stable module registration order."""

    return tuple(parameter for parameter in model.parameters() if parameter.requires_grad)


def flatten_parameters(model: nn.Module, *, detach: bool = False) -> Tensor:
    """Flatten trainable model parameters into one vector.

    With ``detach=False`` the result remains connected to autograd, which is
    needed for the linear correction in the surrogate objective.
    """

    vector = parameters_to_vector(trainable_parameters(model))
    return vector.detach().clone() if detach else vector


def assign_parameters(model: nn.Module, vector: Tensor) -> None:
    """Copy a flat parameter vector into ``model``."""

    parameters = trainable_parameters(model)
    expected = sum(parameter.numel() for parameter in parameters)
    if vector.numel() != expected:
        raise ValueError(
            f"Parameter vector has {vector.numel()} entries; expected {expected}."
        )
    reference = parameters[0]
    vector_to_parameters(
        vector.to(device=reference.device, dtype=reference.dtype),
        parameters,
    )


def flatten_gradients(
    model: nn.Module,
    *,
    fill_none: bool = True,
    detach: bool = True,
) -> Tensor:
    """Flatten gradients currently stored in ``parameter.grad``."""

    pieces: list[Tensor] = []
    for parameter in trainable_parameters(model):
        gradient = parameter.grad
        if gradient is None:
            if not fill_none:
                raise RuntimeError("A trainable parameter has no gradient.")
            gradient = torch.zeros_like(parameter)
        pieces.append(gradient.reshape(-1))
    vector = torch.cat(pieces)
    return vector.detach().clone() if detach else vector


def parameter_count(model: nn.Module) -> int:
    """Number of trainable scalar parameters."""

    return sum(parameter.numel() for parameter in trainable_parameters(model))
