"""Federated surrogate-objective refinement for CERNN."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor

from .compat import zip_strict
from .losses import shared_hidden_l2_penalty
from .model import CERNN, flatten_parameters, parameter_count
from .training import (
    OptimizationResult,
    OptimizerConfig,
    cernn_data_loss,
    data_gradient,
    fit_cernn,
    objective_gradient,
    optimize_objective,
    parameter_change,
    prepare_regression_tensors,
)


@dataclass(frozen=True)
class FederatedClient:
    """In-memory client data used by the simulation implementation."""

    x: Tensor
    y: Tensor

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[0])


@dataclass
class FederatedResult:
    model: CERNN
    history: list[dict[str, Any]]
    communication_bytes: int
    host_fit: OptimizationResult


def equal_iid_partition(
    data_or_size: int | Sequence[Any] | Tensor,
    n_clients: int,
    seed: int = 0,
) -> list[Tensor]:
    """Randomly partition indices so client sizes differ by at most one.

    Remainder observations are assigned to a random subset of clients.
    """

    if isinstance(data_or_size, int):
        n_samples = data_or_size
    else:
        n_samples = len(data_or_size)
    if n_samples < 0:
        raise ValueError("The sample count cannot be negative.")
    if n_clients <= 0:
        raise ValueError("n_clients must be positive.")
    if n_samples < n_clients:
        raise ValueError("n_samples must be at least n_clients.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    shuffled_indices = torch.randperm(n_samples, generator=generator)
    client_order = torch.randperm(n_clients, generator=generator).tolist()
    quotient, remainder = divmod(n_samples, n_clients)
    sizes = [quotient] * n_clients
    for client_index in client_order[:remainder]:
        sizes[client_index] += 1

    partitions: list[Tensor] = []
    offset = 0
    for size in sizes:
        partitions.append(shuffled_indices[offset : offset + size].clone())
        offset += size
    return partitions


def make_federated_clients(
    x: Tensor,
    y: Tensor,
    n_clients: int,
    *,
    seed: int = 0,
) -> list[FederatedClient]:
    """Create random, near-equal IID clients from a centralized sample."""

    x_value = torch.as_tensor(x)
    y_value = torch.as_tensor(y)
    if x_value.shape[0] != y_value.shape[0]:
        raise ValueError("x and y must contain the same number of samples.")
    indices = equal_iid_partition(x_value.shape[0], n_clients, seed)
    return [
        FederatedClient(x=x_value[index], y=y_value[index])
        for index in indices
    ]


def _coerce_client(client: FederatedClient | Sequence[Tensor]) -> FederatedClient:
    if isinstance(client, FederatedClient):
        return client
    if isinstance(client, Sequence) and len(client) == 2:
        return FederatedClient(x=torch.as_tensor(client[0]), y=torch.as_tensor(client[1]))
    raise TypeError("Each client must be FederatedClient or an (x, y) pair.")


def prepare_clients(
    model: CERNN,
    clients: Sequence[FederatedClient | Sequence[Tensor]],
) -> list[FederatedClient]:
    """Validate and move all client tensors to the model device."""

    if not clients:
        raise ValueError("At least one client is required.")
    prepared: list[FederatedClient] = []
    for raw_client in clients:
        client = _coerce_client(raw_client)
        x_value, y_value = prepare_regression_tensors(model, client.x, client.y)
        prepared.append(FederatedClient(x_value, y_value))
    return prepared


def aggregate_client_gradients(
    model: CERNN,
    clients: Sequence[FederatedClient | Sequence[Tensor]],
) -> tuple[Tensor, list[Tensor]]:
    """Return the sample-size-weighted global data gradient and local gradients."""

    prepared = prepare_clients(model, clients)
    local_gradients = [
        data_gradient(model, client.x, client.y)
        for client in prepared
    ]
    total_samples = sum(client.n_samples for client in prepared)
    global_gradient = sum(
        (client.n_samples / total_samples) * gradient
        for client, gradient in zip_strict(prepared, local_gradients)
    )
    return global_gradient, local_gradients


def centralized_federated_objective(
    model: CERNN,
    clients: Sequence[FederatedClient],
    l2_lambda: float,
) -> Tensor:
    """Centralized objective represented as a weighted sum of client losses."""

    total_samples = sum(client.n_samples for client in clients)
    data_term = sum(
        (client.n_samples / total_samples)
        * cernn_data_loss(model, client.x, client.y)
        for client in clients
    )
    return data_term + l2_lambda * shared_hidden_l2_penalty(model)


def make_surrogate_objective(
    model: CERNN,
    host: FederatedClient,
    host_gradient: Tensor,
    global_gradient: Tensor,
    l2_lambda: float,
    expansion_point: Tensor | None = None,
):
    """Construct the host surrogate objective for one FedCERNN round.

    Centering the linear term at ``expansion_point`` changes only an additive
    constant and improves numerical scaling.
    """

    if l2_lambda < 0.0:
        raise ValueError("l2_lambda cannot be negative.")
    current = (
        flatten_parameters(model, detach=True)
        if expansion_point is None
        else expansion_point.detach().clone().to(model.device, model.dtype)
    )
    correction = (host_gradient - global_gradient).detach().to(
        device=model.device,
        dtype=model.dtype,
    )
    if correction.numel() != current.numel():
        raise ValueError("Gradient correction has the wrong length.")

    def objective() -> Tensor:
        displacement = flatten_parameters(model) - current
        return (
            cernn_data_loss(model, host.x, host.y)
            - torch.dot(correction, displacement)
            + l2_lambda * shared_hidden_l2_penalty(model)
        )

    return objective


def surrogate_gradient_match_error(
    model: CERNN,
    clients: Sequence[FederatedClient],
    surrogate_objective,
    l2_lambda: float,
) -> float:
    """Maximum absolute surrogate/centralized gradient difference."""

    surrogate_gradient = objective_gradient(model, surrogate_objective)
    central_gradient = objective_gradient(
        model,
        lambda: centralized_federated_objective(model, clients, l2_lambda),
    )
    return float(torch.max(torch.abs(surrogate_gradient - central_gradient)).cpu())


class FedCERNN:
    """Full-participation cross-silo FedCERNN trainer.

    Client zero is always the host.  Every communicated gradient is a data-loss
    gradient; the shared-hidden-layer penalty is added exactly once by the host.
    """

    def __init__(
        self,
        model: CERNN,
        *,
        l2_lambda: float = 0.0,
        optimizer_config: OptimizerConfig | None = None,
        host_optimizer_config: OptimizerConfig | None = None,
        host_regularization_path: Sequence[float] | None = None,
        adam_steps: int | None = None,
        lbfgs_steps: int | None = None,
        lr: float | None = None,
        tolerance: float | None = None,
    ) -> None:
        if l2_lambda < 0.0:
            raise ValueError("l2_lambda cannot be negative.")
        if optimizer_config is not None and any(
            value is not None
            for value in (adam_steps, lbfgs_steps, lr, tolerance)
        ):
            raise ValueError(
                "Use optimizer_config or scalar optimizer overrides, not both."
            )
        if optimizer_config is None:
            defaults = OptimizerConfig()
            optimizer_config = OptimizerConfig(
                adam_steps=defaults.adam_steps if adam_steps is None else adam_steps,
                adam_lr=defaults.adam_lr if lr is None else lr,
                lbfgs_steps=(
                    defaults.lbfgs_steps if lbfgs_steps is None else lbfgs_steps
                ),
                lbfgs_lr=defaults.lbfgs_lr,
                tolerance=defaults.tolerance if tolerance is None else tolerance,
                history_size=defaults.history_size,
            )
        self.model = model
        self.l2_lambda = float(l2_lambda)
        self.optimizer_config = optimizer_config
        self.host_optimizer_config = host_optimizer_config or optimizer_config
        path = (
            (self.l2_lambda,)
            if host_regularization_path is None
            else tuple(float(value) for value in host_regularization_path)
        )
        if not path or any(value < 0.0 or not math.isfinite(value) for value in path):
            raise ValueError("host_regularization_path must contain finite nonnegative values.")
        if not math.isclose(path[-1], self.l2_lambda, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(
                "The final host_regularization_path value must equal l2_lambda."
            )
        self.host_regularization_path = path

    def fit(
        self,
        clients: Sequence[FederatedClient | Sequence[Tensor]],
        *,
        rounds: int = 1,
        participation_rate: float = 1.0,
        participation_seed: int = 0,
    ) -> FederatedResult:
        if rounds < 0:
            raise ValueError("rounds cannot be negative.")
        if not 0.0 < participation_rate <= 1.0:
            raise ValueError("participation_rate must lie in (0, 1].")
        prepared = prepare_clients(self.model, clients)
        host = prepared[0]

        host_stage_history: list[dict[str, Any]] = []
        for stage_index, stage_lambda in enumerate(self.host_regularization_path):
            host_fit = fit_cernn(
                self.model,
                host.x,
                host.y,
                l2_lambda=stage_lambda,
                optimizer_config=self.host_optimizer_config,
            )
            host_stage_history.append(
                {
                    "round": 0,
                    "stage": (
                        "host_regularization_warm_start"
                        if stage_index + 1 < len(self.host_regularization_path)
                        else "host_fit"
                    ),
                    "regularization_lambda": stage_lambda,
                    "objective": host_fit.final_loss,
                    "gradient_norm": host_fit.gradient_norm,
                    "adam_steps": host_fit.adam_steps,
                    "lbfgs_iterations": host_fit.lbfgs_iterations,
                    "lbfgs_evaluations": host_fit.lbfgs_evaluations,
                    "rms_parameter_change": host_fit.rms_parameter_change,
                    "communication_bytes": 0,
                    "cumulative_communication_bytes": 0,
                }
            )
        history: list[dict[str, Any]] = host_stage_history
        if len(self.host_regularization_path) == 1:
            history[0].pop("regularization_lambda")
        else:
            history[-1]["regularization_path"] = list(self.host_regularization_path)

        scalar_bytes = next(self.model.parameters()).element_size()
        communication_bytes = 0

        for round_index in range(1, rounds + 1):
            remote_indices = list(range(1, len(prepared)))
            active_indices = list(range(len(prepared)))
            if remote_indices and participation_rate < 1.0:
                keep = max(1, int(torch.ceil(torch.tensor(participation_rate * len(remote_indices))).item()))
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int(participation_seed) + round_index)
                chosen = torch.randperm(len(remote_indices), generator=generator)[:keep]
                active_indices = [0] + sorted(remote_indices[int(index)] for index in chosen)
            active_clients = [prepared[index] for index in active_indices]
            bytes_per_round = (
                2 * max(len(active_clients) - 1, 0) * parameter_count(self.model) * scalar_bytes
            )
            expansion_point = flatten_parameters(self.model, detach=True)
            global_gradient, local_gradients = aggregate_client_gradients(
                self.model,
                active_clients,
            )
            surrogate = make_surrogate_objective(
                self.model,
                host,
                local_gradients[0],
                global_gradient,
                self.l2_lambda,
                expansion_point,
            )
            match_error = surrogate_gradient_match_error(
                self.model,
                active_clients,
                surrogate,
                self.l2_lambda,
            )
            optimization = optimize_objective(
                self.model,
                surrogate,
                self.optimizer_config,
            )
            communication_bytes += bytes_per_round
            central_value = centralized_federated_objective(
                self.model,
                active_clients,
                self.l2_lambda,
            )
            history.append(
                {
                    "round": round_index,
                    "stage": "surrogate_refinement",
                    "objective": float(central_value.detach().cpu()),
                    "surrogate_objective": optimization.final_loss,
                    "gradient_norm": optimization.gradient_norm,
                    "adam_steps": optimization.adam_steps,
                    "lbfgs_iterations": optimization.lbfgs_iterations,
                    "lbfgs_evaluations": optimization.lbfgs_evaluations,
                    "rms_parameter_change": optimization.rms_parameter_change,
                    "safeguard_boundary_active": (
                        optimization.safeguard_boundary_active
                    ),
                    "safeguard_rejected_candidate": (
                        optimization.safeguard_rejected_candidate
                    ),
                    "radius_rejected_candidate": (
                        optimization.radius_rejected_candidate
                    ),
                    "radius_projected_candidate": (
                        optimization.radius_projected_candidate
                    ),
                    "objective_rejected_candidate": (
                        optimization.objective_rejected_candidate
                    ),
                    "surrogate_gradient_match_error": match_error,
                    "parameter_change": parameter_change(
                        self.model,
                        expansion_point,
                    ),
                    "communication_bytes": bytes_per_round,
                    "cumulative_communication_bytes": communication_bytes,
                    "active_clients": active_indices,
                }
            )

        return FederatedResult(
            model=self.model,
            history=history,
            communication_bytes=communication_bytes,
            host_fit=host_fit,
        )
