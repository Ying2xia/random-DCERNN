"""Federated and non-federated comparison methods for the experiments.

These implementations prioritize a common model, objective, initialization,
client partition, and deterministic full-client participation.  They are
research baselines, not production distributed systems.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .compat import zip_strict
from .federated import FederatedClient, aggregate_client_gradients
from .losses import composite_expectile_loss, shared_hidden_l2_penalty
from .model import CERNN, assign_parameters, flatten_parameters, parameter_count
from .training import (
    OptimizerConfig,
    data_gradient,
    fit_cernn,
    objective_gradient,
    optimize_objective,
    prepare_regression_tensors,
)


BASELINE_NAMES: tuple[str, ...] = (
    "Intercept-only",
    "Centralized CERNN",
    "Host CERNN",
    "CSL-CERNN",
    "CEASE-CERNN",
    "DANE-CERNN",
    "FedDANE-CERNN",
    "one-shot FedAvg",
    "multi-round FedAvg",
    "FedProx",
    "SCAFFOLD",
    "DCERNN",
)


@dataclass(frozen=True)
class BaselineConfig:
    """Optimizer settings shared by the registered comparison methods.

    ``local_optimizer`` controls the corrected-surrogate solves used by CSL,
    CEASE, DANE, and FedDANE. ``predictive_local_optimizer`` separately
    controls the uncorrected client solves used by FedAvg, FedProx, and
    SCAFFOLD. Keeping these solvers separate prevents a short trust-region
    surrogate profile from accidentally freezing the predictive baselines.

    ``local_steps`` and ``local_lr`` are retained as the number and step size
    of synchronous DCERNN gradient updates within each outer round.
    """

    rounds: int = 5
    local_steps: int = 20
    local_lr: float = 1.0e-2
    l2_lambda: float = 1.0e-4
    fedprox_mu: float = 1.0e-2
    cease_alpha: float = 1.0e-2
    feddane_mu: float = 1.0e-2
    participation_rate: float = 1.0
    participation_seed: int = 0
    centralized_optimizer: OptimizerConfig = field(
        default_factory=OptimizerConfig
    )
    host_optimizer: OptimizerConfig | None = None
    local_optimizer: OptimizerConfig = field(
        default_factory=lambda: OptimizerConfig(
            adam_steps=30,
            adam_lr=1.0e-2,
            lbfgs_steps=10,
            tolerance=1.0e-7,
        )
    )
    predictive_local_optimizer: OptimizerConfig | None = None
    predictive_local_optimizers: Mapping[str, OptimizerConfig] = field(
        default_factory=dict
    )
    corrected_local_optimizers: Mapping[str, OptimizerConfig] = field(default_factory=dict)
    scaffold_solver: str = "legacy"
    scaffold_sgd_steps: int = 100
    scaffold_sgd_lr: float = 0.03

    @property
    def resolved_predictive_optimizer(self) -> OptimizerConfig:
        """Return the solver used by FedAvg, FedProx, and SCAFFOLD."""

        return self.predictive_local_optimizer or self.local_optimizer

    def predictive_optimizer_for(self, method: str) -> OptimizerConfig:
        """Return a method-specific predictive solver when configured."""

        return self.predictive_local_optimizers.get(
            method, self.resolved_predictive_optimizer
        )

    def __post_init__(self) -> None:
        if self.scaffold_solver not in {"legacy", "sgd"}:
            raise ValueError("scaffold_solver must be legacy or sgd.")
        if self.scaffold_sgd_steps < 1 or self.scaffold_sgd_lr <= 0:
            raise ValueError("SCAFFOLD SGD steps and learning rate must be positive.")
        if self.rounds < 1:
            raise ValueError("rounds must be positive.")
        if self.local_steps < 1:
            raise ValueError("local_steps must be positive.")
        if self.local_lr <= 0.0:
            raise ValueError("local_lr must be positive.")
        if (
            self.l2_lambda < 0.0
            or self.fedprox_mu < 0.0
            or self.cease_alpha < 0.0
            or self.feddane_mu < 0.0
        ):
            raise ValueError("Regularization coefficients cannot be negative.")
        if not 0.0 < self.participation_rate <= 1.0:
            raise ValueError("participation_rate must lie in (0, 1].")


@dataclass
class BaselineResult:
    method: str
    model: CERNN
    communication_bytes: int
    history: list[dict[str, Any]]
    host_warm_start: Tensor | None = None


def _coerce_clients(
    model: CERNN,
    clients: Sequence[FederatedClient | Sequence[object]],
) -> list[FederatedClient]:
    if not clients:
        raise ValueError("At least one client is required.")
    result: list[FederatedClient] = []
    for client in clients:
        if isinstance(client, FederatedClient):
            x, y = client.x, client.y
        elif isinstance(client, Sequence) and len(client) == 2:
            x, y = client
        else:
            raise TypeError("A client must be FederatedClient or an (X, y) pair.")
        x_value, y_value = prepare_regression_tensors(
            model, torch.as_tensor(x), torch.as_tensor(y)
        )
        result.append(FederatedClient(x_value, y_value))
    return result


def _model_bytes(model: CERNN) -> int:
    return parameter_count(model) * next(model.parameters()).element_size()


def _active_indices(count: int, round_index: int, config: BaselineConfig) -> list[int]:
    """Select full or partial participants while always retaining the host."""

    if count < 1:
        raise ValueError("At least one client is required.")
    remote = list(range(1, count))
    if not remote:
        return [0]
    keep = int(np.ceil(config.participation_rate * len(remote)))
    if keep >= len(remote):
        return list(range(count))
    rng = np.random.default_rng(int(config.participation_seed) + int(round_index))
    selected = sorted(rng.choice(remote, size=max(1, keep), replace=False).tolist())
    return [0, *selected]


def _weighted_parameter_average(
    models: Sequence[CERNN], sample_sizes: Sequence[int]
) -> Tensor:
    if len(models) != len(sample_sizes) or not models:
        raise ValueError("models and sample_sizes must have equal nonzero length.")
    total = float(sum(int(size) for size in sample_sizes))
    if total <= 0.0:
        raise ValueError("The total sample size must be positive.")
    vectors = [flatten_parameters(model, detach=True) for model in models]
    return sum(
        (int(size) / total) * vector
        for size, vector in zip_strict(sample_sizes, vectors)
    )


def _local_objective(
    model: CERNN,
    client: FederatedClient,
    l2_lambda: float,
    reference: Tensor | None = None,
    proximal_mu: float = 0.0,
) -> Tensor:
    prediction = model(client.x)
    value = composite_expectile_loss(prediction, client.y, model.taus)
    value = value + l2_lambda * shared_hidden_l2_penalty(model)
    if reference is not None and proximal_mu > 0.0:
        difference = flatten_parameters(model) - reference
        value = value + 0.5 * proximal_mu * torch.dot(difference, difference)
    return value


def _optimize_local_model(
    model: CERNN,
    client: FederatedClient,
    *,
    optimizer_config: OptimizerConfig,
    l2_lambda: float,
    proximal_reference: Tensor | None = None,
    proximal_mu: float = 0.0,
    gradient_adjustment: Tensor | None = None,
) -> float:
    """Optimize one client objective with the configured Adam/L-BFGS solver.

    A constant linear term implements the SCAFFOLD control-variate correction.
    Centering it at the received global model affects only an additive constant.
    """

    expansion_point = flatten_parameters(model, detach=True)

    def objective() -> Tensor:
        value = _local_objective(
            model,
            client,
            l2_lambda,
            reference=proximal_reference,
            proximal_mu=proximal_mu,
        )
        if gradient_adjustment is not None:
            displacement = flatten_parameters(model) - expansion_point
            value = value + torch.dot(
                gradient_adjustment.to(device=model.device, dtype=model.dtype),
                displacement,
            )
        return value

    return optimize_objective(model, objective, optimizer_config).final_loss


def _marginal_expectile(sample: np.ndarray, tau: float) -> float:
    """Solve the empirical marginal expectile equation by bisection."""

    values = np.asarray(sample, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot compute an expectile of an empty sample.")
    if not 0.0 < tau < 1.0:
        raise ValueError("tau must lie strictly between zero and one.")

    def moment(level: float) -> float:
        residual = values - level
        weights = np.where(residual >= 0.0, tau, 1.0 - tau)
        return float(np.mean(weights * residual))

    lower = float(values.min())
    upper = float(values.max())
    if lower == upper:
        return lower
    for _ in range(200):
        middle = 0.5 * (lower + upper)
        if moment(middle) > 0.0:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def fit_intercept_only(
    model: CERNN,
    clients: Sequence[FederatedClient | Sequence[object]],
    config: BaselineConfig,
) -> BaselineResult:
    """Fit pooled marginal expectiles as a covariate-free diagnostic.

    This centralized diagnostic is evaluated by the same prediction and metric
    code as the trained methods. Its stored byte count follows the existing
    convention for centralized references and is reported as not applicable,
    rather than as a federated communication claim, in generated tables.
    """

    del config
    prepared = _coerce_clients(model, clients)
    pooled_response = np.concatenate(
        [client.y.detach().cpu().numpy().reshape(-1) for client in prepared]
    )
    with torch.no_grad():
        for head, tau in zip_strict(model.heads, model.tau_values):
            head.weight.zero_()
            head.bias.fill_(_marginal_expectile(pooled_response, float(tau)))
    return BaselineResult(
        method="Intercept-only",
        model=model,
        communication_bytes=0,
        history=[
            {
                "round": 0,
                "stage": "pooled_marginal_expectiles",
                "pooled_data": True,
                "communication_accounting": "not_applicable",
            }
        ],
    )


def fit_centralized(
    model: CERNN,
    clients: Sequence[FederatedClient | Sequence[object]],
    config: BaselineConfig,
) -> BaselineResult:
    prepared = _coerce_clients(model, clients)
    host_optimizer = config.host_optimizer or config.centralized_optimizer
    host_optimization = fit_cernn(
        model,
        prepared[0].x,
        prepared[0].y,
        l2_lambda=config.l2_lambda,
        optimizer_config=host_optimizer,
    )
    host_warm_start = flatten_parameters(model, detach=True)
    x = torch.cat([client.x for client in prepared], dim=0)
    y = torch.cat([client.y for client in prepared], dim=0)
    pooled_optimization = fit_cernn(
        model,
        x,
        y,
        l2_lambda=config.l2_lambda,
        optimizer_config=config.centralized_optimizer,
    )
    return BaselineResult(
        method="Centralized CERNN",
        model=model,
        communication_bytes=0,
        history=[
            {
                "round": 0,
                "stage": "host_warm_start",
                "objective": host_optimization.final_loss,
                "gradient_norm": host_optimization.gradient_norm,
                "adam_steps": host_optimization.adam_steps,
                "lbfgs_iterations": host_optimization.lbfgs_iterations,
                "lbfgs_evaluations": host_optimization.lbfgs_evaluations,
                "rms_parameter_change": host_optimization.rms_parameter_change,
            },
            {
                "round": 0,
                "stage": "pooled_fit",
                "objective": pooled_optimization.final_loss,
                "gradient_norm": pooled_optimization.gradient_norm,
                "adam_steps": pooled_optimization.adam_steps,
                "lbfgs_iterations": pooled_optimization.lbfgs_iterations,
                "lbfgs_evaluations": pooled_optimization.lbfgs_evaluations,
                "rms_parameter_change": pooled_optimization.rms_parameter_change,
            },
        ],
        host_warm_start=host_warm_start,
    )


def fit_host(
    model: CERNN,
    clients: Sequence[FederatedClient | Sequence[object]],
    config: BaselineConfig,
) -> BaselineResult:
    prepared = _coerce_clients(model, clients)
    host_optimizer = config.host_optimizer or config.centralized_optimizer
    optimization = fit_cernn(
        model,
        prepared[0].x,
        prepared[0].y,
        l2_lambda=config.l2_lambda,
        optimizer_config=host_optimizer,
    )
    host_warm_start = flatten_parameters(model, detach=True)
    return BaselineResult(
        method="Host CERNN",
        model=model,
        communication_bytes=0,
        history=[
            {
                "round": 0,
                "stage": "host_warm_start",
                "objective": optimization.final_loss,
                "gradient_norm": optimization.gradient_norm,
                "adam_steps": optimization.adam_steps,
                "lbfgs_iterations": optimization.lbfgs_iterations,
                "lbfgs_evaluations": optimization.lbfgs_evaluations,
                "rms_parameter_change": optimization.rms_parameter_change,
            }
        ],
        host_warm_start=host_warm_start,
    )


def fit_fedavg(
    model: CERNN,
    clients: Sequence[FederatedClient | Sequence[object]],
    config: BaselineConfig,
    *,
    rounds: int,
    method: str,
    proximal_mu: float = 0.0,
) -> BaselineResult:
    prepared = _coerce_clients(model, clients)
    communication = 0
    history: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        active_indices = _active_indices(len(prepared), round_index, config)
        active_clients = [prepared[index] for index in active_indices]
        bytes_per_round = 2 * max(len(active_clients) - 1, 0) * _model_bytes(model)
        global_vector = flatten_parameters(model, detach=True)
        local_models: list[CERNN] = []
        final_losses: list[float] = []
        for client in active_clients:
            local_model = copy.deepcopy(model)
            final_loss = _optimize_local_model(
                local_model,
                client,
                optimizer_config=config.predictive_optimizer_for(method),
                l2_lambda=config.l2_lambda,
                proximal_reference=global_vector if proximal_mu > 0.0 else None,
                proximal_mu=proximal_mu,
            )
            local_models.append(local_model)
            final_losses.append(final_loss)
        average = _weighted_parameter_average(
            local_models, [client.n_samples for client in active_clients]
        )
        assign_parameters(model, average)
        communication += bytes_per_round
        history.append(
            {
                "round": round_index,
                "mean_local_objective": float(np.mean(final_losses)),
                "communication_bytes": bytes_per_round,
                "cumulative_communication_bytes": communication,
                "active_clients": active_indices,
            }
        )
    return BaselineResult(method, model, communication, history)


def fit_scaffold(
    model: CERNN,
    clients: Sequence[FederatedClient | Sequence[object]],
    config: BaselineConfig,
) -> BaselineResult:
    """Full-participation SCAFFOLD with one control vector per client.

    The exact full-batch control update (option I in SCAFFOLD) is used so the
    configured Adam/L-BFGS local solver can be applied without relying on the
    constant-step SGD identity required by the alternative control update.
    """

    prepared = _coerce_clients(model, clients)
    global_control = torch.zeros_like(flatten_parameters(model, detach=True))
    client_controls = [torch.zeros_like(global_control) for _ in prepared]
    communication = 0
    history: list[dict[str, Any]] = []

    for round_index in range(1, config.rounds + 1):
        active_indices = _active_indices(len(prepared), round_index, config)
        active_clients = [prepared[index] for index in active_indices]
        # Model and server control are sent; model and client-control deltas return.
        bytes_per_round = 4 * max(len(active_clients) - 1, 0) * _model_bytes(model)
        local_models: list[CERNN] = []
        control_deltas: list[Tensor] = []
        final_losses: list[float] = []
        for client_index, client in zip_strict(active_indices, active_clients):
            local_model = copy.deepcopy(model)
            adjustment = global_control - client_controls[client_index]
            new_control = objective_gradient(
                model,
                lambda client=client: _local_objective(
                    model,
                    client,
                    config.l2_lambda,
                ),
            )
            if config.scaffold_solver == "sgd":
                # Algorithm 1: finite full-batch corrected SGD, with option I
                # controls evaluated at the common broadcast parameter.
                for _ in range(config.scaffold_sgd_steps):
                    gradient = objective_gradient(
                        local_model,
                        lambda: _local_objective(local_model, client, config.l2_lambda),
                    ) + adjustment
                    if not bool(torch.isfinite(gradient).all()):
                        raise FloatingPointError("Non-finite SCAFFOLD corrected gradient.")
                    vector = flatten_parameters(local_model, detach=True)
                    assign_parameters(local_model, vector - config.scaffold_sgd_lr * gradient)
                final_loss = float(_local_objective(local_model, client, config.l2_lambda).detach())
                if not np.isfinite(final_loss):
                    raise FloatingPointError("Non-finite SCAFFOLD local loss.")
            else:
                final_loss = _optimize_local_model(
                    local_model,
                    client,
                    optimizer_config=config.predictive_optimizer_for("SCAFFOLD"),
                    l2_lambda=config.l2_lambda,
                    gradient_adjustment=adjustment,
                )
            control_deltas.append(new_control - client_controls[client_index])
            client_controls[client_index] = new_control
            local_models.append(local_model)
            final_losses.append(final_loss)

        sample_sizes = [client.n_samples for client in active_clients]
        total_samples = float(sum(sample_sizes))
        average = _weighted_parameter_average(local_models, sample_sizes)
        assign_parameters(model, average)
        global_control = global_control + sum(
            (size / total_samples) * delta
            for size, delta in zip_strict(sample_sizes, control_deltas)
        )
        communication += bytes_per_round
        history.append(
            {
                "round": round_index,
                "mean_local_objective": float(np.mean(final_losses)),
                "control_norm": float(torch.linalg.vector_norm(global_control).cpu()),
                "local_solver": config.scaffold_solver,
                "local_objective_kind": "unadjusted" if config.scaffold_solver == "sgd" else "corrected",
                "communication_bytes": bytes_per_round,
                "cumulative_communication_bytes": communication,
                "active_clients": active_indices,
            }
        )
    return BaselineResult("SCAFFOLD", model, communication, history)


def fit_dcernn(
    model: CERNN,
    clients: Sequence[FederatedClient | Sequence[object]],
    config: BaselineConfig,
) -> BaselineResult:
    """Synchronous multi-round distributed CERNN gradient descent.

    Every global step evaluates data gradients at the same parameter vector,
    aggregates them by client sample size, and adds the shared-hidden penalty
    gradient exactly once on the server.
    """

    prepared = _coerce_clients(model, clients)
    total_steps = config.rounds * config.local_steps
    communication = 0
    history: list[dict[str, Any]] = []
    for step_index in range(1, total_steps + 1):
        active_indices = _active_indices(len(prepared), step_index, config)
        active_clients = [prepared[index] for index in active_indices]
        total_samples = float(sum(client.n_samples for client in active_clients))
        bytes_per_step = 2 * max(len(active_clients) - 1, 0) * _model_bytes(model)
        local_gradients = [data_gradient(model, client.x, client.y) for client in active_clients]
        aggregate = sum(
            (client.n_samples / total_samples) * gradient
            for client, gradient in zip_strict(active_clients, local_gradients)
        )
        if config.l2_lambda > 0.0:
            penalty_gradient = objective_gradient(
                model, lambda: shared_hidden_l2_penalty(model)
            )
            aggregate = aggregate + config.l2_lambda * penalty_gradient
        current = flatten_parameters(model, detach=True)
        assign_parameters(model, current - config.local_lr * aggregate)
        communication += bytes_per_step
        history.append(
            {
                "round": step_index,
                "gradient_norm": float(torch.linalg.vector_norm(aggregate).cpu()),
                "communication_bytes": bytes_per_step,
                "cumulative_communication_bytes": communication,
                "active_clients": active_indices,
            }
        )
    return BaselineResult("DCERNN", model, communication, history)


def _host_initialize(
    model: CERNN,
    clients: Sequence[FederatedClient],
    config: BaselineConfig,
) -> tuple[Tensor, dict[str, Any]]:
    """Fit the common host initializer used by corrected-surrogate methods."""

    result = fit_cernn(
        model,
        clients[0].x,
        clients[0].y,
        l2_lambda=config.l2_lambda,
        optimizer_config=config.host_optimizer or config.centralized_optimizer,
    )
    return flatten_parameters(model, detach=True), {
        "round": 0,
        "stage": "host_fit",
        "objective": result.final_loss,
        "gradient_norm": result.gradient_norm,
        "adam_steps": result.adam_steps,
        "lbfgs_iterations": result.lbfgs_iterations,
        "lbfgs_evaluations": result.lbfgs_evaluations,
        "rms_parameter_change": result.rms_parameter_change,
        "communication_bytes": 0,
        "cumulative_communication_bytes": 0,
    }


def _corrected_local_objective(
    model: CERNN,
    client: FederatedClient,
    *,
    l2_lambda: float,
    expansion_point: Tensor,
    global_gradient: Tensor,
    local_gradient: Tensor,
    proximal_mu: float,
) -> Tensor:
    """DANE/CEASE local subproblem with a data-gradient correction.

    The shared penalty is included exactly once in every local solve.  The
    correction uses data gradients only, so its derivative at the expansion
    point recovers the pooled penalized gradient after the common penalty is
    added.
    """

    value = _local_objective(
        model,
        client,
        l2_lambda,
        reference=expansion_point if proximal_mu > 0.0 else None,
        proximal_mu=proximal_mu,
    )
    displacement = flatten_parameters(model) - expansion_point
    correction = global_gradient - local_gradient
    return value + torch.dot(correction, displacement)


def fit_cease(
    model: CERNN,
    clients: Sequence[FederatedClient | Sequence[object]],
    config: BaselineConfig,
) -> BaselineResult:
    """CEASE Algorithm 3 specialized to the CERNN loss.

    One host subproblem is solved per round after workers communicate data-loss
    gradients.  ``cease_alpha`` is the proximal-point regularizer.  Setting it
    to zero reduces this implementation to the host CSL surrogate.
    """

    prepared = _coerce_clients(model, clients)
    host_start, host_history = _host_initialize(model, prepared, config)
    del host_start
    communication = 0
    history = [host_history]
    host = prepared[0]
    for round_index in range(1, config.rounds + 1):
        active_indices = _active_indices(len(prepared), round_index, config)
        active_clients = [prepared[index] for index in active_indices]
        bytes_per_round = 2 * max(len(active_clients) - 1, 0) * _model_bytes(model)
        expansion = flatten_parameters(model, detach=True)
        global_gradient, local_gradients = aggregate_client_gradients(model, active_clients)

        def objective() -> Tensor:
            return _corrected_local_objective(
                model,
                host,
                l2_lambda=config.l2_lambda,
                expansion_point=expansion,
                global_gradient=global_gradient,
                local_gradient=local_gradients[0],
                proximal_mu=config.cease_alpha,
            )

        optimization = optimize_objective(model, objective, config.local_optimizer)
        communication += bytes_per_round
        history.append(
            {
                "round": round_index,
                "stage": "cease_host_refinement",
                "objective": optimization.final_loss,
                "gradient_norm": optimization.gradient_norm,
                "communication_bytes": bytes_per_round,
                "cumulative_communication_bytes": communication,
                "proximal_alpha": config.cease_alpha,
                "active_clients": active_indices,
            }
        )
    return BaselineResult("CEASE-CERNN", model, communication, history)


def fit_csl(
    model: CERNN,
    clients: Sequence[FederatedClient | Sequence[object]],
    config: BaselineConfig,
) -> BaselineResult:
    """Host-based CSL surrogate (the zero-proximal CEASE update)."""

    result = fit_cease(model, clients, replace(config, cease_alpha=0.0))
    result.method = "CSL-CERNN"
    for item in result.history:
        if item.get("stage") == "cease_host_refinement":
            item["stage"] = "csl_host_refinement"
    return result


def fit_dane(
    model: CERNN,
    clients: Sequence[FederatedClient | Sequence[object]],
    config: BaselineConfig,
    *,
    method: str,
    proximal_mu: float,
) -> BaselineResult:
    """Averaged corrected-local solver used for DANE and FedDANE.

    Each round records both the gradient-collection and local-model exchange;
    four remote parameter-vector payloads are therefore counted.  ``mu=0``
    gives the DANE subproblem, while a positive value is the stabilized
    FedDANE-style variant.
    """

    prepared = _coerce_clients(model, clients)
    host_start, host_history = _host_initialize(model, prepared, config)
    del host_start
    communication = 0
    history = [host_history]
    for round_index in range(1, config.rounds + 1):
        active_indices = _active_indices(len(prepared), round_index, config)
        active_clients = [prepared[index] for index in active_indices]
        bytes_per_round = 4 * max(len(active_clients) - 1, 0) * _model_bytes(model)
        expansion = flatten_parameters(model, detach=True)
        global_gradient, local_gradients = aggregate_client_gradients(model, active_clients)
        local_models: list[CERNN] = []
        residuals: list[float] = []
        for client, local_gradient in zip_strict(active_clients, local_gradients):
            local_model = copy.deepcopy(model)

            def objective(
                local_model: CERNN = local_model,
                client: FederatedClient = client,
                local_gradient: Tensor = local_gradient,
            ) -> Tensor:
                return _corrected_local_objective(
                    local_model,
                    client,
                    l2_lambda=config.l2_lambda,
                    expansion_point=expansion,
                    global_gradient=global_gradient,
                    local_gradient=local_gradient,
                    proximal_mu=proximal_mu,
                )

            result = optimize_objective(
                local_model, objective,
                config.corrected_local_optimizers.get(method, config.local_optimizer),
            )
            local_models.append(local_model)
            residuals.append(result.gradient_norm)
        assign_parameters(
            model,
            _weighted_parameter_average(
                local_models, [client.n_samples for client in active_clients]
            ),
        )
        communication += bytes_per_round
        history.append(
            {
                "round": round_index,
                "stage": "dane_averaged_refinement",
                "mean_local_gradient_norm": float(np.mean(residuals)),
                "communication_bytes": bytes_per_round,
                "cumulative_communication_bytes": communication,
                "proximal_mu": proximal_mu,
                "active_clients": active_indices,
            }
        )
    return BaselineResult(method, model, communication, history)


def canonical_method_name(method: str) -> str:
    aliases = {
        "intercept_only": "Intercept-only",
        "intercept": "Intercept-only",
        "marginal": "Intercept-only",
        "central": "Centralized CERNN",
        "centralized": "Centralized CERNN",
        "centralized_cernn": "Centralized CERNN",
        "host": "Host CERNN",
        "host_cernn": "Host CERNN",
        "one_shot_fedavg": "one-shot FedAvg",
        "fedavg_1": "one-shot FedAvg",
        "fedavg": "multi-round FedAvg",
        "multi_round_fedavg": "multi-round FedAvg",
        "csl": "CSL-CERNN",
        "csl_cernn": "CSL-CERNN",
        "cease": "CEASE-CERNN",
        "cease_cernn": "CEASE-CERNN",
        "dane": "DANE-CERNN",
        "dane_cernn": "DANE-CERNN",
        "feddane": "FedDANE-CERNN",
        "feddane_cernn": "FedDANE-CERNN",
        "fedprox": "FedProx",
        "scaffold": "SCAFFOLD",
        "dcernn": "DCERNN",
    }
    normalized = str(method).strip()
    canonical = aliases.get(normalized.lower().replace("-", "_").replace(" ", "_"))
    if canonical is not None:
        return canonical
    if normalized in BASELINE_NAMES:
        return normalized
    raise ValueError(f"Unknown baseline {method!r}; choose from {BASELINE_NAMES}.")


def fit_baseline(
    method: str,
    model: CERNN,
    clients: Sequence[FederatedClient | Sequence[object]],
    config: BaselineConfig | None = None,
) -> BaselineResult:
    """Fit a named baseline from the common experiment entry point."""

    settings = config or BaselineConfig()
    canonical = canonical_method_name(method)
    if canonical == "Intercept-only":
        return fit_intercept_only(model, clients, settings)
    if canonical == "Centralized CERNN":
        return fit_centralized(model, clients, settings)
    if canonical == "Host CERNN":
        return fit_host(model, clients, settings)
    if canonical == "CSL-CERNN":
        return fit_csl(model, clients, settings)
    if canonical == "CEASE-CERNN":
        return fit_cease(model, clients, settings)
    if canonical == "DANE-CERNN":
        return fit_dane(
            model,
            clients,
            settings,
            method="DANE-CERNN",
            proximal_mu=0.0,
        )
    if canonical == "FedDANE-CERNN":
        return fit_dane(
            model,
            clients,
            settings,
            method="FedDANE-CERNN",
            proximal_mu=settings.feddane_mu,
        )
    if canonical == "one-shot FedAvg":
        return fit_fedavg(
            model,
            clients,
            settings,
            rounds=1,
            method="one-shot FedAvg",
        )
    if canonical == "multi-round FedAvg":
        return fit_fedavg(
            model,
            clients,
            settings,
            rounds=settings.rounds,
            method="multi-round FedAvg",
        )
    if canonical == "FedProx":
        return fit_fedavg(
            model,
            clients,
            settings,
            rounds=settings.rounds,
            method="FedProx",
            proximal_mu=settings.fedprox_mu,
        )
    if canonical == "SCAFFOLD":
        return fit_scaffold(model, clients, settings)
    return fit_dcernn(model, clients, settings)


@torch.no_grad()
def predict(model: CERNN, X: object) -> np.ndarray:
    reference = next(model.parameters())
    values = torch.as_tensor(X, dtype=reference.dtype, device=reference.device)
    return model(values).detach().cpu().numpy()
