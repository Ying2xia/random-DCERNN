"""Optimization and gradient helpers shared by centralized and federated fits."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor, nn

from .compat import zip_strict
from .losses import composite_expectile_loss, shared_hidden_l2_penalty
from .model import CERNN, assign_parameters, flatten_parameters, trainable_parameters

ScalarObjective = Callable[[], Tensor]


@dataclass(frozen=True)
class OptimizerConfig:
    """Adam warm start followed by optional safeguarded L-BFGS refinement."""

    adam_steps: int = 50
    adam_lr: float = 1e-2
    lbfgs_steps: int = 20
    lbfgs_lr: float = 1.0
    tolerance: float = 1e-7
    history_size: int = 20
    stepwise_lbfgs: bool = False
    select_min_gradient_checkpoint: bool = False
    max_rms_parameter_change: float | None = None
    project_rms_parameter_change: bool = False

    def __post_init__(self) -> None:
        if self.adam_steps < 0 or self.lbfgs_steps < 0:
            raise ValueError("Optimizer step counts cannot be negative.")
        if self.adam_lr <= 0.0 or self.lbfgs_lr <= 0.0:
            raise ValueError("Optimizer learning rates must be positive.")
        if self.tolerance < 0.0:
            raise ValueError("tolerance cannot be negative.")
        if (
            self.max_rms_parameter_change is not None
            and (
                self.max_rms_parameter_change < 0.0
                or not math.isfinite(self.max_rms_parameter_change)
            )
        ):
            raise ValueError(
                "max_rms_parameter_change must be finite and nonnegative."
            )


@dataclass(frozen=True)
class OptimizationResult:
    final_loss: float
    gradient_norm: float
    adam_steps: int
    lbfgs_evaluations: int
    converged: bool
    lbfgs_iterations: int = 0
    rms_parameter_change: float = 0.0
    safeguard_boundary_active: bool = False
    safeguard_rejected_candidate: bool = False
    radius_rejected_candidate: bool = False
    radius_projected_candidate: bool = False
    objective_rejected_candidate: bool = False


def prepare_regression_tensors(
    model: nn.Module,
    x: Tensor,
    y: Tensor,
) -> tuple[Tensor, Tensor]:
    """Move regression tensors to the model's dtype and device."""

    reference = next(model.parameters())
    x_value = torch.as_tensor(
        x,
        dtype=reference.dtype,
        device=reference.device,
    )
    y_value = torch.as_tensor(
        y,
        dtype=reference.dtype,
        device=reference.device,
    )
    if x_value.ndim != 2:
        raise ValueError("x must have shape (n, p).")
    if y_value.ndim == 2 and y_value.shape[1] == 1:
        y_value = y_value[:, 0]
    if y_value.ndim != 1 or x_value.shape[0] != y_value.shape[0]:
        raise ValueError("y must have shape (n,) or (n, 1).")
    if x_value.shape[0] == 0:
        raise ValueError("A training data set cannot be empty.")
    return x_value, y_value


def cernn_data_loss(
    model: CERNN,
    x: Tensor,
    y: Tensor,
    *,
    sample_weight: Tensor | None = None,
) -> Tensor:
    """Mean composite expectile data loss, without regularization."""

    x_value, y_value = prepare_regression_tensors(model, x, y)
    weight_value = None
    if sample_weight is not None:
        weight_value = torch.as_tensor(
            sample_weight,
            dtype=model.dtype,
            device=model.device,
        )
    return composite_expectile_loss(
        model(x_value),
        y_value,
        model.taus,
        reduction="mean",
        tau_reduction="mean",
        sample_weight=weight_value,
    )


def objective_gradient(
    model: nn.Module,
    objective: ScalarObjective,
    *,
    create_graph: bool = False,
) -> Tensor:
    """Evaluate a scalar objective gradient without modifying ``.grad``."""

    value = objective()
    if value.ndim != 0:
        raise ValueError("The objective must return a scalar tensor.")
    parameters = trainable_parameters(model)
    gradients = torch.autograd.grad(
        value,
        parameters,
        create_graph=create_graph,
        allow_unused=True,
    )
    pieces = [
        (
            torch.zeros_like(parameter)
            if gradient is None
            else gradient
        ).reshape(-1)
        for parameter, gradient in zip_strict(parameters, gradients)
    ]
    vector = torch.cat(pieces)
    return vector if create_graph else vector.detach().clone()


def data_gradient(model: CERNN, x: Tensor, y: Tensor) -> Tensor:
    """Gradient of local data loss only (never including the L2 penalty)."""

    return objective_gradient(model, lambda: cernn_data_loss(model, x, y))


def _gradient_norm_from_storage(model: nn.Module) -> float:
    squared_norm = torch.zeros(
        (),
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device,
    )
    for parameter in trainable_parameters(model):
        if parameter.grad is not None:
            squared_norm = squared_norm + parameter.grad.square().sum()
    return float(torch.sqrt(squared_norm).detach().cpu())


def _evaluate_objective_and_gradient(
    model: nn.Module,
    objective: ScalarObjective,
) -> tuple[float, float]:
    """Evaluate an objective and its gradient at the current parameter point."""

    model.zero_grad(set_to_none=True)
    value = objective()
    if value.ndim != 0 or not bool(torch.isfinite(value)):
        raise FloatingPointError("The objective became non-finite.")
    value.backward()
    gradient_norm = _gradient_norm_from_storage(model)
    if not math.isfinite(gradient_norm):
        raise FloatingPointError("The objective gradient became non-finite.")
    model.zero_grad(set_to_none=True)
    return float(value.detach().cpu()), gradient_norm


def _rms_parameter_change(current: Tensor, reference: Tensor) -> float:
    """Root-mean-square coordinate displacement between parameter vectors."""

    return float(torch.sqrt(torch.mean((current - reference).square())).cpu())


def _project_to_rms_radius(
    candidate: Tensor,
    reference: Tensor,
    radius: float,
) -> Tensor:
    """Project ``candidate`` onto the RMS ball around ``reference``."""

    displacement = candidate - reference
    current_radius = _rms_parameter_change(candidate, reference)
    if current_radius <= radius:
        return candidate.detach().clone()
    return (reference + displacement * (radius / current_radius)).detach().clone()


def _objective_comparison_tolerance(value: float, reference: Tensor) -> float:
    """Scale-aware roundoff allowance used by the safeguarded L-BFGS path."""

    return float(
        64.0
        * torch.finfo(reference.dtype).eps
        * max(1.0, abs(value))
    )


def optimize_objective(
    model: nn.Module,
    objective: ScalarObjective,
    config: OptimizerConfig | None = None,
) -> OptimizationResult:
    """Minimize ``objective`` in-place using Adam and then L-BFGS."""

    settings = config or OptimizerConfig()
    parameters = trainable_parameters(model)
    optimization_start = flatten_parameters(model, detach=True)
    completed_adam_steps = 0
    last_gradient_norm = float("inf")
    use_stepwise_safeguard = (
        settings.stepwise_lbfgs
        or settings.select_min_gradient_checkpoint
        or settings.max_rms_parameter_change is not None
    )
    last_feasible_adam = optimization_start.detach().clone()
    radius_rejected_candidate = False
    radius_projected_candidate = False
    objective_rejected_candidate = False

    if settings.adam_steps:
        adam = torch.optim.Adam(parameters, lr=settings.adam_lr)
        for _ in range(settings.adam_steps):
            adam.zero_grad(set_to_none=True)
            loss = objective()
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise FloatingPointError("The objective became non-finite.")
            loss.backward()
            last_gradient_norm = _gradient_norm_from_storage(model)
            if last_gradient_norm <= settings.tolerance:
                break
            adam.step()
            completed_adam_steps += 1
            if (
                use_stepwise_safeguard
                and settings.max_rms_parameter_change is not None
            ):
                adam_candidate = flatten_parameters(model, detach=True)
                if (
                    _rms_parameter_change(adam_candidate, optimization_start)
                    > settings.max_rms_parameter_change
                ):
                    if settings.project_rms_parameter_change:
                        projected = _project_to_rms_radius(
                            adam_candidate,
                            optimization_start,
                            settings.max_rms_parameter_change,
                        )
                        assign_parameters(model, projected)
                        last_feasible_adam = projected
                        radius_projected_candidate = True
                    else:
                        radius_rejected_candidate = True
                        assign_parameters(model, last_feasible_adam)
                        break
                else:
                    last_feasible_adam = adam_candidate

    refinement_start = flatten_parameters(model, detach=True)
    lbfgs_evaluations = 0
    lbfgs_iterations = 0
    if settings.lbfgs_steps and not use_stepwise_safeguard:
        lbfgs = torch.optim.LBFGS(
            parameters,
            lr=settings.lbfgs_lr,
            max_iter=settings.lbfgs_steps,
            tolerance_grad=settings.tolerance,
            tolerance_change=max(settings.tolerance * 0.1, 1e-15),
            history_size=settings.history_size,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            nonlocal lbfgs_evaluations
            lbfgs.zero_grad(set_to_none=True)
            value = objective()
            if value.ndim != 0 or not bool(torch.isfinite(value)):
                raise FloatingPointError("The objective became non-finite.")
            value.backward()
            lbfgs_evaluations += 1
            return value

        lbfgs.step(closure)
        state = lbfgs.state.get(parameters[0], {})
        lbfgs_iterations = int(state.get("n_iter", 0))
    elif settings.lbfgs_steps:
        adam_loss, adam_gradient_norm = _evaluate_objective_and_gradient(
            model,
            objective,
        )
        objective_upper_bound = adam_loss + _objective_comparison_tolerance(
            adam_loss,
            refinement_start,
        )
        best_vector = refinement_start.detach().clone()
        best_loss = adam_loss
        best_gradient_norm = adam_gradient_norm
        last_feasible_vector = best_vector

        lbfgs = torch.optim.LBFGS(
            parameters,
            lr=settings.lbfgs_lr,
            max_iter=1,
            tolerance_grad=settings.tolerance,
            tolerance_change=max(settings.tolerance * 0.1, 1e-15),
            history_size=settings.history_size,
            line_search_fn="strong_wolfe",
        )

        def closure() -> Tensor:
            nonlocal lbfgs_evaluations
            lbfgs.zero_grad(set_to_none=True)
            value = objective()
            if value.ndim != 0 or not bool(torch.isfinite(value)):
                raise FloatingPointError("The objective became non-finite.")
            value.backward()
            lbfgs_evaluations += 1
            return value

        for _ in range(settings.lbfgs_steps):
            projected_lbfgs_candidate = False
            lbfgs.step(closure)
            lbfgs_iterations += 1
            candidate_loss, candidate_gradient_norm = (
                _evaluate_objective_and_gradient(model, objective)
            )
            candidate_vector = flatten_parameters(model, detach=True)
            candidate_rms_change = _rms_parameter_change(
                candidate_vector,
                optimization_start,
            )
            if (
                settings.max_rms_parameter_change is not None
                and candidate_rms_change
                > settings.max_rms_parameter_change
            ):
                if settings.project_rms_parameter_change:
                    projected = _project_to_rms_radius(
                        candidate_vector,
                        optimization_start,
                        settings.max_rms_parameter_change,
                    )
                    assign_parameters(model, projected)
                    candidate_vector = projected
                    candidate_loss, candidate_gradient_norm = (
                        _evaluate_objective_and_gradient(model, objective)
                    )
                    radius_projected_candidate = True
                    projected_lbfgs_candidate = True
                else:
                    radius_rejected_candidate = True
                    assign_parameters(model, best_vector)
                    last_feasible_vector = best_vector
                    break

            if candidate_loss <= objective_upper_bound:
                last_feasible_vector = candidate_vector
                if (
                    candidate_gradient_norm < best_gradient_norm
                    or (
                        candidate_gradient_norm == best_gradient_norm
                        and candidate_loss < best_loss
                    )
                ):
                    best_vector = candidate_vector
                    best_loss = candidate_loss
                    best_gradient_norm = candidate_gradient_norm
            else:
                objective_rejected_candidate = True

            # Adam can continue along the projected boundary.  A quasi-Newton
            # line-search history, however, is no longer valid after manual
            # projection, so retain the projected checkpoint and stop L-BFGS.
            if projected_lbfgs_candidate:
                break

        if settings.select_min_gradient_checkpoint:
            assign_parameters(model, best_vector)
        else:
            assign_parameters(model, last_feasible_vector)

    final_loss, last_gradient_norm = _evaluate_objective_and_gradient(
        model,
        objective,
    )
    converged = last_gradient_norm <= settings.tolerance
    final_parameters = flatten_parameters(model, detach=True)
    final_rms_change = _rms_parameter_change(
        final_parameters,
        optimization_start,
    )
    boundary_tolerance = (
        max(1.0e-12, 1.0e-6 * settings.max_rms_parameter_change)
        if settings.max_rms_parameter_change is not None
        else 0.0
    )
    safeguard_boundary_active = (
        settings.max_rms_parameter_change is not None
        and abs(final_rms_change - settings.max_rms_parameter_change)
        <= boundary_tolerance
    )
    return OptimizationResult(
        final_loss=final_loss,
        gradient_norm=last_gradient_norm,
        adam_steps=completed_adam_steps,
        lbfgs_evaluations=lbfgs_evaluations,
        converged=converged,
        lbfgs_iterations=lbfgs_iterations,
        rms_parameter_change=final_rms_change,
        safeguard_boundary_active=safeguard_boundary_active,
        safeguard_rejected_candidate=(
            radius_rejected_candidate or objective_rejected_candidate
        ),
        radius_rejected_candidate=radius_rejected_candidate,
        radius_projected_candidate=radius_projected_candidate,
        objective_rejected_candidate=objective_rejected_candidate,
    )


def fit_cernn(
    model: CERNN,
    x: Tensor,
    y: Tensor,
    *,
    l2_lambda: float = 0.0,
    optimizer_config: OptimizerConfig | None = None,
) -> OptimizationResult:
    """Fit a centralized or local CERNN model in-place."""

    if l2_lambda < 0.0:
        raise ValueError("l2_lambda cannot be negative.")
    x_value, y_value = prepare_regression_tensors(model, x, y)

    def objective() -> Tensor:
        return cernn_data_loss(model, x_value, y_value) + (
            l2_lambda * shared_hidden_l2_penalty(model)
        )

    return optimize_objective(model, objective, optimizer_config)


def parameter_change(model: nn.Module, reference: Tensor) -> float:
    """Euclidean distance from a stored parameter vector."""

    current = flatten_parameters(model, detach=True)
    return float(torch.linalg.vector_norm(current - reference).cpu())
