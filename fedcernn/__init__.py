"""Public API for the FedCERNN research implementation."""

from .federated import (
    FedCERNN,
    FederatedClient,
    FederatedResult,
    aggregate_client_gradients,
    centralized_federated_objective,
    equal_iid_partition,
    make_federated_clients,
    make_surrogate_objective,
    surrogate_gradient_match_error,
)
from .losses import composite_expectile_loss, shared_hidden_l2_penalty
from .model import (
    CERNN,
    assign_parameters,
    flatten_gradients,
    flatten_parameters,
    parameter_count,
)
from .training import (
    OptimizationResult,
    OptimizerConfig,
    cernn_data_loss,
    data_gradient,
    fit_cernn,
    objective_gradient,
    optimize_objective,
)

__all__ = [
    "CERNN",
    "FedCERNN",
    "FederatedClient",
    "FederatedResult",
    "OptimizationResult",
    "OptimizerConfig",
    "aggregate_client_gradients",
    "assign_parameters",
    "cernn_data_loss",
    "centralized_federated_objective",
    "composite_expectile_loss",
    "data_gradient",
    "equal_iid_partition",
    "fit_cernn",
    "flatten_gradients",
    "flatten_parameters",
    "make_federated_clients",
    "make_surrogate_objective",
    "objective_gradient",
    "optimize_objective",
    "parameter_count",
    "shared_hidden_l2_penalty",
    "surrogate_gradient_match_error",
]
