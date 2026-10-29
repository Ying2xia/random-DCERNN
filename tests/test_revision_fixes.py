"""Regression tests for the post-audit experiment fixes."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from fedcernn.baselines import (
    BASELINE_NAMES,
    BaselineConfig,
    _marginal_expectile,
    canonical_method_name,
    fit_baseline,
)
from fedcernn.experiments import _baseline_config, _configured_method_rounds
from fedcernn.federated import make_federated_clients
from fedcernn.model import CERNN
from fedcernn.training import OptimizerConfig


TAUS = (0.1, 0.3, 0.5, 0.7, 0.9)


class MarginalExpectileTests(unittest.TestCase):
    def test_half_expectile_equals_sample_mean(self) -> None:
        sample = np.random.default_rng(0).normal(3.0, 2.0, size=20000)
        self.assertAlmostEqual(
            _marginal_expectile(sample, 0.5), float(sample.mean()), places=8
        )

    def test_solution_satisfies_empirical_moment_equation(self) -> None:
        sample = np.random.default_rng(1).standard_exponential(5000)
        for tau in TAUS:
            level = _marginal_expectile(sample, tau)
            residual = sample - level
            weights = np.where(residual >= 0.0, tau, 1.0 - tau)
            self.assertLess(abs(float(np.mean(weights * residual))), 1.0e-8)

    def test_expectiles_increase_with_level(self) -> None:
        sample = np.random.default_rng(2).normal(size=5000)
        levels = [_marginal_expectile(sample, tau) for tau in TAUS]
        self.assertTrue(all(left < right for left, right in zip(levels, levels[1:])))


class InterceptOnlyTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(size=(600, 4))
        y = x[:, 0] + rng.normal(size=600)
        self.x = torch.as_tensor(x, dtype=torch.float64)
        self.y = torch.as_tensor(y, dtype=torch.float64)
        self.clients = make_federated_clients(self.x, self.y, 4, seed=0)

    def test_registered_and_has_no_network_rounds(self) -> None:
        self.assertIn("Intercept-only", BASELINE_NAMES)
        self.assertEqual(canonical_method_name("intercept"), "Intercept-only")
        self.assertEqual(_configured_method_rounds("Intercept-only", {}), 0)

    def test_predictions_are_pooled_marginal_expectiles(self) -> None:
        model = CERNN(4, (8, 4), TAUS)
        result = fit_baseline("Intercept-only", model, self.clients)
        with torch.no_grad():
            predictions = result.model(self.x).numpy()
        self.assertTrue(np.allclose(predictions, predictions[0][None, :]))
        for value, tau in zip(predictions[0], TAUS):
            self.assertAlmostEqual(
                value, _marginal_expectile(self.y.numpy(), tau), places=8
            )
        self.assertEqual(
            result.history[0]["communication_accounting"], "not_applicable"
        )


class PredictiveSolverSeparationTests(unittest.TestCase):
    def test_predictive_solver_falls_back_when_omitted(self) -> None:
        shared = OptimizerConfig(adam_steps=7)
        config = BaselineConfig(local_optimizer=shared)
        self.assertIs(config.resolved_predictive_optimizer, shared)

    def test_predictive_solver_is_parsed_separately(self) -> None:
        config = _baseline_config(
            {
                "optimization": {"l2_lambda": 1.0e-4},
                "baselines": {
                    "local_optimizer": {
                        "adam_steps": 3,
                        "max_rms_parameter_change": 0.005,
                    },
                    "predictive_local_optimizer": {
                        "adam_steps": 30,
                        "lbfgs_steps": 10,
                    },
                },
            }
        )
        self.assertEqual(config.local_optimizer.adam_steps, 3)
        self.assertEqual(config.local_optimizer.max_rms_parameter_change, 0.005)
        self.assertEqual(config.resolved_predictive_optimizer.adam_steps, 30)
        self.assertIsNone(
            config.resolved_predictive_optimizer.max_rms_parameter_change
        )

    def test_method_specific_predictive_solver_overrides_default(self) -> None:
        config = _baseline_config(
            {
                "baselines": {
                    "predictive_local_optimizer": {"adam_steps": 30},
                    "predictive_local_optimizers": {
                        "SCAFFOLD": {"adam_steps": 100, "adam_lr": 0.003}
                    },
                }
            }
        )
        self.assertEqual(
            config.predictive_optimizer_for("multi-round FedAvg").adam_steps,
            30,
        )
        self.assertEqual(
            config.predictive_optimizer_for("SCAFFOLD").adam_steps,
            100,
        )
        self.assertEqual(
            config.predictive_optimizer_for("SCAFFOLD").adam_lr,
            0.003,
        )


if __name__ == "__main__":
    unittest.main()
