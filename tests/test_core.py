"""Core numerical invariants for CERNN and FedCERNN."""

from __future__ import annotations

import math
import pathlib
import sys
import unittest

import torch

CODE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from fedcernn import (  # noqa: E402
    CERNN,
    FedCERNN,
    FederatedClient,
    OptimizerConfig,
    aggregate_client_gradients,
    centralized_federated_objective,
    composite_expectile_loss,
    equal_iid_partition,
    flatten_parameters,
    make_surrogate_objective,
    objective_gradient,
    optimize_objective,
    parameter_count,
    shared_hidden_l2_penalty,
)


class LossTests(unittest.TestCase):
    def test_zero_residual_has_zero_gradient(self) -> None:
        predictions = torch.tensor(
            [[2.0, 2.0, 2.0], [-1.0, -1.0, -1.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        targets = torch.tensor([2.0, -1.0], dtype=torch.float64)
        loss = composite_expectile_loss(
            predictions,
            targets,
            (0.1, 0.5, 0.9),
        )
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        torch.testing.assert_close(
            predictions.grad,
            torch.zeros_like(predictions),
            atol=0.0,
            rtol=0.0,
        )
class PartitionTests(unittest.TestCase):
    def test_equal_iid_partition_is_complete_and_balanced(self) -> None:
        partitions = equal_iid_partition(103, 10, seed=27)
        sizes = [partition.numel() for partition in partitions]
        self.assertLessEqual(max(sizes) - min(sizes), 1)
        combined = torch.cat(partitions)
        self.assertEqual(combined.numel(), 103)
        torch.testing.assert_close(
            torch.sort(combined).values,
            torch.arange(103),
            atol=0,
            rtol=0,
        )
        repeated = equal_iid_partition(103, 10, seed=27)
        for first, second in zip(partitions, repeated, strict=True):
            torch.testing.assert_close(first, second, atol=0, rtol=0)


class OptimizerSafeguardTests(unittest.TestCase):
    @staticmethod
    def _quadratic_model() -> torch.nn.Linear:
        model = torch.nn.Linear(3, 1, bias=True, dtype=torch.float64)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[2.0, -1.0, 0.5]], dtype=torch.float64))
            model.bias.fill_(-0.75)
        return model

    @staticmethod
    def _quadratic_objective(model: torch.nn.Linear) -> torch.Tensor:
        vector = flatten_parameters(model)
        target = torch.tensor([0.3, -0.2, 0.1, 0.4], dtype=torch.float64)
        weights = torch.tensor([1.0, 4.0, 16.0, 64.0], dtype=torch.float64)
        return torch.sum(weights * (vector - target).square())

    def test_stepwise_candidate_prefix_cannot_worsen_selected_gradient(self) -> None:
        short_model = self._quadratic_model()
        long_model = self._quadratic_model()
        initial_loss = float(self._quadratic_objective(short_model))
        common = dict(
            adam_steps=0,
            lbfgs_lr=0.4,
            tolerance=0.0,
            stepwise_lbfgs=True,
            select_min_gradient_checkpoint=True,
        )
        short = optimize_objective(
            short_model,
            lambda: self._quadratic_objective(short_model),
            OptimizerConfig(lbfgs_steps=5, **common),
        )
        long = optimize_objective(
            long_model,
            lambda: self._quadratic_objective(long_model),
            OptimizerConfig(lbfgs_steps=10, **common),
        )
        self.assertEqual(short.lbfgs_iterations, 5)
        self.assertEqual(long.lbfgs_iterations, 10)
        self.assertLessEqual(short.final_loss, initial_loss + 1e-12)
        self.assertLessEqual(long.final_loss, initial_loss + 1e-12)
        self.assertLessEqual(long.gradient_norm, short.gradient_norm + 1e-12)

    def test_stepwise_rms_radius_restores_a_feasible_checkpoint(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            model.weight.zero_()

        result = optimize_objective(
            model,
            lambda: (model.weight.reshape(()) - 100.0).square(),
            OptimizerConfig(
                adam_steps=0,
                lbfgs_steps=5,
                lbfgs_lr=1.0,
                tolerance=0.0,
                stepwise_lbfgs=True,
                select_min_gradient_checkpoint=True,
                max_rms_parameter_change=0.01,
            ),
        )
        self.assertLessEqual(result.rms_parameter_change, 0.01 + 1e-12)
        self.assertEqual(result.lbfgs_iterations, 1)
        self.assertTrue(result.safeguard_rejected_candidate)
        self.assertTrue(result.radius_rejected_candidate)
        self.assertAlmostEqual(float(model.weight), 0.0, places=14)

    def test_rms_radius_also_safeguards_adam_steps(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            model.weight.zero_()

        result = optimize_objective(
            model,
            lambda: (model.weight.reshape(()) - 100.0).square(),
            OptimizerConfig(
                adam_steps=3,
                adam_lr=1.0,
                lbfgs_steps=0,
                max_rms_parameter_change=0.01,
            ),
        )
        self.assertEqual(result.adam_steps, 1)
        self.assertLessEqual(result.rms_parameter_change, 0.01)
        self.assertTrue(result.safeguard_rejected_candidate)
        self.assertTrue(result.radius_rejected_candidate)
        self.assertAlmostEqual(float(model.weight), 0.0, places=14)

    def test_rms_radius_projection_keeps_adam_within_the_boundary(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            model.weight.zero_()

        result = optimize_objective(
            model,
            lambda: (model.weight.reshape(()) - 100.0).square(),
            OptimizerConfig(
                adam_steps=3,
                adam_lr=1.0,
                lbfgs_steps=0,
                max_rms_parameter_change=0.01,
                project_rms_parameter_change=True,
            ),
        )
        self.assertEqual(result.adam_steps, 3)
        self.assertLessEqual(result.rms_parameter_change, 0.01 + 1e-12)
        self.assertTrue(result.radius_projected_candidate)
        self.assertFalse(result.radius_rejected_candidate)
        self.assertAlmostEqual(float(model.weight), 0.01, places=14)

    def test_rms_radius_projection_retains_a_stepwise_lbfgs_checkpoint(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            model.weight.zero_()

        result = optimize_objective(
            model,
            lambda: (model.weight.reshape(()) - 100.0).square(),
            OptimizerConfig(
                adam_steps=0,
                lbfgs_steps=2,
                lbfgs_lr=1.0,
                tolerance=0.0,
                stepwise_lbfgs=True,
                select_min_gradient_checkpoint=True,
                max_rms_parameter_change=0.01,
                project_rms_parameter_change=True,
            ),
        )
        self.assertEqual(result.lbfgs_iterations, 1)
        self.assertLessEqual(result.rms_parameter_change, 0.01 + 1e-12)
        self.assertTrue(result.radius_projected_candidate)
        self.assertFalse(result.radius_rejected_candidate)

    def test_default_optimizer_path_still_runs(self) -> None:
        model = self._quadratic_model()
        result = optimize_objective(
            model,
            lambda: self._quadratic_objective(model),
            OptimizerConfig(adam_steps=2, lbfgs_steps=3),
        )
        self.assertTrue(math.isfinite(result.final_loss))
        self.assertTrue(math.isfinite(result.gradient_norm))
        self.assertGreater(result.lbfgs_evaluations, 0)
        self.assertFalse(result.safeguard_boundary_active)
        self.assertFalse(result.safeguard_rejected_candidate)
        self.assertFalse(OptimizerConfig().stepwise_lbfgs)


class FederatedGradientTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.model = CERNN(
            input_dim=3,
            hidden_dims=(5, 4),
            taus=(0.2, 0.5, 0.8),
        )
        x = torch.randn(23, 3, dtype=torch.float64)
        y = (
            0.7 * x[:, 0]
            - 0.2 * x[:, 1].square()
            + 0.1 * torch.randn(23, dtype=torch.float64)
        )
        split_sizes = (8, 7, 8)
        self.clients: list[FederatedClient] = []
        offset = 0
        for size in split_sizes:
            self.clients.append(
                FederatedClient(x[offset : offset + size], y[offset : offset + size])
            )
            offset += size

    def test_global_gradient_decomposition(self) -> None:
        global_gradient, _ = aggregate_client_gradients(self.model, self.clients)
        pooled_x = torch.cat([client.x for client in self.clients])
        pooled_y = torch.cat([client.y for client in self.clients])
        pooled_gradient = objective_gradient(
            self.model,
            lambda: composite_expectile_loss(
                self.model(pooled_x),
                pooled_y,
                self.model.taus,
            ),
        )
        torch.testing.assert_close(
            global_gradient,
            pooled_gradient,
            atol=2e-12,
            rtol=2e-10,
        )

    def test_surrogate_gradient_matches_centralized_gradient(self) -> None:
        l2_lambda = 0.037
        expansion_point = flatten_parameters(self.model, detach=True)
        global_gradient, local_gradients = aggregate_client_gradients(
            self.model,
            self.clients,
        )
        surrogate = make_surrogate_objective(
            self.model,
            self.clients[0],
            local_gradients[0],
            global_gradient,
            l2_lambda,
            expansion_point,
        )
        surrogate_gradient = objective_gradient(self.model, surrogate)
        centralized_gradient = objective_gradient(
            self.model,
            lambda: centralized_federated_objective(
                self.model,
                self.clients,
                l2_lambda,
            ),
        )
        torch.testing.assert_close(
            surrogate_gradient,
            centralized_gradient,
            atol=2e-12,
            rtol=2e-10,
        )

    def test_penalty_is_added_once_not_once_per_client(self) -> None:
        l2_lambda = 0.05
        global_gradient, local_gradients = aggregate_client_gradients(
            self.model,
            self.clients,
        )
        penalty_gradient = objective_gradient(
            self.model,
            lambda: shared_hidden_l2_penalty(self.model),
        )
        surrogate = make_surrogate_objective(
            self.model,
            self.clients[0],
            local_gradients[0],
            global_gradient,
            l2_lambda,
            flatten_parameters(self.model, detach=True),
        )
        actual = objective_gradient(self.model, surrogate)
        expected = global_gradient + l2_lambda * penalty_gradient
        repeated_penalty = (
            global_gradient
            + len(self.clients) * l2_lambda * penalty_gradient
        )
        torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-10)
        self.assertGreater(
            float(torch.linalg.vector_norm(actual - repeated_penalty)),
            1e-6,
        )

    def test_fit_reports_model_message_communication(self) -> None:
        config = OptimizerConfig(
            adam_steps=2,
            adam_lr=0.01,
            lbfgs_steps=2,
            tolerance=1e-6,
        )
        trainer = FedCERNN(
            self.model,
            l2_lambda=1e-4,
            optimizer_config=config,
        )
        result = trainer.fit(self.clients, rounds=1)
        expected = (
            2
            * (len(self.clients) - 1)
            * parameter_count(self.model)
            * next(self.model.parameters()).element_size()
        )
        self.assertEqual(result.communication_bytes, expected)
        self.assertEqual(len(result.history), 2)
        self.assertLess(
            result.history[1]["surrogate_gradient_match_error"],
            1e-10,
        )

    def test_host_regularization_path_pretrains_without_communication(self) -> None:
        config = OptimizerConfig(adam_steps=1, lbfgs_steps=0)
        result = FedCERNN(
            self.model,
            l2_lambda=1e-4,
            optimizer_config=config,
            host_regularization_path=(1e-3, 1e-4),
        ).fit(self.clients, rounds=0)
        self.assertEqual(result.communication_bytes, 0)
        self.assertEqual(len(result.history), 2)
        self.assertEqual(result.history[0]["stage"], "host_regularization_warm_start")
        self.assertEqual(result.history[1]["stage"], "host_fit")
        self.assertEqual(result.history[1]["regularization_path"], [1e-3, 1e-4])


if __name__ == "__main__":
    unittest.main()
