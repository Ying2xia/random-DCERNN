import unittest

import numpy as np
import torch

from fedcernn.experiments import _make_model
from fedcernn.metrics import evaluate_predictions
from fedcernn.losses import composite_expectile_loss
from fedcernn.simulation import generate_simulation
from run_dernn_r100 import TAUS, make_model


class DERNNTests(unittest.TestCase):
    def test_single_head_matches_original_middle_head_initialization(self):
        settings = {'model': {'hidden_dims': [32, 16], 'taus': list(TAUS)}}
        full = _make_model(10, settings, TAUS, 73)
        single = make_model(10, settings, 73)
        x = torch.randn(11, 10, dtype=torch.float64)
        torch.testing.assert_close(single(x)[:, 0], full(x)[:, 2], rtol=0, atol=0)
        self.assertEqual(len(single.heads), 1)
        self.assertEqual(single.tau_values, (0.5,))

    def test_training_loss_and_repeated_output_score_are_half_mse(self):
        prediction = torch.tensor([[0.4], [-1.2], [3.0]], dtype=torch.float64, requires_grad=True)
        target = torch.tensor([1.0, 0.0, -2.0], dtype=torch.float64)
        loss = composite_expectile_loss(prediction, target, torch.tensor([0.5]))
        expected = 0.5 * ((prediction[:, 0] - target)**2).mean()
        torch.testing.assert_close(loss, expected)
        torch.testing.assert_close(torch.autograd.grad(loss, prediction)[0],
                                   torch.autograd.grad(expected, prediction)[0])
        repeated = np.repeat(prediction.detach().numpy(), 5, axis=1)
        metrics = evaluate_predictions(target, repeated, TAUS)
        self.assertAlmostEqual(metrics.composite_risk, expected.item())
        # Level-specific scores remain asymmetric; only their average is MSE/2.
        self.assertNotAlmostEqual(metrics.per_tau_risk[0], metrics.per_tau_risk[-1])

    def test_changing_training_levels_does_not_change_data(self):
        for dgp in ('smooth_additive', 'correlated_interaction', 'hierarchical_composition'):
            for error in ('normal', 't3', 'centered_chi2', 'contaminated_normal'):
                full = generate_simulation(20, dgp, error, seed=183, taus=TAUS)
                single = generate_simulation(20, dgp, error, seed=183, taus=(0.5,))
                np.testing.assert_array_equal(full.X, single.X)
                np.testing.assert_array_equal(full.y, single.y)
                np.testing.assert_array_equal(full.true_expectiles[:, 2], single.true_expectiles[:, 0])


if __name__ == '__main__':
    unittest.main()
