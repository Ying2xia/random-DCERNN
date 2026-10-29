from types import SimpleNamespace
import unittest

import numpy as np

from run_dernn_real_data import original_scale_metrics


class RealDERNNMetricsTests(unittest.TestCase):
    def test_original_units_and_meps_survey_weights(self):
        # Nonzero location and nonunit scale catch standardized/original-scale
        # mixing; unequal weights catch accidentally reusing unweighted risk.
        y_test = np.array([-1., 0., 2.])
        y_validation = np.array([1., -0.5])
        pred = np.array([[0.], [0.5], [1.]])
        val_pred = np.array([[0.], [0.]])
        inverse = lambda x: np.asarray(x) * 3. + 7.
        splits = SimpleNamespace(y_test=y_test, y_validation=y_validation,
                                 y_test_thousands=inverse(y_test),
                                 y_validation_thousands=inverse(y_validation),
                                 test_weights=np.array([1., 3., 2.]), inverse_target=inverse)
        losses = .5 * (inverse(y_test) - inverse(pred)[:, 0])**2
        for dataset in ('meps', 'eicu'):
            metrics = original_scale_metrics(dataset, splits, pred, val_pred)
            self.assertAlmostEqual(metrics['composite_risk'], losses.mean())
            self.assertAlmostEqual(metrics['mse_tau_0p5'], 2 * losses.mean())
            self.assertAlmostEqual(metrics['validation_composite_risk'],
                                   (.5 * (inverse(y_validation)-inverse(val_pred)[:, 0])**2).mean())
            self.assertEqual(metrics['crossing_rate'], 0.)
            if dataset == 'meps':
                self.assertAlmostEqual(metrics['survey_weighted_composite_risk'],
                                       np.average(losses, weights=splits.test_weights))
            else:
                self.assertNotIn('survey_weighted_composite_risk', metrics)


if __name__ == '__main__':
    unittest.main()
