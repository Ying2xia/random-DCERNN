import copy
import unittest
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import torch
import numpy as np
from fedcernn.baselines import BaselineConfig, fit_baseline
from fedcernn.experiments import _baseline_config, _fit_real_method, run_experiment
from fedcernn.model import CERNN, flatten_parameters
from fedcernn.losses import composite_expectile_loss, shared_hidden_l2_penalty
from fedcernn.training import objective_gradient


class RealSolverTests(unittest.TestCase):
    def test_sgd_first_step_matches_pooled_gradient(self):
        torch.manual_seed(7)
        model = CERNN(2, (3, 2))
        clients = [(torch.randn(n, 2, dtype=torch.float64), torch.randn(n, dtype=torch.float64)) for n in [9, 11]]
        start = flatten_parameters(model, detach=True)
        gradient = objective_gradient(model, lambda: sum(
            len(y)/20 * composite_expectile_loss(model(x), y, model.taus) for x,y in clients
        ) + 1e-4 * shared_hidden_l2_penalty(model))
        result = fit_baseline('SCAFFOLD', model, clients, BaselineConfig(rounds=1, scaffold_solver='sgd', scaffold_sgd_steps=1, scaffold_sgd_lr=.03))
        torch.testing.assert_close(flatten_parameters(result.model, detach=True), start-.03*gradient, rtol=1e-12, atol=1e-12)

    def test_dane_override_does_not_change_other_local_solver(self):
        config = _baseline_config({'baselines': {'corrected_local_optimizers': {'DANE-CERNN': {'adam_steps': 30, 'lbfgs_steps': 0}}}})
        self.assertEqual(config.corrected_local_optimizers['DANE-CERNN'].lbfgs_steps, 0)
        self.assertEqual(config.local_optimizer.lbfgs_steps, 20)
        self.assertEqual(config.scaffold_solver, 'legacy')

    def test_selection_uses_validation_only_and_refits(self):
        settings = {'baselines': {'validation_candidates': {'DANE-CERNN': [
            {'name':'bad', 'baselines':{'tag':9}}, {'name':'good', 'baselines':{'tag':0}}
        ]}}}
        saved = copy.deepcopy(settings)
        calls = []
        def fit(method, dim, taus, clients, config, seed):
            tag = config['baselines']['tag']; calls.append((tag, seed))
            return tag, 100, 5, []
        with patch('fedcernn.experiments._fit_method', side_effect=fit), patch('fedcernn.experiments.predict', side_effect=lambda m,x:np.full((len(x),1),m)):
            result, meta = _fit_real_method('DANE-CERNN', 2, [.5], [], settings, 42, np.zeros((4,2)), np.zeros(4))
        self.assertEqual(calls, [(9,42),(0,42),(0,42)])
        self.assertEqual(result[0], 0)
        self.assertEqual(meta['selected'], 'good')
        self.assertFalse(meta['test_used_for_selection'])
        self.assertEqual(settings, saved)

    def test_all_nonfinite_candidates_fail(self):
        s={'baselines':{'validation_candidates':{'SCAFFOLD':[{'name':'bad','baselines':{}}]}}}
        with patch('fedcernn.experiments._fit_method', side_effect=FloatingPointError('bad')):
            with self.assertRaises(FloatingPointError):
                _fit_real_method('SCAFFOLD',2,[.5],[],s,42,np.zeros((1,2)),np.zeros(1))

    def test_health_pipeline_saves_selection_and_resumes(self):
        rng = np.random.default_rng(123)
        x, y = rng.normal(size=(24, 2)), rng.normal(size=24)
        splits = SimpleNamespace(
            X_train=x, y_train=y, X_validation=x[:8], y_validation=y[:8],
            X_test=x[8:16], y_test=y[8:16], y_test_thousands=y[8:16],
            y_validation_thousands=y[:8], target_transform='raw', test_weights=np.ones(8),
            inverse_target=lambda z:z,
            clients=lambda n_clients, seed:[(torch.tensor(x),torch.tensor(y))],
        )
        config = {
            'experiment':{'repetitions':1,'base_seed':42,'fail_fast':True,'torch_threads':1},
            'data':{'path':'mock','matrix_path':'mock','n_clients':1},
            'model':{'hidden_dims':[3,2],'taus':[.5]},
            'methods':['SCAFFOLD'],
            'baselines':{'rounds':1,'scaffold_solver':'sgd','scaffold_sgd_steps':1,
                'validation_candidates':{'SCAFFOLD':[
                    {'name':'a','baselines':{'scaffold_sgd_lr':.01}},
                    {'name':'b','baselines':{'scaffold_sgd_lr':.03}},
                ]}},
        }
        for dataset, loader in [('meps','prepare_meps'),('eicu','load_restricted_eicu_matrices')]:
            with self.subTest(dataset=dataset), tempfile.TemporaryDirectory() as directory, patch('fedcernn.experiments.'+loader, return_value=splits):
                config['dataset']=dataset
                first=run_experiment(config,output_directory=directory)
                second=run_experiment(config,output_directory=directory)
                self.assertEqual(len(second),1)
                self.assertEqual(first[0]['status'],'success')
                self.assertIn(second[0]['tuning_json']['selected'],['a','b'])
                self.assertGreaterEqual(second[0]['elapsed_seconds'],0)
