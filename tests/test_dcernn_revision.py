import sys
from pathlib import Path
import unittest
import copy
import json
import tempfile
from unittest.mock import patch
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dcernn_revision import GDConfig,fit_gd,gradient_and_loss
from fedcernn.model import CERNN,flatten_parameters
from fedcernn.baselines import BaselineConfig,fit_dcernn


class RevisionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(12)
        self.model=CERNN(3,hidden_dims=(4,2),taus=(.1,.5,.9))
        self.clients=[(torch.randn(n,3,dtype=torch.float64),torch.randn(n,dtype=torch.float64))
                      for n in (7,13,5)]

    def test_same_algorithm_and_unequal_client_weights(self):
        old=fit_dcernn(copy.deepcopy(self.model),self.clients,
                      BaselineConfig(rounds=1,local_steps=5,local_lr=.03,l2_lambda=.002))
        new=fit_gd(copy.deepcopy(self.model),self.clients,
                   GDConfig(.03,max_steps=5,min_steps=10,check_every=1,l2_lambda=.002))
        torch.testing.assert_close(flatten_parameters(old.model),flatten_parameters(new['model']),rtol=1e-12,atol=1e-12)
        self.assertEqual(new['communication_bytes'],old.communication_bytes*6//5)
        self.assertEqual(new['communication_rounds'],6) # includes endpoint gradient check

    def test_pooled_development_matches_distributed(self):
        cfg=GDConfig(.1,max_steps=7,min_steps=20,check_every=2)
        a=fit_gd(copy.deepcopy(self.model),self.clients,cfg)
        b=fit_gd(copy.deepcopy(self.model),self.clients,cfg,pooled_execution=True)
        torch.testing.assert_close(flatten_parameters(a['model']),flatten_parameters(b['model']),rtol=1e-12,atol=1e-12)

    def test_stopping_and_checkpoint_isolation(self):
        cfg=GDConfig(.01,max_steps=20,min_steps=3,check_every=1,gradient_tolerance=100)
        r=fit_gd(self.model,self.clients,cfg,return_checkpoints=True)
        self.assertEqual(r['steps'],3)
        self.assertEqual(r['stop_reason'],'gradient_tolerance')
        self.assertEqual(r['communication_rounds'],4)
        initial=r['checkpoints'][0].clone()
        for p in self.model.parameters():
            with torch.no_grad():p.zero_()
        torch.testing.assert_close(initial,r['checkpoints'][0])

    def test_cap_is_not_convergence(self):
        r=fit_gd(self.model,self.clients,GDConfig(.01,max_steps=2,min_steps=20))
        self.assertEqual(r['stop_reason'],'max_steps')
        self.assertFalse(r['converged'])

    def test_scheduled_updates_and_scalar_accounting(self):
        from dcernn_final_revision import GDConfig as FinalConfig,fit_gd as final_fit
        expected=copy.deepcopy(self.model)
        for rate in (.1,.1,.01,.01,.001):
            expected=fit_dcernn(expected,self.clients,BaselineConfig(rounds=1,local_steps=1,
                                 local_lr=rate,l2_lambda=.002)).model
        cfg=FinalConfig(.1,max_steps=5,min_steps=20,check_every=1,l2_lambda=.002,
                        decay_steps=(2,4),decay_factor=.1)
        result=final_fit(self.model,self.clients,cfg)
        torch.testing.assert_close(flatten_parameters(expected),flatten_parameters(result['model']),rtol=1e-12,atol=1e-12)
        pbytes=sum(p.numel()*p.element_size() for p in self.model.parameters())
        self.assertEqual(result['communication_bytes'],6*2*(2*pbytes+8))
        self.assertEqual(result['objective_scalar_bytes'],6*2*8)

    def test_simulation_smoke_does_not_mutate_settings(self):
        from run_dcernn_revision import fit_simulation
        job=dict(settings={'model':{'hidden_dims':[32,16]},'simulation':
                            {'dgp':'hierarchical_composition','error':'normal','n_train':100000,
                             'n_test':5000,'n_clients':20}},
                 gd_config=dict(learning_rate=.3,max_steps=6000))
        before=copy.deepcopy(job)
        row=fit_simulation(job,{'seed':123456789},smoke=True)
        self.assertEqual(job,before)
        self.assertEqual(row['steps'],3)
        self.assertEqual(row['n_test'],80)
        self.assertEqual(row['communication_rounds'],4)

    def test_resume_after_failure_and_reject_changed_plan(self):
        import run_dcernn_revision as runner
        job={'name':'resume-test','dataset':'simulation','repetitions':
             [dict(repetition=i,seed=i+1,base_seed=12) for i in range(2)]}
        result=dict(composite_risk=.4,steps=3,stop_reason='max_steps',converged=False,
                    communication_bytes=100,communication_rounds=4,elapsed_seconds=.01)
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'OUT',Path(tmp)):
            plan=Path(tmp)/'frozen_plan.json';plan.write_text('{}')
            with patch.object(runner,'fit_simulation',side_effect=[copy.deepcopy(result),RuntimeError('interruption')]):
                with self.assertRaises(RuntimeError):runner.run_job(job)
            path=Path(tmp)/'runs/resume-test/results.jsonl'
            self.assertEqual(len(path.read_text().splitlines()),1)
            with patch.object(runner,'fit_simulation',return_value=copy.deepcopy(result)) as fit:
                runner.run_job(job);runner.run_job(job)
                self.assertEqual(fit.call_count,1)
            self.assertEqual(len(path.read_text().splitlines()),2)
            plan.write_text('{"changed":true}')
            with self.assertRaises(AssertionError):runner.run_job(job)

    def test_validation_search_never_reads_test(self):
        import numpy as np
        from run_dcernn_revision import real_validation_search
        class Splits:
            X_train=np.random.default_rng(1).normal(size=(80,3))
            y_train=np.random.default_rng(2).normal(size=80)
            X_validation=np.random.default_rng(3).normal(size=(20,3))
            y_validation=np.random.default_rng(4).normal(size=20)
            @property
            def X_test(self):raise AssertionError('Test accessed during tuning')
            @property
            def y_test(self):raise AssertionError('Test accessed during tuning')
        grid={'learning_rates':[.03,.1], 'budgets':[1,2,3], 'near_best_fraction':.01,
              'stopping':dict(max_steps=3,min_steps=1,check_every=1)}
        settings={'model':{'hidden_dims':[4,2]},'data':{'n_clients':4}}
        with patch('run_dernn_real_data.prepare_data',return_value=Splits()):
            _,selected,candidates=real_validation_search(settings,{'seed':123,'dataset':'eicu'},grid,smoke=True)
        self.assertIn(selected['step'],[1,2,3])
        self.assertEqual(len(candidates),2)


if __name__=='__main__':unittest.main()
