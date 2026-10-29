#!/usr/bin/env python3
"""Run/resume the frozen DERNN tau=0.5 supplement to the three primary examples."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from fedcernn.paths import frozen_hash_matches

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/dernn_tau05_r100'
TAUS = (0.1, 0.3, 0.5, 0.7, 0.9)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_plan():
    plan = json.loads((OUT / 'frozen_plan.json').read_text())
    for name, sha in plan['source_sha256'].items():
        if not frozen_hash_matches(ROOT / name, sha):
            raise RuntimeError(f'Frozen source changed: {name}')
    assert len(plan['jobs']) == 12 and plan['expected_fits'] == 1200
    for job in plan['jobs']:
        assert [r['repetition'] for r in job['repetitions']] == list(range(100))
        assert len({r['seed'] for r in job['repetitions']}) == 100
        assert job['settings']['model'] == {'hidden_dims': [32, 16], 'taus': list(TAUS)}
        assert job['settings']['simulation']['n_train'] == 100000
        assert job['settings']['simulation']['n_test'] == 5000
    return plan


def make_model(input_dim, settings, seed):
    """Match the existing centralized model's initial backbone and 0.5 head."""
    from fedcernn.experiments import _make_model
    from fedcernn.model import CERNN
    reference = _make_model(input_dim, settings, TAUS, seed)
    model = CERNN(input_dim, hidden_dims=(32, 16), taus=(0.5,))
    model.backbone.load_state_dict(reference.backbone.state_dict())
    model.heads[0].load_state_dict(reference.heads[2].state_dict())
    return model


def fit_one(job, repetition, smoke=False):
    import numpy as np
    from fedcernn.baselines import fit_centralized, predict
    from fedcernn.experiments import _baseline_config, _partition_clients
    from fedcernn.metrics import evaluate_predictions
    from fedcernn.simulation import generate_simulation

    settings = copy.deepcopy(job['settings'])
    sim = settings['simulation']
    if smoke:
        sim.update(n_train=200, n_test=80, n_clients=4)
        for config in (settings['optimization'], settings['fedcernn']['host_optimizer']):
            config.update(adam_steps=2, lbfgs_steps=2)
    seed = repetition['seed']
    train = generate_simulation(sim['n_train'], sim['dgp'], sim['error'],
                                seed=seed, taus=TAUS, n_features=sim.get('n_features'))
    test = generate_simulation(sim['n_test'], sim['dgp'], sim['error'],
                               seed=(seed + 1) % 2**32, taus=TAUS,
                               n_features=train.n_features)
    clients = _partition_clients(train.X, train.y, sim['n_clients'],
                                 seed=(seed + 2) % 2**32, partition=sim.get('partition'))
    started = time.perf_counter()
    model = make_model(train.n_features, settings, (seed + 3) % 2**32)
    result = fit_centralized(model, clients, _baseline_config(settings))
    single = predict(result.model, test.X)
    assert single.shape == (sim['n_test'], 1)
    metrics = evaluate_predictions(test.y, np.repeat(single, 5, axis=1), TAUS,
                                   true_expectiles=test.true_expectiles).to_flat_dict(TAUS)
    mse = float(np.mean((test.y.reshape(-1) - single[:, 0])**2))
    np.testing.assert_allclose(metrics['composite_risk'], 0.5 * mse, rtol=1e-12)
    assert all(np.isfinite(value) for value in metrics.values())
    settings['model']['taus'] = [0.5]
    settings['methods'] = ['DERNN']
    settings['evaluation'] = {'taus': list(TAUS), 'prediction': 'repeat_single_output'}
    return dict(method='DERNN', status='success', dgp=sim['dgp'], error=sim['error'],
                repetition=repetition['repetition'], seed=seed,
                base_seed=repetition['base_seed'], training_taus=[0.5],
                evaluation_taus=list(TAUS), hidden_dims='32-16', n_expectiles=1,
                n_train=sim['n_train'], n_test=sim['n_test'], n_clients=sim['n_clients'],
                parameter_count=sum(p.numel() for p in model.parameters()),
                scenario_json=settings, source_centralized_run_key=repetition['run_key'],
                communication_bytes=0, communication_rounds=0, rounds=0,
                elapsed_seconds=time.perf_counter()-started, mse_tau_0p5=mse,
                metrics_json=metrics, history_json=result.history, smoke=smoke, **metrics)


def run_job(plan, job, smoke=False):
    import torch
    torch.set_num_threads(1)
    directory = OUT / ('smoke' if smoke else 'runs') / job['name']
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'results.jsonl'
    fingerprint = digest(OUT / 'frozen_plan.json')
    with (directory / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        completed = set()
        if path.exists():
            for line in path.read_text().splitlines():
                record = json.loads(line)
                assert record['plan_sha256'] == fingerprint
                assert record['smoke'] == smoke
                assert record['status'] == 'success'
                key = (record['repetition'], record['seed'])
                assert key not in completed
                completed.add(key)
        todo = job['repetitions'][:1] if smoke else job['repetitions']
        assert completed <= {(r['repetition'], r['seed']) for r in todo}
        for repetition in todo:
            key = (repetition['repetition'], repetition['seed'])
            if key in completed:
                continue
            print(f"{job['name']}: starting {key[0]+1}/{len(todo)}", flush=True)
            record = fit_one(job, repetition, smoke=smoke)
            record['plan_sha256'] = fingerprint
            # Replace the small per-scenario ledger atomically, so interruption
            # cannot leave a partial JSON record that breaks resume.
            previous = path.read_text() if path.exists() else ''
            temp = path.with_suffix('.jsonl.tmp')
            from fedcernn.experiments import _json_safe
            with temp.open('w') as handle:
                handle.write(previous + json.dumps(_json_safe(record), allow_nan=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
            print(f"{job['name']}: {key[0]+1}/{len(todo)}, risk={record['composite_risk']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=[1, 2], default=2)
    parser.add_argument('--dry-run', action='store_true', help='Check plan and original source ledgers; no training.')
    parser.add_argument('--smoke', action='store_true', help='12 tiny pipeline checks in a separate directory.')
    parser.add_argument('--job', help=argparse.SUPPRESS)
    args = parser.parse_args()
    plan = read_plan()
    if args.job:
        job = next(j for j in plan['jobs'] if j['name'] == args.job)
        run_job(plan, job, smoke=args.smoke)
        return
    for name, sha in plan['reference_results_sha256'].items():
        if digest(ROOT / name) != sha:
            raise RuntimeError(f'Reference result ledger changed: {name}')
    print('Validated: 3 examples × 4 errors × 100 repetitions = 1,200 DERNN fits.', flush=True)
    print('Training: tau=0.5 only; evaluation: same prediction at five levels.', flush=True)
    if args.dry_run:
        print('No training started.', flush=True)
        return
    lockname = '.smoke-launcher.lock' if args.smoke else '.launcher.lock'
    with (OUT / lockname).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        logs = OUT / ('smoke_logs' if args.smoke else 'logs')
        logs.mkdir(exist_ok=True)
        stop = threading.Event()
        guard = threading.Lock()
        children = set()

        def launch(job):
            with (logs / (job['name'] + '.log')).open('a') as log:
                with guard:
                    if stop.is_set():
                        return
                    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--job', job['name']]
                    if args.smoke:
                        command.append('--smoke')
                    child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    children.add(child)
                print(f"Starting {job['name']}; log: {logs / (job['name'] + '.log')}", flush=True)
                try:
                    code = child.wait()
                    if code and not stop.is_set():
                        raise RuntimeError(f"{job['name']} failed (exit {code}); see log.")
                    if not code:
                        print(f"Completed {job['name']}", flush=True)
                finally:
                    with guard:
                        children.discard(child)

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
        futures = [pool.submit(launch, job) for job in plan['jobs']]
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except BaseException:
            stop.set()
            with guard:
                for child in children:
                    if child.poll() is None:
                        child.terminate()
            for future in futures:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    print(f"Completed. Results: {OUT / ('smoke' if args.smoke else 'runs')}", flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped. Run the same command to resume completed fits.', file=sys.stderr)
        sys.exit(130)
