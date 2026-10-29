#!/usr/bin/env python3
"""Run/resume the Chapter 5 DERNN supplement: MEPS 10 runs, eICU 20 runs."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from fedcernn.paths import frozen_hash_matches

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
OUT = ROOT / 'results/dernn_real_data'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_plan(dataset='all'):
    plan = json.loads((OUT / 'frozen_plan.json').read_text())
    for name, sha in plan['source_sha256'].items():
        if not frozen_hash_matches(ROOT / name, sha):
            raise RuntimeError(f'Frozen source changed: {name}')
    for name, job in plan['datasets'].items():
        expected = 10 if name == 'meps' else 20
        assert len(job['repetitions']) == expected
        assert [r['repetition'] for r in job['repetitions']] == list(range(expected))
        assert len({r['seed'] for r in job['repetitions']}) == expected
        assert job['settings']['optimization']['l2_lambda'] == 0.0001
        assert job['settings']['model']['hidden_dims'] == [32, 16]
        if dataset in ('all', name):
            if digest(ROOT / job['reference_path']) != job['reference_sha256']:
                raise RuntimeError(f'{name}: reference ledger changed.')
            if digest(WORKSPACE / job['data_path']) != job['data_sha256']:
                raise RuntimeError(f'{name}: source data changed.')
    return plan


def prepare_data(dataset, settings, seed):
    from fedcernn.data_meps import prepare_meps
    from fedcernn.data_eicu import load_restricted_eicu_matrices
    data = settings['data']
    if dataset == 'meps':
        if data.get('target_transform', 'raw') != 'raw':
            raise ValueError('This supplement uses the Chapter 5 raw-scale MEPS analysis.')
        return prepare_meps(WORKSPACE / data['path'], seed=seed,
                            feature_columns=data.get('feature_columns'), target_transform='raw')
    return load_restricted_eicu_matrices(WORKSPACE / data['matrix_path'])


def original_scale_metrics(dataset, splits, test_prediction, validation_prediction):
    """Accept a single standardized output; evaluate at all five levels."""
    import numpy as np
    from fedcernn.metrics import evaluate_predictions
    from run_dernn_r100 import TAUS

    pred = splits.inverse_target(test_prediction)
    val_pred = splits.inverse_target(validation_prediction)
    assert pred.shape == (len(splits.y_test), 1)
    assert val_pred.shape == (len(splits.y_validation), 1)
    target = splits.y_test_thousands if dataset == 'meps' else splits.inverse_target(splits.y_test)
    val_target = (splits.y_validation_thousands if dataset == 'meps'
                  else splits.inverse_target(splits.y_validation))
    repeated = np.repeat(pred, 5, axis=1)
    metrics = evaluate_predictions(target, repeated, TAUS).to_flat_dict(TAUS)
    validation = evaluate_predictions(val_target, np.repeat(val_pred, 5, axis=1), TAUS).to_flat_dict(TAUS)
    metrics.update({f'validation_{key}': value for key, value in validation.items()})
    metrics['mse_tau_0p5'] = float(np.mean((np.asarray(target).reshape(-1) - pred[:, 0])**2))
    np.testing.assert_allclose(metrics['composite_risk'], 0.5 * metrics['mse_tau_0p5'], rtol=1e-12)
    if dataset == 'meps':
        weighted = evaluate_predictions(target, repeated, TAUS,
                                        sample_weight=splits.test_weights).to_flat_dict(TAUS)
        metrics.update({f'survey_weighted_{key}': value for key, value in weighted.items()})
    assert all(np.isfinite(value) for value in metrics.values())
    return metrics


def fit_one(dataset, job, repetition, smoke=False):
    from fedcernn.baselines import fit_centralized, predict
    from fedcernn.experiments import _baseline_config
    from run_dernn_r100 import TAUS, make_model

    settings = copy.deepcopy(job['settings'])
    seed = repetition['seed']
    splits = prepare_data(dataset, settings, seed)
    assert len(splits.y_train) == repetition['n_train']
    assert len(splits.y_test) == repetition['n_test']
    clients = splits.clients(settings['data']['n_clients'], seed=(seed + 1) % 2**32)
    if smoke:
        # Keep the actual preprocessing/splits, but fit at most 20 rows/client.
        clients = [(x[:20], y[:20]) for x, y in clients]
        settings['optimization'].update(adam_steps=2, lbfgs_steps=2)
        if 'host_optimizer' in settings.get('fedcernn', {}):
            settings['fedcernn']['host_optimizer'].update(adam_steps=2, lbfgs_steps=2)
    started = time.perf_counter()
    model = make_model(splits.X_train.shape[1], settings, (seed + 2) % 2**32)
    result = fit_centralized(model, clients, _baseline_config(settings))
    metrics = original_scale_metrics(dataset, splits, predict(result.model, splits.X_test),
                                     predict(result.model, splits.X_validation))
    elapsed = time.perf_counter() - started
    settings['model']['taus'] = [0.5]
    settings['methods'] = ['DERNN']
    settings['evaluation'] = {'taus': list(TAUS), 'prediction': 'repeat_single_output'}
    return dict(dataset=dataset, method='DERNN', status='success', smoke=smoke,
                repetition=repetition['repetition'], seed=seed, base_seed=repetition['base_seed'],
                source_centralized_run_key=repetition['run_key'], training_taus=[0.5],
                evaluation_taus=list(TAUS), n_expectiles=1, hidden_dims='32-16',
                n_train=sum(len(y) for _, y in clients), n_test=len(splits.y_test),
                n_clients=len(clients), parameter_count=sum(p.numel() for p in model.parameters()),
                elapsed_seconds=elapsed, communication_bytes=0, communication_rounds=0,
                scenario_json=settings, metrics_json=metrics, history_json=result.history,
                timestamp_utc=datetime.now(timezone.utc).isoformat(), **metrics)


def write_summary(dataset, rows, directory):
    import numpy as np
    from scipy.stats import t
    from fedcernn.experiments import _json_safe
    n = len(rows)
    summary = {'dataset': dataset, 'method': 'DERNN', 'n': n, 'training_tau': 0.5}
    for metric in list(rows[0]['metrics_json']) + ['elapsed_seconds']:
        values = np.asarray([r[metric] for r in rows], dtype=float)
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if n > 1 else None
        radius = float(t.ppf(.975, n-1) * sd / np.sqrt(n)) if n > 1 else None
        summary[metric] = dict(mean=mean, sd=sd,
                               ci_low=mean-radius if radius is not None else None,
                               ci_high=mean+radius if radius is not None else None)
    (directory / 'summary.json').write_text(json.dumps(_json_safe(summary), indent=2) + '\n')
    if not rows[0]['smoke']:
        r = summary['composite_risk']
        row = (f"DERNN ($\\tau=0.5$) & {r['mean']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
               f" & {summary['calibration']['mean']:.4f} & {summary['crossing_rate']['mean']:.4f}"
               " & N/A & N/A \\\\\n")
        (directory / 'dernn_table_row.tex').write_text(row)


def run_dataset(dataset, job, smoke=False):
    from fedcernn.experiments import _json_safe
    directory = OUT / ('smoke' if smoke else 'runs') / dataset
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'results.jsonl'
    fingerprint = digest(OUT / 'frozen_plan.json')
    todo = job['repetitions'][:1] if smoke else job['repetitions']
    expected = {(r['repetition'], r['seed']) for r in todo}
    with (directory / '.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'{dataset}: another DERNN process is already writing these results.')
        rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        completed = set()
        for row in rows:
            assert row['plan_sha256'] == fingerprint and row['smoke'] == smoke
            assert row['dataset'] == dataset and row['status'] == 'success'
            key = (row['repetition'], row['seed'])
            assert key in expected and key not in completed
            completed.add(key)
        for repetition in todo:
            key = (repetition['repetition'], repetition['seed'])
            if key in completed:
                continue
            print(f'{dataset}: starting {key[0]+1}/{len(todo)}', flush=True)
            record = fit_one(dataset, job, repetition, smoke=smoke)
            record['plan_sha256'] = fingerprint
            rows.append(record)
            temp = path.with_suffix('.jsonl.tmp')
            with temp.open('w') as handle:
                for row in rows:
                    handle.write(json.dumps(_json_safe(row), allow_nan=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, path)
            print(f"{dataset}: completed {key[0]+1}/{len(todo)}, risk={record['composite_risk']:.6f}", flush=True)
        assert len(rows) == len(todo)
        write_summary(dataset, rows, directory)
    print(f'{dataset}: results saved in {directory}', flush=True)


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['all', 'meps', 'eicu'], default='all')
    parser.add_argument('--dry-run', action='store_true', help='Check data, reference ledgers and code; no training.')
    parser.add_argument('--smoke', action='store_true', help='One reduced fit per dataset, stored separately.')
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = read_plan(args.dataset)
    selected = list(plan['datasets']) if args.dataset == 'all' else [args.dataset]
    for dataset in selected:
        print(f"Validated {dataset}: {len(plan['datasets'][dataset]['repetitions'])} DERNN runs, tau=0.5.", flush=True)
    if args.dry_run:
        print('No training started.', flush=True)
        return
    for dataset in selected:
        run_dataset(dataset, plan['datasets'][dataset], smoke=args.smoke)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nStopped. Run the same command to resume completed fits.')
        raise SystemExit(130)
