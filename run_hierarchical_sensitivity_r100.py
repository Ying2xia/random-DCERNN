#!/usr/bin/env python3
"""Run/resume the frozen sensitivity studies for the current Example 3."""
from pathlib import Path
import argparse
import concurrent.futures
import copy
import fcntl
import hashlib
import json
import subprocess
import sys
import threading
from fedcernn.paths import frozen_hash_matches, resolve_project_path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/hierarchical_sensitivity_r100'
STUDIES = ('scale', 'outputs', 'depth', 'rounds', 'solver')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_plan():
    import yaml
    from fedcernn.experiments import expand_scenarios, validate_config, _scenario_seed_key, derive_seed
    plan = json.loads((OUT / 'frozen_plan.json').read_text())
    for path, sha in plan['source_sha256'].items():
        if not frozen_hash_matches(ROOT / path, sha):
            raise RuntimeError(f'Frozen source changed: {path}')
    total = 0
    seed_streams = set()
    for job in plan['jobs']:
        path = ROOT / job['config']
        if not frozen_hash_matches(path, job['sha256']):
            raise RuntimeError(f'Frozen config changed: {path}')
        config = yaml.safe_load(path.read_text())
        validate_config(config)
        scenarios = expand_scenarios(config)
        assert len(scenarios) == 1
        scenario = scenarios[0]
        settings = scenario.settings
        assert settings['simulation']['dgp'] == 'hierarchical_composition'
        assert settings['simulation']['error'] == 'contaminated_normal'
        assert settings['optimization']['l2_lambda'] == 0.001
        assert settings['methods'] == job['methods']
        assert config['experiment']['repetitions'] == 100
        assert config['experiment']['base_seed'] == plan['base_seed']
        assert resolve_project_path(config['experiment']['output_directory']) == OUT / 'runs' / job['name']
        seed_streams.add(tuple(derive_seed(plan['base_seed'], _scenario_seed_key(scenario), r)
                               for r in range(100)))
        total += 100 * len(job['methods'])
    assert len(plan['jobs']) == 17
    assert total == plan['expected_fits'] == 4100
    assert len(seed_streams) == 1
    assert len(set(next(iter(seed_streams)))) == 100
    return plan


def select_jobs(plan, study):
    # The common reference is fitted once and used by all studies.
    return [job for job in plan['jobs']
            if study == 'all' or job['study'] in ('reference', study)]


def smoke_config(job):
    import yaml
    config = copy.deepcopy(yaml.safe_load((ROOT / job['config']).read_text()))
    config['experiment'].update(repetitions=1, output_directory=f"code/results/hierarchical_sensitivity_r100/smoke/{job['name']}")
    config['simulation'].update(n_train=120, n_test=60, n_clients=4)
    # Exercise every architecture, output grid and refinement count cheaply.
    for optimizer in [config['optimization'], config['fedcernn']['host_optimizer'],
                      config['fedcernn']['surrogate_optimizer']]:
        optimizer.update(adam_steps=2, lbfgs_steps=2)
    directory = OUT / 'smoke_configs'
    directory.mkdir(exist_ok=True)
    path = directory / (job['name'] + '.yaml')
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=('all',) + STUDIES, default='all')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--dry-run', action='store_true', help='Validate without starting training.')
    parser.add_argument('--smoke', action='store_true', help='Small pipeline checks, kept separate from formal runs.')
    args = parser.parse_args()
    plan = validate_plan()
    jobs = select_jobs(plan, args.study)
    count = 100 * sum(len(job['methods']) for job in jobs)
    print(f'Validated {len(jobs)} settings, {count} fits (before skipping completed records).', flush=True)
    print('Current Example 3; contaminated-normal errors; 100 repetitions per setting.', flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"  {job['name']}: {', '.join(job['methods'])}")
        print('No training started.', flush=True)
        return
    lockname = '.smoke-launcher.lock' if args.smoke else '.launcher.lock'
    with (OUT / lockname).open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A sensitivity launcher is already running. Wait for it to finish.')
        logs = OUT / ('smoke_logs' if args.smoke else 'logs')
        logs.mkdir(exist_ok=True)
        stop = threading.Event()
        guard = threading.Lock()
        children = set()

        def run_job(job):
            config_path = smoke_config(job) if args.smoke else ROOT / job['config']
            with (logs / (job['name'] + '.log')).open('a') as log:
                with guard:
                    if stop.is_set():
                        return
                    process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'run_experiment.py'),
                                                '--config', str(config_path)], cwd=ROOT,
                                               stdout=log, stderr=subprocess.STDOUT)
                    children.add(process)
                print(f"Starting {job['name']}; log: {logs / (job['name'] + '.log')}", flush=True)
                try:
                    code = process.wait()
                    if code and not stop.is_set():
                        raise RuntimeError(f"{job['name']} failed (exit {code}); see its log.")
                    if not code:
                        print(f"Completed {job['name']}", flush=True)
                finally:
                    with guard:
                        children.discard(process)

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
        futures = [pool.submit(run_job, job) for job in jobs]
        try:
            for future in concurrent.futures.as_completed(futures):
                future.result()
        except BaseException:
            stop.set()
            with guard:
                for process in children:
                    if process.poll() is None:
                        process.terminate()
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
