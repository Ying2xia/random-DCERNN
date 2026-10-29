#!/usr/bin/env python3
"""Run the frozen 100-repetition hierarchical example; completed fits resume."""
from pathlib import Path
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import subprocess
import sys
import threading
from fedcernn.paths import frozen_hash_matches

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/hierarchical_composition_r100'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,choices=[1,2],default=2)
    parser.add_argument('--dry-run',action='store_true',help='Validate the frozen plan without fitting.')
    args=parser.parse_args()
    plan=json.loads((OUT/'frozen_plan.json').read_text())
    for name,digest in plan['source_sha256'].items():
        if not frozen_hash_matches(ROOT/name, digest):
            raise RuntimeError(f'Frozen experiment code changed: {name}. Check before resuming.')
    from fedcernn.experiments import run_experiment
    total=0
    for job in plan['configs']:
        path=ROOT/job['config']
        if not frozen_hash_matches(path, job['sha256']):
            raise RuntimeError(f'Frozen configuration changed: {path}')
        scenarios=run_experiment(path,dry_run=True)
        assert len(scenarios)==1
        settings=scenarios[0]['settings']
        assert settings['simulation']['n_train']==100000
        assert settings['simulation']['n_clients']==20
        assert settings['methods']==job['methods']
        total+=100*len(job['methods'])
    assert total==plan['expected_fits']==4800
    print('Validated: 4 errors × 12 methods × 100 repetitions = 4,800 fits.',flush=True)
    if args.dry_run:
        print('Validation only; no training started.',flush=True)
        return
    with (OUT/'.launcher.lock').open('a+') as lockfile:
        try:fcntl.flock(lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise SystemExit('This experiment is already running.')
        logs=OUT/'logs';logs.mkdir(exist_ok=True)
        stop=threading.Event();guard=threading.Lock();children=set()
        def run_job(job):
            name=job['name']
            with (logs/(name+'.log')).open('a') as log:
                with guard:
                    if stop.is_set():return
                    process=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_experiment.py'),'--config',str(ROOT/job['config'])],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    children.add(process)
                print(f'Starting {name}; log: {logs/(name+".log")}',flush=True)
                try:
                    code=process.wait()
                    if code!=0 and not stop.is_set():
                        raise RuntimeError(f'{name} failed (exit {code}); see its log.')
                    if code==0:print(f'Completed {name}',flush=True)
                finally:
                    with guard:children.discard(process)
        pool=concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
        futures=[pool.submit(run_job,job) for job in plan['configs']]
        try:
            for future in concurrent.futures.as_completed(futures):future.result()
        except BaseException:
            stop.set()
            with guard:
                for process in children:
                    if process.poll() is None:process.terminate()
            for future in futures:future.cancel()
            raise
        finally:pool.shutdown(wait=True,cancel_futures=True)
    print(f'All groups completed. Results: {OUT}',flush=True)

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('\nStopped. Rerun the same command to resume completed records.',file=sys.stderr)
        sys.exit(130)
