#!/usr/bin/env python3
"""Resume only the revised DCERNN: simulations (1200) and real data (30)."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from fedcernn.paths import frozen_hash_matches

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/dcernn_revision'
TAUS=[.1,.3,.5,.7,.9]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def read_plan():
    plan=json.loads((OUT/'frozen_plan.json').read_text())
    for name,sha in plan['source_sha256'].items():
        if not frozen_hash_matches(ROOT/name,sha):raise RuntimeError(f'Frozen source changed: {name}')
    assert sum(len(j['repetitions']) for j in plan['jobs'] if j['dataset']=='simulation')==1200
    assert sum(len(j['repetitions']) for j in plan['jobs'] if j['dataset']!='simulation')==30
    for job in plan['jobs']:
        keys=[(r['repetition'],r['seed']) for r in job['repetitions']]
        assert len(set(keys))==len(keys)
    return plan


def metrics(model,x,y,*,truth=None,scale=None):
    import numpy as np
    import torch
    from fedcernn.metrics import evaluate_predictions
    with torch.no_grad():pred=model(torch.as_tensor(x,dtype=torch.float64)).cpu().numpy()
    if scale is not None:pred=scale(pred)
    return evaluate_predictions(y,pred,TAUS,true_expectiles=truth).to_flat_dict(TAUS)


def clean_fit(result):
    return {k:v for k,v in result.items() if k not in ('model','checkpoints')}


def fit_simulation(job,repetition,smoke=False):
    import torch
    from fedcernn.simulation import generate_simulation
    from fedcernn.experiments import _make_model,_partition_clients
    from dcernn_final_revision import GDConfig,fit_gd
    settings=copy.deepcopy(job['settings'])
    sim=settings['simulation'];seed=repetition['seed']
    n=200 if smoke else sim['n_train']
    nt=80 if smoke else sim['n_test']
    train=generate_simulation(n,sim['dgp'],sim['error'],seed=seed,taus=TAUS)
    clients=_partition_clients(train.X,train.y,4 if smoke else sim['n_clients'],(seed+2)%2**32)
    model=_make_model(train.n_features,settings,TAUS,(seed+3)%2**32)
    configuration=copy.deepcopy(job['gd_config'])
    if smoke:configuration.update(max_steps=3,min_steps=10,check_every=1)
    result=fit_gd(model,clients,GDConfig(**configuration))
    settings['dcernn_revision']=configuration
    if result['stop_reason'] in ('nonfinite','diverged'):raise FloatingPointError('DCERNN diverged; no test scoring or silent fallback.')
    # Formal test generation occurs only after training and stopping are complete.
    test=generate_simulation(nt,sim['dgp'],sim['error'],seed=(seed+1)%2**32,
                             taus=TAUS,n_features=train.n_features)
    values=metrics(model,test.X,test.y,truth=test.true_expectiles)
    return dict(dataset='simulation',dgp=sim['dgp'],error=sim['error'],
                n_train=n,n_test=nt,n_clients=len(clients),gd_config=configuration,
                parameter_count=sum(p.numel() for p in model.parameters()),
                scenario_json=settings,metrics_json=values,**values,**{k:v for k,v in clean_fit(result).items() if k!='history'},
                history_json=result['history'])


def real_validation_search(settings,repetition,grid,*,smoke=False,preflight=False):
    """Each outer repeat uses its own validation set; never the outer test."""
    import numpy as np
    import torch
    from fedcernn.experiments import _make_model,_partition_clients
    from fedcernn.metrics import expectile_risk
    from run_dernn_real_data import prepare_data
    from dcernn_final_revision import GDConfig,fit_gd
    seed=repetition['seed'];dataset=repetition['dataset']
    splits=prepare_data(dataset,settings,seed)
    x,y=splits.X_train,splits.y_train
    vx,vy=splits.X_validation,splits.y_validation
    if preflight:
        # Small independent inner holdout from TRAIN only. Outer validation and
        # test are not used in development preflights, nor to change the grid.
        rng=np.random.default_rng(2026091703)
        order=rng.permutation(len(y))
        nval=min(3000,len(y)//5)
        validation_indices=order[:nval]
        train_indices=order[nval:nval+10000]
        vx,vy=x[validation_indices],y[validation_indices]
        x,y=x[train_indices],y[train_indices]
    if smoke:x,y=x[:200],y[:200]
    nclients=4 if smoke else settings['data']['n_clients']
    clients=_partition_clients(x,y,nclients,(seed+1)%2**32)
    vx=torch.as_tensor(vx,dtype=torch.float64)
    candidates=[]
    snapshots={}
    for rate in grid['learning_rates']:
        model=_make_model(x.shape[1],settings,TAUS,(seed+2)%2**32)
        cfg=dict(grid['stopping']);cfg.update(learning_rate=rate,l2_lambda=0.0001)
        if smoke:cfg.update(max_steps=3,min_steps=1,check_every=1)
        def observe(m,step):
            with torch.no_grad():risk,_=expectile_risk(vy,m(vx),TAUS)
            return dict(validation_risk=risk)
        result=fit_gd(model,clients,GDConfig(**cfg),observer=observe,return_checkpoints=True,
                      pooled_execution=preflight)
        # A diverged configuration is excluded even if its initial checkpoint
        # had a finite score. Fail visibly if no eligible stable fit survives.
        checkpoints=[]
        if result['stop_reason'] not in ('nonfinite','diverged'):
            budgets=[1,2,3] if smoke else grid['budgets']
            for h in result['history']:
                if h['step'] in budgets or (h['step']==result['steps'] and result['converged']):
                    checkpoints.append(h)
        candidates.append(dict(learning_rate=rate,fit=clean_fit(result),eligible=checkpoints))
        for h in checkpoints:snapshots[(rate,h['step'])]=result['checkpoints'][h['step']]
    eligible=[dict(learning_rate=c['learning_rate'],**h) for c in candidates for h in c['eligible']]
    if not eligible:raise RuntimeError('All DCERNN validation candidates failed.')
    best=min(h['validation_risk'] for h in eligible)
    near=[h for h in eligible if h['validation_risk']<=best*(1+grid['near_best_fraction'])]
    selected=min(near,key=lambda h:(h['step'],h['validation_risk'],h['learning_rate']))
    return splits,selected,candidates


def fit_real(job,repetition,smoke=False):
    import numpy as np
    import torch
    from fedcernn.experiments import _make_model
    from fedcernn.metrics import evaluate_predictions
    from dcernn_final_revision import GDConfig,fit_gd
    settings=copy.deepcopy(job['settings'])
    rep={**repetition,'dataset':job['dataset']}
    splits,selected,candidates=real_validation_search(settings,rep,job['validation_grid'],smoke=smoke)
    if not smoke and (len(splits.y_train)!=repetition['n_train'] or len(splits.y_test)!=repetition['n_test']):
        raise RuntimeError('Real-data sample counts no longer match the frozen outer split.')
    if smoke:
        from dataclasses import replace
        overrides=dict(X_test=splits.X_train[-50:],y_test=splits.y_train[-50:])
        if job['dataset']=='meps':
            overrides.update(y_test_thousands=splits.y_train_thousands[-50:],
                             test_weights=splits.train_weights[-50:],
                             test_person_ids=splits.train_person_ids[-50:])
        splits=replace(splits,**overrides)
    seed=repetition['seed']
    clients=splits.clients(settings['data']['n_clients'],seed=(seed+1)%2**32)
    if smoke:clients=[(x[:20],y[:20]) for x,y in clients]
    model=_make_model(splits.X_train.shape[1],settings,TAUS,(seed+2)%2**32)
    cfg=copy.deepcopy(job['validation_grid']['stopping'])
    cfg.update(learning_rate=selected['learning_rate'],max_steps=selected['step'],l2_lambda=.0001)
    if smoke:cfg.update(min_steps=1,check_every=1)
    # One explicitly accounted final fit. Validation-search costs stay visible
    # separately, matching other validation-tuned baselines' fit-only reporting.
    result=fit_gd(model,clients,GDConfig(**cfg))
    settings['dcernn_revision']={'chosen':cfg,'validation_grid':job['validation_grid']}
    if result['stop_reason'] in ('nonfinite','diverged'):raise FloatingPointError('Final fit diverged.')
    target=(splits.y_test_thousands if job['dataset']=='meps' else splits.inverse_target(splits.y_test))
    valtarget=(splits.y_validation_thousands if job['dataset']=='meps' else splits.inverse_target(splits.y_validation))
    values=metrics(model,splits.X_test,target,scale=splits.inverse_target)
    validation=metrics(model,splits.X_validation,valtarget,scale=splits.inverse_target)
    values.update({f'validation_{k}':v for k,v in validation.items()})
    if not smoke:
        np.testing.assert_allclose(validation['composite_risk'],
                                   selected['validation_risk']*splits.target_scale**2,
                                   rtol=1e-9,atol=1e-10,
                                   err_msg='Final fit failed to reproduce the selected validation checkpoint')
    if job['dataset']=='meps':
        with torch.no_grad():pred=splits.inverse_target(model(torch.from_numpy(splits.X_test)).cpu().numpy())
        weighted=evaluate_predictions(target,pred,TAUS,sample_weight=splits.test_weights).to_flat_dict(TAUS)
        values.update({f'survey_weighted_{k}':v for k,v in weighted.items()})
    tuning_bytes=sum(c['fit']['communication_bytes'] for c in candidates)
    tuning_seconds=sum(c['fit']['elapsed_seconds'] for c in candidates)
    return dict(dataset=job['dataset'],n_train=sum(len(y) for _,y in clients),n_test=len(splits.y_test),
                n_clients=len(clients),gd_config=cfg,scenario_json=settings,metrics_json=values,**values,
                parameter_count=sum(p.numel() for p in model.parameters()),
                **{k:v for k,v in clean_fit(result).items() if k!='history'},history_json=result['history'],
                validation_selection=selected,validation_candidates=candidates,
                tuning_communication_bytes=tuning_bytes,tuning_seconds=tuning_seconds,
                total_including_tuning_bytes=tuning_bytes+result['communication_bytes'],
                source_original_run_key=repetition['run_key'])


def run_job(job,smoke=False,limit=None):
    import torch
    torch.set_num_threads(1)
    directory=OUT/('smoke' if smoke else 'runs')/job['name']
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'results.jsonl'
    fingerprint=digest(OUT/'frozen_plan.json')
    todo=job['repetitions'][:1] if smoke else job['repetitions']
    if limit is not None:todo=todo[:limit]
    expected={(r['repetition'],r['seed']) for r in job['repetitions']}
    with (directory/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        done=set()
        for r in rows:
            key=(r['repetition'],r['seed'])
            assert key in expected and key not in done and r['status']=='success'
            assert r['plan_sha256']==fingerprint and r['smoke']==smoke
            done.add(key)
        for rep in todo:
            key=(rep['repetition'],rep['seed'])
            if key in done:continue
            print(f"{job['name']}: starting {rep['repetition']+1}/{len(job['repetitions'])}",flush=True)
            started=time.perf_counter()
            result=fit_simulation(job,rep,smoke) if job['dataset']=='simulation' else fit_real(job,rep,smoke)
            result.update(method='DCERNN',status='success',smoke=smoke,plan_sha256=fingerprint,
                          repetition=rep['repetition'],seed=rep['seed'],base_seed=rep['base_seed'],
                          hidden_dims='32-16',n_expectiles=5,training_taus=TAUS,
                          wall_seconds_including_tuning=time.perf_counter()-started,
                          timestamp_utc=datetime.now(timezone.utc).isoformat())
            result['rounds']=result['communication_rounds']
            rows.append(result)
            tmp=path.with_suffix('.tmp')
            with tmp.open('w') as f:
                for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
                f.flush();os.fsync(f.fileno())
            os.replace(tmp,path)
            done.add(key)
            print(f"{job['name']}: done {rep['repetition']+1}, risk={result['composite_risk']:.6g}, steps={result['steps']}, {result['stop_reason']}",flush=True)
        if rows:
            import numpy as np
            summary=dict(n=len(rows),mean_risk=float(np.mean([r['composite_risk'] for r in rows])),
                         sd_risk=float(np.std([r['composite_risk'] for r in rows],ddof=1)) if len(rows)>1 else None,
                         mean_bytes=float(np.mean([r['communication_bytes'] for r in rows])),
                         mean_fit_seconds=float(np.mean([r['elapsed_seconds'] for r in rows])),
                         converged_count=sum(r['converged'] for r in rows),
                         max_steps_count=sum(r['stop_reason']=='max_steps' for r in rows))
            (directory/'summary.json').write_text(json.dumps(summary,indent=2))
    return job['name']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--chapter',choices=['all','4','5'],default='all')
    p.add_argument('--dataset',choices=['all','meps','eicu'],default='all')
    p.add_argument('--workers',type=int,choices=[1,2],default=1)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    plan=read_plan()
    jobs=[j for j in plan['jobs'] if (args.chapter=='all' or (j['dataset']=='simulation')==(args.chapter=='4'))
          and (args.dataset=='all' or j['dataset']==args.dataset)]
    if not jobs:raise SystemExit('No jobs match these filters.')
    if any(j['dataset']!='simulation' for j in jobs):
        real=json.loads((ROOT/'results/dernn_real_data/frozen_plan.json').read_text())
        for j in jobs:
            if j['dataset']!='simulation':
                d=real['datasets'][j['dataset']]
                if digest(ROOT.parent/d['data_path'])!=d['data_sha256']:raise RuntimeError('Real data changed.')
    print(f"Validated {len(jobs)} settings / {sum(len(j['repetitions']) for j in jobs)} formal fits; completed fits are skipped.",flush=True)
    if args.dry_run:
        for j in jobs:print(j['name'],j.get('gd_config',j.get('validation_grid')))
        return
    with (OUT/('.smoke-launcher.lock' if args.smoke else '.launcher.lock')).open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.workers==1:
            for job in jobs:run_job(job,args.smoke)
        else:
            pool=ProcessPoolExecutor(max_workers=args.workers)
            try:
                for f in as_completed([pool.submit(run_job,j,args.smoke) for j in jobs]):
                    print('Completed',f.result(),flush=True)
            except BaseException:
                # Python 3.12 has no public terminate_workers(). Do not keep
                # running other long jobs after a worker fails or Ctrl-C.
                for process in list(pool._processes.values()):process.terminate()
                pool.shutdown(wait=True,cancel_futures=True)
                raise
            else:pool.shutdown(wait=True)
    print('Completed:',OUT/('smoke' if args.smoke else 'runs'),flush=True)


if __name__=='__main__':main()
