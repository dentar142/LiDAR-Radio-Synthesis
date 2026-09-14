#!/usr/bin/env python3
"""H15: target-separated preparation, GPU prediction and frozen final scoring."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .run_h11_sparse_learning_curves import BandData, atomic_csv, atomic_json, derive_seed, randomized_spatial_order, fit_physical_sparse
from .run_h13_statistical_revalidation import load_corrected_h13_band, balanced_spatial_fold_labels, sha256_file
from .h14_models import predict_candidate

BANDS = ('n41', 'n79')
MODES = ('infill', 'spatial30', 'spatial60')
BUDGETS = (30, 100, 300, 1000, 3000)
REPEATS = range(8)
NW = {'id':'nw50', 'family':'gaussian_nw', 'bandwidth_m':50., 'neighbors':128}


def position_groups(xy):
    return np.unique(np.floor(np.asarray(xy)).astype(np.int64), axis=0, return_inverse=True)[1]


def make_splits(xy, band):
    groups = position_groups(xy)
    result = {}
    for mode in MODES:
        cell = 1. if mode == 'infill' else 50.
        labels = balanced_spatial_fold_labels(xy, n_folds=5, cell_size_m=cell,
            seed=derive_seed(20260905, band, 'H15', str(cell)))
        test = labels == 0
        distance = cKDTree(xy[test]).query(xy, workers=1)[0]
        buffer = 0. if mode == 'infill' else float(mode.replace('spatial',''))
        pool = (~test) & (distance >= buffer)
        if np.intersect1d(groups[test], groups[pool]).size:
            raise RuntimeError('position group crosses train/test')
        result[mode] = (np.flatnonzero(pool), np.flatnonzero(test), buffer)
    if not np.array_equal(result['spatial30'][1], result['spatial60'][1]):
        raise RuntimeError('spatial test support mismatch')
    return result


def prepare(a):
    out=a.output_dir
    if (out/'manifest.json').exists():
        raise RuntimeError('prepare requires fresh output directory')
    out.mkdir(parents=True,exist_ok=True)
    manifest={'schema':'h15-v1', 'status':'PREPARING','budgets':BUDGETS,
        'bands':BANDS,'modes':MODES,'repeats':8,'runs':[], 'ineligible':[], 'sources':{}}
    for band in BANDS:
        data,source=load_corrected_h13_band(a.data_root,a.rt_root,band)
        source_dir=out/'inputs'/band; source_dir.mkdir(parents=True)
        xy=data.points[['x','y']].to_numpy(float)
        groups=position_groups(xy)
        frame=data.points[['point_id','x','y','z','date','trajectory_group']].copy()
        frame['position_group']=groups
        atomic_csv(source_dir/'features.csv',frame)
        np.savez_compressed(source_dir/'rt.npz',gains=data.gains,los=data.los,tx=data.tx)
        atomic_json(source_dir/'configs.json',data.configs)
        # Truth exists only in a separate scorer-owned artifact; never passed to prediction.
        truth_dir=out/'scorer_truth'; truth_dir.mkdir(exist_ok=True)
        atomic_csv(truth_dir/f'{band}.csv',data.points[['point_id','observed_dbm']])
        source['prepared_features_sha256']=sha256_file(source_dir/'features.csv')
        source['prepared_rt_sha256']=sha256_file(source_dir/'rt.npz')
        source['prepared_configs_sha256']=sha256_file(source_dir/'configs.json')
        source['scorer_truth_sha256']=sha256_file(truth_dir/f'{band}.csv')
        manifest['sources'][band]=source
        for mode,(pool,test,buffer) in make_splits(xy,band).items():
            np.save(source_dir/f'test_{mode}.npy',test)
            for repeat in REPEATS:
                seed=derive_seed(20260905,band,mode,repeat,'H15-order')
                maximum=min(max(BUDGETS),len(pool))
                order=pool[randomized_spatial_order(xy[pool],maximum,seed)]
                for budget in BUDGETS:
                    key=f'{band}_{mode}_b{budget}_r{repeat:02d}'
                    if len(pool)<budget:
                        manifest['ineligible'].append({'id':key,'band':band,'mode':mode,'budget':budget,'repeat':repeat,'reason':'insufficient_buffered_pool','pool_n':len(pool)})
                        continue
                    selected=order[:budget]
                    folder=out/'inputs'/'runs'/key; folder.mkdir(parents=True)
                    train=frame.iloc[selected][['point_id']].copy()
                    train['index']=selected
                    train['observed_dbm']=data.points.observed_dbm.to_numpy(float)[selected]
                    atomic_csv(folder/'training.csv',train)
                    nearest=cKDTree(xy[selected]).query(xy[test],workers=1)[0]
                    spec={'id':key,'band':band,'mode':mode,'budget':budget,'repeat':repeat,
                        'seed':derive_seed(20260905,band,mode,repeat,budget,'H15-fit'),
                        'train_n':budget,'train_position_groups':len(np.unique(groups[selected])),
                        'test_n':len(test),'train_pool_n':len(pool),'buffer_m':buffer,
                        'minimum_train_test_m':float(nearest.min()),'median_train_test_m':float(np.median(nearest)),
                        'training_sha256':sha256_file(folder/'training.csv'),
                        'test_sha256':sha256_file(source_dir/f'test_{mode}.npy')}
                    atomic_json(folder/'spec.json',spec)
                    manifest['runs'].append(spec)
        print(json.dumps({'prepared':band,'rows':len(xy),'eligible_runs':len(manifest['runs'])}),flush=True)
    manifest['status']='PREPARED'
    atomic_json(out/'manifest.json',manifest)


def load_fit_input(root, key):
    manifest=json.loads((root/'manifest.json').read_text())
    spec=json.loads((root/'inputs'/'runs'/key/'spec.json').read_text())
    frozen=next(r for r in manifest['runs'] if r['id']==key)
    if spec!=frozen: raise RuntimeError('run contract changed')
    band=spec['band']; source=manifest['sources'][band]; directory=root/'inputs'/band
    for filename,field in [('features.csv','prepared_features_sha256'),('rt.npz','prepared_rt_sha256'),('configs.json','prepared_configs_sha256')]:
        if sha256_file(directory/filename)!=source[field]: raise RuntimeError('input digest changed')
    training_path=root/'inputs'/'runs'/key/'training.csv'
    test_path=directory/f"test_{spec['mode']}.npy"
    if sha256_file(training_path)!=spec['training_sha256'] or sha256_file(test_path)!=spec['test_sha256']:
        raise RuntimeError('split hash changed')
    points=pd.read_csv(directory/'features.csv')
    training=pd.read_csv(training_path); train_index=training['index'].to_numpy(int)
    test_index=np.load(test_path,allow_pickle=False)
    if not np.array_equal(points.point_id.to_numpy()[train_index].astype(str),training.point_id.to_numpy().astype(str)):
        raise RuntimeError('training IDs mismatch')
    if np.intersect1d(train_index,test_index).size or np.intersect1d(points.position_group.to_numpy()[train_index],points.position_group.to_numpy()[test_index]).size:
        raise RuntimeError('train/test leakage')
    points['observed_dbm']=np.nan
    points.loc[train_index,'observed_dbm']=training.observed_dbm.to_numpy(float)
    rt=np.load(directory/'rt.npz',allow_pickle=False)
    data=BandData(band,points,json.loads((directory/'configs.json').read_text()),rt['gains'],rt['los'],rt['tx'],np.arange(len(points)))
    return data,spec,train_index,test_index


def build_features(data):
    xyz=data.points[['x','y','z']].to_numpy(float)
    raw=data.gains.T
    return np.column_stack((np.where(np.isfinite(raw),raw,0.),np.isfinite(raw).astype(float),data.los.astype(float),np.log10(np.maximum(1.,np.linalg.norm(xyz-data.tx,axis=1)))))


def run(a):
    from .h15_models import predict_models
    start=time.time(); data,spec,tr,te=load_fit_input(a.root,a.key)
    destination=a.output_dir or a.root/'runs'/a.key
    destination.mkdir(parents=True,exist_ok=True)
    xy=data.points[['x','y']].to_numpy(float); y=data.points.observed_dbm.to_numpy(float)
    fit=np.zeros(len(xy),bool);fit[tr]=True
    assert np.isnan(y[~fit]).all()
    union=np.r_[tr,te]
    fallback=predict_candidate(NW,xy[tr],y[tr],xy[union],None,None,spec['seed'])
    n=len(tr); physical={}; details={}; raw_outputs={}
    for name,family,twc in [('RT','BASE',False),('S4W_TWC','S4W',True),('S5_TWC','S5',True)]:
        value,info=fit_physical_sparse(data,y,fit,family,twc=twc,class_ridge=5.)
        covered=np.zeros(len(union),bool) if value is None else np.isfinite(value[union])
        mean=fallback.copy()
        if value is not None: mean[covered]=value[union][covered]
        physical[name]=(mean[:n],mean[n:])
        details[name]={'fit':info,'train_path_coverage':float(covered[:n].mean()),'test_path_coverage':float(covered[n:].mean()),'fallback':'train_only_gaussian_nw50'}
        raw_outputs[name+'_MEAN']=mean[n:]
        residual=predict_candidate(NW,xy[tr],y[tr]-mean[:n],xy[te],None,None,spec['seed'])
        raw_outputs[name+'_NW']=mean[n:]+residual
    chosen=min(('S4W_TWC','S5_TWC'),key=lambda k:float(np.mean((y[tr]-physical[k][0])**2)))
    means={'RT':physical['RT'],'MAT_TWC':physical[chosen]}
    aux=build_features(data)
    atomic_json(destination/'progress.json',{'status':'FITTING','spec':spec,'selected_material_proxy':chosen})
    predictions,info=predict_models(xy[tr],y[tr],xy[te],aux[tr],aux[te],means,spec['seed'],device=a.device,quick=a.quick)
    predictions.update(raw_outputs);predictions['GAUSSIAN_NW50']=fallback[n:]
    frame=data.points.iloc[te][['point_id','x','y','date','trajectory_group']].reset_index(drop=True)
    frame['nearest_train_m']=cKDTree(xy[tr]).query(xy[te],workers=1)[0]
    for name,values in predictions.items():
        values=np.asarray(values)
        if values.shape!=(len(te),) or not np.isfinite(values).all(): raise RuntimeError('invalid complete prediction '+name)
        frame[name]=values
    path=destination/'predictions.csv.gz';frame.to_csv(path,index=False,compression='gzip')
    atomic_json(destination/'parameters.json',{'physical':details,'selected_material_proxy_train_only':chosen,'models':info})
    atomic_json(destination/'status.json',{'status':'COMPLETE','spec':spec,'quick':a.quick,'methods':list(predictions),
        'predictions_sha256':sha256_file(path),'elapsed_seconds':time.time()-start})
    print(json.dumps({'status':'COMPLETE','id':a.key,'elapsed_seconds':time.time()-start,'methods':len(predictions)}),flush=True)


def plan(a):
    manifest=json.loads((a.root/'manifest.json').read_text())
    if a.smoke:
        specs=[r for r in manifest['runs'] if r['repeat']==0 and r['mode']=='infill' and r['budget'] in (30,3000)]
    else: specs=manifest['runs']
    specs=sorted(specs,key=lambda r:(-r['budget'],r['repeat'],r['band'],r['mode']))
    jobs=[]
    for spec in specs:
        command=[str(a.python),str(a.release/'src'/'run_h15_physical_neural.py'),'run','--root',str(a.root),'--key',spec['id'],'--device','cuda']
        if a.smoke: command+=['--output-dir',str(a.root/'smoke'/spec['id'])]
        jobs.append({'id':'run_'+spec['id'],'command':command,'kind':'gpu','min_free_mib':12000,'cwd':str(a.release),
            'depends':[],'timeout_seconds':7200,'eta_group':f"b{spec['budget']}"})
    if not a.smoke:
        jobs.append({'id':'score_h15','command':[str(a.python),str(a.release/'src'/'run_h15_physical_neural.py'),'score','--root',str(a.root)],
            'kind':'cpu','cwd':str(a.release),'depends':[j['id'] for j in jobs],'timeout_seconds':7200,'eta_group':'scoring'})
    result={'schema':'h15-queue-v1','cpu_workers':1,'gpu_candidates':[1,2,3],'jobs':jobs,
        'manifest_sha256':sha256_file(a.root/'manifest.json'),'scope':'exploratory_fixed_Tx_existing_data','smoke':a.smoke}
    atomic_json(a.output_dir,result)
    print(json.dumps({'jobs':len(jobs),'output':str(a.output_dir)}))


def score(a):
    manifest=json.loads((a.root/'manifest.json').read_text())
    out=a.root/'report';out.mkdir(exist_ok=True)
    rows=[];support={};names=None;prediction_hashes=[]
    truths={}
    for band in BANDS:
        p=a.root/'scorer_truth'/f'{band}.csv'
        if sha256_file(p)!=manifest['sources'][band]['scorer_truth_sha256']:raise RuntimeError('truth changed')
        truths[band]=pd.read_csv(p).set_index('point_id').observed_dbm
    for spec in manifest['runs']:
        directory=a.root/'runs'/spec['id'];status=json.loads((directory/'status.json').read_text())
        if status['status']!='COMPLETE' or status['quick'] or status['spec']!=spec: raise RuntimeError('incomplete/changed run')
        path=directory/'predictions.csv.gz';digest=sha256_file(path)
        if digest!=status['predictions_sha256']:raise RuntimeError('prediction changed before scoring')
        frame=pd.read_csv(path);methods=sorted(status['methods'])
        if names is None:names=methods
        if names!=methods:raise RuntimeError('method matrix changed')
        ids=frame.point_id.to_numpy();key=(spec['band'],spec['mode'])
        if key in support and not np.array_equal(support[key],ids): raise RuntimeError('test support changed')
        support[key]=ids
        if not frame.point_id.is_unique or len(frame)!=spec['test_n']:raise RuntimeError('test IDs invalid')
        observed=truths[spec['band']].loc[ids].to_numpy(float)
        parameters=json.loads((directory/'parameters.json').read_text())
        for method in methods:
            pred=frame[method].to_numpy(float)
            if not np.isfinite(pred).all():raise RuntimeError('nonfinite prediction')
            e=pred-observed
            rows.append({k:spec[k] for k in ['band','mode','budget','repeat']}|{'method':method,'n':len(e),
                'mae_db':float(np.mean(abs(e))),'rmse_db':float(np.sqrt(np.mean(e**2))),
                'medae_db':float(np.median(abs(e))),'p90_db':float(np.quantile(abs(e),.9)),'bias_db':float(np.mean(e)),
                'train_positions':spec['train_position_groups'],'nearest_train_median_m':spec['median_train_test_m'],
                'rt_path_coverage':parameters['physical']['RT']['test_path_coverage'],'elapsed_seconds':status['elapsed_seconds']})
        prediction_hashes.append({'run':spec['id'],'sha256':digest})
    metrics=pd.DataFrame(rows);atomic_csv(out/'repeat_metrics.csv',metrics)
    summary=metrics.groupby(['band','mode','budget','method']).agg(repeats=('repeat','nunique'),n=('n','first'),
        mae_mean=('mae_db','mean'),mae_sd=('mae_db','std'),rmse_mean=('rmse_db','mean'),rmse_sd=('rmse_db','std'),
        medae_mean=('medae_db','mean'),p90_mean=('p90_db','mean'),rt_path_coverage=('rt_path_coverage','mean')).reset_index()
    if not summary.repeats.eq(8).all():raise RuntimeError('incomplete repeat matrix')
    atomic_csv(out/'summary.csv',summary)
    reference=metrics[metrics.method=='GP_XY_M32'][['band','mode','budget','repeat','mae_db','rmse_db']]
    paired=metrics.merge(reference,on=['band','mode','budget','repeat'],suffixes=('','_gp'),validate='many_to_one')
    paired['delta_mae_db']=paired.mae_db-paired.mae_db_gp;paired['delta_rmse_db']=paired.rmse_db-paired.rmse_db_gp
    atomic_csv(out/'paired_vs_gp.csv',paired)
    best=summary.sort_values('rmse_mean').groupby(['band','mode','budget']).first().reset_index()
    atomic_csv(out/'descriptive_best_not_deployed.csv',best)
    atomic_json(out/'audit.json',{'status':'PASS','runs':len(manifest['runs']),'methods':names,'ineligible':manifest['ineligible'],
        'same_test_support':True,'complete_eight_repeats':True,'prediction_hashes':prediction_hashes,
        'scope':'exploratory_fixed_historical_Tx','new_RadioDiff_replication':False})
    print(json.dumps({'status':'PASS','runs':len(manifest['runs']),'methods':len(names),'rows':len(metrics)}))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','plan','score'])
    for flag in ['root','output-dir','data-root','rt-root','release','python']:p.add_argument('--'+flag,type=Path)
    p.add_argument('--key');p.add_argument('--device',default='cpu');p.add_argument('--quick',action='store_true');p.add_argument('--smoke',action='store_true')
    a=p.parse_args();globals()[a.command](a)


if __name__=='__main__':main()
