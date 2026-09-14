"""Small deterministic local experts and train-only risk routing primitives.

These are engineering hypotheses for H17, not a new Gaussian-process implementation.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree

def _check_xy(a, name):
    a=np.asarray(a,float)
    if a.ndim!=2 or a.shape[1]!=2 or not np.isfinite(a).all(): raise ValueError(name)
    return a

def local_predict(train_xy, train_y, query_xy, *, bandwidth_m=50., k=32, linear=False, adaptive=False,
                  train_physics=None, query_physics=None):
    x=_check_xy(train_xy,'train_xy'); q=_check_xy(query_xy,'query_xy'); y=np.asarray(train_y,float)
    if y.ndim!=1 or len(y)!=len(x) or not np.isfinite(y).all() or len(x)==0: raise ValueError('train_y')
    if not np.isfinite(bandwidth_m) or bandwidth_m<=0 or k<1: raise ValueError('parameters')
    k=min(int(k),len(x),128); d,ix=cKDTree(x).query(q,k=k,workers=1)
    if k==1: d=d[:,None]; ix=ix[:,None]
    scale=float(bandwidth_m)
    scales=np.maximum(d[:,-1],5.) if adaptive else np.full(len(q),scale)
    logw=-0.5*(d/np.maximum(scales[:,None],1e-12))**2
    if train_physics is not None or query_physics is not None:
        tp=np.asarray(train_physics,float); qp=np.asarray(query_physics,float)
        if tp.ndim!=1 or qp.ndim!=1 or len(tp)!=len(x) or len(qp)!=len(q): raise ValueError('physics')
        raw=tp[ix]; good=np.isfinite(raw)&np.isfinite(qp[:,None]); finite_one=np.isfinite(raw)|np.isfinite(qp[:,None])
        diff=np.zeros_like(raw); np.subtract(qp[:,None],raw,out=diff,where=good)
        logw+=np.where(good,-.5*(diff/10.)**2,np.where(finite_one,np.log(.25),0.))
    out=np.empty(len(q))
    lo,hi=float(y.min()),float(y.max())
    for j in range(len(q)):
        ww=np.exp(logw[j]-np.max(logw[j])); yy=y[ix[j]]; total=ww.sum()
        if not np.isfinite(total) or total<=0: out[j]=yy[0]; continue
        pred=float(np.dot(ww,yy)/total)
        if linear:
            z=(x[ix[j]]-q[j])/max(scales[j],5.); A=np.column_stack((np.ones(k),z)); reg=np.diag([0.,1.,1.])
            try: pred=float(np.linalg.solve(A.T@(ww[:,None]*A)+reg,A.T@(ww*yy))[0])
            except np.linalg.LinAlgError: pass
        out[j]=np.clip(pred,lo,hi)
    return out

def risk_features(train_xy, query_xy, query_predictions, query_has_path):
    x=_check_xy(train_xy,'train_xy'); q=_check_xy(query_xy,'query_xy'); p=np.asarray(query_predictions,float)
    if len(x)==0 or p.ndim!=2 or len(p)!=len(q) or not np.isfinite(p).all(): raise ValueError('predictions')
    tree=cKDTree(x); d,_=tree.query(q,k=1,workers=1); kk=min(16,len(x)); kd,_=tree.query(q,k=kk,workers=1)
    kth=kd if kk==1 else kd[:,-1]; path=np.asarray(query_has_path,bool)
    if len(path)!=len(q): raise ValueError('query_has_path')
    return np.column_stack((np.log1p(d),np.log1p(kth),p.std(axis=1),path.astype(float)))

def local_risk_weights(meta_train_features,oof_errors,query_features,*,metric='MAE',k=64,shrinkage=32):
    X=np.asarray(meta_train_features,float)
    Q=np.asarray(query_features,float); e=np.asarray(oof_errors,float)
    if metric.upper() not in ('MAE','RMSE') or k<1 or shrinkage<0 or e.ndim!=2 or X.ndim!=2 or Q.ndim!=2 or X.shape[1]!=Q.shape[1] or len(X)!=len(e) or len(X)==0 or e.shape[1]==0 or not np.isfinite(X).all() or not np.isfinite(Q).all() or not np.isfinite(e).all(): raise ValueError('risk inputs')
    loss=np.abs(e) if metric.upper()=='MAE' else e**2
    mu=X.mean(0); sd=np.where(X.std(0)>1e-12,X.std(0),1.); Z=(X-mu)/sd; zq=(Q-mu)/sd
    kk=min(int(k),len(X),128); d,ix=cKDTree(Z).query(zq,k=kk,workers=1)
    if kk==1: d=d[:,None]; ix=ix[:,None]
    global_r=loss.mean(axis=0); risks=[]; ess=[]
    for j in range(len(Q)):
        w=np.exp(-.5*d[j]**2); local=(w[:,None]*loss[ix[j]]).sum(axis=0)/max(w.sum(),1e-12); n=float(w.sum()**2/max(np.dot(w,w),1e-12))
        risks.append((local*n+global_r*shrinkage)/(n+shrinkage) if n+shrinkage>0 else global_r)
        ess.append(n)
    r=np.asarray(risks,float).reshape(len(Q),e.shape[1])
    if metric.upper()=='RMSE':r=np.sqrt(r)
    r=np.maximum(r,1e-6); weights=(1/r**2); weights/=weights.sum(axis=1,keepdims=True)
    return weights,{'metric':metric,'global_risk':global_r.tolist(),'effective_sample_size':float(np.mean(ess)) if ess else 0.,'feature_mean':mu.tolist(),'feature_scale':sd.tolist()}
