"""One-model binary ownership training and conditional evaluation."""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from dfl_transfer_features import swap

PARAMS=dict(objective='binary:logistic',eval_metric='logloss',tree_method='hist',
    max_depth=2,n_estimators=120,learning_rate=.04,reg_lambda=10,min_child_weight=15,
    subsample=.8,colsample_bytree=.9,random_state=42,n_jobs=2)


def fit(inputs,target,groups,seed=42):
    target=np.asarray(target,int);groups=np.asarray(groups)
    if len(inputs)!=len(target) or len(groups)!=len(target) or set(target)!={0,1}:
        raise ValueError('Both binary classes and aligned groups required')
    weight=np.zeros(len(target))
    unique=np.unique(groups)
    for group in unique:
        for label in (0,1):
            eligible=(groups==group)&(target==label)
            if not eligible.any():raise ValueError('Every training group needs both classes')
            weight[eligible]=len(target)/(len(unique)*2*eligible.sum())
    augmented=pd.concat([inputs,swap(inputs)],ignore_index=True)
    model=XGBClassifier(**dict(PARAMS,random_state=seed))
    model.fit(augmented,np.r_[target,1-target],sample_weight=np.r_[weight,weight])
    return model,weight


def probability(model,inputs):
    if not np.array_equal(model.classes_,[0,1]):raise ValueError('Unexpected binary class order')
    p=(model.predict_proba(inputs)[:,1]+1-model.predict_proba(swap(inputs))[:,1])/2
    if not np.isfinite(p).all() or (p<0).any() or (p>1).any():raise ValueError('Invalid conditional probability')
    return p


def nearest(inputs):
    return inputs['h__share'].fillna(.5).to_numpy()


def binary_metrics(labels,home_probability):
    labels=np.asarray(labels);p=np.asarray(home_probability,float)
    known=np.isin(labels,['H','A'])
    y=(labels[known]=='H').astype(int);p=p[known]
    if not len(y) or set(y)!={0,1} or not np.isfinite(p).all():raise ValueError('Invalid conditional evaluation support')
    pred=(p>=.5).astype(int)
    recall=[float((pred[y==k]==k).mean()) for k in (0,1)]
    clipped=np.clip(p,1e-6,1-1e-6)
    bins=np.minimum((p*10).astype(int),9)
    ece=sum(float((bins==i).mean())*abs(float(p[bins==i].mean())-float(y[bins==i].mean())) for i in range(10) if (bins==i).any())
    return dict(live_frames=len(y),balanced_accuracy=sum(recall)/2,home_recall=recall[1],away_recall=recall[0],
        accuracy=float((y==pred).mean()),brier=float(((p-y)**2).mean()),
        log_loss=float(-(y*np.log(clipped)+(1-y)*np.log(1-clipped)).mean()),
        calibration_error_10_bins=ece,predicted_home_fraction=float(pred.mean()))
