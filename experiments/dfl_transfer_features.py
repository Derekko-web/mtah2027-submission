"""Frozen XY-only shared representation for the DFL ownership trial."""
import numpy as np
import pandas as pd

MAX_REACH=100


def statistics(values,radius,future):
    values=np.asarray(values,float)
    padded=np.pad(values,(0,radius) if future else (radius,0),constant_values=np.nan)
    windows=np.lib.stride_tricks.sliding_window_view(padded,radius+1)
    finite=np.isfinite(windows);count=finite.sum(axis=1)
    mean=np.full(len(values),np.nan)
    np.divide(np.where(finite,windows,0).sum(axis=1),count,out=mean,where=count>0)
    variance=np.full(len(values),np.nan)
    np.divide(np.where(finite,(windows-mean[:,None])**2,0).sum(axis=1),count,out=variance,where=count>0)
    available=np.minimum(np.arange(len(values))+1,radius+1)
    if future:available=available[::-1]
    return mean,np.sqrt(variance),count/available


def geometry(positions,ball,teams,blocks):
    """Metres in any common origin; anonymous per-team sets; no provider scalars."""
    positions,ball=np.asarray(positions,float),np.asarray(ball,float)
    teams=np.asarray(teams)
    n=len(ball)
    if positions.shape!=(n,len(teams),2) or ball.shape!=(n,2) or set(teams)!={0,1}:
        raise ValueError('XY/team shape differs')
    names=['ball_seen','ball_displacement5','ball_displacement25']
    fields=['n_tracks','distance1','distance2','distance3','within1','within3','within5','share']
    names += [p+f for p in ('h__','a__') for f in fields]
    base={name:np.full(n,np.nan) for name in names}
    occupied=np.zeros(n,bool)
    for left,right in blocks:
        left,right=max(0,left),min(n,right)
        if left>=right:continue
        if occupied[left:right].any():raise ValueError('Overlapping half blocks')
        occupied[left:right]=True
        seen=np.isfinite(ball[left:right]).all(axis=1)
        base['ball_seen'][left:right]=seen
        for step in (5,25):
            if left+step>=right:continue
            speed=np.linalg.norm(ball[left+step:right]-ball[left:right-step],axis=1)/(step/25)
            speed[~np.isfinite(speed)|(speed>80)]=np.nan
            base[f'ball_displacement{step}'][left+step:right]=speed
        for team,prefix in enumerate(('h__','a__')):
            for start in range(left,right,4096):
                end=min(start+4096,right)
                xy=positions[start:end,teams==team]
                finite=np.isfinite(xy).all(axis=2)
                base[prefix+'n_tracks'][start:end]=finite.sum(axis=1)
                distance=np.linalg.norm(xy-ball[start:end,None],axis=2)
                safe=np.where(np.isfinite(distance)&finite,distance,np.inf)
                ordered=np.sort(safe,axis=1)
                for rank in range(3):
                    result=ordered[:,rank].copy() if ordered.shape[1]>rank else np.full(end-start,np.inf)
                    result[~np.isfinite(result)]=np.nan
                    base[prefix+f'distance{rank+1}'][start:end]=result
                support=np.isfinite(ball[start:end]).all(axis=1)&finite.any(axis=1)
                for radius in (1,3,5):
                    result=(safe<=radius).sum(axis=1).astype(float)
                    result[~support]=np.nan
                    base[prefix+f'within{radius}'][start:end]=result
        dh,da=base['h__distance1'][left:right],base['a__distance1'][left:right]
        base['h__share'][left:right]=(da+.1)/(dh+da+.2)
        base['a__share'][left:right]=(dh+.1)/(dh+da+.2)
    extra={}
    selected=['ball_seen','ball_displacement25','h__distance1','a__distance1','h__share','a__share']
    for name in selected:
        for radius in (25,75):
            for future,direction in ((False,'past'),(True,'future')):
                keys=[name+f'_{direction}{radius}_{suffix}' for suffix in ('mean','std','fraction')]
                for key in keys:extra[key]=np.full(n,np.nan)
                for left,right in blocks:
                    left,right=max(0,left),min(n,right)
                    if left>=right:continue
                    values=statistics(base[name][left:right],radius,future)
                    for key,value in zip(keys,values):extra[key][left:right]=value
    return pd.DataFrame({**base,**extra}).astype(np.float32)


def swap(features):
    opposite=lambda name:('a__'+name[3:] if name.startswith('h__') else 'h__'+name[3:] if name.startswith('a__') else name)
    return features[[opposite(name) for name in features]].set_axis(features.columns,axis=1)


def corrupt(positions,ball,seed):
    """One fixed observation-only schedule, independent of all targets/clocks."""
    rng=np.random.default_rng(seed)
    positions=np.array(positions,dtype=np.float32,copy=True)
    ball=np.array(ball,dtype=np.float32,copy=True)
    n=len(ball)
    positions+=rng.normal(0,.2,positions.shape).astype(np.float32)
    ball+=rng.normal(0,.2,ball.shape).astype(np.float32)
    drops=rng.random(((n+49)//50,positions.shape[1]))<.2
    positions[np.repeat(drops,50,axis=0)[:n]]=np.nan
    visible=bool(rng.random()<.6)
    for i in range(n):
        if rng.random()<(1/75 if visible else 1/50):visible=not visible
        if not visible:ball[i]=np.nan
    return positions,ball
