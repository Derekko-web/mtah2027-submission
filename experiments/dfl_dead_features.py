"""Anonymous, team-invariant XY motion for direct external live/dead supervision."""
import warnings
import numpy as np
import pandas as pd
from dfl_transfer_features import geometry,statistics


def anonymous_motion(positions,step,blocks):
    positions=np.asarray(positions,float);n,m,_=positions.shape
    speed=np.full((n,m),np.nan)
    for left,right in blocks:
        left,right=max(0,left),min(n,right)
        for start in range(left+step,right,512):
            stop=min(right,start+512);now=positions[start:stop];past=positions[start-step:stop-step]
            distance=np.linalg.norm(now[:,:,None]-past[:,None,:],axis=3)
            distance[~np.isfinite(distance)]=np.inf
            previous=distance.argmin(axis=2);current=distance.argmin(axis=1)
            nearest=np.take_along_axis(distance,previous[:,:,None],axis=2)[:,:,0]
            back=np.take_along_axis(current,previous,axis=1)
            good=(back==np.arange(m)[None,:])&np.isfinite(nearest)&(nearest/(step/25)<=12)
            speed[start:stop]=np.where(good,nearest/(step/25),np.nan)
    finite=np.isfinite(speed);count=finite.sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        median=np.nanmedian(speed,axis=1);p75=np.nanquantile(speed,.75,axis=1)
    moving=np.divide((speed>1).sum(axis=1),count,out=np.full(n,np.nan),where=count>0)
    observed=np.isfinite(positions).all(axis=2).sum(axis=1)
    matched=np.divide(count,observed,out=np.full(n,np.nan),where=observed>0)
    return dict(speedmedian=median,speedp75=p75,movingfraction=moving,matchedfraction=matched)


def build(positions,ball,teams,blocks):
    positions=np.asarray(positions,float);ball=np.asarray(ball,float);teams=np.asarray(teams)
    base=geometry(positions,ball,teams,blocks);values={};paired={}
    for name in base:
        if name.startswith('h__'):
            pair=base[[name,'a__'+name[3:]]]
            values[name[3:]+'_min']=pair.min(axis=1).to_numpy();values[name[3:]+'_max']=pair.max(axis=1).to_numpy()
        elif not name.startswith('a__'):values[name]=base[name].to_numpy()
    for team in (0,1):
        xy=positions[:,teams==team];fields={}
        for step in (5,25):
            fields.update({key+f'_lag{step}':value for key,value in anonymous_motion(xy,step,blocks).items()})
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            center=np.nanmean(xy,axis=1);radius=np.linalg.norm(xy-center[:,None],axis=2)
            fields['shape_radius_median']=np.nanmedian(radius,axis=1);fields['shape_radius_p90']=np.nanquantile(radius,.9,axis=1)
        paired[team]=fields
    for name in paired[0]:
        pair=pd.DataFrame({0:paired[0][name],1:paired[1][name]})
        values[name+'_min']=pair.min(axis=1).to_numpy();values[name+'_max']=pair.max(axis=1).to_numpy()
    for name in ('speedmedian_lag5','speedmedian_lag25','movingfraction_lag25'):
        for teamstat in ('min','max'):
            column=name+'_'+teamstat
            for radius in (25,75):
                for future,direction in ((False,'past'),(True,'future')):
                    arrays=[np.full(len(ball),np.nan) for _ in range(3)]
                    for left,right in blocks:
                        left,right=max(0,left),min(len(ball),right)
                        if left>=right:continue
                        for target,part in zip(arrays,statistics(values[column][left:right],radius,future)):target[left:right]=part
                    for suffix,value in zip(('mean','std','fraction'),arrays):values[column+f'_{direction}{radius}_{suffix}']=value
    return pd.DataFrame(values).astype(np.float32)
