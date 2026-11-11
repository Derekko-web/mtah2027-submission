"""Label-free local formation matching between independent camera timelines."""

from difflib import SequenceMatcher
import unicodedata

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from games import GAME,GAMES,require,source_frames
from starter.baseline import team_names,team_of_each_track

PITCH = np.array(GAMES[GAME]["pitch"])


def normalized_name(name):
    return "".join(c for c in unicodedata.normalize("NFKD",name.lower()) if c.isalnum())


def reconcile_teams(tactical_names,broadcast_names,broadcast_teams):
    a,b = [list(map(normalized_name,names)) for names in (tactical_names,broadcast_names)]
    score = np.array([[SequenceMatcher(None,x,y).ratio() for y in b] for x in a])
    direct,reverse = score[0,0]+score[1,1],score[0,1]+score[1,0]
    chosen = [score[0,0],score[1,1]] if direct >= reverse else [score[0,1],score[1,0]]
    if min(chosen) < .6 or abs(direct-reverse) < .5:
        raise ValueError("Team metadata cannot be reconciled confidently")
    return np.asarray(broadcast_teams) if direct >= reverse else 1-np.asarray(broadcast_teams)


def load_source(source,n_frames=None):
    n_frames = source_frames(source) if n_frames is None else n_frames
    meta,tracking,homography = require(["atd_meta","atd_raw","homography"],source=source)
    teams = team_of_each_track(meta)
    xy = np.full((n_frames,len(teams),2),np.nan,dtype=np.float32)
    speeds = np.full((n_frames,len(teams)),np.nan,dtype=np.float32)
    ball = np.full((n_frames,2),np.nan,dtype=np.float32)
    count = 0
    with open(tracking) as handle:
        for frame,line in enumerate(handle):
            if frame >= n_frames: break
            frame_id,players,ball_raw = line.strip().split(":")
            if int(frame_id) != frame+1: raise ValueError("Source frame alignment differs")
            values = np.fromstring(players.replace(";",","),sep=",").reshape(-1,3)
            if len(values) != len(teams): raise ValueError("Source track layout differs")
            xy[frame],speeds[frame] = values[:,:2]*PITCH,values[:,2]
            ball[frame] = np.fromstring(ball_raw,sep=",")[:2]*PITCH
            count += 1
    if count != n_frames: raise ValueError("Source tracking ended early")
    valid_xy = np.isfinite(xy).all(axis=2) & (xy >= -5).all(axis=2) & (xy <= PITCH+5).all(axis=2)
    xy[~valid_xy] = np.nan
    quality = pd.read_csv(homography,usecols=["frame","valid"]).iloc[:n_frames]
    if len(quality) != n_frames or not np.array_equal(quality.frame,np.arange(n_frames)):
        raise ValueError("Source homography frame alignment differs")
    return dict(positions=xy,speeds=speeds,ball=ball,teams=teams,
                valid=quality.valid.to_numpy(dtype=bool),names=team_names(meta))


def transform(points,orientation):
    result = np.asarray(points).copy()
    if orientation & 1: result[...,0] = PITCH[0]-result[...,0]
    if orientation & 2: result[...,1] = PITCH[1]-result[...,1]
    return result


def frame_usable(source):
    valid = np.asarray(source["valid"],dtype=bool).copy()
    finite = np.isfinite(source["positions"]).all(axis=2)
    for team in (0,1): valid &= finite[:,source["teams"] == team].sum(axis=1) >= 4
    return valid


def formation_cost(tactical_frame,broadcast_frames,tactical_teams,broadcast_teams,orientation):
    """Directed same-team set error, robust to different visible player counts."""
    broadcast_frames = transform(broadcast_frames,orientation)
    errors = []
    for team in (0,1):
        current = tactical_frame[tactical_teams == team]
        current = current[np.isfinite(current).all(axis=1)]
        other = broadcast_frames[:,broadcast_teams == team]
        observed = np.isfinite(other).all(axis=2)
        if len(current) < 4:
            return np.full(len(other),np.inf)
        distance = np.zeros(other.shape[:2])
        if observed.any(): distance[observed] = cKDTree(current).query(other[observed])[0]
        count = observed.sum(axis=1)
        error = np.sum(np.minimum(distance,10)**2,axis=1)/np.maximum(count,1)
        error[count < 4] = np.inf
        errors.append(error)
    return np.sqrt((errors[0]+errors[1])/2)


def formation_anchors(tactical,broadcast,profiles,progress=None):
    """Search all broadcast moments; use only local tactical +/-50-frame context."""
    txy,bxy = tactical["positions"],broadcast["positions"]
    usable_t,usable_b = frame_usable(tactical),frame_usable(broadcast)
    tf = np.arange(25,len(txy)-25,25)
    bf = np.arange(25,len(bxy)-25,25)
    bf = bf[usable_b[bf]]
    if not len(bf): raise ValueError("No eligible broadcast formation anchors")
    table = np.full((len(tf),4),np.nan)
    for index,frame in enumerate(tf):
        if not usable_t[frame]: continue
        costs = np.stack([formation_cost(txy[frame],bxy[bf],tactical["teams"],broadcast["teams"],orientation)
                          for orientation in range(4)])
        flattened = int(np.argmin(costs))
        orientation,candidate = np.unravel_index(flattened,costs.shape)
        best = costs[orientation,candidate]
        far = np.abs(bf-bf[candidate]) > 250
        runner = np.min(costs[:,far]) if far.any() else np.inf
        table[index] = [bf[candidate],orientation,best,runner-best]
        if progress is not None and index % 50 == 0: progress(index,len(tf))
    maximum_cost = max(config["maximum_cost"] for config in profiles.values())
    minimum_margin = min(config["minimum_margin"] for config in profiles.values())
    fine = np.full((len(tf),3),np.nan)
    for index in range(1,len(tf)-1):
        neighborhood = table[index-1:index+2]
        if not np.isfinite(neighborhood[:,:3]).all(): continue
        if (neighborhood[:,2] > maximum_cost).any() or (neighborhood[:,3] < minimum_margin).any(): continue
        if not np.all(neighborhood[:,1] == table[index,1]): continue
        if (np.abs(np.diff(neighborhood[:,0])-25) > 25).any(): continue
        frame,orientation = tf[index],int(table[index,1])
        candidates = np.arange(int(table[index,0])-25,int(table[index,0])+26)
        candidates = candidates[(candidates >= 25) & (candidates < len(bxy)-25)]
        costs = []
        for offset in (-25,0,25):
            value = formation_cost(txy[frame+offset],bxy[candidates+offset],tactical["teams"],broadcast["teams"],orientation)
            value[~usable_b[candidates+offset] | ~usable_t[frame+offset]] = np.inf
            costs.append(value**2)
        score = np.sqrt(np.mean(costs,axis=0))
        best = int(score.argmin())
        if not np.isfinite(score[best]): continue
        possible = candidates[score <= score[best]+.05]
        fine[index] = [candidates[best],score[best],int(possible[-1]-possible[0])]
    return dict(tactical_frames=tf,coarse=table,fine=fine)


def map_anchors(anchors,tactical,broadcast,profile):
    tf,coarse,fine = [anchors[name] for name in ("tactical_frames","coarse","fine")]
    allowed = np.isfinite(fine).all(axis=1) & (fine[:,1] <= profile["maximum_cost"]) & (fine[:,2] <= profile["maximum_fine_span"])
    for index in range(1,len(tf)-1):
        allowed[index] &= bool((coarse[index-1:index+2,2] <= profile["maximum_cost"]).all() and
                               (coarse[index-1:index+2,3] >= profile["minimum_margin"]).all())
    n = len(tactical["positions"])
    mapped,orientation = np.full(n,-1,int),np.full(n,-1,int)
    score,margin = np.full(n,np.nan),np.full(n,np.nan)
    for index in range(len(tf)-1):
        if not allowed[index:index+2].all() or coarse[index,1] != coarse[index+1,1]: continue
        start,end = tf[index],tf[index+1]
        rate = (fine[index+1,0]-fine[index,0])/(end-start)
        if not .8 <= rate <= 1.2: continue
        frames = np.arange(start,end+1)
        fraction = (frames-start)/(end-start)
        mapped[frames] = np.rint(fine[index,0]+fraction*(fine[index+1,0]-fine[index,0])).astype(int)
        orientation[frames] = int(coarse[index,1])
        score[frames] = (1-fraction)*fine[index,1]+fraction*fine[index+1,1]
        margin[frames] = np.minimum(coarse[index,3],coarse[index+1,3])
    valid = mapped >= 0
    valid[valid] &= frame_usable(broadcast)[mapped[valid]]
    mapped[~valid],orientation[~valid] = -1,-1
    score[~valid],margin[~valid] = np.nan,np.nan
    return dict(frames=mapped,orientation=orientation,cost=score,margin=margin,anchor_allowed=allowed)
