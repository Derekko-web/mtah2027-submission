"""Keyed DFL XML ingestion and prospective label support; no competition targets."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET
import numpy as np

SECTIONS=('firstHalf','secondHalf')


def digest(path,algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def fetch_file(record,folder):
    """Exclusive download, preserve incomplete bytes; verify publisher length/MD5."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/record['name']
    if path.exists():
        if path.stat().st_size!=record['size'] or digest(path,'md5')!=record['computed_md5']:
            raise ValueError('Existing download identity differs')
        return path
    partial=path.with_suffix(path.suffix+'.partial')
    request=urllib.request.Request(record['download_url'],headers={'User-Agent':'MTAH-local-research/1.0'})
    with urllib.request.urlopen(request,timeout=120) as response,partial.open('xb') as handle:
        for chunk in iter(lambda:response.read(1024*1024),b''):handle.write(chunk)
    if partial.stat().st_size!=record['size'] or digest(partial,'md5')!=record['computed_md5']:
        raise ValueError('Downloaded file fails publisher identity')
    partial.rename(path)
    return path


def match_metadata(path,expected):
    root=ET.parse(path).getroot()
    general=root.find('.//General');environment=root.find('.//Environment')
    if general is None or environment is None:raise ValueError('Missing DFL match metadata sections')
    ids={value for node in root.iter() for key,value in node.attrib.items() if key.lower()=='matchid'}
    if ids and ids!={expected}:raise ValueError('Match metadata body identity differs')
    home=general.get('HomeTeamId');away=general.get('AwayTeamId',general.get('GuestTeamId'))
    if not home or not away or home==away:raise ValueError('Invalid metadata team mapping')
    pitch=np.array([float(environment.get('PitchX')),float(environment.get('PitchY'))])
    if not (90<=pitch[0]<=120 and 50<=pitch[1]<=90):raise ValueError('Invalid pitch dimensions')
    players={}
    for node in root.findall('.//Teams/Team'):
        if node.get('TeamId') not in (home,away):continue
        team=0 if node.get('TeamId')==home else 1
        for player in node.findall('./Players/Player'):
            person=player.get('PersonId')
            if person is None or person in players:raise ValueError('Missing or duplicated person metadata')
            players[person]=team
    if sum(t==0 for t in players.values())<11 or sum(t==1 for t in players.values())<11:
        raise ValueError('Incomplete metadata teamsheets')
    order=sorted(players)
    return dict(match_id=expected,home_team_id=home,away_team_id=away,pitch=pitch,
        people=order,teams=np.array([players[p] for p in order],dtype=np.int8),body_match_id_present=bool(ids))


def number(value):
    try:return float(value)
    except (ValueError,TypeError):return np.nan


def timestamp(value):
    if not value:raise ValueError('Missing frame timestamp')
    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    # A fixed clock origin makes naive source times portable. This does not
    # assert the match's civil timezone; only keyed differences enter validation.
    if parsed.tzinfo is None:parsed=parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def convert_targets(status,owner):
    """Unknown never becomes D or gets forward-filled; provider codes are targets."""
    status,owner=np.asarray(status),np.asarray(owner)
    result=np.full(len(status),'X',dtype='<U1')
    result[status==0]='D'
    result[(status==1)&(owner==1)]='H'
    result[(status==1)&(owner==2)]='A'
    return result


def parse_match(info_path,position_path,expected):
    meta=match_metadata(info_path,expected)
    balls={section:{} for section in SECTIONS};body_ids=set();unknown_sections=set();keys=set()
    for event,node in ET.iterparse(position_path,events=('start','end')):
        if event=='start':
            for key,value in node.attrib.items():
                if key.lower()=='matchid':body_ids.add(value)
            continue
        if node.tag.rsplit('}',1)[-1]!='FrameSet':continue
        if node.get('TeamId','').lower()=='ball':
            section=node.get('GameSection')
            if section not in SECTIONS:unknown_sections.add(section)
            else:
                for frame in node:
                    if frame.tag.rsplit('}',1)[-1]!='Frame':continue
                    keys.update(frame.attrib)
                    index=int(frame.get('N'))
                    if index in balls[section]:raise ValueError('Duplicate ball frame key')
                    balls[section][index]=(number(frame.get('X')),number(frame.get('Y')),
                        number(frame.get('BallStatus')),number(frame.get('BallPossession')),timestamp(frame.get('T')))
        node.clear()
    if body_ids and body_ids!={expected}:raise ValueError('Position body match identity differs')
    if unknown_sections:raise ValueError('Unexpected ball game section')
    if not {'BallStatus','BallPossession','N','T','X','Y'}<=keys:raise ValueError('Missing direct target or XY fields')
    outputs={}
    for section in SECTIONS:
        if not balls[section]:raise ValueError('Missing ball half')
        first,last=min(balls[section]),max(balls[section]);n=last-first+1
        if n>250000:raise ValueError('Unbounded or corrupt half frame range')
        data=np.full((n,5),np.nan)
        for frame,value in balls[section].items():data[frame-first]=value
        ball=data[:,:2].astype(np.float32)
        good=np.isfinite(ball).all(axis=1)&(np.abs(ball)<=meta['pitch']/2+5).all(axis=1)
        ball[~good]=np.nan
        outputs[section]=dict(frames=np.arange(first,last+1),timestamps=data[:,4],
            ball=ball,status=data[:,2],owner=data[:,3],labels=convert_targets(data[:,2],data[:,3]),
            positions=np.full((n,len(meta['teams']),2),np.nan,dtype=np.float32),
            teams=meta['teams'],pitch=meta['pitch'])
    del balls
    lookup={person:i for i,person in enumerate(meta['people'])}
    seen={s:np.zeros(v['positions'].shape[:2],bool) for s,v in outputs.items()}
    outside=unrecognized=0
    for _,node in ET.iterparse(position_path,events=('end',)):
        if node.tag.rsplit('}',1)[-1]!='FrameSet':continue
        if node.get('TeamId','').lower()=='ball':node.clear();continue
        section=node.get('GameSection');person=node.get('PersonId')
        if section not in outputs or person not in lookup:
            unrecognized+=1;node.clear();continue
        current=outputs[section];column=lookup[person];start=current['frames'][0]
        expected_team=meta['home_team_id'] if meta['teams'][column]==0 else meta['away_team_id']
        if node.get('TeamId')!=expected_team:raise ValueError('Player/team metadata identity differs')
        for frame in node:
            if frame.tag.rsplit('}',1)[-1]!='Frame':continue
            i=int(frame.get('N'))-start
            if not 0<=i<len(current['frames']):outside+=1;continue
            if seen[section][i,column]:raise ValueError('Duplicate player/half/frame key')
            seen[section][i,column]=True
            if np.isfinite(current['timestamps'][i]) and abs(timestamp(frame.get('T'))-current['timestamps'][i])>0.001:
                raise ValueError('Player/ball keyed timestamp mismatch')
            xy=np.array([number(frame.get('X')),number(frame.get('Y'))])
            if np.isfinite(xy).all() and (np.abs(xy)<=meta['pitch']/2+5).all():current['positions'][i,column]=xy
        node.clear()
    return outputs,dict(match_id=expected,metadata_body_id_present=meta['body_match_id_present'],
        positions_body_id_present=bool(body_ids),ball_attribute_names=sorted(keys),
        player_frames_outside_ball_domain=outside,unrecognized_player_framesets=unrecognized)


def run_counts(labels):
    labels=np.asarray(labels)
    starts=np.r_[0,np.flatnonzero(labels[1:]!=labels[:-1])+1]
    stops=np.r_[starts[1:],len(labels)]
    result={}
    for label in ('H','A','D','X'):
        lengths=stops[labels[starts]==label]-starts[labels[starts]==label]
        result[label]=dict(spells=len(lengths),frames=int(lengths.sum()),
            length_quantiles=np.quantile(lengths,[0,.25,.5,.75,1]).tolist() if len(lengths) else [])
    return result


def quality_summary(outputs,rule):
    halves={};total_frames=valid_status=live=valid_owner=live_geometry=live_ball=0
    all_counts={label:dict(frames=0,spells=0) for label in ('H','A','D','X')}
    cadence_ok=cadence_total=0
    for section,current in outputs.items():
        status,owner=current['status'],current['owner'];active=status==1
        finite=np.isfinite(current['positions']).all(axis=2)
        counts=run_counts(current['labels'])
        for label in counts:
            for key in ('frames','spells'):all_counts[label][key]+=counts[label][key]
        geometry=np.ones(len(status),bool)
        for team in (0,1):geometry&=finite[:,current['teams']==team].sum(axis=1)>=4
        timestamps=current['timestamps'];present=np.flatnonzero(np.isfinite(timestamps))
        delta=np.diff(timestamps[present]);expected=np.diff(present)*.04
        good=np.abs(delta-expected)<=.0001
        cadence_ok+=int(good.sum());cadence_total+=len(good)
        total_frames+=len(status);valid_status+=int(np.isin(status,[0,1]).sum());live+=int(active.sum())
        valid_owner+=int((active&np.isin(owner,[1,2])).sum())
        live_geometry+=int((active&geometry).sum());live_ball+=int((active&np.isfinite(current['ball']).all(axis=1)).sum())
        halves[section]=dict(canonical_frames=len(status),present_ball_records=len(present),
            missing_ball_records=len(status)-len(present),runs=counts,
            status_values={str(k):int(v) for k,v in zip(*np.unique(status[np.isfinite(status)],return_counts=True))},
            owner_values={str(k):int(v) for k,v in zip(*np.unique(owner[np.isfinite(owner)],return_counts=True))},
            cadence_pairs=len(good),cadence_bad=int((~good).sum()))
    ratios=dict(status_valid_fraction=valid_status/total_frames if total_frames else 0,
        active_owner_valid_fraction=valid_owner/live if live else 0,
        active_geometry_fraction=live_geometry/live if live else 0,
        active_ball_xy_fraction=live_ball/live if live else 0,
        cadence_fraction=cadence_ok/cadence_total if cadence_total else 0)
    checks=dict(two_halves=set(outputs)==set(SECTIONS),
        half_length=all(v['canonical_frames']>=rule['minimum_half_frames'] for v in halves.values()),
        status=ratios['status_valid_fraction']>=rule['minimum_valid_status_fraction'],
        owner=ratios['active_owner_valid_fraction']>=rule['minimum_active_owner_fraction'],
        geometry=ratios['active_geometry_fraction']>=rule['minimum_active_geometry_fraction'],
        ball_xy=ratios['active_ball_xy_fraction']>=rule['minimum_active_ball_xy_fraction'],
        cadence=ratios['cadence_fraction']>=rule['minimum_cadence_fraction'],
        live_classes=all(all_counts[label]['frames']>=rule['minimum_live_frames_per_team'] and
            all_counts[label]['spells']>=rule['minimum_spells_per_team'] for label in ('H','A')),
        dead_present=all_counts['D']['frames']>=rule['minimum_dead_frames'])
    return dict(halves=halves,canonical_frames=total_frames,live_frames=live,
        class_counts=all_counts,ratios=ratios,checks=checks,passed=all(checks.values()))
