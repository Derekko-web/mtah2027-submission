"""Synthetic XML proves keyed alignment, explicit unknowns and target separation."""
import copy
import os
import time
from datetime import datetime,timedelta,timezone
import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from dfl_transfer_data import parse_match,convert_targets,quality_summary,run_counts
from dfl_transfer_data import timestamp

RULE=dict(minimum_half_frames=3,minimum_valid_status_fraction=.99,minimum_active_owner_fraction=.98,
    minimum_active_geometry_fraction=.95,minimum_active_ball_xy_fraction=.9,minimum_cadence_fraction=.999,
    minimum_live_frames_per_team=1,minimum_spells_per_team=1,minimum_dead_frames=1)


def fixture(folder):
    match='DFL-MAT-SYNTH'
    root=ET.Element('PutDataRequest');info=ET.SubElement(root,'MatchInformation')
    ET.SubElement(info,'General',MatchId=match,HomeTeamId='home',GuestTeamId='away')
    ET.SubElement(info,'Environment',PitchX='105',PitchY='68')
    teams=ET.SubElement(info,'Teams')
    for t in ('home','away'):
        players=ET.SubElement(ET.SubElement(teams,'Team',TeamId=t),'Players')
        for i in range(11):ET.SubElement(players,'Player',PersonId=t+str(i))
    ip=folder/'info.xml';ET.ElementTree(root).write(ip)
    root=ET.Element('PutDataRequest');positions=ET.SubElement(root,'Positions',MatchId=match)
    for half,start in [('firstHalf',10),('secondHalf',100)]:
        ball=ET.SubElement(positions,'FrameSet',TeamId='BALL',GameSection=half,PersonId='ball')
        for j in (2,0,1):
            n=start+j;time=(datetime(2022,1,1,tzinfo=timezone.utc)+timedelta(seconds=n*.04)).isoformat()
            ET.SubElement(ball,'Frame',N=str(n),T=time,X=str(j),Y='0',BallStatus='0' if j==2 else '1',BallPossession=str(j+1))
        for t in ('home','away'):
            for i in range(11):
                fs=ET.SubElement(positions,'FrameSet',TeamId=t,GameSection=half,PersonId=t+str(i))
                for j in (1,2,0):
                    n=start+j;time=(datetime(2022,1,1,tzinfo=timezone.utc)+timedelta(seconds=n*.04)).isoformat()
                    ET.SubElement(fs,'Frame',N=str(n),T=time,X=str(i-5),Y='5' if t=='home' else '-5')
    pp=folder/'positions.xml';ET.ElementTree(root).write(pp)
    return ip,pp,match


class DFLTransferDataTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.folder=Path(self.temp.name)
        self.ip,self.pp,self.match=fixture(self.folder)
    def tearDown(self):self.temp.cleanup()
    def mutate(self,fn):
        tree=ET.parse(self.pp);fn(tree.getroot());tree.write(self.pp)
    def test_targets_unknown_never_forward_filled(self):
        result=convert_targets([1,1,0,np.nan,1,2],[1,2,np.nan,1,0,2])
        np.testing.assert_array_equal(result,['H','A','D','X','X','X'])
    def test_naive_timestamp_independent_of_host_timezone(self):
        previous=os.environ.get('TZ');values=[]
        try:
            for zone in ('UTC','America/Toronto'):
                os.environ['TZ']=zone;time.tzset();values.append(timestamp('2022-01-01T12:00:00'))
            self.assertEqual(values[0],values[1])
        finally:
            if previous is None:os.environ.pop('TZ',None)
            else:os.environ['TZ']=previous
            time.tzset()
    def test_frame_order_is_keyed_not_positional(self):
        outputs,meta=parse_match(self.ip,self.pp,self.match)
        for half in outputs.values():
            np.testing.assert_array_equal(half['labels'],['H','A','D'])
            np.testing.assert_array_equal(half['ball'][:,0],[0,1,2])
            self.assertTrue(np.isfinite(half['positions']).all())
        self.assertEqual(meta['unrecognized_player_framesets'],0)
        self.assertTrue(quality_summary(outputs,RULE)['passed'])
    def test_missing_ball_frame_remains_gap_in_canonical_denominator(self):
        def remove(root):
            fs=root.find('.//FrameSet');fs.remove(next(f for f in fs if f.get('N')=='11'))
        self.mutate(remove);out,_=parse_match(self.ip,self.pp,self.match)
        self.assertEqual(out['firstHalf']['labels'][1],'X');self.assertTrue(np.isnan(out['firstHalf']['ball'][1]).all())
        q=quality_summary(out,RULE);self.assertEqual(q['canonical_frames'],6);self.assertFalse(q['checks']['status'])
    def test_duplicate_ball_frame_rejected(self):
        self.mutate(lambda root:root.find('.//FrameSet').append(copy.deepcopy(root.find('.//Frame'))))
        with self.assertRaisesRegex(ValueError,'Duplicate ball'):parse_match(self.ip,self.pp,self.match)
    def test_duplicate_player_frame_rejected(self):
        def duplicate(root):
            fs=root.findall('.//FrameSet')[1];fs.append(copy.deepcopy(fs[0]))
        self.mutate(duplicate)
        with self.assertRaisesRegex(ValueError,'Duplicate player'):parse_match(self.ip,self.pp,self.match)
    def test_body_match_mismatch_rejected(self):
        self.mutate(lambda root:root.find('.//Positions').set('MatchId','DFL-MAT-WRONG'))
        with self.assertRaisesRegex(ValueError,'body match identity'):parse_match(self.ip,self.pp,self.match)
    def test_keyed_player_timestamp_mismatch_rejected(self):
        self.mutate(lambda root:root.findall('.//FrameSet')[1][0].set('T','2022-01-01T01:00:00+00:00'))
        with self.assertRaisesRegex(ValueError,'timestamp mismatch'):parse_match(self.ip,self.pp,self.match)
    def test_targets_do_not_change_observation_arrays(self):
        before,_=parse_match(self.ip,self.pp,self.match)
        def change(root):
            for fs in root.findall('.//FrameSet'):
                for frame in fs:
                    frame.set('S','999');frame.set('Z','999');frame.set('A','999')
                    if fs.get('TeamId')=='BALL':frame.set('BallPossession','0');frame.set('BallStatus','1')
        self.mutate(change);after,_=parse_match(self.ip,self.pp,self.match)
        for section in before:
            for key in ('positions','ball','teams','frames','timestamps','pitch'):
                np.testing.assert_array_equal(before[section][key],after[section][key])
            self.assertTrue((after[section]['labels']=='X').all())
    def test_unknowns_split_spells(self):
        value=run_counts(np.array(list('HHXHHDAA')))
        self.assertEqual(value['H']['spells'],2);self.assertEqual(value['X']['frames'],1)
    def test_missing_team_cannot_pass_geometry_support(self):
        outputs,_=parse_match(self.ip,self.pp,self.match)
        for half in outputs.values():half['positions'][:,half['teams']==1]=np.nan
        self.assertFalse(quality_summary(outputs,RULE)['checks']['geometry'])
    def test_every_quality_gate_is_conjunctive(self):
        outputs,_=parse_match(self.ip,self.pp,self.match)
        self.assertTrue(quality_summary(outputs,RULE)['passed'])
        for key in ('minimum_half_frames','minimum_live_frames_per_team','minimum_spells_per_team','minimum_dead_frames'):
            self.assertFalse(quality_summary(outputs,dict(RULE,**{key:10000}))['passed'])


if __name__=='__main__':unittest.main()
