import unittest
import numpy as np
import pandas as pd
from dfl_transfer_features import geometry,swap,corrupt


class SharedGeometryTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(35)
        self.players=rng.normal(0,12,(501,22,2))
        self.ball=rng.normal(0,3,(501,2))
        self.teams=np.repeat([0,1],11)
        self.players[20:40,:5]=np.nan
        self.ball[30:50]=np.nan
        self.blocks=[(0,501)]
    def view(self,p=None,b=None,t=None,blocks=None):
        return geometry(self.players if p is None else p,self.ball if b is None else b,
            self.teams if t is None else t,self.blocks if blocks is None else blocks)
    def test_team_swap_and_player_order(self):
        expected=self.view()
        pd.testing.assert_frame_equal(self.view(t=1-self.teams),swap(expected),check_exact=True)
        order=np.random.default_rng(41).permutation(22)
        pd.testing.assert_frame_equal(self.view(p=self.players[:,order],t=self.teams[order]),expected,check_exact=True)
    def test_origin_rotation_and_reflection(self):
        expected=self.view()
        for transform in (np.array([[0.,-1.],[1.,0.]]),np.diag([-1.,1.])):
            actual=self.view(p=self.players@transform+[105.,68.],b=self.ball@transform+[105.,68.])
            np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5,equal_nan=True)
    def test_bounded_observation_reach(self):
        expected=self.view().iloc[250]
        p=self.players.copy();b=self.ball.copy()
        p[:150]=999;p[351:]=999;b[:150]=999;b[351:]=999
        pd.testing.assert_series_equal(self.view(p,b).iloc[250],expected,check_exact=True)
    def test_prefix_exact(self):
        actual=self.view(self.players[:351],self.ball[:351])
        pd.testing.assert_frame_equal(actual.iloc[100:251],self.view().iloc[100:251],check_exact=True)
    def test_half_reset(self):
        expected=self.view(blocks=[(0,250),(250,501)])
        p=self.players.copy();b=self.ball.copy();p[:250]=999;b[:250]=999
        pd.testing.assert_frame_equal(self.view(p,b,blocks=[(0,250),(250,501)]).iloc[250:],expected.iloc[250:],check_exact=True)
    def test_missing_ball_retains_unknown_geometry(self):
        actual=self.view(b=np.full_like(self.ball,np.nan))
        self.assertTrue((actual.ball_seen==0).all())
        for column in ('h__distance1','a__distance3','h__within1','a__share'):
            self.assertTrue(actual[column].isna().all())
    def test_corruption_deterministic_observation_only(self):
        p,b=corrupt(self.players,self.ball,42)
        pp,bb=corrupt(self.players,self.ball,42)
        np.testing.assert_array_equal(p,pp);np.testing.assert_array_equal(b,bb)
        self.assertGreater(np.isnan(b).sum(),np.isnan(self.ball).sum())
        self.assertEqual(len(self.view(p,b).columns),91)


if __name__=='__main__':unittest.main()
