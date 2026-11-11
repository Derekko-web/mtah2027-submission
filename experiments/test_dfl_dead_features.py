"""Synthetic invariance, locality and missingness checks for XY-only D features."""
import unittest
import numpy as np
from dfl_dead_features import build,anonymous_motion


class DeadFeatureTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(4);self.teams=np.repeat([0,1],6)
        origin=rng.normal(size=(12,2))*20
        drift=rng.normal(size=(12,2))*.05
        self.xy=origin[None]+np.arange(480)[:,None,None]*drift[None]
        self.ball=np.c_[np.arange(480)*.05,np.sin(np.arange(480)*.02)]
        self.blocks=[(0,480)]

    def test_team_swap_player_order_and_coordinate_origin(self):
        a=build(self.xy,self.ball,self.teams,self.blocks)
        b=build(self.xy[:,::-1]+10,self.ball+10,1-self.teams[::-1],self.blocks)
        np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-5,equal_nan=True)

    def test_rotation_and_prefix_footprint(self):
        a=build(self.xy,self.ball,self.teams,self.blocks)
        rotation=np.array([[0.,-1.],[1.,0.]])
        b=build(self.xy@rotation,self.ball@rotation,self.teams,self.blocks)
        np.testing.assert_allclose(a,b,atol=1e-5,rtol=1e-5,equal_nan=True)
        b=build(self.xy[:400],self.ball[:400],self.teams,[(0,400)])
        np.testing.assert_array_equal(a.iloc[100:300],b.iloc[100:300])

    def test_motion_never_crosses_half_or_invents_missing_tracks(self):
        xy=np.zeros((100,4,2));xy[50:]=10
        r=anonymous_motion(xy,25,[(0,50),(50,100)])
        self.assertTrue(np.isnan(r['speedmedian'][50:75]).all())
        xy[:]=np.nan;r=anonymous_motion(xy,5,[(0,100)])
        self.assertTrue(np.isnan(r['speedmedian']).all())


if __name__=='__main__':unittest.main()
