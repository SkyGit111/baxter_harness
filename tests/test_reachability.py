import copy
import math
import unittest
from cix_harness.reachability import scan,yaw_orientation


class ReachabilityTests(unittest.TestCase):
    def setUp(self):
        self.cfg={'arms':{a:{'primitives':{'tool_orientation_xyzw':[1,0,0,0]}} for a in ('left','right')}}
        self.states={a:{'gripping':False,'joint_rad':[0]*7} for a in ('left','right')}
        self.grid={'x':[.4,.5,.6],'y':[-.1,.1],'z':[-.1,.1]}
        self.dest={a:[.5,0,-.1] for a in self.states}
    def run_scan(self,evaluator,**kw):
        return scan(self.cfg,self.states,'<robot/>',self.grid,self.dest,evaluator,**kw)
    @staticmethod
    def accept(arm,point,destination,q,skill):
        return {'checked_skills':['pick',skill],'complete_path_checked':True}

    def test_holes_and_z_layers_are_preserved_without_certifying_a_box(self):
        def evaluate(arm,point,destination,q,skill):
            if point[0]==.5 or point[2]>.0:raise ValueError('blocked route')
            return self.accept(arm,point,destination,q,skill)
        report=self.run_scan(evaluate)
        self.assertFalse(report['certified']);self.assertFalse(report['applied'])
        for result in report['arms'].values():
            self.assertEqual(result['sample_count'],12)
            self.assertEqual(result['candidate_count'],4)
            self.assertEqual(result['sampled_candidate_z_values_m'],[-.1])
            self.assertEqual([layer['candidate_count'] for layer in result['z_slices']],[4,0])
            self.assertFalse(result['continuous_volume_certified'])
            self.assertTrue(all(s['status']=='no_candidate_found' for s in result['samples'] if s['point_base_m'][0]==.5))

    def test_successful_pick_without_placement_never_passes(self):
        report=self.run_scan(lambda *args:{'checked_skills':['pick'],'complete_path_checked':True})
        self.assertEqual(report['arms']['right']['candidate_count'],0)
        self.assertIn('entire requested operation',report['arms']['right']['samples'][0]['attempts'][0]['reason'])

    def test_multi_yaw_passes_when_original_orientation_fails(self):
        def evaluate(arm,point,destination,q,skill):
            if abs(q[1])<.7:raise ValueError('head clearance')
            return self.accept(arm,point,destination,q,skill)
        report=self.run_scan(evaluate,yaw_degrees=[0,90],arms=['right'])
        first=report['arms']['right']['samples'][0]
        self.assertEqual(first['status'],'candidate')
        self.assertEqual([a['status'] for a in first['attempts']],['rejected','candidate'])
        self.assertEqual(first['attempts'][0]['reason'],'head clearance')

    def test_yaw_composition_preserves_downward_tool_axis(self):
        from cix_harness.pi05_contract import rotate
        for angle in (0,45,90,180,270):
            q=yaw_orientation([1,0,0,0],angle)
            xyz=rotate(q,[0,0,1])
            for a,b in zip(xyz,[0,0,-1]):self.assertAlmostEqual(a,b)

    def test_changes_in_geometry_or_start_pose_invalidate_fingerprint(self):
        before=self.run_scan(self.accept)['fingerprints']
        self.states['left']['joint_rad'][0]=.1
        after=self.run_scan(self.accept)['fingerprints']
        self.assertNotEqual(before['start_states_sha256'],after['start_states_sha256'])
        self.assertEqual(before['config_sha256'],after['config_sha256'])

    def test_nonfinite_or_incomplete_evidence_fails_closed(self):
        def invalid(*args):return {'checked_skills':['pick','stack'],'complete_path_checked':True,'clearance':math.nan}
        report=self.run_scan(invalid,arms=['left'],yaw_degrees=[0])
        self.assertEqual(report['arms']['left']['candidate_count'],0)
        for grid in ({'x':[.1,.1],'y':[0],'z':[0]},{'x':[math.nan],'y':[0],'z':[0]}):
            self.grid=grid
            with self.assertRaises(ValueError):self.run_scan(self.accept)


if __name__=='__main__':unittest.main()
