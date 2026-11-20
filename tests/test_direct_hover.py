import unittest
from unittest.mock import patch

from cix_harness.planning import stage_phase
from cix_harness.primitive_service import IKPreflightError
from cix_harness.primitives import plan, tcp_from_eef
from tests.test_table_geometry import service, state, profile


class DirectHoverTests(unittest.TestCase):
    def setUp(self):
        self.p=profile()
        self.p.update(approach_path='direct_hover', contact_band_m=.04, max_leg_m=.48)
        self.target=[.70,.20,-.12]

    def test_high_start_blends_turn_and_translation_then_splits_vertical_contact(self):
        st=state()
        st['eef_m']=[.54,.12,.24]
        st['eef_xyzw']=[0,1,0,0]
        stages=plan('pick',st,self.p,self.target)
        first=stages[0]
        self.assertEqual(first['tcp_m'][:2],self.target[:2])
        self.assertLess(first['tcp_m'][2],st['eef_m'][2])
        self.assertEqual(first['xyzw'],self.p['tool_orientation_xyzw'])
        moves=[s for s in stages if s['kind']=='move']
        phases=[stage_phase(stages,i,'pick') for i,s in enumerate(stages) if s['kind']=='move']
        self.assertEqual(phases,['transit','transit','approach','contact','lift_clear'])
        self.assertAlmostEqual(moves[1]['tcp_m'][2]-moves[2]['tcp_m'][2],.04)
        self.assertAlmostEqual(moves[3]['tcp_m'][2]-moves[2]['tcp_m'][2],.04)
        self.assertTrue(all(s['tcp_m'][:2]==self.target[:2] for s in moves))
        self.assertEqual(stages[-1],{'kind':'verify_grasp','held':True})

    def test_low_start_lifts_in_place_and_stack_preserves_holding_orientation(self):
        st=state(True)
        st['eef_m']=[.58,.10,-.08]
        st['eef_xyzw']=[0,1,0,0]
        stages=plan('stack',st,self.p,self.target)
        self.assertEqual(stages[0]['motion_phase'],'clearance')
        self.assertEqual(stages[0]['tcp_m'][:2],st['eef_m'][:2])
        self.assertGreater(stages[0]['tcp_m'][2],st['eef_m'][2])
        self.assertTrue(all(s['xyzw']==st['eef_xyzw'] for s in stages if s['kind']=='move'))
        self.assertEqual([s.get('motion_phase') for s in stages if s['kind']=='move'][-3:],
                         ['placement','contact','retreat'])

    def test_turning_with_offset_follows_tcp_line_at_transfer_height(self):
        self.p['tcp_offset_tool_m']=[.02,0,0]
        srv,robot,modules=service(self.p)
        start=tcp_from_eef(state()['eef_m'],state()['eef_xyzw'],self.p)
        end=[.66,-.05,start[2]]
        stage={'kind':'move','tcp_m':end,'xyzw':[0,1,0,0],'motion_phase':'transit'}
        with patch.dict('sys.modules',modules):
            _,paths=srv._preflight([stage],{'state':state()},srv.observation[1])
        for i,(_,target_pose) in enumerate(paths[0],start=1):
            pose=target_pose.pose
            p,q=pose.position,pose.orientation
            actual=tcp_from_eef([p.x,p.y,p.z],[q.x,q.y,q.z,q.w],self.p)
            for a,b,x in zip(start,end,actual):
                self.assertAlmostEqual(x,a+(b-a)*i/len(paths[0]))
        robot.move_segment.assert_not_called()

    def test_online_preflight_retries_staged_and_exposes_checked_schedule(self):
        srv,robot,modules=service(self.p)
        robot.cfg['planning']={'trajectory_mode':'continuous'}
        # Constant joint feedback hits every curve-metadata path in this isolated
        # test. The first candidate specifically simulates an infeasible diagonal.
        robot.continuous_ik.forward.return_value=robot.limb.endpoint_pose()
        original=srv._preflight
        def fail_direct(stages,*args,**kwargs):
            if any('motion_phase' in s for s in stages):
                raise IKPreflightError('diagonal has no continuous IK')
            return original(stages,*args,**kwargs)
        with patch.dict('sys.modules',modules),patch.object(srv,'_preflight',side_effect=fail_direct):
            receipt=srv.prepare('o','pick',target_base_m=self.target)
        self.assertEqual(receipt['status'],'ready')
        self.assertEqual([r['approach_path'] for r in receipt['planning_candidates']],['direct_hover','staged'])
        self.assertEqual(receipt['selected_approach_path'],'staged')
        self.assertAlmostEqual(receipt['nominal_motion_seconds'],
                               sum(s['nominal_duration_s'] for s in receipt['motion_schedule']))
        self.assertGreater(receipt['nominal_motion_seconds'],0)
        robot.move_segment.assert_not_called()
        robot.grip.command_position.assert_not_called()

    def test_budget_rejection_can_still_use_a_fully_checked_shorter_candidate(self):
        st=state()
        st['eef_m'][2]=-.08
        srv,robot,modules=service(self.p)
        robot.limb.endpoint_pose.return_value['position'].z=st['eef_m'][2]
        srv.observation[0]['state']=st
        staged=plan('pick',st,dict(self.p,approach_path='staged'),self.target)
        robot.motion_remaining=sum(s['kind']!='verify_grasp' for s in staged)
        with patch.dict('sys.modules',modules):
            result=srv.prepare('o','pick',target_base_m=self.target)
        self.assertEqual(result['selected_approach_path'],'staged')
        self.assertEqual(result['planning_candidates'][0]['reason'],'stage_budget')
        self.assertGreater(robot.continuous_ik.solve.call_count,0)
        robot.move_segment.assert_not_called()
