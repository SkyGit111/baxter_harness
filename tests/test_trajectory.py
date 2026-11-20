import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch

from cix_harness.baxter import Baxter
from cix_harness.trajectory import JointTrajectory,check_controller_feedback,ControllerSafetyError,motion_limits


class TrajectoryTest(unittest.TestCase):
    def test_isolated_short_ik_bend_does_not_slow_entire_leg(self):
        points = [{'a': i * .001, 'b': .0007 if i == 120 else 0.} for i in range(1, 301)]
        trajectory = JointTrajectory({'a': 0., 'b': 0.}, points, .12, .06)
        self.assertLess(trajectory.duration, 8.)  # Old global time dilation scheduled 142.87 s.
        self.assertAlmostEqual(trajectory.sample(trajectory.knots[120])['b'], .0007)
        self.assertEqual(trajectory.sample(trajectory.duration), points[-1])
        route = [dict(a=0., b=0.)] + points
        self.assertEqual(list(trajectory.sampled_positions(.005)), route)
        dt = .0005
        for i in range(1, 300):
            t = trajectory.knots[i]  # Includes both sides of the short reversal.
            previous,current,following = [trajectory.sample(t+offset) for offset in (-dt, 0., dt)]
            for name in current:
                self.assertLessEqual(abs(following[name]-previous[name])/(2*dt), .12001)
                self.assertLessEqual(abs(following[name]-2*current[name]+previous[name])/dt**2, .06001)

    def test_dense_straight_path_passes_internal_points_without_stopping(self):
        path=JointTrajectory({'j':0.},[{'j':i/1000} for i in range(1,301)],.03,.06)
        self.assertLess(path.duration,15.)  # Old per-knot 0.15 s floor costs 45 s.
        for knot in path.knots[10:-10:10]:
            velocity=(path.sample(knot+.0001)['j']-path.sample(knot-.0001)['j'])/.0002
            self.assertGreater(velocity,.005)
        self.assertLessEqual(path.peak_velocity,.0300001)
        self.assertLessEqual(path.peak_acceleration,.0600001)

    def test_shape_preserving_path_and_derivatives_at_reversals(self):
        points=[{'a':.02,'b':-.01},{'a':.04,'b':.01},{'a':.01,'b':.01},{'a':.01,'b':.04}]
        trajectory=JointTrajectory({'a':0.,'b':0.},points,.12,.15)
        for i,(begin,end) in enumerate(zip(trajectory.knots,trajectory.knots[1:])):
            for j in range(101):
                q=trajectory.sample(begin+(end-begin)*j/100)
                for name in q:
                    a,b=trajectory.points[i][name],trajectory.points[i+1][name]
                    self.assertGreaterEqual(q[name],min(a,b)-1e-12)
                    self.assertLessEqual(q[name],max(a,b)+1e-12)
        dt=1e-4
        for j in range(1,1000):
            t=trajectory.duration*j/1000
            a,b,c=[trajectory.sample(t+x) for x in (-dt,0,dt)]
            for name in b:
                self.assertLessEqual(abs(c[name]-a[name])/(2*dt),.12001)
                self.assertLessEqual(abs(c[name]-2*b[name]+a[name])/(dt*dt),.1501)
        samples=list(trajectory.sampled_positions(.005))
        for a,b in zip(samples,samples[1:]):
            self.assertLessEqual(max(abs(a[n]-b[n]) for n in a),.005001)

    def test_stationary_and_invalid_trajectories(self):
        still=JointTrajectory({'j':.1},[{'j':.1}]*10,.03)
        self.assertEqual(still.sample(.1),{'j':.1})
        self.assertEqual(still.duration,.15)
        for point in ({'j':float('nan')},{'other':0}):
            with self.assertRaises(ValueError):JointTrajectory({'j':0},[point],.03)

    def test_contact_limits_cannot_exceed_global_limit(self):
        cfg={'joint_speed_rad_s':.06,'joint_speed_by_phase_rad_s':{'contact':.02}}
        self.assertEqual(motion_limits(cfg,'contact'),(.02,.12))
        self.assertEqual(motion_limits(cfg,'transit'),(.06,.12))
        cfg['joint_speed_by_phase_rad_s']['contact']=.08
        with self.assertRaises(ValueError):motion_limits(cfg,'contact')

    def test_clear_lift_speed_respects_existing_loaded_limit(self):
        phases={'contact':.02,'transit':.08}
        cfg={'joint_speed_rad_s':.12,'joint_speed_by_phase_rad_s':phases}
        self.assertEqual(motion_limits(cfg,'lift')[0],.02)
        self.assertEqual(motion_limits(cfg,'lift_clear')[0],.08)
        phases['lift']=.04
        self.assertEqual(motion_limits(cfg,'lift_clear')[0],.04)
        phases['lift_clear']=.06
        self.assertEqual(motion_limits(cfg,'lift_clear')[0],.06)

    def test_factory_head_avoidance_is_reported_before_tracking_failure(self):
        state={'avoidance':(1.,{'other_arm':False,'objects':['collision_head_link_1']}),'collision':(1.,False)}
        with self.assertRaisesRegex(ControllerSafetyError,'collision_head_link_1'):check_controller_feedback(state,1.1)
        state['avoidance']=(1.,{'other_arm':False,'objects':[]})
        check_controller_feedback(state,1.1)
        with self.assertRaisesRegex(ControllerSafetyError,'stale'):check_controller_feedback(state,1.6)
        state['collision']=(1.,True)
        with self.assertRaisesRegex(ControllerSafetyError,'contact'):check_controller_feedback(state,1.1)

    def test_continuous_execution_reports_waits_and_preserves_limits(self):
        robot=object.__new__(Baxter);robot.cfg={'max_translation_m':.12,'joint_speed_rad_s':.06}
        for name in ('ready','check_controller_feedback','inside','guard_pose','audit'):
            setattr(robot,name,Mock())
        robot.limb=Mock();clock=[0.];measured={'j':0.};commands=[]
        robot.joints=lambda:dict(measured);robot.endpoint_position=lambda:[0.,0.,0.]
        robot.arrival_error=lambda target:(abs(.2-measured['j']),0.)
        def follow(q):
            measured.update(q);commands.append((clock[0],q['j']))
        robot.limb.set_joint_positions.side_effect=follow
        path=[({'j':i/1000},Mock()) for i in range(1,201)]
        with patch('cix_harness.baxter.time.monotonic',side_effect=lambda:clock[0]), \
             patch('cix_harness.baxter.time.sleep',side_effect=lambda dt:clock.__setitem__(0,clock[0]+dt)):
            result=robot.move_segment({'j':0.},path,[0.,0.,0.],continuous=True,motion_phase='transit')
        self.assertEqual(measured['j'],.2)
        self.assertLess(result['elapsed_s'],result['legacy_schedule_seconds']/2)
        self.assertLessEqual(result['peak_command_speed_rad_s'],.06001)
        self.assertGreaterEqual(result['settling_s'],.2)
        self.assertGreater(result['loop_timings_s']['sleep'],0.)
        self.assertEqual(result['motion_phase'],'transit')
        velocities=[(b-a)/(tb-ta) for (ta,a),(tb,b) in zip(commands,commands[1:])]
        self.assertLessEqual(max(abs(b-a)/.01 for a,b in zip(velocities,velocities[1:])),.1201)

    def test_factory_fault_sends_no_motor_command(self):
        robot=object.__new__(Baxter);robot.cfg={'max_translation_m':.12};robot.limb=Mock()
        robot.ready=Mock();robot.audit=Mock();robot.controller_feedback={}
        with self.assertRaisesRegex(ControllerSafetyError,'telemetry'):
            robot.move_segment({'j':0.},[({'j':.1},Mock())],[0.,0.,0.],continuous=True)
        robot.limb.set_joint_positions.assert_not_called()


if __name__=='__main__':unittest.main()
