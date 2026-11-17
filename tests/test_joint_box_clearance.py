"""A curved joint path can leave every small box along its diagonal chord."""
import unittest

from cix_harness.collision import ArmCollisionError,Capsule,InterArmGuard,check_capsules
from tests.test_dual_bridge import configuration


class PointModel:
    def __init__(self,arm):self.arm=arm
    def capsules(self,joints,holding=False):
        point=(joints['a'],joints['b'],0.) if self.arm=='left' else (.025,.075,0.)
        return [Capsule(self.arm+'_tool',point,point,.002)]
    def capsule_displacement_bounds(self,deltas):
        return {self.arm+'_tool':sum(abs(v) for v in deltas.values()) if self.arm=='left' else 0.}


class JointBoxClearance(unittest.TestCase):
    def test_independent_joint_phases_cannot_hide_between_diagonal_subboxes(self):
        cfg=configuration()['inter_arm_clearance'];cfg['minimum_distance_m']=.005
        models={a:PointModel(a) for a in ('left','right')}
        obstacle=models['right'].capsules({'peer':0.})
        # All five old chord-subdivision boxes pass despite the obstacle being
        # inside the original joint rectangle.
        for i in range(5):
            centre=(i+.5)*.02
            check_capsules(models['left'].capsules({'a':centre,'b':centre}),obstacle,.005,
                           active_padding={'left_tool':.02})
        # q_a=.1*t^2 and q_b=.1*(2*t-t^2) are both monotone on [0,1],
        # stay within the original knots, and hit the obstacle at t=.5.
        with self.assertRaises(ArmCollisionError):
            check_capsules(models['left'].capsules({'a':.025,'b':.075}),obstacle,.005)
        peer={'joint_names':['peer'],'joint_rad':[0.],'gripping':False,'gripper_percent':100.}
        guard=InterArmGuard(cfg,models)
        with self.assertRaises(ArmCollisionError):
            guard.path_clearance('left',{'a':0.,'b':0.},[[({'a':.1,'b':.1},None)]],peer,peer)
