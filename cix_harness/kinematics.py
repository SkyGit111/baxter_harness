"""Locally seeded, joint-limited KDL IK; no robot commands in this module."""
import math


class ContinuousIK:
    def __init__(self, arm):
        # Noetic's Python 3 parser must precede the obsolete Python 2 copy in
        # this laboratory's Baxter workspace. Do not change the system install.
        import sys
        from pathlib import Path
        noetic = '/opt/ros/noetic/lib/python3/dist-packages'
        if Path(noetic, 'urdf_parser_py').is_dir() and noetic not in sys.path[:1]:
            sys.path.insert(0, noetic)
        import PyKDL as kdl
        from urdf_parser_py.urdf import URDF
        from .upstream.kdl_parser import kdl_tree_from_urdf_model
        self.kdl = kdl
        robot = URDF.from_parameter_server(key='robot_description')
        chain = kdl_tree_from_urdf_model(robot).getChain('base', arm+'_gripper')
        self.names = [chain.getSegment(i).getJoint().getName() for i in range(chain.getNrOfSegments())
                      if chain.getSegment(i).getJoint().getType() != getattr(kdl.Joint, 'None')]
        if len(self.names) != 7:
            raise RuntimeError('Expected seven moving joints in the Baxter gripper chain')
        self.lower, self.upper = kdl.JntArray(7), kdl.JntArray(7)
        for i,name in enumerate(self.names):
            limits = robot.joint_map[name].limit
            self.lower[i], self.upper[i] = limits.lower, limits.upper
        self.fk = kdl.ChainFkSolverPos_recursive(chain)
        self.velocity = kdl.ChainIkSolverVel_pinv(chain)
        self.solver = kdl.ChainIkSolverPos_NR_JL(chain,self.lower,self.upper,self.fk,self.velocity,150,1e-6)

    def frame(self, position, orientation):
        k = self.kdl
        return k.Frame(k.Rotation.Quaternion(orientation.x,orientation.y,orientation.z,orientation.w),
                       k.Vector(position.x,position.y,position.z))

    def solve(self, target, current, observed):
        k = self.kdl
        if set(current) != set(self.names):
            raise ValueError('KDL chain and measured joint names differ')
        seed = k.JntArray(7)
        for i,name in enumerate(self.names):
            seed[i] = current[name]
            if not self.lower[i] <= seed[i] <= self.upper[i]:
                raise ValueError('Measured joint lies outside URDF limits')
        measured = k.Frame()
        if self.fk.JntToCart(seed,measured) < 0:
            raise RuntimeError('KDL current forward kinematics failed')
        error = k.diff(measured,self.frame(observed['position'],observed['orientation']))
        if error.vel.Norm() > .005 or error.rot.Norm() > .02:
            raise ValueError('KDL and measured endpoint disagree; check model and tool frame')
        goal = self.frame(target.pose.position,target.pose.orientation)
        output = k.JntArray(7)
        if self.solver.CartToJnt(seed,goal,output) < 0:
            raise ValueError('No continuous joint-limited KDL solution')
        for i in range(7):
            if not math.isfinite(output[i]) or not self.lower[i] <= output[i] <= self.upper[i]:
                raise ValueError('KDL output violates joint limits')
            if abs(output[i]-seed[i]) > .15:
                raise ValueError('KDL output exceeds 0.15 rad joint continuity limit')
        verified = k.Frame()
        if self.fk.JntToCart(output,verified) < 0:
            raise RuntimeError('KDL result forward kinematics failed')
        residual = k.diff(verified,goal)
        if residual.vel.Norm() > .001 or residual.rot.Norm() > .005:
            raise ValueError('KDL result does not reach the requested pose')
        return {name:float(output[i]) for i,name in enumerate(self.names)}
